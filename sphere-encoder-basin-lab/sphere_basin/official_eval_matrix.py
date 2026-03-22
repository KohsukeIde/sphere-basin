from __future__ import annotations

import argparse
import json
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

from .config import load_yaml
from .launch import _normalize_gpu_groups, _run_eval
from .parse_eval import parse_eval_dir
from .paths import ensure_workspace_compat, find_job_dir


def _ckpt_epoch(ckpt_name: str) -> str:
    return ckpt_name[:-4] if ckpt_name.endswith('.pth') else ckpt_name


def _has_eval_result(
    job_path: Path,
    *,
    ckpt_epoch: str,
    forward_steps: int,
    cfg: float,
    cfg_position: str,
    cache_sampling_noise: bool,
    use_sampling_scheduler: bool,
    use_ema_model: bool,
) -> bool:
    eval_dir = job_path / 'eval'
    if not eval_dir.exists():
        return False
    df = parse_eval_dir(eval_dir)
    if df.empty:
        return False
    out = df.copy()
    for col in ['forward_steps', 'cfg']:
        if col in out.columns:
            out[col] = out[col].astype(float)
    for col in ['cache_sampling_noise', 'use_sampling_scheduler', 'use_ema']:
        if col in out.columns:
            out[col] = out[col].astype(str).str.lower()
    if 'cfg_position' in out.columns:
        out['cfg_position'] = out['cfg_position'].astype(str)
    mask = (
        (out.get('ckpt_epoch') == ckpt_epoch)
        & (out.get('task_mode') == 'generation')
        & (out.get('forward_steps') == float(forward_steps))
        & (out.get('cfg') == float(cfg))
        & (out.get('cfg_position') == str(cfg_position))
        & (out.get('cache_sampling_noise') == str(cache_sampling_noise).lower())
        & (out.get('use_sampling_scheduler') == str(use_sampling_scheduler).lower())
        & (out.get('use_ema') == str(use_ema_model).lower())
    )
    return bool(mask.any())


def _build_tasks(cfg: dict[str, Any], workspace_root: Path) -> list[dict[str, Any]]:
    eval_cfg = dict(cfg['eval'])
    tasks: list[dict[str, Any]] = []
    for task in cfg['tasks']:
        job_dir = str(task['job_dir'])
        job_path = find_job_dir(workspace_root, job_dir)
        ckpt_fname = str(task['ckpt_fname'])
        ckpt_path = job_path / 'ckpt' / ckpt_fname
        if not ckpt_path.exists():
            raise FileNotFoundError(f'missing checkpoint for {job_dir}: {ckpt_path}')
        tasks.append(
            {
                'job_dir': job_dir,
                'job_path': job_path,
                'ckpt_fname': ckpt_fname,
                'ckpt_epoch': _ckpt_epoch(ckpt_fname),
                'regime_name': str(task['regime_name']),
                'cache_sampling_noise': bool(task['cache_sampling_noise']),
                'use_sampling_scheduler': bool(task['use_sampling_scheduler']),
                'eval_cfg': eval_cfg,
            }
        )
    return tasks


def _task_status(task: dict[str, Any], *, state: str, note: str | None = None) -> dict[str, Any]:
    out = {
        'job_dir': task['job_dir'],
        'ckpt_epoch': task['ckpt_epoch'],
        'ckpt_fname': task['ckpt_fname'],
        'regime_name': task['regime_name'],
        'cache_sampling_noise': task['cache_sampling_noise'],
        'use_sampling_scheduler': task['use_sampling_scheduler'],
        'state': state,
    }
    if note is not None:
        out['note'] = note
    return out


def _run_task(
    task: dict[str, Any],
    *,
    sphere_repo: str,
    dev_dir: str,
    dist_mode: str,
    dry_run: bool,
    force: bool,
    gpu_group: str | None,
) -> dict[str, Any]:
    eval_cfg = dict(task['eval_cfg'])
    forward_steps = list(eval_cfg.get('forward_steps', [4]))
    if not force:
        done = all(
            _has_eval_result(
                task['job_path'],
                ckpt_epoch=task['ckpt_epoch'],
                forward_steps=int(fwd),
                cfg=float(eval_cfg.get('cfg_min', 1.0)),
                cfg_position=str(eval_cfg.get('cfg_position', 'combo')),
                cache_sampling_noise=bool(task['cache_sampling_noise']),
                use_sampling_scheduler=bool(task['use_sampling_scheduler']),
                use_ema_model=bool(eval_cfg.get('use_ema_model', False)),
            )
            for fwd in forward_steps
        )
        if done:
            return _task_status(task, state='skipped', note='existing eval rows')

    _run_eval(
        sphere_repo=sphere_repo,
        job_dir=task['job_dir'],
        dev_dir=dev_dir,
        dist_mode=dist_mode,
        forward_steps=forward_steps,
        report_fid=list(eval_cfg.get('report_fid', ['rfid', 'gfid'])),
        use_cfg=bool(eval_cfg.get('use_cfg', False)),
        cfg_min=float(eval_cfg.get('cfg_min', 1.0)),
        cfg_max=float(eval_cfg.get('cfg_max', 1.0)),
        cfg_position=str(eval_cfg.get('cfg_position', 'combo')),
        rm_folder_after_eval=bool(eval_cfg.get('rm_folder_after_eval', True)),
        use_sampling_scheduler=bool(task['use_sampling_scheduler']),
        cache_sampling_noise=bool(task['cache_sampling_noise']),
        num_eval_samples=int(eval_cfg.get('num_eval_samples', 50000)),
        batch_size_per_rank=int(eval_cfg.get('batch_size_per_rank', 25)),
        use_ema_model=bool(eval_cfg.get('use_ema_model', False)),
        ckpt_fname=task['ckpt_fname'],
        dry_run=dry_run,
        gpu_group=gpu_group,
    )
    return _task_status(task, state='completed' if not dry_run else 'dry_run')


def _schedule_tasks(
    tasks: list[dict[str, Any]],
    *,
    sphere_repo: str,
    dev_dir: str,
    dist_mode: str,
    gpu_groups: list[str | None],
    dry_run: bool,
    force: bool,
) -> list[dict[str, Any]]:
    if not tasks:
        return []
    if len(gpu_groups) <= 1:
        gpu_group = gpu_groups[0] if gpu_groups else None
        return [
            _run_task(
                task,
                sphere_repo=sphere_repo,
                dev_dir=dev_dir,
                dist_mode=dist_mode,
                dry_run=dry_run,
                force=force,
                gpu_group=gpu_group,
            )
            for task in tasks
        ]

    pending = list(tasks)
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(gpu_groups)) as pool:
        future_to_group = {}
        for gpu_group in gpu_groups:
            if not pending:
                break
            task = pending.pop(0)
            future = pool.submit(
                _run_task,
                task,
                sphere_repo=sphere_repo,
                dev_dir=dev_dir,
                dist_mode=dist_mode,
                dry_run=dry_run,
                force=force,
                gpu_group=gpu_group,
            )
            future_to_group[future] = gpu_group

        while future_to_group:
            done, _ = wait(future_to_group.keys(), return_when=FIRST_COMPLETED)
            for future in done:
                gpu_group = future_to_group.pop(future)
                results.append(future.result())
                if pending:
                    task = pending.pop(0)
                    next_future = pool.submit(
                        _run_task,
                        task,
                        sphere_repo=sphere_repo,
                        dev_dir=dev_dir,
                        dist_mode=dist_mode,
                        dry_run=dry_run,
                        force=force,
                        gpu_group=gpu_group,
                    )
                    future_to_group[next_future] = gpu_group
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description='Run a matrix of official eval jobs.')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--sphere-repo', type=str, required=True)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    experiment = cfg['experiment']
    workspace_root = ensure_workspace_compat(experiment.get('dev_dir'))
    dist_mode = str(experiment.get('dist_mode', 'local'))
    gpu_groups = _normalize_gpu_groups(experiment.get('gpu_groups'))
    tasks = _build_tasks(cfg, workspace_root)
    results = _schedule_tasks(
        tasks,
        sphere_repo=args.sphere_repo,
        dev_dir=str(workspace_root),
        dist_mode=dist_mode,
        gpu_groups=gpu_groups,
        dry_run=args.dry_run,
        force=args.force,
    )

    out_dir = workspace_root / 'research_summary'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'official_eval_manifest.json'
    payload = {
        'config_path': str(Path(args.config).resolve()),
        'sphere_repo': args.sphere_repo,
        'workspace_root': str(workspace_root),
        'task_count': len(tasks),
        'results': results,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(f'wrote official eval manifest to: {out_path}')


if __name__ == '__main__':
    main()
