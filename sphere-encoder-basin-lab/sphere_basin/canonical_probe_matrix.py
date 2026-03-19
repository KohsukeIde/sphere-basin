from __future__ import annotations

import argparse
import json
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

from .config import load_yaml
from .launch import _normalize_gpu_groups, _run_probe
from .paths import ensure_workspace_compat, find_job_dir


def _ckpt_epoch(ckpt_name: str) -> str:
    return ckpt_name[:-4] if ckpt_name.endswith('.pth') else ckpt_name


def _ckpt_fname(ckpt_name: str) -> str:
    return ckpt_name if ckpt_name.endswith('.pth') else f'{ckpt_name}.pth'


def _probe_output_path(
    job_path: Path,
    *,
    ckpt_epoch: str,
    cfg: float,
    cfg_position: str,
    use_sampling_scheduler: bool,
    cache_sampling_noise: bool,
    use_ema_model: bool,
    seed: int | None,
) -> Path:
    suffix = '' if seed is None else f'_seed={seed}'
    return (
        job_path
        / 'research'
        / (
            f'probe_ckpt={ckpt_epoch}'
            f'_cfg={cfg}-{cfg_position}'
            f'_sched={use_sampling_scheduler}'
            f'_cache={cache_sampling_noise}'
            f'_ema={use_ema_model}'
            f'{suffix}.json'
        )
    )


def _build_tasks(cfg: dict[str, Any], workspace_root: Path) -> list[dict[str, Any]]:
    probe_cfg = dict(cfg['probe'])
    regimes = list(probe_cfg.pop('regimes', []))
    tasks: list[dict[str, Any]] = []
    for job_spec in cfg['jobs']:
        job_dir = str(job_spec['job_dir'])
        job_path = find_job_dir(workspace_root, job_dir)
        for ckpt in job_spec['checkpoints']:
            ckpt_epoch = _ckpt_epoch(str(ckpt))
            ckpt_fname = _ckpt_fname(str(ckpt))
            ckpt_path = job_path / 'ckpt' / ckpt_fname
            if not ckpt_path.exists():
                raise FileNotFoundError(f'missing checkpoint for {job_dir}: {ckpt_path}')
            for regime in regimes:
                cache_sampling_noise = bool(regime['cache_sampling_noise'])
                use_sampling_scheduler = bool(regime['use_sampling_scheduler'])
                tasks.append(
                    {
                        'job_dir': job_dir,
                        'job_path': job_path,
                        'ckpt_epoch': ckpt_epoch,
                        'ckpt_fname': ckpt_fname,
                        'regime_name': str(regime['name']),
                        'cache_sampling_noise': cache_sampling_noise,
                        'use_sampling_scheduler': use_sampling_scheduler,
                        'expected_probe_path': _probe_output_path(
                            job_path,
                            ckpt_epoch=ckpt_epoch,
                            cfg=float(probe_cfg.get('cfg', 1.0)),
                            cfg_position=str(probe_cfg.get('cfg_position', 'combo')),
                            use_sampling_scheduler=use_sampling_scheduler,
                            cache_sampling_noise=cache_sampling_noise,
                            use_ema_model=bool(probe_cfg.get('use_ema_model', False)),
                            seed=(
                                None
                                if probe_cfg.get('seed') is None
                                else int(probe_cfg.get('seed', 0))
                            ),
                        ),
                        'probe_cfg': probe_cfg,
                    }
                )
    return tasks


def _task_status(task: dict[str, Any], *, state: str, probe_file: Path | None = None, note: str | None = None) -> dict[str, Any]:
    out = {
        'job_dir': task['job_dir'],
        'ckpt_epoch': task['ckpt_epoch'],
        'regime_name': task['regime_name'],
        'cache_sampling_noise': task['cache_sampling_noise'],
        'use_sampling_scheduler': task['use_sampling_scheduler'],
        'state': state,
        'expected_probe_path': str(task['expected_probe_path']),
    }
    if probe_file is not None:
        out['probe_file'] = str(probe_file)
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
    probe_file = Path(task['expected_probe_path'])
    if probe_file.exists() and not force:
        return _task_status(task, state='skipped', probe_file=probe_file, note='existing probe file')

    probe_cfg = dict(task['probe_cfg'])
    _run_probe(
        sphere_repo=sphere_repo,
        job_dir=task['job_dir'],
        dev_dir=dev_dir,
        dist_mode=dist_mode,
        forward_steps=list(probe_cfg.get('forward_steps', [1, 4])),
        num_prior_samples=int(probe_cfg.get('num_prior_samples', 4096)),
        num_data_samples=int(probe_cfg.get('num_data_samples', 4096)),
        batch_size_per_rank=int(probe_cfg.get('batch_size_per_rank', 64)),
        contraction_noise_scalers=list(probe_cfg.get('contraction_noise_scalers', [0.25, 0.5, 0.75, 1.0])),
        cfg=float(probe_cfg.get('cfg', 1.0)),
        cfg_position=str(probe_cfg.get('cfg_position', 'combo')),
        taus_deg=list(probe_cfg.get('taus_deg', [5.0, 10.0, 20.0, 30.0, 45.0, 60.0])),
        use_sampling_scheduler=bool(task['use_sampling_scheduler']),
        cache_sampling_noise=bool(task['cache_sampling_noise']),
        use_ema_model=bool(probe_cfg.get('use_ema_model', False)),
        num_workers=int(probe_cfg.get('num_workers', 4)),
        seed=int(probe_cfg.get('seed', 0)),
        ckpt_fname=str(task['ckpt_fname']),
        dry_run=dry_run,
        gpu_group=gpu_group,
    )
    if dry_run:
        return _task_status(task, state='dry_run', probe_file=probe_file)
    if not probe_file.exists():
        raise FileNotFoundError(f'expected probe output was not created: {probe_file}')
    return _task_status(task, state='completed', probe_file=probe_file)


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
    parser = argparse.ArgumentParser(description='Run a canonical post-hoc probe matrix.')
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
    out_path = out_dir / 'canonical_probe_matrix_manifest.json'
    payload = {
        'config_path': str(Path(args.config).resolve()),
        'sphere_repo': args.sphere_repo,
        'workspace_root': str(workspace_root),
        'task_count': len(tasks),
        'results': results,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(f'wrote canonical probe manifest to: {out_path}')


if __name__ == '__main__':
    main()
