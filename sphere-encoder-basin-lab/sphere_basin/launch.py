from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

from .config import load_yaml
from .paths import (
    canonical_job_name,
    default_workspace_root,
    ensure_workspace_compat,
    project_root,
    resolve_dev_dir,
    slugify,
)


def _bool_to_str(v: Any) -> str:
    if isinstance(v, bool):
        return 'True' if v else 'False'
    return str(v)


def _kv_args(d: dict[str, Any]) -> list[str]:
    argv: list[str] = []
    for k, v in d.items():
        flag = f'--{k}'
        if isinstance(v, list):
            argv.append(flag)
            argv.extend(_bool_to_str(x) for x in v)
        else:
            argv.extend([flag, _bool_to_str(v)])
    return argv


def _format_gpu_group(gpu_group: Any) -> str | None:
    if gpu_group is None:
        return None
    if isinstance(gpu_group, str):
        text = gpu_group.strip()
        return text or None
    if isinstance(gpu_group, int):
        return str(gpu_group)
    if isinstance(gpu_group, (list, tuple)):
        vals = [str(x).strip() for x in gpu_group if str(x).strip()]
        return ','.join(vals) if vals else None
    return str(gpu_group)


def _normalize_gpu_groups(raw: Any) -> list[str | None]:
    if raw is None:
        return [None]
    if isinstance(raw, (str, int)):
        return [_format_gpu_group(raw)]
    groups = [_format_gpu_group(item) for item in raw]
    return groups or [None]


def _nproc_for_gpu_group(gpu_group: str | None) -> int:
    if gpu_group is not None:
        return len([x for x in gpu_group.split(',') if x.strip()])

    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible:
        return len([x for x in visible.split(',') if x.strip()])
    return 1


def _venv_bin_dir() -> Path | None:
    path = project_root() / '.venv' / 'bin'
    return path if path.exists() else None


def _build_subprocess_env(cwd: str, gpu_group: str | None = None) -> dict[str, str]:
    env = os.environ.copy()
    venv_bin = _venv_bin_dir()
    if venv_bin is not None:
        env['PATH'] = f'{venv_bin}{os.pathsep}{env.get("PATH", "")}'

    env.setdefault('CUDA_LAUNCH_BLOCKING', '1')
    env.setdefault('OMP_NUM_THREADS', '1')
    env.setdefault('PYTHONIOENCODING', 'UTF-8')
    env.setdefault('NCCL_IB_DISABLE', '1')

    py_path = env.get('PYTHONPATH', '')
    env['PYTHONPATH'] = f'{cwd}{os.pathsep}{py_path}' if py_path else cwd

    if gpu_group is not None:
        env['CUDA_VISIBLE_DEVICES'] = gpu_group

    return env


def _build_repo_cmd(
    entry_script: str,
    script_args: list[str],
    dist_mode: str = 'local',
    gpu_group: str | None = None,
) -> list[str]:
    if dist_mode == 'local':
        return [
            'torchrun',
            '--standalone',
            '--nnodes',
            '1',
            '--nproc_per_node',
            str(_nproc_for_gpu_group(gpu_group)),
            entry_script,
            *script_args,
        ]
    return ['./run.sh', '--dist-mode', dist_mode, entry_script, *script_args]


def _run(
    cmd: list[str],
    cwd: str,
    dry_run: bool = False,
    gpu_group: str | None = None,
) -> None:
    env = _build_subprocess_env(cwd=cwd, gpu_group=gpu_group)
    prefix = f'CUDA_VISIBLE_DEVICES={gpu_group} ' if gpu_group is not None else ''
    printable = prefix + ' '.join(shlex.quote(x) for x in cmd)
    print(f'>>> {printable}')
    if dry_run:
        return
    subprocess.run(cmd, cwd=cwd, check=True, env=env)


def _manifest_path(job_path: Path) -> Path:
    research_dir = job_path / 'research'
    research_dir.mkdir(parents=True, exist_ok=True)
    return research_dir / 'manifest.json'


def _write_manifest(job_path: Path, data: dict[str, Any]) -> None:
    path = _manifest_path(job_path)
    path.write_text(json.dumps(data, indent=2), encoding='utf-8')


def _rewrite_promoted_cfg(job_path: Path, workspace_root: Path) -> None:
    cfg_path = job_path / 'cfg.json'
    if not cfg_path.exists():
        return

    cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
    cfg['dev_dir'] = str(workspace_root)
    cfg['out_dir'] = 'experiments'
    cfg['vis_dir'] = str(job_path / 'vis')
    cfg['ckpt_dir'] = str(job_path / 'ckpt')
    cfg_path.write_text(json.dumps(cfg, indent=4), encoding='utf-8')


def _cleanup_empty_dirs(path: Path, stop_at: Path) -> None:
    cur = path
    while cur != stop_at and cur.exists():
        try:
            cur.rmdir()
        except OSError:
            break
        cur = cur.parent


def _promote_staged_job(
    workspace_root: Path,
    src_out_dir: str,
    default_job: str,
    target_job: str,
    dry_run: bool = False,
) -> Path:
    src = workspace_root / src_out_dir / default_job
    dst = workspace_root / 'experiments' / target_job
    if dst.exists():
        raise FileExistsError(f'target job already exists: {dst}')
    if not src.exists():
        raise FileNotFoundError(f'expected training output not found: {src}')
    print(f'>>> promote {src} -> {dst}')
    if dry_run:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    src.rename(dst)
    _cleanup_empty_dirs(src.parent, workspace_root)
    return dst


def _build_train_spec(
    *,
    train_common: dict[str, Any],
    dev_dir: str,
    run_slug: str,
    target_job: str,
    manifest: dict[str, Any],
    workspace_root: Path,
) -> dict[str, Any]:
    train_args = dict(train_common)
    train_args['dev_dir'] = dev_dir
    train_args['out_dir'] = str(Path('_staging') / run_slug)
    if bool(train_args.get('use_wandb')):
        train_args.setdefault('wandb_name', target_job)
    default_job = canonical_job_name(train_args)
    target_path = workspace_root / 'experiments' / target_job
    return {
        'train_args': train_args,
        'default_job': default_job,
        'target_job': target_job,
        'target_path': target_path,
        'manifest': manifest,
    }


def _execute_train_spec(
    spec: dict[str, Any],
    *,
    sphere_repo: str,
    workspace_root: Path,
    dist_mode: str,
    skip_existing: bool,
    dry_run: bool,
    gpu_group: str | None,
) -> str:
    target_job = spec['target_job']
    if skip_existing and spec['target_path'].exists():
        print(f'>>> skip existing: {target_job}')
        return target_job

    train_args = dict(spec['train_args'])
    _run(
        _build_repo_cmd(
            entry_script='train.py',
            script_args=_kv_args(train_args),
            dist_mode=dist_mode,
            gpu_group=gpu_group,
        ),
        cwd=sphere_repo,
        dry_run=dry_run,
        gpu_group=gpu_group,
    )
    if dry_run:
        return target_job

    final_path = _promote_staged_job(
        workspace_root=workspace_root,
        src_out_dir=train_args['out_dir'],
        default_job=spec['default_job'],
        target_job=target_job,
        dry_run=False,
    )
    _rewrite_promoted_cfg(final_path, workspace_root)
    _write_manifest(final_path, spec['manifest'])
    return target_job


def _schedule_train_specs(
    specs: list[dict[str, Any]],
    *,
    sphere_repo: str,
    workspace_root: Path,
    dist_mode: str,
    skip_existing: bool,
    dry_run: bool,
    gpu_groups: list[str | None],
) -> list[str]:
    if not specs:
        return []

    if len(gpu_groups) <= 1:
        gpu_group = gpu_groups[0] if gpu_groups else None
        return [
            _execute_train_spec(
                spec,
                sphere_repo=sphere_repo,
                workspace_root=workspace_root,
                dist_mode=dist_mode,
                skip_existing=skip_existing,
                dry_run=dry_run,
                gpu_group=gpu_group,
            )
            for spec in specs
        ]

    jobs = list(specs)
    results: list[str] = []
    with ThreadPoolExecutor(max_workers=len(gpu_groups)) as pool:
        future_to_group = {}
        for gpu_group in gpu_groups:
            if not jobs:
                break
            spec = jobs.pop(0)
            future = pool.submit(
                _execute_train_spec,
                spec,
                sphere_repo=sphere_repo,
                workspace_root=workspace_root,
                dist_mode=dist_mode,
                skip_existing=skip_existing,
                dry_run=dry_run,
                gpu_group=gpu_group,
            )
            future_to_group[future] = gpu_group

        while future_to_group:
            done, _ = wait(future_to_group.keys(), return_when=FIRST_COMPLETED)
            for future in done:
                gpu_group = future_to_group.pop(future)
                results.append(future.result())
                if jobs:
                    spec = jobs.pop(0)
                    next_future = pool.submit(
                        _execute_train_spec,
                        spec,
                        sphere_repo=sphere_repo,
                        workspace_root=workspace_root,
                        dist_mode=dist_mode,
                        skip_existing=skip_existing,
                        dry_run=dry_run,
                        gpu_group=gpu_group,
                    )
                    future_to_group[next_future] = gpu_group
    return results


def _build_alpha_specs(
    cfg: dict[str, Any],
    workspace_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    experiment = cfg['experiment']
    train_common = dict(cfg['train'])
    tag = slugify(experiment.get('tag', 'sweep'))
    dev_dir = str(resolve_dev_dir(experiment.get('dev_dir')))
    specs: list[dict[str, Any]] = []

    for alpha in list(cfg['sweep']['alpha_values']):
        target_job = f"{canonical_job_name({**train_common, 'dev_dir': dev_dir})}-a{alpha}-{tag}"
        manifest = {
            'sweep_name': experiment.get('name', 'alpha_sweep'),
            'tag': tag,
            'alpha': alpha,
            'variant': 'full',
            'job_dir': target_job,
        }
        specs.append(
            _build_train_spec(
                train_common={**train_common, 'noise_sigma_max_angle': alpha},
                dev_dir=dev_dir,
                run_slug=f'a{alpha}-{tag}',
                target_job=target_job,
                manifest=manifest,
                workspace_root=workspace_root,
            )
        )

    meta = {
        'dist_mode': experiment.get('dist_mode', 'local'),
        'skip_existing': bool(experiment.get('skip_existing', True)),
        'gpu_groups': _normalize_gpu_groups(experiment.get('gpu_groups')),
    }
    return specs, meta


def _build_loss_specs(
    cfg: dict[str, Any],
    workspace_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    experiment = cfg['experiment']
    train_common = dict(cfg['train'])
    tag = slugify(experiment.get('tag', 'ablation'))
    dev_dir = str(resolve_dev_dir(experiment.get('dev_dir')))
    specs: list[dict[str, Any]] = []

    for name, patch in dict(cfg['variants']).items():
        train_args = {**train_common, **patch}
        alpha = train_args.get('noise_sigma_max_angle', 'na')
        target_job = f"{canonical_job_name({**train_args, 'dev_dir': dev_dir})}-a{alpha}-{slugify(name)}-{tag}"
        manifest = {
            'sweep_name': experiment.get('name', 'loss_ablation'),
            'tag': tag,
            'alpha': alpha,
            'variant': name,
            'job_dir': target_job,
        }
        specs.append(
            _build_train_spec(
                train_common=train_args,
                dev_dir=dev_dir,
                run_slug=f'a{alpha}-{slugify(name)}-{tag}',
                target_job=target_job,
                manifest=manifest,
                workspace_root=workspace_root,
            )
        )

    meta = {
        'dist_mode': experiment.get('dist_mode', 'local'),
        'skip_existing': bool(experiment.get('skip_existing', True)),
        'gpu_groups': _normalize_gpu_groups(experiment.get('gpu_groups')),
    }
    return specs, meta


def _run_alpha_training(
    cfg: dict[str, Any],
    *,
    sphere_repo: str,
    dry_run: bool,
) -> list[str]:
    workspace_root = ensure_workspace_compat(cfg['experiment'].get('dev_dir'))
    specs, meta = _build_alpha_specs(cfg, workspace_root)
    return _schedule_train_specs(
        specs,
        sphere_repo=sphere_repo,
        workspace_root=workspace_root,
        dist_mode=meta['dist_mode'],
        skip_existing=meta['skip_existing'],
        dry_run=dry_run,
        gpu_groups=meta['gpu_groups'],
    )


def _run_loss_training(
    cfg: dict[str, Any],
    *,
    sphere_repo: str,
    dry_run: bool,
) -> list[str]:
    workspace_root = ensure_workspace_compat(cfg['experiment'].get('dev_dir'))
    specs, meta = _build_loss_specs(cfg, workspace_root)
    return _schedule_train_specs(
        specs,
        sphere_repo=sphere_repo,
        workspace_root=workspace_root,
        dist_mode=meta['dist_mode'],
        skip_existing=meta['skip_existing'],
        dry_run=dry_run,
        gpu_groups=meta['gpu_groups'],
    )


def cmd_train_alpha(args: argparse.Namespace) -> None:
    cfg = load_yaml(args.config)
    jobs = _run_alpha_training(cfg, sphere_repo=args.sphere_repo, dry_run=args.dry_run)
    print(f'>>> alpha jobs: {", ".join(jobs)}')


def cmd_train_loss(args: argparse.Namespace) -> None:
    cfg = load_yaml(args.config)
    jobs = _run_loss_training(cfg, sphere_repo=args.sphere_repo, dry_run=args.dry_run)
    print(f'>>> loss jobs: {", ".join(jobs)}')


def _run_eval(
    *,
    sphere_repo: str,
    job_dir: str,
    dev_dir: str,
    dist_mode: str,
    forward_steps: list[int],
    report_fid: list[str],
    use_cfg: bool,
    cfg_min: float,
    cfg_max: float,
    cfg_position: str,
    rm_folder_after_eval: bool,
    use_sampling_scheduler: bool,
    cache_sampling_noise: bool,
    num_eval_samples: int,
    batch_size_per_rank: int,
    use_ema_model: bool,
    ckpt_fname: str | None,
    dry_run: bool,
    gpu_group: str | None,
) -> None:
    cmd = _build_repo_cmd(
        entry_script='eval.py',
        script_args=[
            '--dev_dir',
            dev_dir,
            '--job_dir',
            job_dir,
            '--forward_steps',
            *[str(x) for x in forward_steps],
            '--report_fid',
            *report_fid,
            '--use_cfg',
            _bool_to_str(use_cfg),
            '--cfg_min',
            str(cfg_min),
            '--cfg_max',
            str(cfg_max),
            '--cfg_position',
            cfg_position,
            '--rm_folder_after_eval',
            _bool_to_str(rm_folder_after_eval),
            '--use_sampling_scheduler',
            _bool_to_str(use_sampling_scheduler),
            '--cache_sampling_noise',
            _bool_to_str(cache_sampling_noise),
            '--num_eval_samples',
            str(num_eval_samples),
            '--batch_size_per_rank',
            str(batch_size_per_rank),
            '--use_ema_model',
            _bool_to_str(use_ema_model),
            *([] if ckpt_fname is None else ['--ckpt_fname', ckpt_fname]),
        ],
        dist_mode=dist_mode,
        gpu_group=gpu_group,
    )
    _run(cmd, cwd=sphere_repo, dry_run=dry_run, gpu_group=gpu_group)


def _run_probe(
    *,
    sphere_repo: str,
    job_dir: str,
    dev_dir: str,
    dist_mode: str,
    forward_steps: list[int],
    num_prior_samples: int,
    num_data_samples: int,
    batch_size_per_rank: int,
    contraction_noise_scalers: list[float],
    use_sampling_scheduler: bool,
    cache_sampling_noise: bool,
    use_ema_model: bool,
    num_workers: int,
    ckpt_fname: str | None,
    dry_run: bool,
    gpu_group: str | None,
) -> None:
    cmd = _build_repo_cmd(
        entry_script='research/probe_projector.py',
        script_args=[
            '--dev_dir',
            dev_dir,
            '--job_dir',
            job_dir,
            '--forward_steps',
            *[str(x) for x in forward_steps],
            '--num_prior_samples',
            str(num_prior_samples),
            '--num_data_samples',
            str(num_data_samples),
            '--batch_size_per_rank',
            str(batch_size_per_rank),
            '--contraction_noise_scalers',
            *[str(x) for x in contraction_noise_scalers],
            '--use_sampling_scheduler',
            _bool_to_str(use_sampling_scheduler),
            '--cache_sampling_noise',
            _bool_to_str(cache_sampling_noise),
            '--use_ema_model',
            _bool_to_str(use_ema_model),
            '--num_workers',
            str(num_workers),
            *([] if ckpt_fname is None else ['--ckpt_fname', ckpt_fname]),
        ],
        dist_mode=dist_mode,
        gpu_group=gpu_group,
    )
    _run(cmd, cwd=sphere_repo, dry_run=dry_run, gpu_group=gpu_group)


def _normalize_analysis_gpu_groups(
    analysis: dict[str, Any],
    experiment: dict[str, Any],
) -> list[str | None]:
    if 'gpu_groups' in analysis:
        return _normalize_gpu_groups(analysis.get('gpu_groups'))
    if 'gpu_group' in analysis:
        return _normalize_gpu_groups(analysis.get('gpu_group'))
    return _normalize_gpu_groups(experiment.get('gpu_groups'))


def _run_analysis_job(
    *,
    sphere_repo: str,
    job_dir: str,
    dev_dir: str,
    dist_mode: str,
    analysis: dict[str, Any],
    dry_run: bool,
    gpu_group: str | None,
) -> str:
    _run_eval(
        sphere_repo=sphere_repo,
        job_dir=job_dir,
        dev_dir=dev_dir,
        dist_mode=dist_mode,
        forward_steps=list(analysis.get('eval_forward_steps', [1, 4])),
        report_fid=list(analysis.get('report_fid', ['rfid', 'gfid'])),
        use_cfg=bool(analysis.get('use_cfg', False)),
        cfg_min=float(analysis.get('cfg_min', 1.0)),
        cfg_max=float(analysis.get('cfg_max', 1.0)),
        cfg_position=str(analysis.get('cfg_position', 'combo')),
        rm_folder_after_eval=bool(analysis.get('rm_folder_after_eval', True)),
        use_sampling_scheduler=bool(analysis.get('use_sampling_scheduler', False)),
        cache_sampling_noise=bool(analysis.get('cache_sampling_noise', False)),
        num_eval_samples=int(analysis.get('num_eval_samples', 10000)),
        batch_size_per_rank=int(analysis.get('eval_batch_size_per_rank', 25)),
        use_ema_model=bool(analysis.get('use_ema_model', False)),
        ckpt_fname=analysis.get('ckpt_fname'),
        dry_run=dry_run,
        gpu_group=gpu_group,
    )
    _run_probe(
        sphere_repo=sphere_repo,
        job_dir=job_dir,
        dev_dir=dev_dir,
        dist_mode=dist_mode,
        forward_steps=list(analysis.get('probe_forward_steps', [1, 4])),
        num_prior_samples=int(analysis.get('num_prior_samples', 4096)),
        num_data_samples=int(analysis.get('num_data_samples', 4096)),
        batch_size_per_rank=int(analysis.get('probe_batch_size_per_rank', 64)),
        contraction_noise_scalers=list(analysis.get('contraction_noise_scalers', [0.25, 0.5, 0.75, 1.0])),
        use_sampling_scheduler=bool(analysis.get('use_sampling_scheduler', False)),
        cache_sampling_noise=bool(analysis.get('cache_sampling_noise', False)),
        use_ema_model=bool(analysis.get('use_ema_model', False)),
        num_workers=int(analysis.get('num_workers', 4)),
        ckpt_fname=analysis.get('ckpt_fname'),
        dry_run=dry_run,
        gpu_group=gpu_group,
    )
    return job_dir


def _schedule_analysis_jobs(
    jobs: list[str],
    *,
    sphere_repo: str,
    dev_dir: str,
    dist_mode: str,
    analysis: dict[str, Any],
    dry_run: bool,
    gpu_groups: list[str | None],
) -> list[str]:
    if not jobs:
        return []

    if len(gpu_groups) <= 1:
        gpu_group = gpu_groups[0] if gpu_groups else None
        return [
            _run_analysis_job(
                sphere_repo=sphere_repo,
                job_dir=job_dir,
                dev_dir=dev_dir,
                dist_mode=dist_mode,
                analysis=analysis,
                dry_run=dry_run,
                gpu_group=gpu_group,
            )
            for job_dir in jobs
        ]

    pending_jobs = list(jobs)
    results: list[str] = []
    with ThreadPoolExecutor(max_workers=len(gpu_groups)) as pool:
        future_to_group = {}
        for gpu_group in gpu_groups:
            if not pending_jobs:
                break
            job_dir = pending_jobs.pop(0)
            future = pool.submit(
                _run_analysis_job,
                sphere_repo=sphere_repo,
                job_dir=job_dir,
                dev_dir=dev_dir,
                dist_mode=dist_mode,
                analysis=analysis,
                dry_run=dry_run,
                gpu_group=gpu_group,
            )
            future_to_group[future] = gpu_group

        while future_to_group:
            done, _ = wait(future_to_group.keys(), return_when=FIRST_COMPLETED)
            for future in done:
                gpu_group = future_to_group.pop(future)
                results.append(future.result())
                if pending_jobs:
                    job_dir = pending_jobs.pop(0)
                    next_future = pool.submit(
                        _run_analysis_job,
                        sphere_repo=sphere_repo,
                        job_dir=job_dir,
                        dev_dir=dev_dir,
                        dist_mode=dist_mode,
                        analysis=analysis,
                        dry_run=dry_run,
                        gpu_group=gpu_group,
                    )
                    future_to_group[next_future] = gpu_group
    return results


def cmd_official_eval(args: argparse.Namespace) -> None:
    _run_eval(
        sphere_repo=args.sphere_repo,
        job_dir=args.job_dir,
        dev_dir=str(ensure_workspace_compat(args.dev_dir)),
        dist_mode=args.dist_mode,
        forward_steps=args.forward_steps,
        report_fid=args.report_fid,
        use_cfg=args.use_cfg,
        cfg_min=args.cfg_min,
        cfg_max=args.cfg_max,
        cfg_position=args.cfg_position,
        rm_folder_after_eval=args.rm_folder_after_eval,
        use_sampling_scheduler=args.use_sampling_scheduler,
        cache_sampling_noise=args.cache_sampling_noise,
        num_eval_samples=args.num_eval_samples,
        batch_size_per_rank=args.batch_size_per_rank,
        use_ema_model=args.use_ema_model,
        ckpt_fname=args.ckpt_fname,
        dry_run=args.dry_run,
        gpu_group=_format_gpu_group(args.gpu_group),
    )


def cmd_probe(args: argparse.Namespace) -> None:
    _run_probe(
        sphere_repo=args.sphere_repo,
        job_dir=args.job_dir,
        dev_dir=str(ensure_workspace_compat(args.dev_dir)),
        dist_mode=args.dist_mode,
        forward_steps=args.forward_steps,
        num_prior_samples=args.num_prior_samples,
        num_data_samples=args.num_data_samples,
        batch_size_per_rank=args.batch_size_per_rank,
        contraction_noise_scalers=args.contraction_noise_scalers,
        use_sampling_scheduler=args.use_sampling_scheduler,
        cache_sampling_noise=args.cache_sampling_noise,
        use_ema_model=args.use_ema_model,
        num_workers=args.num_workers,
        ckpt_fname=args.ckpt_fname,
        dry_run=args.dry_run,
        gpu_group=_format_gpu_group(args.gpu_group),
    )


def cmd_run_e1(args: argparse.Namespace) -> None:
    cfg = load_yaml(args.config)
    experiment = cfg['experiment']
    analysis = cfg.get('analysis', {})
    workspace_root = ensure_workspace_compat(experiment.get('dev_dir'))

    if args.skip_train:
        specs, _ = _build_alpha_specs(cfg, workspace_root)
        jobs = [spec['target_job'] for spec in specs]
    else:
        jobs = _run_alpha_training(cfg, sphere_repo=args.sphere_repo, dry_run=args.dry_run)

    analysis_gpu_groups = _normalize_analysis_gpu_groups(analysis, experiment)
    dist_mode = experiment.get('dist_mode', 'local')
    dev_dir = str(workspace_root)

    _schedule_analysis_jobs(
        jobs,
        sphere_repo=args.sphere_repo,
        dev_dir=dev_dir,
        dist_mode=dist_mode,
        analysis=analysis,
        dry_run=args.dry_run,
        gpu_groups=analysis_gpu_groups,
    )

    aggregate_cmd = [sys.executable, '-m', 'sphere_basin.aggregate', '--workspace', str(workspace_root)]
    plot_cmd = [
        sys.executable,
        '-m',
        'sphere_basin.plot_phase_diagram',
        '--csv',
        str(workspace_root / 'research_summary' / 'summary_generation.csv'),
        '--forward-steps',
        str(analysis.get('plot_forward_steps', 4)),
        '--tau-deg',
        str(analysis.get('plot_tau_deg', 2.0)),
    ]
    _run(aggregate_cmd, cwd=str(project_root()), dry_run=args.dry_run)
    _run(plot_cmd, cwd=str(project_root()), dry_run=args.dry_run)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Sphere Basin launcher')
    sub = parser.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('train-alpha')
    p.add_argument('--config', type=str, required=True)
    p.add_argument('--sphere-repo', type=str, required=True)
    p.add_argument('--dry-run', action='store_true')
    p.set_defaults(func=cmd_train_alpha)

    p = sub.add_parser('train-loss')
    p.add_argument('--config', type=str, required=True)
    p.add_argument('--sphere-repo', type=str, required=True)
    p.add_argument('--dry-run', action='store_true')
    p.set_defaults(func=cmd_train_loss)

    p = sub.add_parser('official-eval')
    p.add_argument('--sphere-repo', type=str, required=True)
    p.add_argument('--job-dir', type=str, required=True)
    p.add_argument('--dev-dir', type=str, default=str(default_workspace_root()))
    p.add_argument('--dist-mode', type=str, default='local')
    p.add_argument('--gpu-group', type=str, default=None)
    p.add_argument('--forward-steps', type=int, nargs='+', default=[1, 4])
    p.add_argument('--report-fid', type=str, nargs='+', default=['rfid', 'gfid'])
    p.add_argument('--use-cfg', action='store_true')
    p.add_argument('--cfg-min', type=float, default=1.0)
    p.add_argument('--cfg-max', type=float, default=1.0)
    p.add_argument('--cfg-position', type=str, default='combo')
    p.add_argument('--rm-folder-after-eval', action='store_true')
    p.add_argument('--use-sampling-scheduler', action='store_true')
    p.add_argument('--cache-sampling-noise', action='store_true')
    p.add_argument('--num-eval-samples', type=int, default=10000)
    p.add_argument('--batch-size-per-rank', type=int, default=25)
    p.add_argument('--use-ema-model', action='store_true')
    p.add_argument('--ckpt-fname', type=str, default=None)
    p.add_argument('--dry-run', action='store_true')
    p.set_defaults(func=cmd_official_eval)

    p = sub.add_parser('probe')
    p.add_argument('--sphere-repo', type=str, required=True)
    p.add_argument('--job-dir', type=str, required=True)
    p.add_argument('--dev-dir', type=str, default=str(default_workspace_root()))
    p.add_argument('--dist-mode', type=str, default='local')
    p.add_argument('--gpu-group', type=str, default=None)
    p.add_argument('--forward-steps', type=int, nargs='+', default=[1, 4])
    p.add_argument('--num-prior-samples', type=int, default=4096)
    p.add_argument('--num-data-samples', type=int, default=4096)
    p.add_argument('--batch-size-per-rank', type=int, default=64)
    p.add_argument('--contraction-noise-scalers', type=float, nargs='+', default=[0.25, 0.5, 0.75, 1.0])
    p.add_argument('--use-sampling-scheduler', action='store_true')
    p.add_argument('--cache-sampling-noise', action='store_true')
    p.add_argument('--use-ema-model', action='store_true')
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--ckpt-fname', type=str, default=None)
    p.add_argument('--dry-run', action='store_true')
    p.set_defaults(func=cmd_probe)

    p = sub.add_parser('run-e1')
    p.add_argument('--config', type=str, required=True)
    p.add_argument('--sphere-repo', type=str, required=True)
    p.add_argument('--skip-train', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    p.set_defaults(func=cmd_run_e1)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
