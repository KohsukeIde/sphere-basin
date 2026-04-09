from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml

from .config import load_yaml
from .launch import (
    _build_train_spec,
    _build_loss_specs,
    _normalize_gpu_groups,
    _run,
    _schedule_train_specs,
)
from .paths import ensure_workspace_compat, find_job_dir, project_root, resolve_dev_dir


SUMMARY_FILES_TO_RESTORE = [
    'canonical_master.csv',
    'canonical_prior_long.csv',
    'canonical_contraction_long.csv',
    'canonical_probe_matrix_manifest.json',
    'official_eval_master.csv',
    'official_eval_manifest.json',
]


def _write_status(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')


def _backup_summary_files(workspace_root: Path) -> tuple[Path, list[str]]:
    summary_root = workspace_root / 'research_summary'
    backup_root = summary_root / '_imagenet100_followup_backup'
    backup_root.mkdir(parents=True, exist_ok=True)
    backed_up: list[str] = []
    for rel in SUMMARY_FILES_TO_RESTORE:
        src = summary_root / rel
        if src.exists():
            dst = backup_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            backed_up.append(rel)
    return backup_root, backed_up


def _restore_summary_files(workspace_root: Path, backup_root: Path, backed_up: list[str]) -> None:
    summary_root = workspace_root / 'research_summary'
    for rel in backed_up:
        src = backup_root / rel
        dst = summary_root / rel
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def _make_resume_ckpt(base_ckpt: Path, out_path: Path, next_epoch: int) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(base_ckpt, map_location='cpu')
    ckpt['epoch'] = int(next_epoch)
    torch.save(ckpt, out_path)
    return out_path


def _resume_full_a85(
    *,
    cfg: dict[str, Any],
    sphere_repo: str,
    workspace_root: Path,
    dry_run: bool,
) -> str:
    exp_cfg = cfg['experiment']
    resume_cfg = cfg['resume_full']
    train_common = dict(cfg['train_common'])
    source_job_dir = str(resume_cfg['job_dir'])
    target_job_dir = str(resume_cfg['target_job'])
    job_path = find_job_dir(workspace_root, source_job_dir)
    target_ckpt_epoch = str(resume_cfg.get('target_ckpt_epoch', 'ep0049'))
    target_ckpt = workspace_root / 'experiments' / target_job_dir / 'ckpt' / f'{target_ckpt_epoch}.pth'
    if bool(exp_cfg.get('skip_existing', True)) and target_ckpt.exists():
        print(f'>>> skip existing resumed checkpoint: {target_ckpt}')
        return target_job_dir

    resume_from_name = str(resume_cfg.get('resume_ckpt', 'ep0024.pth'))
    base_ckpt = job_path / 'ckpt' / resume_from_name
    if not base_ckpt.exists():
        raise FileNotFoundError(f'missing resume checkpoint: {base_ckpt}')

    next_epoch = int(base_ckpt.stem.replace('ep', '')) + 1
    patched_resume = (
        workspace_root
        / 'research_summary'
        / '_resume_ckpts'
        / f'{source_job_dir}-{base_ckpt.stem}-resume-next={next_epoch:04d}.pth'
    )
    if not dry_run:
        _make_resume_ckpt(base_ckpt, patched_resume, next_epoch=next_epoch)

    train_args = {
        **train_common,
        'noise_sigma_max_angle': int(resume_cfg['alpha']),
        'epochs': int(resume_cfg['target_epochs']),
        'wandb_group': str(resume_cfg.get('wandb_group', 'imagenet100_followup')),
        'init_from': 'resume',
        'auto_resume': False,
        'resume_from': str(patched_resume),
    }
    manifest = {
        'sweep_name': str(exp_cfg.get('name', 'imagenet100_followup_minimal')),
        'tag': 'followup',
        'alpha': int(resume_cfg['alpha']),
        'variant': 'full_resume',
        'job_dir': target_job_dir,
        'source_job_dir': source_job_dir,
        'source_ckpt': resume_from_name,
    }
    spec = _build_train_spec(
        train_common=train_args,
        dev_dir=str(resolve_dev_dir(exp_cfg.get('dev_dir'))),
        run_slug=str(resume_cfg.get('run_slug', 'a85-ep0049-followup')),
        target_job=target_job_dir,
        manifest=manifest,
        workspace_root=workspace_root,
    )
    jobs = _schedule_train_specs(
        [spec],
        sphere_repo=sphere_repo,
        workspace_root=workspace_root,
        dist_mode=str(exp_cfg.get('dist_mode', 'local')),
        skip_existing=bool(exp_cfg.get('skip_existing', True)),
        dry_run=dry_run,
        gpu_groups=_normalize_gpu_groups(exp_cfg.get('gpu_groups')),
        retry_attempts=int(exp_cfg.get('retry_attempts', 0)),
    )
    return jobs[0]


def _build_ablation_cfg(base_cfg: dict[str, Any], workspace_root: Path) -> Path:
    exp_cfg = base_cfg['experiment']
    ablation = base_cfg['ablation']
    payload = {
        'experiment': {
            'name': str(ablation.get('experiment_name', 'imagenet100_consistency_followup')),
            'dist_mode': str(exp_cfg.get('dist_mode', 'local')),
            'dev_dir': str(exp_cfg.get('dev_dir', 'workspace')),
            'tag': str(ablation.get('tag', 'followup')),
            'skip_existing': bool(exp_cfg.get('skip_existing', True)),
            'retry_attempts': int(exp_cfg.get('retry_attempts', 0)),
            'gpu_groups': exp_cfg.get('gpu_groups', [[0, 1]]),
        },
        'train': dict(ablation['train']),
        'variants': dict(ablation['variants']),
    }
    out = workspace_root / 'research_summary' / 'imagenet100_followup_loss_ablation.yaml'
    out.write_text(yaml.safe_dump(payload, sort_keys=False), encoding='utf-8')
    return out


def _refresh_dataset(*, cfg: dict[str, Any], sphere_repo: str, dry_run: bool) -> None:
    dataset_cfg = dict(cfg['dataset'])
    exp_cfg = dict(cfg['experiment'])
    workspace_root = ensure_workspace_compat(exp_cfg.get('dev_dir'))
    fid_stats_path = (
        workspace_root
        / 'fid_stats'
        / f"fid_stats_{dataset_cfg.get('fid_stats_mode', 'extr')}_{dataset_cfg.get('dataset_name', 'imagenet-100')}_{int(dataset_cfg.get('image_size', 160))}px.npz"
    )
    cmd = [
        sys.executable,
        '-m',
        'sphere_basin.prepare_imagenet100_cmc',
        '--source-root',
        str(dataset_cfg['source_root']),
        '--dev-dir',
        str(exp_cfg.get('dev_dir', 'workspace')),
        '--dataset-name',
        str(dataset_cfg.get('dataset_name', 'imagenet-100')),
        '--image-size',
        str(int(dataset_cfg.get('image_size', 160))),
        '--class-list-path',
        str(dataset_cfg.get('class_list_path', 'references/imagenet100_cmc.txt')),
        '--fid-stats-mode',
        str(dataset_cfg.get('fid_stats_mode', 'extr')),
        '--fid-stats-batch-size',
        str(int(dataset_cfg.get('fid_stats_batch_size', 64))),
        '--force',
    ]
    if fid_stats_path.exists():
        cmd.append('--skip-fid-stats')
    if bool(dataset_cfg.get('fid_stats_cuda', True)):
        cmd.append('--fid-stats-cuda')
    _run(cmd, cwd=str(project_root()), dry_run=dry_run, gpu_group=None)


def _build_canonical_cfg(*, cfg: dict[str, Any], workspace_root: Path, new_jobs: list[str]) -> Path:
    canonical = dict(cfg['canonical'])
    payload = {
        'experiment': {
            'name': 'imagenet100_followup_canonical',
            'dev_dir': 'workspace',
            'dist_mode': 'local',
            'gpu_groups': [[0, 1]],
        },
        'probe': {
            'cfg': canonical.get('cfg', 1.0),
            'cfg_position': canonical.get('cfg_position', 'combo'),
            'seed': int(canonical.get('seed', 0)),
            'taus_deg': list(canonical.get('taus_deg', [5.0, 10.0, 20.0, 30.0, 45.0, 60.0])),
            'forward_steps': list(canonical.get('forward_steps', [1, 4])),
            'num_prior_samples': int(canonical.get('num_prior_samples', 2048)),
            'num_data_samples': int(canonical.get('num_data_samples', 2048)),
            'batch_size_per_rank': int(canonical.get('batch_size_per_rank', 8)),
            'contraction_noise_scalers': list(canonical.get('contraction_noise_scalers', [0.25, 0.5, 0.75, 1.0])),
            'num_workers': int(canonical.get('num_workers', 4)),
            'use_ema_model': bool(canonical.get('use_ema_model', False)),
            'regimes': [
                {'name': 'independent-fixed', 'cache_sampling_noise': False, 'use_sampling_scheduler': False},
                {'name': 'shared-fixed', 'cache_sampling_noise': True, 'use_sampling_scheduler': False},
            ],
        },
        'master': {
            'contraction_noise_scaler': 1.0,
            'phase_ckpt_epoch': 'ep0049',
            'phase_regime_name': 'shared-fixed',
            'phase_forward_steps': 4,
            'phase_tau_deg': 60.0,
        },
        'jobs': [
            {'job_dir': 'sphere-small-small-imagenet-100-160px-a83-second-axis', 'checkpoints': ['ep0024']},
            {'job_dir': 'sphere-small-small-imagenet-100-160px-a85-second-axis', 'checkpoints': ['ep0024']},
            {'job_dir': str(cfg['resume_full']['target_job']), 'checkpoints': ['ep0049']},
        ] + [{'job_dir': job_dir, 'checkpoints': ['ep0024']} for job_dir in new_jobs],
    }
    out = workspace_root / 'research_summary' / 'imagenet100_followup_canonical.yaml'
    out.write_text(yaml.safe_dump(payload, sort_keys=False), encoding='utf-8')
    return out


def _build_official_eval_cfg(*, cfg: dict[str, Any], workspace_root: Path, new_jobs: list[str]) -> Path:
    eval_cfg = dict(cfg['official_eval'])
    a85_ablation = new_jobs[0]
    a85_followup = str(cfg['resume_full']['target_job'])
    payload = {
        'experiment': {
            'name': 'imagenet100_followup_official_eval',
            'dev_dir': 'workspace',
            'dist_mode': 'local',
            'gpu_groups': [[0, 1]],
        },
        'eval': {
            'forward_steps': list(eval_cfg.get('forward_steps', [4])),
            'report_fid': list(eval_cfg.get('report_fid', ['gfid'])),
            'fid_stats_used_from': str(eval_cfg.get('fid_stats_used_from', 'extr')),
            'use_cfg': bool(eval_cfg.get('use_cfg', False)),
            'cfg_min': float(eval_cfg.get('cfg_min', 1.0)),
            'cfg_max': float(eval_cfg.get('cfg_max', 1.0)),
            'cfg_position': str(eval_cfg.get('cfg_position', 'combo')),
            'rm_folder_after_eval': bool(eval_cfg.get('rm_folder_after_eval', True)),
            'num_eval_samples': int(eval_cfg.get('num_eval_samples', 10000)),
            'batch_size_per_rank': int(eval_cfg.get('batch_size_per_rank', 8)),
            'use_ema_model': bool(eval_cfg.get('use_ema_model', False)),
        },
        'tasks': [
            {
                'job_dir': 'sphere-small-small-imagenet-100-160px-a83-second-axis',
                'ckpt_fname': 'ep0024.pth',
                'regime_name': 'shared-fixed',
                'cache_sampling_noise': True,
                'use_sampling_scheduler': False,
            },
            {
                'job_dir': 'sphere-small-small-imagenet-100-160px-a85-second-axis',
                'ckpt_fname': 'ep0024.pth',
                'regime_name': 'shared-fixed',
                'cache_sampling_noise': True,
                'use_sampling_scheduler': False,
            },
            {
                'job_dir': a85_followup,
                'ckpt_fname': 'ep0049.pth',
                'regime_name': 'shared-fixed',
                'cache_sampling_noise': True,
                'use_sampling_scheduler': False,
            },
            {
                'job_dir': a85_followup,
                'ckpt_fname': 'ep0049.pth',
                'regime_name': 'independent-fixed',
                'cache_sampling_noise': False,
                'use_sampling_scheduler': False,
            },
            {
                'job_dir': a85_ablation,
                'ckpt_fname': 'ep0024.pth',
                'regime_name': 'shared-fixed',
                'cache_sampling_noise': True,
                'use_sampling_scheduler': False,
            },
        ],
    }
    out = workspace_root / 'research_summary' / 'imagenet100_followup_official_eval.yaml'
    out.write_text(yaml.safe_dump(payload, sort_keys=False), encoding='utf-8')
    return out


def _write_compare_csv(
    *,
    canonical_master_path: Path,
    official_eval_master_path: Path,
    out_path: Path,
    full_followup_job: str,
    new_jobs: list[str],
) -> None:
    canonical = pd.read_csv(canonical_master_path)
    official = pd.read_csv(official_eval_master_path)

    mapping = {
        ('sphere-small-small-imagenet-100-160px-a83-second-axis', 'ep0024', 'shared-fixed'): 'a83_full_ep24_shared',
        ('sphere-small-small-imagenet-100-160px-a85-second-axis', 'ep0024', 'shared-fixed'): 'a85_full_ep24_shared',
        (full_followup_job, 'ep0049', 'shared-fixed'): 'a85_full_ep49_shared',
        (full_followup_job, 'ep0049', 'independent-fixed'): 'a85_full_ep49_independent',
        (new_jobs[0], 'ep0024', 'shared-fixed'): 'a85_no_pix_con_ep24_shared',
    }

    canonical = canonical[
        (canonical['forward_steps'] == 4)
        & (canonical['tau_deg'] == 60.0)
    ].copy()
    canonical['variant_group'] = canonical.apply(
        lambda r: mapping.get((r['job_dir'], r['ckpt_epoch'], r['regime_name'])),
        axis=1,
    )
    canonical = canonical[canonical['variant_group'].notna()].copy()

    official = official[['job_dir', 'ckpt_epoch', 'regime_name', 'forward_steps', 'fid', 'isc_mean', 'isc_std']].copy()
    official['variant_group'] = official.apply(
        lambda r: mapping.get((r['job_dir'], r['ckpt_epoch'], r['regime_name'])),
        axis=1,
    )
    official = official[
        (official['forward_steps'] == 4)
        & (official['variant_group'].notna())
    ].rename(
        columns={
            'fid': 'official_fid',
            'isc_mean': 'official_isc_mean',
            'isc_std': 'official_isc_std',
        }
    )

    merged = canonical.merge(
        official[['job_dir', 'ckpt_epoch', 'regime_name', 'official_fid', 'official_isc_mean', 'official_isc_std']],
        on=['job_dir', 'ckpt_epoch', 'regime_name'],
        how='left',
    )
    merged = merged.sort_values('variant_group')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description='Run minimal ImageNet-100 follow-up for accept-only second-axis decisions.')
    parser.add_argument(
        '--config',
        type=str,
        default=str(project_root() / 'configs' / 'imagenet100_followup_minimal.yaml'),
    )
    parser.add_argument('--sphere-repo', type=str, required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    workspace_root = ensure_workspace_compat(cfg['experiment'].get('dev_dir'))
    status_path = workspace_root / 'research_summary' / 'imagenet100_followup_status.json'
    backup_root, backed_up = _backup_summary_files(workspace_root)

    status: dict[str, Any] = {
        'config': str(Path(args.config).resolve()),
        'sphere_repo': args.sphere_repo,
        'dry_run': args.dry_run,
        'state': 'running',
        'steps': [],
        'backed_up_summary_files': backed_up,
    }
    _write_status(status_path, status)

    try:
        status['current_step'] = 'refresh_dataset'
        status['steps'].append({'name': 'refresh_dataset', 'state': 'running'})
        _write_status(status_path, status)
        _refresh_dataset(cfg=cfg, sphere_repo=args.sphere_repo, dry_run=args.dry_run)
        status['steps'][-1]['state'] = 'completed'
        _write_status(status_path, status)

        status['current_step'] = 'resume_a85_full'
        status['steps'].append({'name': 'resume_a85_full', 'state': 'running'})
        _write_status(status_path, status)
        resumed_job = _resume_full_a85(
            cfg=cfg,
            sphere_repo=args.sphere_repo,
            workspace_root=workspace_root,
            dry_run=args.dry_run,
        )
        status['steps'][-1]['state'] = 'completed'
        status['resumed_job'] = resumed_job
        _write_status(status_path, status)

        status['current_step'] = 'train_a85_no_pix_con'
        status['steps'].append({'name': 'train_a85_no_pix_con', 'state': 'running'})
        _write_status(status_path, status)
        ablation_cfg = _build_ablation_cfg(cfg, workspace_root)
        status['ablation_config'] = str(ablation_cfg)
        _write_status(status_path, status)
        loss_cfg = load_yaml(ablation_cfg)
        specs, meta = _build_loss_specs(loss_cfg, workspace_root)
        new_jobs = [spec['target_job'] for spec in specs]
        trained_jobs = _schedule_train_specs(
            specs,
            sphere_repo=args.sphere_repo,
            workspace_root=workspace_root,
            dist_mode=meta['dist_mode'],
            skip_existing=meta['skip_existing'],
            dry_run=args.dry_run,
            gpu_groups=meta['gpu_groups'],
            retry_attempts=meta['retry_attempts'],
        )
        status['steps'][-1]['state'] = 'completed'
        status['new_jobs'] = new_jobs
        status['trained_jobs'] = trained_jobs
        _write_status(status_path, status)

        status['current_step'] = 'canonical_probe'
        status['steps'].append({'name': 'canonical_probe', 'state': 'running'})
        canonical_cfg = _build_canonical_cfg(cfg=cfg, workspace_root=workspace_root, new_jobs=new_jobs)
        status['canonical_config'] = str(canonical_cfg)
        _write_status(status_path, status)
        _run(
            [sys.executable, '-m', 'sphere_basin.canonical_probe_matrix', '--config', str(canonical_cfg), '--sphere-repo', args.sphere_repo],
            cwd=str(project_root()),
            dry_run=args.dry_run,
            gpu_group=None,
        )
        _run(
            [sys.executable, '-m', 'sphere_basin.canonical_master', '--config', str(canonical_cfg)],
            cwd=str(project_root()),
            dry_run=args.dry_run,
            gpu_group=None,
        )
        status['steps'][-1]['state'] = 'completed'
        _write_status(status_path, status)

        status['current_step'] = 'official_eval'
        status['steps'].append({'name': 'official_eval', 'state': 'running'})
        official_cfg = _build_official_eval_cfg(cfg=cfg, workspace_root=workspace_root, new_jobs=new_jobs)
        status['official_eval_config'] = str(official_cfg)
        _write_status(status_path, status)
        _run(
            [sys.executable, '-m', 'sphere_basin.official_eval_matrix', '--config', str(official_cfg), '--sphere-repo', args.sphere_repo],
            cwd=str(project_root()),
            dry_run=args.dry_run,
            gpu_group=None,
        )
        _run(
            [sys.executable, '-m', 'sphere_basin.official_eval_master', '--config', str(official_cfg)],
            cwd=str(project_root()),
            dry_run=args.dry_run,
            gpu_group=None,
        )
        status['steps'][-1]['state'] = 'completed'
        _write_status(status_path, status)

        if not args.dry_run:
            summary_root = workspace_root / 'research_summary'
            canonical_master = summary_root / 'canonical_master.csv'
            official_master = summary_root / 'official_eval_master.csv'
            canonical_copy = summary_root / 'imagenet100_followup_canonical_master.csv'
            official_copy = summary_root / 'imagenet100_followup_official_eval_master.csv'
            compare_path = summary_root / 'imagenet100_followup_compare.csv'
            shutil.copy2(canonical_master, canonical_copy)
            shutil.copy2(official_master, official_copy)
            _write_compare_csv(
                canonical_master_path=canonical_master,
                official_eval_master_path=official_master,
                out_path=compare_path,
                full_followup_job=resumed_job,
                new_jobs=new_jobs,
            )
            status['canonical_master_copy'] = str(canonical_copy)
            status['official_eval_master_copy'] = str(official_copy)
            status['compare_csv'] = str(compare_path)

    except BaseException as exc:
        status['state'] = 'failed'
        status['error'] = f'{type(exc).__name__}: {exc}'
        if status.get('steps'):
            status['steps'][-1]['state'] = 'failed'
        _write_status(status_path, status)
        raise
    finally:
        if not args.dry_run:
            _restore_summary_files(workspace_root, backup_root, backed_up)

    status['state'] = 'completed'
    status['current_step'] = None
    _write_status(status_path, status)


if __name__ == '__main__':
    main()
