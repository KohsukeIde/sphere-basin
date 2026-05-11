from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .config import load_yaml
from .launch import _build_loss_specs, _run, _schedule_train_specs
from .paths import ensure_workspace_compat, project_root
from .unattended_imagenet100_followup import (
    _backup_summary_files,
    _refresh_dataset,
    _restore_summary_files,
)


REFERENCE_JOBS = {
    'a83_full': ('sphere-small-small-imagenet-100-160px-a83-second-axis', 'ep0024'),
    'a85_full': ('sphere-small-small-imagenet-100-160px-a85-second-axis', 'ep0024'),
    'a85_full_late': ('sphere-small-small-imagenet-100-160px-a85-second-axis-ep0049-followup', 'ep0049'),
    'a85_no_pix_con': ('sphere-small-small-imagenet-100-160px-a85-no_pix_con-followup', 'ep0024'),
}


def _write_status(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')


def _build_ablation_cfg(base_cfg: dict[str, Any], workspace_root: Path) -> Path:
    exp_cfg = base_cfg['experiment']
    ablation = base_cfg['ablation']
    payload = {
        'experiment': {
            'name': str(ablation.get('experiment_name', 'imagenet100_a83_no_pix_check')),
            'dist_mode': str(exp_cfg.get('dist_mode', 'local')),
            'dev_dir': str(exp_cfg.get('dev_dir', 'workspace')),
            'tag': str(ablation.get('tag', 'no_pix_check')),
            'skip_existing': bool(exp_cfg.get('skip_existing', True)),
            'retry_attempts': int(exp_cfg.get('retry_attempts', 0)),
            'gpu_groups': exp_cfg.get('gpu_groups', [[0, 1]]),
        },
        'train': dict(ablation['train']),
        'variants': dict(ablation['variants']),
    }
    out = workspace_root / 'research_summary' / 'imagenet100_a83_no_pix_check_loss_ablation.yaml'
    out.write_text(yaml.safe_dump(payload, sort_keys=False), encoding='utf-8')
    return out


def _build_canonical_cfg(*, cfg: dict[str, Any], workspace_root: Path, new_jobs: list[str]) -> Path:
    canonical = dict(cfg['canonical'])
    jobs = [
        {'job_dir': job_dir, 'checkpoints': [ckpt]}
        for job_dir, ckpt in REFERENCE_JOBS.values()
    ]
    jobs.extend({'job_dir': job_dir, 'checkpoints': ['ep0024']} for job_dir in new_jobs)
    payload = {
        'experiment': {
            'name': 'imagenet100_a83_no_pix_check_canonical',
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
            'phase_ckpt_epoch': 'ep0024',
            'phase_regime_name': 'shared-fixed',
            'phase_forward_steps': 4,
            'phase_tau_deg': 60.0,
        },
        'jobs': jobs,
    }
    out = workspace_root / 'research_summary' / 'imagenet100_a83_no_pix_check_canonical.yaml'
    out.write_text(yaml.safe_dump(payload, sort_keys=False), encoding='utf-8')
    return out


def _eval_task(job_dir: str, ckpt_epoch: str, regime_name: str) -> dict[str, Any]:
    shared = regime_name == 'shared-fixed'
    return {
        'job_dir': job_dir,
        'ckpt_fname': f'{ckpt_epoch}.pth',
        'regime_name': regime_name,
        'cache_sampling_noise': shared,
        'use_sampling_scheduler': False,
    }


def _build_official_eval_cfg(*, cfg: dict[str, Any], workspace_root: Path, new_jobs: list[str]) -> Path:
    eval_cfg = dict(cfg['official_eval'])
    task_pairs: list[tuple[str, str]] = list(REFERENCE_JOBS.values())
    task_pairs.extend((job_dir, 'ep0024') for job_dir in new_jobs)
    tasks = [
        _eval_task(job_dir, ckpt, regime)
        for job_dir, ckpt in task_pairs
        for regime in ('shared-fixed', 'independent-fixed')
    ]
    payload = {
        'experiment': {
            'name': 'imagenet100_a83_no_pix_check_official_eval',
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
        'tasks': tasks,
    }
    out = workspace_root / 'research_summary' / 'imagenet100_a83_no_pix_check_official_eval.yaml'
    out.write_text(yaml.safe_dump(payload, sort_keys=False), encoding='utf-8')
    return out


def _write_compare_csv(
    *,
    canonical_master_path: Path,
    official_eval_master_path: Path,
    out_path: Path,
    new_jobs: list[str],
) -> None:
    canonical = pd.read_csv(canonical_master_path)
    official = pd.read_csv(official_eval_master_path)

    mapping = {
        (job_dir, ckpt, regime): f'{label}_{ckpt}_{regime.replace("-fixed", "")}'
        for label, (job_dir, ckpt) in REFERENCE_JOBS.items()
        for regime in ('shared-fixed', 'independent-fixed')
    }
    for job_dir in new_jobs:
        for regime in ('shared-fixed', 'independent-fixed'):
            mapping[(job_dir, 'ep0024', regime)] = f'a83_no_pix_con_ep0024_{regime.replace("-fixed", "")}'

    canonical = canonical[
        (canonical['forward_steps'] == 4)
        & (canonical['tau_deg'] == 60.0)
    ].copy()
    canonical['variant_group'] = canonical.apply(
        lambda row: mapping.get((row['job_dir'], row['ckpt_epoch'], row['regime_name'])),
        axis=1,
    )
    canonical = canonical[canonical['variant_group'].notna()].copy()

    official = official[['job_dir', 'ckpt_epoch', 'regime_name', 'forward_steps', 'fid', 'isc_mean', 'isc_std']].copy()
    official['variant_group'] = official.apply(
        lambda row: mapping.get((row['job_dir'], row['ckpt_epoch'], row['regime_name'])),
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
    merged = merged.sort_values(['variant_group', 'job_dir', 'regime_name'])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description='Run ImageNet-100 a83 no_pix_con check and shared/independent eval matrix.')
    parser.add_argument(
        '--config',
        type=str,
        default=str(project_root() / 'configs' / 'imagenet100_a83_no_pix_check.yaml'),
    )
    parser.add_argument('--sphere-repo', type=str, required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    workspace_root = ensure_workspace_compat(cfg['experiment'].get('dev_dir'))
    summary_root = workspace_root / 'research_summary'
    status_path = summary_root / 'imagenet100_a83_no_pix_check_status.json'
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

        status['current_step'] = 'train_a83_no_pix_con'
        status['steps'].append({'name': 'train_a83_no_pix_con', 'state': 'running'})
        _write_status(status_path, status)
        ablation_cfg = _build_ablation_cfg(cfg, workspace_root)
        loss_cfg = load_yaml(ablation_cfg)
        specs, meta = _build_loss_specs(loss_cfg, workspace_root)
        new_jobs = [str(spec['target_job']) for spec in specs]
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
        status['ablation_config'] = str(ablation_cfg)
        status['new_jobs'] = new_jobs
        status['trained_jobs'] = trained_jobs
        status['steps'][-1]['state'] = 'completed'
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
            canonical_master = summary_root / 'canonical_master.csv'
            official_master = summary_root / 'official_eval_master.csv'
            canonical_copy = summary_root / 'imagenet100_a83_no_pix_check_canonical_master.csv'
            official_copy = summary_root / 'imagenet100_a83_no_pix_check_official_eval_master.csv'
            compare_path = summary_root / 'imagenet100_a83_no_pix_check_compare.csv'
            shutil.copy2(canonical_master, canonical_copy)
            shutil.copy2(official_master, official_copy)
            _write_compare_csv(
                canonical_master_path=canonical_master,
                official_eval_master_path=official_master,
                out_path=compare_path,
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
