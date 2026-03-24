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


def _write_status(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')


def _build_canonical_cfg(*, workspace_root: Path, new_jobs: list[str]) -> Path:
    cfg = {
        'experiment': {
            'name': 'a83_consistency_pathway_canonical',
            'dev_dir': 'workspace',
            'dist_mode': 'local',
            'gpu_groups': [[0], [1]],
        },
        'probe': {
            'cfg': 1.0,
            'cfg_position': 'combo',
            'seed': 0,
            'taus_deg': [5.0, 10.0, 20.0, 30.0, 45.0, 60.0],
            'forward_steps': [1, 4],
            'num_prior_samples': 4096,
            'num_data_samples': 4096,
            'batch_size_per_rank': 32,
            'contraction_noise_scalers': [0.25, 0.5, 0.75, 1.0],
            'num_workers': 4,
            'use_ema_model': False,
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
            {'job_dir': 'sphere-small-small-cifar-10-32px-a83-pilot', 'checkpoints': ['ep0024', 'ep0049']},
        ] + [{'job_dir': job_dir, 'checkpoints': ['ep0024', 'ep0049']} for job_dir in new_jobs],
    }
    out = workspace_root / 'research_summary' / 'a83_consistency_pathway_canonical.yaml'
    out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
    return out


def _build_official_eval_cfg(*, workspace_root: Path, new_jobs: list[str]) -> Path:
    cfg = {
        'experiment': {
            'name': 'a83_consistency_pathway_official_eval',
            'dev_dir': 'workspace',
            'dist_mode': 'local',
            'gpu_groups': [[0, 1]],
        },
        'eval': {
            'forward_steps': [4],
            'report_fid': ['rfid', 'gfid'],
            'use_cfg': False,
            'cfg_min': 1.0,
            'cfg_max': 1.0,
            'cfg_position': 'combo',
            'rm_folder_after_eval': True,
            'num_eval_samples': 50000,
            'batch_size_per_rank': 25,
            'use_ema_model': False,
        },
        'tasks': [
            {
                'job_dir': 'sphere-small-small-cifar-10-32px-a83-pilot',
                'ckpt_fname': 'ep0049.pth',
                'regime_name': 'shared-fixed',
                'cache_sampling_noise': True,
                'use_sampling_scheduler': False,
            },
        ] + [
            {
                'job_dir': job_dir,
                'ckpt_fname': 'ep0049.pth',
                'regime_name': 'shared-fixed',
                'cache_sampling_noise': True,
                'use_sampling_scheduler': False,
            }
            for job_dir in new_jobs
        ],
    }
    out = workspace_root / 'research_summary' / 'a83_consistency_pathway_official_eval.yaml'
    out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
    return out


def _write_compare_csv(
    *,
    canonical_master_path: Path,
    official_eval_master_path: Path,
    out_path: Path,
    variant_job_map: dict[str, str],
) -> None:
    canonical = pd.read_csv(canonical_master_path)
    official = pd.read_csv(official_eval_master_path)

    keep_jobs = {'sphere-small-small-cifar-10-32px-a83-pilot', *variant_job_map.values()}
    canonical = canonical[
        (canonical['job_dir'].isin(keep_jobs))
        & (canonical['ckpt_epoch'].isin(['ep0024', 'ep0049']))
        & (canonical['forward_steps'] == 4)
        & (canonical['tau_deg'] == 60.0)
        & (canonical['regime_name'] == 'shared-fixed')
    ].copy()
    official = official[
        (official['job_dir'].isin(keep_jobs))
        & (official['ckpt_epoch'] == 'ep0049')
        & (official['regime_name'] == 'shared-fixed')
        & (official['forward_steps'] == 4)
    ][['job_dir', 'ckpt_epoch', 'fid']].rename(columns={'fid': 'official_fid'})

    merged = canonical.merge(official, on=['job_dir', 'ckpt_epoch'], how='left')
    variant_group_map = {'sphere-small-small-cifar-10-32px-a83-pilot': 'a83_full'}
    for variant_name, job_dir in variant_job_map.items():
        variant_group_map[job_dir] = f'a83_{variant_name}'
    merged['variant_group'] = merged['job_dir'].map(variant_group_map)
    merged = merged.sort_values(['variant_group', 'ckpt_epoch'])
    out_path.to_csv(merged, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description='Run a83 no_pix_con/recon_only short pathway ablation.')
    parser.add_argument('--config', type=str, default=str(project_root() / 'configs' / 'a83_consistency_pathway_short.yaml'))
    parser.add_argument('--sphere-repo', type=str, required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    workspace_root = ensure_workspace_compat(cfg['experiment'].get('dev_dir'))
    status_path = workspace_root / 'research_summary' / 'a83_consistency_pathway_status.json'
    specs, meta = _build_loss_specs(cfg, workspace_root)
    new_jobs = [spec['target_job'] for spec in specs]
    variant_job_map = {str(spec['manifest']['variant']): str(spec['target_job']) for spec in specs}

    status: dict[str, Any] = {
        'config': str(Path(args.config).resolve()),
        'sphere_repo': args.sphere_repo,
        'new_jobs': new_jobs,
        'variant_job_map': variant_job_map,
        'dry_run': args.dry_run,
        'steps': [],
        'state': 'running',
    }
    _write_status(status_path, status)

    try:
        status['current_step'] = 'train_a83_consistency_pathway'
        status['steps'].append({'name': 'train_a83_consistency_pathway', 'state': 'running'})
        _write_status(status_path, status)
        jobs = _schedule_train_specs(
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
        status['trained_jobs'] = jobs
        _write_status(status_path, status)

        canonical_cfg = _build_canonical_cfg(workspace_root=workspace_root, new_jobs=new_jobs)
        status['canonical_config'] = str(canonical_cfg)
        status['current_step'] = 'canonical_probe'
        status['steps'].append({'name': 'canonical_probe', 'state': 'running'})
        _write_status(status_path, status)
        _run([sys.executable, '-m', 'sphere_basin.canonical_probe_matrix', '--config', str(canonical_cfg), '--sphere-repo', args.sphere_repo], cwd=str(project_root()), dry_run=args.dry_run)
        _run([sys.executable, '-m', 'sphere_basin.canonical_master', '--config', str(canonical_cfg)], cwd=str(project_root()), dry_run=args.dry_run)
        status['steps'][-1]['state'] = 'completed'
        _write_status(status_path, status)

        official_cfg = _build_official_eval_cfg(workspace_root=workspace_root, new_jobs=new_jobs)
        status['official_eval_config'] = str(official_cfg)
        status['current_step'] = 'official_eval'
        status['steps'].append({'name': 'official_eval', 'state': 'running'})
        _write_status(status_path, status)
        _run([sys.executable, '-m', 'sphere_basin.official_eval_matrix', '--config', str(official_cfg), '--sphere-repo', args.sphere_repo], cwd=str(project_root()), dry_run=args.dry_run)
        _run([sys.executable, '-m', 'sphere_basin.official_eval_master', '--config', str(official_cfg)], cwd=str(project_root()), dry_run=args.dry_run)
        status['steps'][-1]['state'] = 'completed'
        _write_status(status_path, status)

        if not args.dry_run:
            canonical_master = workspace_root / 'research_summary' / 'canonical_master.csv'
            official_master = workspace_root / 'research_summary' / 'official_eval_master.csv'
            a83_canonical = workspace_root / 'research_summary' / 'a83_consistency_pathway_canonical_master.csv'
            a83_official = workspace_root / 'research_summary' / 'a83_consistency_pathway_official_eval_master.csv'
            shutil.copy2(canonical_master, a83_canonical)
            shutil.copy2(official_master, a83_official)
            compare_path = workspace_root / 'research_summary' / 'a83_consistency_pathway_compare.csv'
            _write_compare_csv(
                canonical_master_path=canonical_master,
                official_eval_master_path=official_master,
                out_path=compare_path,
                variant_job_map=variant_job_map,
            )
            status['canonical_master_copy'] = str(a83_canonical)
            status['official_eval_master_copy'] = str(a83_official)
            status['compare_csv'] = str(compare_path)

    except Exception as exc:
        status['state'] = 'failed'
        status['error'] = f'{type(exc).__name__}: {exc}'
        if status.get('steps'):
            status['steps'][-1]['state'] = 'failed'
        _write_status(status_path, status)
        raise

    status['state'] = 'completed'
    status['current_step'] = None
    _write_status(status_path, status)


if __name__ == '__main__':
    main()
