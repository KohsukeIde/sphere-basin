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
from .launch import _build_train_spec, _normalize_gpu_groups, _run, _schedule_train_specs
from .paths import ensure_workspace_compat, project_root, resolve_dev_dir


def _write_status(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')


def _build_train_specs(cfg: dict[str, Any], workspace_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    experiment = cfg['experiment']
    train_common = dict(cfg['train'])
    dev_dir = str(resolve_dev_dir(experiment.get('dev_dir')))
    specs: list[dict[str, Any]] = []
    for job in cfg['jobs']:
        alpha = int(job['alpha'])
        target_job = str(job['target_job'])
        name = str(job['name'])
        manifest = {
            'sweep_name': experiment.get('name', 'no_lat_con_causal'),
            'tag': experiment.get('tag', 'causal'),
            'alpha': alpha,
            'variant': name,
            'job_dir': target_job,
        }
        specs.append(
            _build_train_spec(
                train_common={**train_common, 'noise_sigma_max_angle': alpha},
                dev_dir=dev_dir,
                run_slug=f'{name}-{experiment.get("tag", "causal")}',
                target_job=target_job,
                manifest=manifest,
                workspace_root=workspace_root,
            )
        )
    meta = {
        'dist_mode': experiment.get('dist_mode', 'local'),
        'skip_existing': bool(experiment.get('skip_existing', True)),
        'gpu_groups': _normalize_gpu_groups(experiment.get('gpu_groups')),
        'retry_attempts': int(experiment.get('retry_attempts', 0)),
    }
    return specs, meta


def _build_canonical_cfg(
    *,
    workspace_root: Path,
    new_jobs: list[str],
) -> Path:
    cfg = {
        'experiment': {
            'name': 'no_lat_con_canonical',
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
                {
                    'name': 'independent-fixed',
                    'cache_sampling_noise': False,
                    'use_sampling_scheduler': False,
                },
                {
                    'name': 'shared-fixed',
                    'cache_sampling_noise': True,
                    'use_sampling_scheduler': False,
                },
            ],
        },
        'master': {
            'contraction_noise_scaler': 1.0,
            'phase_ckpt_epoch': 'ep0074',
            'phase_regime_name': 'shared-fixed',
            'phase_forward_steps': 4,
            'phase_tau_deg': 60.0,
        },
        'jobs': [
            {'job_dir': 'sphere-small-small-cifar-10-32px-a80-pilot', 'checkpoints': ['ep0024', 'ep0074']},
            {'job_dir': 'sphere-small-small-cifar-10-32px-a83-pilot', 'checkpoints': ['ep0024', 'ep0074']},
        ]
        + [
            {'job_dir': job_dir, 'checkpoints': ['ep0024', 'ep0074']}
            for job_dir in new_jobs
        ],
    }
    out = workspace_root / 'research_summary' / 'no_lat_con_canonical.yaml'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
    return out


def _build_official_eval_cfg(
    *,
    workspace_root: Path,
    new_jobs: list[str],
) -> Path:
    cfg = {
        'experiment': {
            'name': 'no_lat_con_official_eval',
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
                'job_dir': 'sphere-small-small-cifar-10-32px-a80-pilot',
                'ckpt_fname': 'ep0074.pth',
                'regime_name': 'shared-fixed',
                'cache_sampling_noise': True,
                'use_sampling_scheduler': False,
            },
            {
                'job_dir': 'sphere-small-small-cifar-10-32px-a83-pilot',
                'ckpt_fname': 'ep0074.pth',
                'regime_name': 'shared-fixed',
                'cache_sampling_noise': True,
                'use_sampling_scheduler': False,
            },
        ]
        + [
            {
                'job_dir': job_dir,
                'ckpt_fname': 'ep0074.pth',
                'regime_name': 'shared-fixed',
                'cache_sampling_noise': True,
                'use_sampling_scheduler': False,
            }
            for job_dir in new_jobs
        ],
    }
    out = workspace_root / 'research_summary' / 'no_lat_con_official_eval.yaml'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
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

    keep_jobs = {
        'sphere-small-small-cifar-10-32px-a80-pilot',
        'sphere-small-small-cifar-10-32px-a83-pilot',
        *new_jobs,
    }
    canonical = canonical[
        (canonical['job_dir'].isin(keep_jobs))
        & (canonical['ckpt_epoch'].isin(['ep0024', 'ep0074']))
        & (canonical['forward_steps'] == 4)
        & (canonical['tau_deg'] == 60.0)
        & (canonical['regime_name'] == 'shared-fixed')
    ].copy()
    official = official[
        (official['job_dir'].isin(keep_jobs))
        & (official['ckpt_epoch'] == 'ep0074')
        & (official['regime_name'] == 'shared-fixed')
        & (official['forward_steps'] == 4)
    ].copy()
    official = official[
        [
            'job_dir',
            'ckpt_epoch',
            'fid',
            'isc_mean',
            'isc_std',
        ]
    ].rename(
        columns={
            'fid': 'official_fid',
            'isc_mean': 'official_isc_mean',
            'isc_std': 'official_isc_std',
        }
    )

    merged = canonical.merge(official, on=['job_dir', 'ckpt_epoch'], how='left')
    merged['alpha'] = pd.to_numeric(merged['alpha'], errors='coerce')
    merged['variant_group'] = merged['job_dir'].map(
        {
            'sphere-small-small-cifar-10-32px-a80-pilot': 'a80_full',
            'sphere-small-small-cifar-10-32px-a83-pilot': 'a83_full',
            new_jobs[0]: 'a80_no_lat_con',
            new_jobs[1]: 'a83_no_lat_con',
        }
    )
    merged = merged.sort_values(['variant_group', 'ckpt_epoch'])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Run no-latent-consistency causal ablation and rebuild canonical/official eval tables.'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=str(project_root() / 'configs' / 'no_lat_con_causal.yaml'),
    )
    parser.add_argument('--sphere-repo', type=str, required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = load_yaml(cfg_path)
    workspace_root = ensure_workspace_compat(cfg['experiment'].get('dev_dir'))
    status_path = workspace_root / 'research_summary' / 'no_lat_con_status.json'

    specs, meta = _build_train_specs(cfg, workspace_root)
    new_jobs = [spec['target_job'] for spec in specs]

    status: dict[str, Any] = {
        'config': str(cfg_path),
        'sphere_repo': args.sphere_repo,
        'new_jobs': new_jobs,
        'dry_run': args.dry_run,
        'steps': [],
        'state': 'running',
    }
    _write_status(status_path, status)

    try:
        status['current_step'] = 'train_no_lat_con'
        status['steps'].append({'name': 'train_no_lat_con', 'state': 'running'})
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
        _run(
            [
                sys.executable,
                '-m',
                'sphere_basin.canonical_probe_matrix',
                '--config',
                str(canonical_cfg),
                '--sphere-repo',
                args.sphere_repo,
            ],
            cwd=str(project_root()),
            dry_run=args.dry_run,
        )
        _run(
            [
                sys.executable,
                '-m',
                'sphere_basin.canonical_master',
                '--config',
                str(canonical_cfg),
            ],
            cwd=str(project_root()),
            dry_run=args.dry_run,
        )
        status['steps'][-1]['state'] = 'completed'
        _write_status(status_path, status)

        official_cfg = _build_official_eval_cfg(workspace_root=workspace_root, new_jobs=new_jobs)
        status['official_eval_config'] = str(official_cfg)
        status['current_step'] = 'official_eval'
        status['steps'].append({'name': 'official_eval', 'state': 'running'})
        _write_status(status_path, status)
        _run(
            [
                sys.executable,
                '-m',
                'sphere_basin.official_eval_matrix',
                '--config',
                str(official_cfg),
                '--sphere-repo',
                args.sphere_repo,
            ],
            cwd=str(project_root()),
            dry_run=args.dry_run,
        )
        _run(
            [
                sys.executable,
                '-m',
                'sphere_basin.official_eval_master',
                '--config',
                str(official_cfg),
            ],
            cwd=str(project_root()),
            dry_run=args.dry_run,
        )
        status['steps'][-1]['state'] = 'completed'
        _write_status(status_path, status)

        if not args.dry_run:
            canonical_master = workspace_root / 'research_summary' / 'canonical_master.csv'
            official_master = workspace_root / 'research_summary' / 'official_eval_master.csv'
            no_lat_canonical = workspace_root / 'research_summary' / 'no_lat_con_canonical_master.csv'
            no_lat_official = workspace_root / 'research_summary' / 'no_lat_con_official_eval_master.csv'
            shutil.copy2(canonical_master, no_lat_canonical)
            shutil.copy2(official_master, no_lat_official)
            compare_path = workspace_root / 'research_summary' / 'no_lat_con_compare.csv'
            _write_compare_csv(
                canonical_master_path=canonical_master,
                official_eval_master_path=official_master,
                out_path=compare_path,
                new_jobs=new_jobs,
            )
            status['canonical_master_copy'] = str(no_lat_canonical)
            status['official_eval_master_copy'] = str(no_lat_official)
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
