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
            'sweep_name': experiment.get('name', 'imagenet100_second_axis'),
            'tag': experiment.get('tag', 'second-axis'),
            'alpha': alpha,
            'variant': name,
            'job_dir': target_job,
        }
        specs.append(
            _build_train_spec(
                train_common={**train_common, 'noise_sigma_max_angle': alpha},
                dev_dir=dev_dir,
                run_slug=f'{name}-{experiment.get("tag", "second-axis")}',
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


def _build_canonical_cfg(*, cfg: dict[str, Any], workspace_root: Path, new_jobs: list[str]) -> Path:
    canonical = dict(cfg['canonical'])
    payload = {
        'experiment': {
            'name': 'imagenet100_second_axis_canonical',
            'dev_dir': 'workspace',
            'dist_mode': 'local',
            'gpu_groups': [[0], [1]],
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
            'phase_ckpt_epoch': canonical.get('phase_ckpt_epoch', 'ep0024'),
            'phase_regime_name': canonical.get('phase_regime_name', 'shared-fixed'),
            'phase_forward_steps': int(canonical.get('phase_forward_steps', 4)),
            'phase_tau_deg': float(canonical.get('phase_tau_deg', 60.0)),
        },
        'jobs': [
            {'job_dir': job_dir, 'checkpoints': ['ep0024']}
            for job_dir in new_jobs
        ],
    }
    out = workspace_root / 'research_summary' / 'imagenet100_second_axis_canonical.yaml'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(payload, sort_keys=False), encoding='utf-8')
    return out


def _build_official_eval_cfg(*, cfg: dict[str, Any], workspace_root: Path, new_jobs: list[str]) -> Path:
    eval_cfg = dict(cfg['official_eval'])
    payload = {
        'experiment': {
            'name': 'imagenet100_second_axis_official_eval',
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
                'job_dir': job_dir,
                'ckpt_fname': 'ep0024.pth',
                'regime_name': 'shared-fixed',
                'cache_sampling_noise': True,
                'use_sampling_scheduler': False,
            }
            for job_dir in new_jobs
        ],
    }
    out = workspace_root / 'research_summary' / 'imagenet100_second_axis_official_eval.yaml'
    out.parent.mkdir(parents=True, exist_ok=True)
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
    canonical = canonical[
        (canonical['job_dir'].isin(new_jobs))
        & (canonical['ckpt_epoch'] == 'ep0024')
        & (canonical['forward_steps'] == 4)
        & (canonical['tau_deg'] == 60.0)
        & (canonical['regime_name'] == 'shared-fixed')
    ].copy()
    official = official[
        (official['job_dir'].isin(new_jobs))
        & (official['ckpt_epoch'] == 'ep0024')
        & (official['regime_name'] == 'shared-fixed')
        & (official['forward_steps'] == 4)
    ][['job_dir', 'ckpt_epoch', 'fid', 'isc_mean', 'isc_std']].rename(
        columns={
            'fid': 'official_fid',
            'isc_mean': 'official_isc_mean',
            'isc_std': 'official_isc_std',
        }
    )
    merged = canonical.merge(official, on=['job_dir', 'ckpt_epoch'], how='left')
    merged['variant_group'] = merged['job_dir'].map(
        {
            new_jobs[0]: 'a83_second_axis',
            new_jobs[1]: 'a85_second_axis',
        }
    )
    merged = merged.sort_values(['variant_group'])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Prepare ImageNet-100 (CMC split) and run the second-axis a83/a85 pilot chain.'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=str(project_root() / 'configs' / 'imagenet100_second_axis.yaml'),
    )
    parser.add_argument('--sphere-repo', type=str, required=True)
    parser.add_argument('--skip-fid-stats', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = load_yaml(config_path)
    workspace_root = ensure_workspace_compat(cfg['experiment'].get('dev_dir'))
    status_path = workspace_root / 'research_summary' / 'imagenet100_second_axis_status.json'
    status: dict[str, Any] = {
        'config': str(config_path),
        'sphere_repo': args.sphere_repo,
        'dry_run': args.dry_run,
        'skip_fid_stats': args.skip_fid_stats,
        'steps': [],
        'state': 'running',
    }
    _write_status(status_path, status)

    specs, meta = _build_train_specs(cfg, workspace_root)
    new_jobs = [spec['target_job'] for spec in specs]

    try:
        dataset_cfg = cfg['dataset']
        status['current_step'] = 'prepare_dataset'
        status['steps'].append({'name': 'prepare_dataset', 'state': 'running'})
        _write_status(status_path, status)
        prepare_cmd = [
            sys.executable,
            '-m',
            'sphere_basin.prepare_imagenet100_cmc',
            '--source-root',
            str(dataset_cfg['source_root']),
            '--dev-dir',
            str(resolve_dev_dir(cfg['experiment'].get('dev_dir'))),
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
            *(['--fid-stats-cuda'] if bool(dataset_cfg.get('fid_stats_cuda', True)) else []),
            *(['--skip-fid-stats'] if args.skip_fid_stats else []),
        ]
        _run(prepare_cmd, cwd=str(project_root()), dry_run=args.dry_run)
        status['steps'][-1]['state'] = 'completed'
        _write_status(status_path, status)

        status['current_step'] = 'train_second_axis'
        status['steps'].append({'name': 'train_second_axis', 'state': 'running'})
        _write_status(status_path, status)
        jobs = _schedule_train_specs(
            specs,
            sphere_repo=args.sphere_repo,
            workspace_root=workspace_root,
            dist_mode=meta['dist_mode'],
            skip_existing=meta['skip_existing'],
            dry_run=args.dry_run,
            gpu_groups=meta['gpu_groups'],
        )
        status['steps'][-1]['state'] = 'completed'
        status['trained_jobs'] = jobs
        _write_status(status_path, status)

        canonical_cfg = _build_canonical_cfg(cfg=cfg, workspace_root=workspace_root, new_jobs=new_jobs)
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

        official_cfg = _build_official_eval_cfg(cfg=cfg, workspace_root=workspace_root, new_jobs=new_jobs)
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
            compare_path = workspace_root / 'research_summary' / 'imagenet100_second_axis_compare.csv'
            _write_compare_csv(
                canonical_master_path=canonical_master,
                official_eval_master_path=official_master,
                out_path=compare_path,
                new_jobs=new_jobs,
            )
            status['compare_csv'] = str(compare_path)
            status['canonical_master'] = str(canonical_master)
            status['official_eval_master'] = str(official_master)
        status['state'] = 'completed'
        status['current_step'] = None
        _write_status(status_path, status)
    except Exception as exc:
        status['state'] = 'failed'
        status['error'] = f'{type(exc).__name__}: {exc}'
        if status.get('steps'):
            status['steps'][-1]['state'] = 'failed'
        _write_status(status_path, status)
        raise


if __name__ == '__main__':
    main()
