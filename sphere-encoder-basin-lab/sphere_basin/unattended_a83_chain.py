from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .config import load_yaml
from .launch import _run
from .paths import ensure_workspace_compat, project_root


def _write_status(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Run focus reprobe, then extend a83 to ep74 and rebuild canonical tables.'
    )
    parser.add_argument('--sphere-repo', type=str, required=True)
    parser.add_argument(
        '--focus-config',
        type=str,
        default=str(project_root() / 'configs' / 'canonical_probe_focus.yaml'),
    )
    parser.add_argument('--job-dir', type=str, default='sphere-small-small-cifar-10-32px-a83-pilot')
    parser.add_argument('--target-ckpt-epoch', type=int, default=74)
    parser.add_argument('--train-gpu-group', type=str, default='0,1')
    parser.add_argument('--train-ckpt-save-interval', type=int, default=5)
    parser.add_argument('--wandb-name', type=str, default='sphere-small-small-cifar-10-32px-a83-pilot-ep74-auto-chain')
    parser.add_argument('--force-focus-probe', action='store_true')
    parser.add_argument('--force-followup-canonical', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    focus_config = Path(args.focus_config).resolve()
    cfg = load_yaml(focus_config)
    workspace_root = ensure_workspace_compat(cfg['experiment'].get('dev_dir'))
    status_path = workspace_root / 'research_summary' / 'a83_ep74_chain_status.json'

    status: dict[str, Any] = {
        'focus_config': str(focus_config),
        'sphere_repo': args.sphere_repo,
        'job_dir': args.job_dir,
        'target_ckpt_epoch': args.target_ckpt_epoch,
        'train_gpu_group': args.train_gpu_group,
        'train_ckpt_save_interval': args.train_ckpt_save_interval,
        'wandb_name': args.wandb_name,
        'dry_run': args.dry_run,
        'steps': [],
        'state': 'running',
    }
    _write_status(status_path, status)

    steps = [
        {
            'name': 'focus_canonical_probe',
            'cmd': [
                sys.executable,
                '-m',
                'sphere_basin.canonical_probe_matrix',
                '--config',
                str(focus_config),
                '--sphere-repo',
                args.sphere_repo,
                *(['--force'] if args.force_focus_probe else []),
            ],
        },
        {
            'name': 'focus_canonical_master',
            'cmd': [
                sys.executable,
                '-m',
                'sphere_basin.canonical_master',
                '--config',
                str(focus_config),
            ],
        },
        {
            'name': 'a83_followup_ep74',
            'cmd': [
                sys.executable,
                '-m',
                'sphere_basin.unattended_followup',
                '--sphere-repo',
                args.sphere_repo,
                '--job-dir',
                args.job_dir,
                '--target-ckpt-epoch',
                str(args.target_ckpt_epoch),
                '--train-gpu-group',
                args.train_gpu_group,
                '--canonical-config',
                str(focus_config),
                '--wandb-name',
                args.wandb_name,
                '--train-ckpt-save-interval',
                str(args.train_ckpt_save_interval),
                *(['--force-canonical'] if args.force_followup_canonical else []),
            ],
        },
    ]

    try:
        for step in steps:
            status['current_step'] = step['name']
            status['steps'].append({'name': step['name'], 'state': 'running'})
            _write_status(status_path, status)
            _run(step['cmd'], cwd=str(project_root()), dry_run=args.dry_run)
            status['steps'][-1]['state'] = 'completed'
            _write_status(status_path, status)
    except Exception as exc:
        status['state'] = 'failed'
        status['error'] = f'{type(exc).__name__}: {exc}'
        if status.get('steps'):
            status['steps'][-1]['state'] = 'failed'
        _write_status(status_path, status)
        raise

    status['state'] = 'completed'
    status['current_step'] = None
    status['final_canonical_master'] = str(
        workspace_root / 'research_summary' / 'canonical_master.csv'
    )
    status['final_compare_csv'] = str(
        workspace_root
        / 'research_summary'
        / f'{args.job_dir}-ep{args.target_ckpt_epoch:04d}-followup_compare.csv'
    )
    _write_status(status_path, status)


if __name__ == '__main__':
    main()
