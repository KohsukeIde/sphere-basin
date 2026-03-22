from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .launch import _run
from .paths import project_root, ensure_workspace_compat
from .config import load_yaml


def _write_status(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description='Run official eval matrix and build official eval master.')
    parser.add_argument(
        '--config',
        type=str,
        default=str(project_root() / 'configs' / 'official_eval_keypoints.yaml'),
    )
    parser.add_argument('--sphere-repo', type=str, required=True)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = load_yaml(config_path)
    workspace_root = ensure_workspace_compat(cfg['experiment'].get('dev_dir'))
    status_path = workspace_root / 'research_summary' / 'official_eval_status.json'
    status: dict[str, Any] = {
        'config': str(config_path),
        'sphere_repo': args.sphere_repo,
        'dry_run': args.dry_run,
        'steps': [],
        'state': 'running',
    }
    _write_status(status_path, status)

    steps = [
        {
            'name': 'official_eval_matrix',
            'cmd': [
                sys.executable,
                '-m',
                'sphere_basin.official_eval_matrix',
                '--config',
                str(config_path),
                '--sphere-repo',
                args.sphere_repo,
                *(['--force'] if args.force else []),
            ],
        },
        {
            'name': 'official_eval_master',
            'cmd': [
                sys.executable,
                '-m',
                'sphere_basin.official_eval_master',
                '--config',
                str(config_path),
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
    status['final_master'] = str(
        workspace_root / 'research_summary' / 'official_eval_master.csv'
    )
    _write_status(status_path, status)


if __name__ == '__main__':
    main()
