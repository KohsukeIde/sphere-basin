from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .paths import ensure_workspace_compat, project_root


def _write_status(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')


def _run(cmd: list[str], cwd: Path) -> None:
    import subprocess

    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description='Run the next mechanistic follow-up chain.')
    parser.add_argument('--sphere-repo', type=str, required=True)
    parser.add_argument('--workspace', type=str, default=str(project_root() / 'workspace'))
    args = parser.parse_args()

    workspace = ensure_workspace_compat(args.workspace)
    status_path = workspace / 'research_summary' / 'mech_followup_status.json'
    root = project_root()
    status: dict[str, Any] = {
        'sphere_repo': args.sphere_repo,
        'workspace': str(workspace),
        'steps': [],
        'state': 'running',
    }
    _write_status(status_path, status)

    steps = [
        ('continuous_summary', [sys.executable, '-m', 'sphere_basin.analyze_continuous_projector', '--workspace', str(workspace)]),
        ('projector_cycle', [sys.executable, '-m', 'sphere_basin.projector_cycle_matrix', '--config', str(root / 'configs' / 'projector_cycle_matrix.yaml'), '--sphere-repo', args.sphere_repo]),
        ('a83_consistency_pathway', [sys.executable, '-m', 'sphere_basin.unattended_a83_consistency_pathway', '--config', str(root / 'configs' / 'a83_consistency_pathway_short.yaml'), '--sphere-repo', args.sphere_repo]),
        ('imagenet100_second_axis', [sys.executable, '-m', 'sphere_basin.unattended_imagenet100_second_axis', '--config', str(root / 'configs' / 'imagenet100_second_axis.yaml'), '--sphere-repo', args.sphere_repo]),
    ]

    try:
        for name, cmd in steps:
            status['current_step'] = name
            status['steps'].append({'name': name, 'state': 'running'})
            _write_status(status_path, status)
            _run(cmd, root)
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
    _write_status(status_path, status)


if __name__ == '__main__':
    main()
