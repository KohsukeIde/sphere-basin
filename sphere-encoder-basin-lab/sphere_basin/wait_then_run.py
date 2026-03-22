from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .paths import project_root


def _write_status(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')


def _read_state(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Wait for one status JSON to complete, then run a follow-up command.'
    )
    parser.add_argument('--wait-status', type=str, required=True)
    parser.add_argument('--poll-seconds', type=float, default=60.0)
    parser.add_argument('--status-out', type=str, required=True)
    parser.add_argument('--cwd', type=str, default=str(project_root()))
    parser.add_argument('cmd', nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.cmd and args.cmd[0] == '--':
        args.cmd = args.cmd[1:]
    if not args.cmd:
        raise SystemExit('missing follow-up command')

    wait_status = Path(args.wait_status).resolve()
    status_out = Path(args.status_out).resolve()
    cwd = Path(args.cwd).resolve()

    status: dict[str, Any] = {
        'wait_status': str(wait_status),
        'cwd': str(cwd),
        'cmd': args.cmd,
        'poll_seconds': args.poll_seconds,
        'state': 'waiting',
    }
    _write_status(status_out, status)

    while True:
        if not wait_status.exists():
            status['state'] = 'waiting'
            status['note'] = 'wait status file not found yet'
            _write_status(status_out, status)
            time.sleep(args.poll_seconds)
            continue

        upstream = _read_state(wait_status)
        upstream_state = str(upstream.get('state'))
        status['upstream_state'] = upstream_state
        status['upstream_current_step'] = upstream.get('current_step')
        _write_status(status_out, status)

        if upstream_state == 'completed':
            break
        if upstream_state == 'failed':
            status['state'] = 'failed'
            status['error'] = f'upstream failed: {upstream.get("error")}'
            _write_status(status_out, status)
            raise SystemExit(1)
        time.sleep(args.poll_seconds)

    status['state'] = 'running_followup'
    _write_status(status_out, status)
    subprocess.run(args.cmd, cwd=str(cwd), check=True)
    status['state'] = 'completed'
    _write_status(status_out, status)


if __name__ == '__main__':
    main()
