from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .config import load_yaml
from .launch import _bool_to_str, _build_repo_cmd, _format_gpu_group, _run
from .paths import ensure_workspace_compat, find_job_dir, project_root


DERIVED_TRAIN_CFG_KEYS = {
    'latent_resolution',
}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def _extract_train_flags(train_py_path: Path) -> set[str]:
    text = train_py_path.read_text(encoding='utf-8')
    return set(re.findall(r'parser\.add_argument\(\s*"--([a-zA-Z0-9_]+)"', text))


def _kv_args_from_cfg(cfg: dict[str, Any], *, allowed_flags: set[str]) -> list[str]:
    argv: list[str] = []
    for key in sorted(allowed_flags):
        if key in DERIVED_TRAIN_CFG_KEYS:
            continue
        if key not in cfg:
            continue
        value = cfg[key]
        if value is None:
            continue
        flag = f'--{key}'
        if isinstance(value, list):
            argv.append(flag)
            argv.extend(_bool_to_str(x) for x in value)
        else:
            argv.extend([flag, _bool_to_str(value)])
    return argv


def _ckpt_epoch_num(ckpt_name: str) -> int:
    match = re.search(r'ep(\d+)', ckpt_name)
    if match is None:
        raise ValueError(f'could not parse checkpoint epoch from {ckpt_name}')
    return int(match.group(1))


def _latest_ckpt(job_path: Path) -> Path | None:
    ckpts = sorted((job_path / 'ckpt').glob('ep*.pth'))
    return ckpts[-1] if ckpts else None


def _training_metrics_has_epoch(job_path: Path, epoch_num: int) -> bool:
    path = job_path / 'research' / 'training_metrics.jsonl'
    if not path.exists():
        return False
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        try:
            epoch = int(float(row.get('epoch')))
        except (TypeError, ValueError):
            continue
        if epoch == epoch_num:
            return True
    return False


def _build_followup_canonical_config(
    *,
    base_config_path: Path,
    workspace_root: Path,
    job_dir: str,
    target_ckpt_epoch: int,
) -> Path:
    cfg = load_yaml(base_config_path)
    updated = copy.deepcopy(cfg)
    target_ckpt = f'ep{target_ckpt_epoch:04d}'
    for job_spec in updated['jobs']:
        if str(job_spec['job_dir']) != job_dir:
            continue
        checkpoints = [str(x) for x in job_spec.get('checkpoints', [])]
        if target_ckpt not in checkpoints:
            checkpoints.append(target_ckpt)
        checkpoints = sorted(set(checkpoints))
        job_spec['checkpoints'] = checkpoints
        break
    else:
        updated['jobs'].append({'job_dir': job_dir, 'checkpoints': [target_ckpt]})

    out_path = (
        workspace_root
        / 'research_summary'
        / f'canonical_probe_matrix_{job_dir}-{target_ckpt}.yaml'
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(updated, sort_keys=False), encoding='utf-8')
    return out_path


def _write_followup_compare_csv(
    *,
    canonical_master_path: Path,
    out_path: Path,
    job_dir: str,
    target_ckpt_epoch: int,
) -> None:
    df = pd.read_csv(canonical_master_path)
    target_ckpt = f'ep{target_ckpt_epoch:04d}'
    keep_jobs = {
        'sphere-small-small-cifar-10-32px-a80-pilot',
        job_dir,
        'sphere-small-small-cifar-10-32px-a85-pilot',
    }
    keep_ckpts = {'ep0024', target_ckpt}
    out = df[
        (df['job_dir'].isin(keep_jobs))
        & (df['ckpt_epoch'].isin(keep_ckpts))
        & (df['forward_steps'] == 4)
        & (df['tau_deg'] == 60.0)
    ].copy()
    out = out.sort_values(['regime_name', 'alpha', 'ckpt_epoch'])
    out.to_csv(out_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Resume one training job to a target checkpoint, then rerun canonical post-hoc probes.'
    )
    parser.add_argument('--sphere-repo', type=str, required=True)
    parser.add_argument('--job-dir', type=str, required=True)
    parser.add_argument('--target-ckpt-epoch', type=int, required=True)
    parser.add_argument('--train-gpu-group', type=str, default='0,1')
    parser.add_argument('--canonical-config', type=str, required=True)
    parser.add_argument('--wandb-name', type=str, default=None)
    parser.add_argument('--train-ckpt-save-interval', type=int, default=None)
    parser.add_argument('--force-canonical', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    canonical_config_path = Path(args.canonical_config).resolve()
    base_cfg = load_yaml(canonical_config_path)
    workspace_root = ensure_workspace_compat(base_cfg['experiment'].get('dev_dir'))
    job_path = find_job_dir(workspace_root, args.job_dir)
    job_cfg_path = job_path / 'cfg.json'
    job_cfg = _load_json(job_cfg_path)

    latest_ckpt = _latest_ckpt(job_path)
    latest_epoch = _ckpt_epoch_num(latest_ckpt.name) if latest_ckpt is not None else -1
    target_ckpt_name = f'ep{args.target_ckpt_epoch:04d}.pth'
    target_ckpt_path = job_path / 'ckpt' / target_ckpt_name

    if latest_epoch < args.target_ckpt_epoch:
        train_py_path = Path(args.sphere_repo) / 'train.py'
        allowed_flags = _extract_train_flags(train_py_path)
        train_cfg = dict(job_cfg)
        train_cfg['epochs'] = args.target_ckpt_epoch + 1
        train_cfg['auto_resume'] = True
        train_cfg['init_from'] = 'resume'
        if latest_ckpt is not None:
            train_cfg['resume_from'] = str(latest_ckpt)
        if args.wandb_name:
            train_cfg['wandb_name'] = args.wandb_name
        if args.train_ckpt_save_interval is not None:
            train_cfg['ckpt_save_interval'] = int(args.train_ckpt_save_interval)
        train_argv = _kv_args_from_cfg(train_cfg, allowed_flags=allowed_flags)
        cmd = _build_repo_cmd(
            entry_script='train.py',
            script_args=train_argv,
            dist_mode='local',
            gpu_group=_format_gpu_group(args.train_gpu_group),
        )
        _run(
            cmd,
            cwd=args.sphere_repo,
            dry_run=args.dry_run,
            gpu_group=_format_gpu_group(args.train_gpu_group),
        )
    else:
        print(
            f'>>> skip training: latest checkpoint already at ep{latest_epoch:04d} '
            f'for {args.job_dir}'
        )

    if not args.dry_run:
        if not target_ckpt_path.exists():
            raise FileNotFoundError(f'target checkpoint missing after training: {target_ckpt_path}')
        if not _training_metrics_has_epoch(job_path, args.target_ckpt_epoch):
            raise RuntimeError(
                f'training_metrics.jsonl does not contain epoch {args.target_ckpt_epoch} '
                f'for {args.job_dir}'
            )

    followup_config_path = _build_followup_canonical_config(
        base_config_path=canonical_config_path,
        workspace_root=workspace_root,
        job_dir=args.job_dir,
        target_ckpt_epoch=args.target_ckpt_epoch,
    )
    print(f'>>> generated follow-up canonical config: {followup_config_path}')

    canonical_probe_cmd = [
        sys.executable,
        '-m',
        'sphere_basin.canonical_probe_matrix',
        '--config',
        str(followup_config_path),
        '--sphere-repo',
        args.sphere_repo,
    ]
    if args.force_canonical:
        canonical_probe_cmd.append('--force')
    if args.dry_run:
        canonical_probe_cmd.append('--dry-run')
    _run(canonical_probe_cmd, cwd=str(project_root()), dry_run=args.dry_run)

    canonical_master_cmd = [
        sys.executable,
        '-m',
        'sphere_basin.canonical_master',
        '--config',
        str(followup_config_path),
    ]
    _run(canonical_master_cmd, cwd=str(project_root()), dry_run=args.dry_run)

    if not args.dry_run:
        compare_path = workspace_root / 'research_summary' / f'{args.job_dir}-ep{args.target_ckpt_epoch:04d}-followup_compare.csv'
        _write_followup_compare_csv(
            canonical_master_path=workspace_root / 'research_summary' / 'canonical_master.csv',
            out_path=compare_path,
            job_dir=args.job_dir,
            target_ckpt_epoch=args.target_ckpt_epoch,
        )
        print(f'>>> wrote follow-up compare csv: {compare_path}')


if __name__ == '__main__':
    main()
