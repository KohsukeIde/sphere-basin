from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import load_yaml
from .parse_eval import parse_eval_dir
from .paths import ensure_workspace_compat, find_job_dir


def _load_json(path: Path) -> dict[str, Any]:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def _latest_by_keys(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out['_row_order'] = range(len(out))
    if 'timestamp' in out.columns:
        out['_sort_ts'] = pd.to_datetime(out['timestamp'], errors='coerce')
        out = out.sort_values(['_sort_ts', '_row_order'])
    else:
        out = out.sort_values(['_row_order'])
    out = out.drop_duplicates(subset=keys, keep='last')
    return out.drop(columns=['_row_order', '_sort_ts'], errors='ignore')


def _job_meta(job_dir: str, job_path: Path) -> dict[str, Any]:
    cfg = _load_json(job_path / 'cfg.json')
    return {
        'job_dir': job_dir,
        'job_path': str(job_path),
        'alpha': cfg.get('noise_sigma_max_angle'),
        'dataset_name': cfg.get('dataset_name'),
        'image_size': cfg.get('image_size'),
        'vit_enc_model_size': cfg.get('vit_enc_model_size'),
        'vit_dec_model_size': cfg.get('vit_dec_model_size'),
    }


def _load_canonical_lookup(workspace_root: Path) -> pd.DataFrame:
    path = workspace_root / 'research_summary' / 'canonical_master.csv'
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return df
    for col in ['forward_steps', 'tau_deg']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def build_table(cfg: dict[str, Any], workspace_root: Path) -> pd.DataFrame:
    eval_cfg = dict(cfg['eval'])
    forward_steps = [int(x) for x in eval_cfg.get('forward_steps', [4])]
    canonical = _load_canonical_lookup(workspace_root)

    rows: list[dict[str, Any]] = []
    for task in cfg['tasks']:
        job_dir = str(task['job_dir'])
        job_path = find_job_dir(workspace_root, job_dir)
        meta = _job_meta(job_dir, job_path)
        df_eval = parse_eval_dir(job_path / 'eval')
        if df_eval.empty:
            continue
        out = df_eval.copy()
        for col in ['fid', 'isc_mean', 'isc_std', 'forward_steps', 'cfg']:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors='coerce')
        for col in ['cache_sampling_noise', 'use_sampling_scheduler', 'use_ema']:
            if col in out.columns:
                out[col] = out[col].astype(str).str.lower()

        ckpt_epoch = task['ckpt_fname'].replace('.pth', '')
        mask = (
            (out['ckpt_epoch'] == ckpt_epoch)
            & (out['task_mode'] == 'generation')
            & (out['forward_steps'].isin(forward_steps))
            & (out['cfg'] == float(eval_cfg.get('cfg_min', 1.0)))
            & (out['cfg_position'] == str(eval_cfg.get('cfg_position', 'combo')))
            & (out['cache_sampling_noise'] == str(bool(task['cache_sampling_noise'])).lower())
            & (out['use_sampling_scheduler'] == str(bool(task['use_sampling_scheduler'])).lower())
            & (out['use_ema'] == str(bool(eval_cfg.get('use_ema_model', False))).lower())
        )
        out = out[mask].copy()
        if out.empty:
            continue
        out = _latest_by_keys(
            out,
            [
                'ckpt_epoch',
                'task_mode',
                'forward_steps',
                'cfg',
                'cfg_position',
                'cache_sampling_noise',
                'use_sampling_scheduler',
                'use_ema',
            ],
        )
        out['job_dir'] = job_dir
        out['regime_name'] = str(task['regime_name'])
        out['noise_mode'] = 'shared' if bool(task['cache_sampling_noise']) else 'independent'
        out['schedule_mode'] = 'decay' if bool(task['use_sampling_scheduler']) else 'fixed'
        for key, val in meta.items():
            out[key] = val
        rows.append(out)

    if not rows:
        return pd.DataFrame()

    df = pd.concat(rows, ignore_index=True)
    if not canonical.empty:
        merged = canonical[
            (canonical['forward_steps'].isin(forward_steps))
            & (canonical['tau_deg'] == 60.0)
        ][
            [
                'job_dir',
                'ckpt_epoch',
                'regime_name',
                'forward_steps',
                'nn_terminal_capture_mass',
                'nn_preterminal_capture_mass',
                'terminal_cdf_mass',
                'nn_terminal_angle_mean_deg',
                'nn_preterminal_angle_mean_deg',
                'curvature_mean_deg',
                'kappa_mean',
            ]
        ].copy()
        df = df.merge(
            merged,
            on=['job_dir', 'ckpt_epoch', 'regime_name', 'forward_steps'],
            how='left',
        )
    return df.sort_values(['regime_name', 'alpha', 'ckpt_epoch']).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description='Build a master table for official eval key checkpoints.')
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    workspace_root = ensure_workspace_compat(cfg['experiment'].get('dev_dir'))
    out_dir = workspace_root / 'research_summary'
    out_dir.mkdir(parents=True, exist_ok=True)

    df = build_table(cfg, workspace_root)
    out_path = out_dir / 'official_eval_master.csv'
    df.to_csv(out_path, index=False)
    print(f'wrote official eval master to: {out_path}')


if __name__ == '__main__':
    main()
