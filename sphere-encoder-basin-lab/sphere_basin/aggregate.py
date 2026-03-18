from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .parse_eval import parse_eval_dir
from .paths import default_workspace_root, resolve_dev_dir


def _load_json(path: Path) -> dict[str, Any]:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def _latest_by_keys(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if df.empty:
        return df

    dedupe_keys = [key for key in keys if key in df.columns]
    if not dedupe_keys:
        return df

    out = df.copy()
    out['_row_order'] = range(len(out))
    if 'timestamp' in out.columns:
        out['_sort_ts'] = pd.to_datetime(out['timestamp'], errors='coerce')
        out = out.sort_values(['_sort_ts', '_row_order'])
    else:
        out = out.sort_values(['_row_order'])
    out = out.drop_duplicates(subset=dedupe_keys, keep='last')
    return out.drop(columns=['_row_order', '_sort_ts'], errors='ignore')


def collect_train_summary(job_dir: Path) -> dict[str, Any]:
    path = job_dir / 'log.jsonl'
    if not path.exists():
        return {}

    last_entry: dict[str, Any] | None = None
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            last_entry = json.loads(line)

    if last_entry is None:
        return {}

    summary: dict[str, Any] = {'job_dir': job_dir.name, 'train_log_path': str(path)}
    for key, value in last_entry.items():
        try:
            summary[f'train_{key}'] = float(value)
        except (TypeError, ValueError):
            summary[f'train_{key}'] = value
    return summary


def collect_probe_rows(job_dir: Path) -> tuple[list[dict], list[dict]]:
    research_dir = job_dir / 'research'
    prior_rows: list[dict] = []
    contraction_rows: list[dict] = []
    if not research_dir.exists():
        return prior_rows, contraction_rows
    manifest = _load_json(research_dir / 'manifest.json') if (research_dir / 'manifest.json').exists() else {}
    for path in sorted(research_dir.glob('probe_*.json')):
        data = _load_json(path)
        meta = data.get('meta', {})
        for row in data.get('prior_rows', []):
            merged = dict(manifest)
            merged.update(meta)
            merged.update(row)
            merged['job_dir'] = job_dir.name
            merged['probe_file'] = str(path)
            prior_rows.append(merged)
        for row in data.get('contraction_rows', []):
            merged = dict(manifest)
            merged.update(meta)
            merged.update(row)
            merged['job_dir'] = job_dir.name
            merged['probe_file'] = str(path)
            contraction_rows.append(merged)
    return prior_rows, contraction_rows


def main() -> None:
    parser = argparse.ArgumentParser(description='Aggregate eval + probe outputs.')
    parser.add_argument('--workspace', type=str, default=str(default_workspace_root()))
    args = parser.parse_args()

    workspace = resolve_dev_dir(args.workspace)
    exp_root = workspace / 'experiments'
    out_root = workspace / 'research_summary'
    out_root.mkdir(parents=True, exist_ok=True)

    eval_rows = []
    prior_rows = []
    contraction_rows = []
    train_rows = []

    for job_dir in sorted(exp_root.glob('*')):
        if not job_dir.is_dir():
            continue
        train_summary = collect_train_summary(job_dir)
        if train_summary:
            train_rows.append(train_summary)
        eval_dir = job_dir / 'eval'
        if eval_dir.exists():
            df_eval = parse_eval_dir(eval_dir)
            if not df_eval.empty:
                df_eval['job_dir'] = job_dir.name
                eval_rows.append(df_eval)
        p_rows, c_rows = collect_probe_rows(job_dir)
        prior_rows.extend(p_rows)
        contraction_rows.extend(c_rows)

    df_eval = pd.concat(eval_rows, ignore_index=True) if eval_rows else pd.DataFrame()
    df_prior = pd.DataFrame(prior_rows)
    df_contr = pd.DataFrame(contraction_rows)
    df_train = pd.DataFrame(train_rows)

    if not df_eval.empty:
        for col in ['fid', 'isc_mean', 'isc_std', 'forward_steps', 'cfg']:
            if col in df_eval.columns:
                df_eval[col] = pd.to_numeric(df_eval[col], errors='coerce')
        for col in ['cfg_position', 'cache_sampling_noise', 'use_sampling_scheduler', 'use_ema']:
            if col in df_eval.columns:
                df_eval[col] = df_eval[col].astype(str).str.lower()
        df_eval = _latest_by_keys(
            df_eval,
            [
                'job_dir',
                'task_mode',
                'forward_steps',
                'cfg',
                'cfg_position',
                'cache_sampling_noise',
                'use_sampling_scheduler',
                'use_ema',
                'seed_sampling',
            ],
        )
        df_eval.to_csv(out_root / 'summary_eval.csv', index=False)

    if not df_train.empty:
        df_train.to_csv(out_root / 'summary_train.csv', index=False)

    if not df_prior.empty:
        for col in ['forward_steps', 'tau_deg', 'terminal_angle_mean_deg', 'capture_mass', 'curvature_mean_deg', 'path_length_mean_deg']:
            if col in df_prior.columns:
                df_prior[col] = pd.to_numeric(df_prior[col], errors='coerce')
        for col in ['cfg_position', 'cache_sampling_noise', 'use_sampling_scheduler', 'use_ema']:
            if col in df_prior.columns:
                df_prior[col] = df_prior[col].astype(str).str.lower()
        df_prior.to_csv(out_root / 'summary_prior_probe.csv', index=False)

    if not df_contr.empty:
        for col in ['noise_scaler', 'tau_deg', 'kappa_mean', 'capture_mass', 'angle_in_mean_deg', 'angle_out_mean_deg']:
            if col in df_contr.columns:
                df_contr[col] = pd.to_numeric(df_contr[col], errors='coerce')
        df_contr.to_csv(out_root / 'summary_contraction_probe.csv', index=False)

    if not df_eval.empty and not df_prior.empty:
        keys = ['job_dir', 'forward_steps', 'cfg_position']
        extra = [k for k in ['cache_sampling_noise', 'use_sampling_scheduler', 'use_ema'] if k in df_eval.columns and k in df_prior.columns]
        merge_keys = keys + extra
        merged = df_eval.merge(df_prior, on=merge_keys, how='left', suffixes=('_eval', '_probe'))
        if not df_train.empty:
            merged = merged.merge(df_train, on='job_dir', how='left')
        merged.to_csv(out_root / 'summary_generation.csv', index=False)

    print(f'wrote summaries to: {out_root}')


if __name__ == '__main__':
    main()
