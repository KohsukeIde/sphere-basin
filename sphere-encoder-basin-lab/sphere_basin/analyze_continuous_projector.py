from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .paths import ensure_workspace_compat, project_root


def _auc_by_tau(df: pd.DataFrame, value_col: str) -> float | None:
    out = df[['tau_deg', value_col]].dropna().sort_values('tau_deg')
    if out.empty or out.shape[0] < 2:
        return None
    x = out['tau_deg'].to_numpy(dtype=float)
    y = out[value_col].to_numpy(dtype=float)
    span = float(x[-1] - x[0])
    if span <= 0:
        return None
    return float(np.trapezoid(y, x) / span)


def build_summary(canonical_path: Path, official_path: Path) -> pd.DataFrame:
    canonical = pd.read_csv(canonical_path)
    official = pd.read_csv(official_path) if official_path.exists() else pd.DataFrame()

    keep_jobs = {
        'sphere-small-small-cifar-10-32px-a80-pilot',
        'sphere-small-small-cifar-10-32px-a80-no-lat-con-causal',
        'sphere-small-small-cifar-10-32px-a80-no_pix_con-causal',
        'sphere-small-small-cifar-10-32px-a80-recon_only-causal',
        'sphere-small-small-cifar-10-32px-a83-pilot',
        'sphere-small-small-cifar-10-32px-a85-pilot',
    }
    canonical = canonical[
        (canonical['job_dir'].isin(keep_jobs))
        & (canonical['regime_name'] == 'shared-fixed')
        & (canonical['forward_steps'] == 4)
    ].copy()
    if canonical.empty:
        return canonical

    rows = []
    keys = ['job_dir', 'ckpt_epoch', 'regime_name', 'forward_steps']
    for key, group in canonical.groupby(keys):
        row60 = group[group['tau_deg'] == 60.0].iloc[-1]
        row = {
            'job_dir': key[0],
            'ckpt_epoch': key[1],
            'regime_name': key[2],
            'forward_steps': key[3],
            'alpha': row60.get('alpha'),
            'terminal_angle_mean_deg': row60.get('terminal_angle_mean_deg'),
            'terminal_angle_median_deg': row60.get('terminal_angle_median_deg'),
            'terminal_angle_p25_deg': row60.get('terminal_angle_p25_deg'),
            'terminal_angle_p75_deg': row60.get('terminal_angle_p75_deg'),
            'nn_preterminal_angle_mean_deg': row60.get('nn_preterminal_angle_mean_deg'),
            'nn_terminal_angle_mean_deg': row60.get('nn_terminal_angle_mean_deg'),
            'projector_delta_mean_deg': (
                float(row60['nn_preterminal_angle_mean_deg']) - float(row60['nn_terminal_angle_mean_deg'])
            ),
            'projector_delta_improvement_mean_deg': row60.get('nn_terminal_angle_improvement_mean_deg'),
            'terminal_cdf_auc': _auc_by_tau(group, 'terminal_cdf_mass'),
            'nn_capture_auc': _auc_by_tau(group, 'nn_terminal_capture_mass'),
            'kappa_mean': row60.get('kappa_mean'),
            'curvature_mean_deg': row60.get('curvature_mean_deg'),
            'nn_preterminal_capture_mass_tau60': row60.get('nn_preterminal_capture_mass'),
            'nn_terminal_capture_mass_tau60': row60.get('nn_terminal_capture_mass'),
            'terminal_cdf_mass_tau60': row60.get('terminal_cdf_mass'),
            'fid_light': row60.get('fid'),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    if not official.empty:
        official = official[
            (official['regime_name'] == 'shared-fixed') & (official['forward_steps'] == 4)
        ][['job_dir', 'ckpt_epoch', 'fid']].rename(columns={'fid': 'official_fid'})
        df = df.merge(official, on=['job_dir', 'ckpt_epoch'], how='left')
    order = {
        'sphere-small-small-cifar-10-32px-a80-pilot': 0,
        'sphere-small-small-cifar-10-32px-a80-no-lat-con-causal': 1,
        'sphere-small-small-cifar-10-32px-a80-no_pix_con-causal': 2,
        'sphere-small-small-cifar-10-32px-a80-recon_only-causal': 3,
        'sphere-small-small-cifar-10-32px-a83-pilot': 4,
        'sphere-small-small-cifar-10-32px-a85-pilot': 5,
    }
    df['_order'] = df['job_dir'].map(order).fillna(999)
    return df.sort_values(['_order', 'ckpt_epoch']).drop(columns=['_order'])


def main() -> None:
    parser = argparse.ArgumentParser(description='Summarize continuous terminal/projector metrics.')
    parser.add_argument(
        '--workspace',
        type=str,
        default=str(project_root() / 'workspace'),
    )
    args = parser.parse_args()

    workspace = ensure_workspace_compat(args.workspace)
    canonical = workspace / 'research_summary' / 'canonical_master.csv'
    official = workspace / 'research_summary' / 'official_eval_master.csv'
    out_path = workspace / 'research_summary' / 'continuous_projector_summary.csv'
    df = build_summary(canonical, official)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f'wrote continuous summary to: {out_path}')


if __name__ == '__main__':
    main()
