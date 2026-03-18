from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _to_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors='coerce')
    return out


def _filter_phase_rows(
    df: pd.DataFrame,
    forward_steps: int,
    tau_deg: float,
    capture_col: str,
) -> pd.DataFrame:
    out = _to_numeric(
        df,
        [
            'alpha',
            'fid',
            'forward_steps',
            'tau_deg',
            capture_col,
            'terminal_capture_mass',
            'terminal_angle_mean_deg',
            'train_dist_loss',
        ],
    )
    if 'task_mode' in out.columns:
        out = out[out['task_mode'] == 'generation']
    if 'forward_steps' in out.columns:
        out = out[out['forward_steps'] == forward_steps]
    if 'tau_deg' in out.columns:
        out = out[out['tau_deg'] == tau_deg]
    out = out.sort_values('alpha')
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description='Plot alpha phase diagrams.')
    parser.add_argument('--csv', type=str, required=True)
    parser.add_argument('--out-dir', type=str, default=None)
    parser.add_argument('--forward-steps', type=int, default=4)
    parser.add_argument('--tau-deg', type=float, default=30.0)
    parser.add_argument('--capture-col', type=str, default='capture_mass')
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir) if args.out_dir else csv_path.parent / 'plots'
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    if 'alpha' not in df.columns:
        raise ValueError('expected an alpha column in the aggregated CSV')

    phase = _filter_phase_rows(
        df,
        forward_steps=args.forward_steps,
        tau_deg=args.tau_deg,
        capture_col=args.capture_col,
    )
    if phase.empty:
        raise ValueError('no rows matched the requested forward-steps / tau filter')

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(phase['alpha'], phase['fid'], marker='o')
    axes[0].set_xlabel('alpha (deg)')
    axes[0].set_ylabel('gFID / FID')
    axes[0].set_title(f'gFID vs alpha (T={args.forward_steps})')

    axes[1].plot(phase['alpha'], phase[args.capture_col], marker='o')
    axes[1].set_xlabel('alpha (deg)')
    axes[1].set_ylabel(f'Capture @ {args.tau_deg:g} deg')
    axes[1].set_title(f'NN manifold capture vs alpha (T={args.forward_steps})')

    fig.tight_layout()
    fig.savefig(out_dir / 'phase_diagram_main.png', dpi=160)
    plt.close(fig)

    if 'train_dist_loss' in phase.columns and phase['train_dist_loss'].notna().any():
        plt.figure(figsize=(6, 4))
        plt.plot(phase['alpha'], phase['train_dist_loss'], marker='o')
        plt.xlabel('alpha (deg)')
        plt.ylabel('train dist_loss')
        plt.title('Reconstruction proxy vs alpha')
        plt.tight_layout()
        plt.savefig(out_dir / 'alpha_vs_recon_proxy.png', dpi=160)
        plt.close()

    if args.capture_col in phase.columns and 'fid' in phase.columns:
        plt.figure(figsize=(6, 4))
        plt.scatter(phase[args.capture_col], phase['fid'])
        plt.xlabel(f'Capture @ {args.tau_deg:g} deg')
        plt.ylabel('gFID / FID')
        plt.title('FID vs NN manifold capture')
        plt.tight_layout()
        plt.savefig(out_dir / 'capture_vs_fid.png', dpi=160)
        plt.close()

    print(f'plots saved to: {out_dir}')


if __name__ == '__main__':
    main()
