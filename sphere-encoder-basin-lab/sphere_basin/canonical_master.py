from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import load_yaml
from .paths import ensure_workspace_compat, find_job_dir


def _load_json(path: Path) -> dict[str, Any]:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def _normalize_ckpt_epoch(ckpt_name: str) -> str:
    return ckpt_name[:-4] if ckpt_name.endswith('.pth') else ckpt_name


def _epoch_number(ckpt_epoch: str) -> float | None:
    digits = ''.join(ch for ch in ckpt_epoch if ch.isdigit())
    if not digits:
        return None
    return float(int(digits))


def _load_training_rows(job_path: Path) -> dict[float, dict[str, Any]]:
    path = job_path / 'research' / 'training_metrics.jsonl'
    if not path.exists():
        return {}
    rows: dict[float, dict[str, Any]] = {}
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            epoch_raw = row.get('epoch')
            try:
                epoch = float(epoch_raw)
            except (TypeError, ValueError):
                continue
            rows[epoch] = row
    return rows


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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
        'batch_size_per_rank': cfg.get('batch_size_per_rank'),
        'sigma_max': cfg.get('noise_sigma_max'),
        'latent_channels': cfg.get('lat_ch'),
        'latent_size': cfg.get('latent_size'),
    }


def _probe_path_candidates(
    job_path: Path,
    *,
    ckpt_epoch: str,
    cfg: float,
    cfg_position: str,
    use_sampling_scheduler: bool,
    cache_sampling_noise: bool,
    use_ema_model: bool,
    seed: int | None,
) -> list[Path]:
    base = (
        f'probe_ckpt={ckpt_epoch}'
        f'_cfg={cfg}-{cfg_position}'
        f'_sched={use_sampling_scheduler}'
        f'_cache={cache_sampling_noise}'
        f'_ema={use_ema_model}'
    )
    paths: list[Path] = []
    if seed is not None:
        paths.append(job_path / 'research' / f'{base}_seed={seed}.json')
    paths.append(job_path / 'research' / f'{base}.json')
    return paths


def _build_expected_targets(cfg: dict[str, Any], workspace_root: Path) -> list[dict[str, Any]]:
    probe_cfg = cfg['probe']
    targets: list[dict[str, Any]] = []
    for job_spec in cfg['jobs']:
        job_dir = str(job_spec['job_dir'])
        job_path = find_job_dir(workspace_root, job_dir)
        meta = _job_meta(job_dir, job_path)
        training_rows = _load_training_rows(job_path)
        for ckpt in job_spec['checkpoints']:
            ckpt_epoch = _normalize_ckpt_epoch(str(ckpt))
            train_epoch = _epoch_number(ckpt_epoch)
            train_row = training_rows.get(train_epoch, {})
            for regime in probe_cfg['regimes']:
                probe_paths = _probe_path_candidates(
                    job_path,
                    ckpt_epoch=ckpt_epoch,
                    cfg=float(probe_cfg.get('cfg', 1.0)),
                    cfg_position=str(probe_cfg.get('cfg_position', 'combo')),
                    use_sampling_scheduler=bool(regime['use_sampling_scheduler']),
                    cache_sampling_noise=bool(regime['cache_sampling_noise']),
                    use_ema_model=bool(probe_cfg.get('use_ema_model', False)),
                    seed=(
                        None
                        if probe_cfg.get('seed') is None
                        else int(probe_cfg.get('seed', 0))
                    ),
                )
                probe_path = next((path for path in probe_paths if path.exists()), probe_paths[0])
                targets.append(
                    {
                        **meta,
                        'ckpt_epoch': ckpt_epoch,
                        'train_epoch': train_epoch,
                        'train_row': train_row,
                        'regime_name': str(regime['name']),
                        'noise_mode': 'shared' if bool(regime['cache_sampling_noise']) else 'independent',
                        'schedule_mode': 'decay' if bool(regime['use_sampling_scheduler']) else 'fixed',
                        'cache_sampling_noise': bool(regime['cache_sampling_noise']),
                        'use_sampling_scheduler': bool(regime['use_sampling_scheduler']),
                        'use_ema': bool(probe_cfg.get('use_ema_model', False)),
                        'probe_seed': probe_cfg.get('seed'),
                        'probe_path': probe_path,
                    }
                )
    return targets


def _merge_train_metrics(base: dict[str, Any], train_row: dict[str, Any], forward_steps: int) -> dict[str, Any]:
    out = dict(base)
    out['train_epoch_available'] = bool(train_row)
    if not train_row:
        out['fid'] = None
        out['recon_mse'] = None
        out['recon_l1'] = None
        out['isc_mean'] = None
        out['isc_std'] = None
        out['train_step'] = None
        return out
    out['train_step'] = _to_float(train_row.get('step'))
    out['recon_mse'] = _to_float(train_row.get('Eval/Recon_MSE'))
    out['recon_l1'] = _to_float(train_row.get('Eval/Recon_L1'))
    out['fid'] = _to_float(train_row.get(f'Eval/FID_{forward_steps}step'))
    out['isc_mean'] = _to_float(train_row.get(f'Eval/ISC_mean_{forward_steps}step'))
    out['isc_std'] = _to_float(train_row.get(f'Eval/ISC_std_{forward_steps}step'))
    return out


def build_tables(cfg: dict[str, Any], workspace_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    master_cfg = dict(cfg.get('master', {}))
    target_noise_scaler = float(master_cfg.get('contraction_noise_scaler', 1.0))
    targets = _build_expected_targets(cfg, workspace_root)

    prior_rows: list[dict[str, Any]] = []
    contraction_rows: list[dict[str, Any]] = []
    master_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []

    for target in targets:
        probe_path = Path(target['probe_path'])
        manifest_rows.append(
            {
                'job_dir': target['job_dir'],
                'alpha': target['alpha'],
                'ckpt_epoch': target['ckpt_epoch'],
                'regime_name': target['regime_name'],
                'probe_path': str(probe_path),
                'exists': probe_path.exists(),
            }
        )
        if not probe_path.exists():
            continue

        data = _load_json(probe_path)
        meta = dict(data.get('meta', {}))
        base = {
            k: v
            for k, v in target.items()
            if k not in {'train_row', 'probe_path'}
        }
        base.update(meta)
        base['probe_file'] = str(probe_path)

        contraction_lookup: dict[tuple[float, float], dict[str, Any]] = {}
        for row in data.get('contraction_rows', []):
            merged = dict(base)
            merged.update(row)
            contraction_rows.append(merged)
            contraction_lookup[(float(row['noise_scaler']), float(row['tau_deg']))] = merged

        for row in data.get('prior_rows', []):
            merged = dict(base)
            merged.update(row)
            prior_rows.append(merged)

            selected_contraction = contraction_lookup.get((target_noise_scaler, float(row['tau_deg'])))
            master_row = {
                **merged,
                'nn_capture_mass': merged.get('capture_mass'),
                'terminal_cdf_mass': merged.get('terminal_capture_mass'),
                'nn_terminal_capture_mass': merged.get('capture_mass'),
                'nn_preterminal_capture_mass': merged.get('preterminal_capture_mass'),
                'nn_terminal_angle_mean_deg': merged.get('nn_angle_after_mean_deg'),
                'nn_terminal_angle_improvement_mean_deg': merged.get('nn_angle_improvement_mean_deg'),
                'nn_preterminal_angle_mean_deg': merged.get('nn_angle_preterminal_mean_deg'),
                'nn_preterminal_angle_improvement_mean_deg': merged.get('nn_angle_preterminal_improvement_mean_deg'),
                'nn_preterminal_improved_mass': merged.get('nn_preterminal_improved_mass'),
                'target_contraction_noise_scaler': target_noise_scaler,
            }
            master_row = _merge_train_metrics(master_row, target['train_row'], int(row['forward_steps']))
            if selected_contraction is not None:
                master_row['kappa_mean'] = selected_contraction.get('kappa_mean')
                master_row['kappa_std'] = selected_contraction.get('kappa_std')
                master_row['contraction_angle_in_mean_deg'] = selected_contraction.get('angle_in_mean_deg')
                master_row['contraction_angle_out_mean_deg'] = selected_contraction.get('angle_out_mean_deg')
                master_row['contraction_capture_mass'] = selected_contraction.get('capture_mass')
                master_row['contraction_probe_file'] = selected_contraction.get('probe_file')
            else:
                master_row['kappa_mean'] = None
                master_row['kappa_std'] = None
                master_row['contraction_angle_in_mean_deg'] = None
                master_row['contraction_angle_out_mean_deg'] = None
                master_row['contraction_capture_mass'] = None
                master_row['contraction_probe_file'] = None
            master_rows.append(master_row)

    df_prior = pd.DataFrame(prior_rows)
    df_contraction = pd.DataFrame(contraction_rows)
    df_master = pd.DataFrame(master_rows)
    df_manifest = pd.DataFrame(manifest_rows)
    return df_prior, df_contraction, df_master, df_manifest


def _phase_export(df_master: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    master_cfg = dict(cfg.get('master', {}))
    out = df_master.copy()
    if out.empty:
        return out
    out = out[out['ckpt_epoch'] == str(master_cfg.get('phase_ckpt_epoch', 'ep0024'))]
    out = out[out['regime_name'] == str(master_cfg.get('phase_regime_name', 'independent-fixed'))]
    out = out[out['forward_steps'] == int(master_cfg.get('phase_forward_steps', 4))]
    out = out[out['tau_deg'] == float(master_cfg.get('phase_tau_deg', 60.0))]
    out = out.sort_values('alpha')
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description='Build canonical post-hoc probe tables.')
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    workspace_root = ensure_workspace_compat(cfg['experiment'].get('dev_dir'))
    out_dir = workspace_root / 'research_summary'
    out_dir.mkdir(parents=True, exist_ok=True)

    df_prior, df_contraction, df_master, df_manifest = build_tables(cfg, workspace_root)

    df_prior.to_csv(out_dir / 'canonical_prior_long.csv', index=False)
    df_contraction.to_csv(out_dir / 'canonical_contraction_long.csv', index=False)
    df_master.to_csv(out_dir / 'canonical_master.csv', index=False)
    df_manifest.to_csv(out_dir / 'canonical_probe_manifest.csv', index=False)

    phase = _phase_export(df_master, cfg)
    phase.to_csv(out_dir / 'canonical_phase.csv', index=False)

    print(f'wrote canonical tables to: {out_dir}')


if __name__ == '__main__':
    main()
