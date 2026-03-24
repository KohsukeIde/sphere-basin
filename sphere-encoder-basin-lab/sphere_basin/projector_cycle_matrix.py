from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from .config import load_yaml
from .paths import ensure_workspace_compat, project_root


def _normalize_rows(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.clip(norms, 1e-8, None)


def _nearest_neighbor_angles_deg(queries: np.ndarray, bank: np.ndarray) -> np.ndarray:
    queries = _normalize_rows(np.asarray(queries, dtype=np.float32))
    bank = _normalize_rows(np.asarray(bank, dtype=np.float32))
    dots = np.clip(queries @ bank.T, -1.0, 1.0)
    return np.rad2deg(np.arccos(dots.max(axis=1))).astype(np.float32)


def _effective_rank(features: np.ndarray) -> float:
    x = np.asarray(features, dtype=np.float32)
    x = x - x.mean(axis=0, keepdims=True)
    s = np.linalg.svd(x, compute_uv=False)
    power = np.square(s)
    total = float(power.sum())
    if total <= 0.0:
        return 0.0
    p = power / total
    entropy = -float(np.sum(p * np.log(np.clip(p, 1e-12, None))))
    return float(np.exp(entropy))


def _pairwise_angle_summary(features: np.ndarray) -> tuple[float, float]:
    x = _normalize_rows(np.asarray(features, dtype=np.float32))
    dots = np.clip(x @ x.T, -1.0, 1.0)
    tri = np.triu_indices(x.shape[0], k=1)
    angles = np.rad2deg(np.arccos(dots[tri])).astype(np.float32)
    if angles.size == 0:
        return 0.0, 0.0
    return float(angles.mean()), float(angles.std())


def _summarize(prefix: str, values: np.ndarray) -> dict[str, float]:
    return {
        f'{prefix}_mean': float(values.mean()),
        f'{prefix}_median': float(np.median(values)),
        f'{prefix}_p25': float(np.percentile(values, 25.0)),
        f'{prefix}_p75': float(np.percentile(values, 75.0)),
    }


def _prepare_imports(sphere_repo: Path) -> None:
    sys.path.insert(0, str(sphere_repo))


@torch.no_grad()
def _build_train_latent_bank(model, loader, device, num_data_samples: int) -> np.ndarray:
    local_rows = []
    for batch in loader:
        imgs, labels = batch[0].to(device), batch[1].to(device)
        y = labels if model.num_classes > 0 else None
        z_clean = model.encoder(imgs, y)
        v = model.spherify(z_clean, sampling=False)
        local_rows.append(v.flatten(1).float().cpu().numpy())
    bank = np.concatenate(local_rows, axis=0) if local_rows else np.empty((0, 0), dtype=np.float32)
    if num_data_samples > 0 and bank.shape[0] > num_data_samples:
        bank = bank[:num_data_samples]
    return _normalize_rows(bank.astype(np.float32, copy=False))


@torch.no_grad()
def _run_task(task: dict[str, Any], cfg: dict[str, Any], sphere_repo: Path, workspace: Path) -> list[dict[str, Any]]:
    _prepare_imports(sphere_repo)
    from research.compat import (
        apply_cfg_on_decode,
        apply_cfg_on_encode,
        build_analysis_loader,
        build_model,
        destroy_dist,
        init_dist,
        load_job_args,
        sample_latent_trajectory,
    )
    from sphere.lpips import LPIPS

    probe_cfg = cfg['probe']
    _, _, ddp_world_size, device = init_dist('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        cli_overrides = {
            'job_dir': task['job_dir'],
            'ckpt_fname': task['ckpt_fname'],
        }
        args, _ = load_job_args(str(workspace), task['job_dir'], cli_overrides)
        model, ckpt_epoch = build_model(args, device=device, use_ema_model=False, compile_model=False)
        loader = build_analysis_loader(
            args,
            ddp_rank=0,
            ddp_world_size=ddp_world_size,
            split='train',
            max_samples=int(probe_cfg.get('num_data_samples', 4096)),
            batch_size_per_rank=int(probe_cfg.get('batch_size_per_rank', 32)),
            num_workers=int(probe_cfg.get('num_workers', 4)),
        )
        train_bank = _build_train_latent_bank(
            model,
            loader=loader,
            device=device,
            num_data_samples=int(probe_cfg.get('num_data_samples', 4096)),
        )
        lpips_root = Path(str(probe_cfg.get('lpips_ckpt_path', workspace / 'pretrained' / 'lpips')))
        if not lpips_root.is_absolute():
            if lpips_root.parts and lpips_root.parts[0] == workspace.name:
                lpips_root = workspace.parent / lpips_root
            else:
                lpips_root = workspace / lpips_root
        lpips = LPIPS(ckpt_pth=str(lpips_root)).to(device).eval()

        total = int(probe_cfg.get('num_prior_samples', 1024))
        batch = int(probe_cfg.get('batch_size_per_rank', 32))
        assert total % batch == 0, 'num_prior_samples must divide batch_size_per_rank'

        z_pre_list = []
        z_t1_list = []
        z_t2_list = []
        lpips_01 = []
        lpips_12 = []
        l2_01 = []
        l2_12 = []

        fixed_class = probe_cfg.get('fixed_class')
        if fixed_class is None and model.num_classes > 0:
            fixed_class = 0

        for _ in range(total // batch):
            zs, xs = sample_latent_trajectory(
                model=model,
                batch_size=batch,
                device=device,
                forward_steps=int(probe_cfg.get('forward_steps', 4)),
                cfg=float(probe_cfg.get('cfg', 1.0)),
                cfg_position=str(probe_cfg.get('cfg_position', 'combo')),
                use_sampling_scheduler=bool(probe_cfg.get('use_sampling_scheduler', False)),
                cache_sampling_noise=bool(probe_cfg.get('cache_sampling_noise', True)),
                fixed_class=fixed_class,
            )
            z_pre = zs[-2].to(device)
            x0 = xs[-1].to(device)
            y = None
            y_uncond = None
            if model.num_classes > 0:
                y = torch.full((batch,), int(fixed_class), device=device, dtype=torch.long)
                y_uncond = torch.full_like(y, model.num_classes)
            z_t1 = apply_cfg_on_encode(model, x0, y, y_uncond, float(probe_cfg.get('cfg', 1.0)), str(probe_cfg.get('cfg_position', 'combo')))
            z_t1 = model.spherify(z_t1, sampling=False)
            x1 = apply_cfg_on_decode(model, z_t1, y, y_uncond, float(probe_cfg.get('cfg', 1.0)), str(probe_cfg.get('cfg_position', 'combo')))
            x1 = torch.clamp(x1 * 0.5 + 0.5, 0, 1)
            z_t2 = apply_cfg_on_encode(model, x1, y, y_uncond, float(probe_cfg.get('cfg', 1.0)), str(probe_cfg.get('cfg_position', 'combo')))
            z_t2 = model.spherify(z_t2, sampling=False)
            x2 = apply_cfg_on_decode(model, z_t2, y, y_uncond, float(probe_cfg.get('cfg', 1.0)), str(probe_cfg.get('cfg_position', 'combo')))
            x2 = torch.clamp(x2 * 0.5 + 0.5, 0, 1)

            z_pre_list.append(z_pre.flatten(1).cpu().numpy())
            z_t1_list.append(z_t1.flatten(1).cpu().numpy())
            z_t2_list.append(z_t2.flatten(1).cpu().numpy())
            lpips_01.append(lpips(x0, x1).flatten().cpu().numpy())
            lpips_12.append(lpips(x1, x2).flatten().cpu().numpy())
            l2_01.append(F.mse_loss(x0, x1, reduction='none').flatten(1).mean(dim=1).cpu().numpy())
            l2_12.append(F.mse_loss(x1, x2, reduction='none').flatten(1).mean(dim=1).cpu().numpy())

        z_pre = np.concatenate(z_pre_list, axis=0)
        z_t1 = np.concatenate(z_t1_list, axis=0)
        z_t2 = np.concatenate(z_t2_list, axis=0)
        lpips_01 = np.concatenate(lpips_01, axis=0)
        lpips_12 = np.concatenate(lpips_12, axis=0)
        l2_01 = np.concatenate(l2_01, axis=0)
        l2_12 = np.concatenate(l2_12, axis=0)

        rows = []
        for stage_name, latents in [('preterminal', z_pre), ('terminal1', z_t1), ('terminal2', z_t2)]:
            nn = _nearest_neighbor_angles_deg(latents, train_bank)
            pair_mean, pair_std = _pairwise_angle_summary(latents)
            row = {
                'label': task['label'],
                'job_dir': task['job_dir'],
                'ckpt_epoch': ckpt_epoch,
                'stage': stage_name,
                'alpha': getattr(args, 'noise_sigma_max_angle', None),
                'nn_angle_mean_deg': float(nn.mean()),
                'nn_angle_median_deg': float(np.median(nn)),
                'nn_angle_p25_deg': float(np.percentile(nn, 25.0)),
                'nn_angle_p75_deg': float(np.percentile(nn, 75.0)),
                'effective_rank': _effective_rank(latents),
                'pairwise_angle_mean_deg': pair_mean,
                'pairwise_angle_std_deg': pair_std,
            }
            rows.append(row)

        rows.append(
            {
                'label': task['label'],
                'job_dir': task['job_dir'],
                'ckpt_epoch': ckpt_epoch,
                'stage': 'cycle_delta',
                **_summarize('lpips_01', lpips_01),
                **_summarize('lpips_12', lpips_12),
                **_summarize('l2_01', l2_01),
                **_summarize('l2_12', l2_12),
            }
        )
        return rows
    finally:
        destroy_dist()


def main() -> None:
    parser = argparse.ArgumentParser(description='Run projector-cycle intervention summaries.')
    parser.add_argument('--config', type=str, default=str(project_root() / 'configs' / 'projector_cycle_matrix.yaml'))
    parser.add_argument('--sphere-repo', type=str, required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    workspace = ensure_workspace_compat(cfg['experiment'].get('dev_dir'))
    sphere_repo = Path(args.sphere_repo).resolve()
    rows = []
    for task in cfg['tasks']:
        rows.extend(_run_task(task, cfg, sphere_repo, workspace))

    out_dir = workspace / 'research_summary'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / 'projector_cycle_summary.csv'
    if rows:
        fieldnames = sorted({k for row in rows for k in row.keys()})
        with out_csv.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    else:
        out_csv.write_text('', encoding='utf-8')
    print(f'wrote projector cycle summary to: {out_csv}')


if __name__ == '__main__':
    main()
