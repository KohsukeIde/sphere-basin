from __future__ import annotations

import argparse
import json
import logging
import os
import os.path as osp
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist

from cli_utils import str2bool
from research.compat import (
    ambient_curvature_deg,
    build_analysis_loader,
    build_model,
    destroy_dist,
    init_dist,
    load_job_args,
    sample_latent_trajectory,
    step_angles_deg,
)
from sphere.utils import vector_compute_angle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def gather_numpy(local_arr: np.ndarray) -> np.ndarray:
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if world_size == 1:
        return local_arr
    gather_list = [None for _ in range(world_size)]
    dist.all_gather_object(gather_list, local_arr)
    return np.concatenate(gather_list, axis=0)


def _normalize_rows(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.clip(norms, 1e-8, None)


def _summarize_array(prefix: str, values: np.ndarray) -> dict[str, float]:
    return {
        f'{prefix}_mean_deg': float(values.mean()),
        f'{prefix}_std_deg': float(values.std()),
        f'{prefix}_median_deg': float(np.median(values)),
        f'{prefix}_p25_deg': float(np.percentile(values, 25.0)),
        f'{prefix}_p75_deg': float(np.percentile(values, 75.0)),
    }


def _nearest_neighbor_angles_deg(
    queries: np.ndarray,
    bank: np.ndarray,
    *,
    chunk_size: int = 1024,
) -> np.ndarray:
    if queries.size == 0 or bank.size == 0:
        return np.empty((0,), dtype=np.float32)

    queries = _normalize_rows(np.asarray(queries, dtype=np.float32))
    bank = _normalize_rows(np.asarray(bank, dtype=np.float32))
    out = []
    for start in range(0, queries.shape[0], chunk_size):
        chunk = queries[start : start + chunk_size]
        dots = chunk @ bank.T
        max_dots = np.clip(dots.max(axis=1), -1.0, 1.0)
        out.append(np.rad2deg(np.arccos(max_dots)).astype(np.float32))
    return np.concatenate(out, axis=0)


def _noise_mode_name(cache_sampling_noise: bool) -> str:
    return 'shared' if cache_sampling_noise else 'independent'


def _schedule_mode_name(use_sampling_scheduler: bool) -> str:
    return 'decay' if use_sampling_scheduler else 'fixed'


@torch.no_grad()
def build_train_latent_bank(model, loader, device, num_data_samples: int) -> np.ndarray:
    local_rows = []
    for batch in loader:
        imgs, labels = batch[0].to(device), batch[1].to(device)
        y = labels if model.num_classes > 0 else None
        z_clean = model.encoder(imgs, y)
        v = model.spherify(z_clean, sampling=False)
        local_rows.append(v.flatten(1).float().cpu().numpy())

    local_bank = (
        np.concatenate(local_rows, axis=0)
        if local_rows
        else np.empty((0, 0), dtype=np.float32)
    )
    bank = gather_numpy(local_bank)
    if num_data_samples > 0 and bank.shape[0] > num_data_samples:
        bank = bank[:num_data_samples]
    return _normalize_rows(bank.astype(np.float32, copy=False))


parser = argparse.ArgumentParser(description='Probe basin capture and contraction metrics.')
parser.add_argument('--dev_dir', type=str, default='workspace')
parser.add_argument('--job_dir', type=str, required=True)
parser.add_argument('--ckpt_fname', type=str, default=None)
parser.add_argument('--batch_size_per_rank', type=int, default=64)
parser.add_argument('--num_prior_samples', type=int, default=4096)
parser.add_argument('--num_data_samples', type=int, default=4096)
parser.add_argument('--forward_steps', type=int, nargs='+', default=[1, 4])
parser.add_argument('--taus_deg', type=float, nargs='+', default=[5.0, 10.0, 20.0, 30.0, 45.0, 60.0])
parser.add_argument('--contraction_noise_scalers', type=float, nargs='+', default=[0.25, 0.5, 0.75, 1.0])
parser.add_argument('--cfg', type=float, default=1.0)
parser.add_argument('--cfg_position', type=str, default='combo')
parser.add_argument('--use_sampling_scheduler', type=str2bool, default=False)
parser.add_argument('--cache_sampling_noise', type=str2bool, default=False)
parser.add_argument('--use_ema_model', type=str2bool, default=False)
parser.add_argument('--compile_model', type=str2bool, default=False)
parser.add_argument('--num_workers', type=int, default=4)
cli_args = parser.parse_args()


@torch.no_grad()
def probe_prior(model, args, device, ddp_world_size, train_latent_bank: np.ndarray | None = None):
    rows = []
    total = args.num_prior_samples
    batch = args.batch_size_per_rank
    assert total % (ddp_world_size * batch) == 0, 'num_prior_samples must divide world_size*batch'
    num_batches = total // (ddp_world_size * batch)
    for fwd in args.forward_steps:
        terminal_angles = []
        curvatures = []
        path_lengths = []
        initial_latents = []
        terminal_latents = []
        for _ in range(num_batches):
            zs, _ = sample_latent_trajectory(
                model=model,
                batch_size=batch,
                device=device,
                forward_steps=fwd,
                cfg=args.cfg,
                cfg_position=args.cfg_position,
                use_sampling_scheduler=args.use_sampling_scheduler,
                cache_sampling_noise=args.cache_sampling_noise,
            )
            step_angles = step_angles_deg(zs)
            initial_latents.append(zs[0].flatten(1).cpu().numpy())
            terminal = step_angles[-1].flatten().cpu().numpy()
            terminal_angles.append(terminal)
            terminal_latents.append(zs[-1].flatten(1).cpu().numpy())
            curvatures.append(ambient_curvature_deg(zs).flatten().cpu().numpy())
            path_lengths.append(np.stack([sa.flatten().cpu().numpy() for sa in step_angles], axis=1).sum(axis=1))
        initial_latents = gather_numpy(np.concatenate(initial_latents))
        terminal_angles = gather_numpy(np.concatenate(terminal_angles))
        terminal_latents = gather_numpy(np.concatenate(terminal_latents))
        curvatures = gather_numpy(np.concatenate(curvatures))
        path_lengths = gather_numpy(np.concatenate(path_lengths))
        terminal_summary = _summarize_array('terminal_angle', terminal_angles)

        nn_before = None
        nn_after = None
        nn_improvement = None
        nn_summary: dict[str, float] = {}
        nn_improved_mass = None
        if train_latent_bank is not None and train_latent_bank.size > 0:
            nn_before = _nearest_neighbor_angles_deg(initial_latents, train_latent_bank)
            nn_after = _nearest_neighbor_angles_deg(terminal_latents, train_latent_bank)
            nn_improvement = nn_before - nn_after
            nn_summary.update(_summarize_array('nn_angle_before', nn_before))
            nn_summary.update(_summarize_array('nn_angle_after', nn_after))
            nn_summary.update(_summarize_array('nn_angle_improvement', nn_improvement))
            nn_improved_mass = float((nn_improvement > 0.0).mean())

        for tau in args.taus_deg:
            row = {
                'mode': 'prior',
                'forward_steps': fwd,
                'tau_deg': tau,
                'terminal_capture_mass': float((terminal_angles <= tau).mean()),
                'curvature_mean_deg': float(curvatures.mean()),
                'path_length_mean_deg': float(path_lengths.mean()),
                'cfg_position': args.cfg_position,
                'cache_sampling_noise': bool(args.cache_sampling_noise),
                'use_sampling_scheduler': bool(args.use_sampling_scheduler),
                'use_ema': bool(args.use_ema_model),
                **terminal_summary,
            }
            if nn_after is not None and nn_before is not None and nn_improvement is not None:
                row.update(nn_summary)
                row['capture_mass'] = float((nn_after <= tau).mean())
                row['nn_improved_mass'] = nn_improved_mass
            else:
                row['capture_mass'] = 0.0
            rows.append(row)
    return rows


@torch.no_grad()
def probe_contraction(model, args, loader, device):
    rows = []
    for noise_scaler in args.contraction_noise_scalers:
        seen = 0
        ratios = []
        angle_ins = []
        angle_outs = []
        for batch in loader:
            imgs, labels = batch[0].to(device), batch[1].to(device)
            y = labels if model.num_classes > 0 else None
            z_clean = model.encoder(imgs, y)
            v = model.spherify(z_clean, sampling=False)
            u = model.spherify(z_clean, sampling=True, noise_scaler=noise_scaler, cache_noise=False)
            x_noisy = model.decoder(u, y)
            z_ret = model.encoder(x_noisy, y)
            v_ret = model.spherify(z_ret, sampling=False)
            angle_in = vector_compute_angle(u, v).flatten().cpu().numpy()
            angle_out = vector_compute_angle(v_ret, v).flatten().cpu().numpy()
            ratios.append(angle_out / np.clip(angle_in, 1e-6, None))
            angle_ins.append(angle_in)
            angle_outs.append(angle_out)
            seen += imgs.shape[0]
            if seen >= args.num_data_samples:
                break
        ratios = gather_numpy(np.concatenate(ratios))
        angle_ins = gather_numpy(np.concatenate(angle_ins))
        angle_outs = gather_numpy(np.concatenate(angle_outs))
        for tau in args.taus_deg:
            rows.append({
                'mode': 'contraction',
                'noise_scaler': noise_scaler,
                'tau_deg': tau,
                'kappa_mean': float(ratios.mean()),
                'kappa_std': float(ratios.std()),
                'angle_in_mean_deg': float(angle_ins.mean()),
                'angle_out_mean_deg': float(angle_outs.mean()),
                'capture_mass': float((angle_outs <= tau).mean()),
                'cfg_position': args.cfg_position,
                'cache_sampling_noise': bool(args.cache_sampling_noise),
                'use_sampling_scheduler': bool(args.use_sampling_scheduler),
                'use_ema': bool(args.use_ema_model),
            })
    return rows


def main(cli_args) -> None:
    ddp_rank, _, ddp_world_size, device = init_dist('cuda' if torch.cuda.is_available() else 'cpu')
    cli_dict = vars(cli_args)
    args, exp_dir = load_job_args(cli_args.dev_dir, cli_args.job_dir, cli_dict)
    model, ckpt_epoch = build_model(args, device=device, use_ema_model=cli_args.use_ema_model, compile_model=cli_args.compile_model)
    loader = build_analysis_loader(
        args,
        ddp_rank=ddp_rank,
        ddp_world_size=ddp_world_size,
        split='train',
        max_samples=cli_args.num_data_samples,
        batch_size_per_rank=cli_args.batch_size_per_rank,
        num_workers=cli_args.num_workers,
    )
    train_latent_bank = build_train_latent_bank(
        model,
        loader=loader,
        device=device,
        num_data_samples=cli_args.num_data_samples,
    )
    prior_rows = probe_prior(
        model,
        args=cli_args,
        device=device,
        ddp_world_size=ddp_world_size,
        train_latent_bank=train_latent_bank,
    )
    contraction_rows = probe_contraction(model, args=cli_args, loader=loader, device=device)
    if ddp_rank == 0:
        out_dir = osp.join(exp_dir, 'research')
        os.makedirs(out_dir, exist_ok=True)
        save_name = (
            f'probe_ckpt={ckpt_epoch}'
            f'_cfg={cli_args.cfg}-{cli_args.cfg_position}'
            f'_sched={cli_args.use_sampling_scheduler}'
            f'_cache={cli_args.cache_sampling_noise}'
            f'_ema={cli_args.use_ema_model}.json'
        )
        payload = {
            'meta': {
                'job_dir': cli_args.job_dir,
                'ckpt_epoch': ckpt_epoch,
                'probe_mode': 'posthoc',
                'cfg': cli_args.cfg,
                'cfg_position': cli_args.cfg_position,
                'cache_sampling_noise': bool(cli_args.cache_sampling_noise),
                'noise_mode': _noise_mode_name(bool(cli_args.cache_sampling_noise)),
                'use_sampling_scheduler': bool(cli_args.use_sampling_scheduler),
                'schedule_mode': _schedule_mode_name(
                    bool(cli_args.use_sampling_scheduler)
                ),
                'regime_name': (
                    f"{_noise_mode_name(bool(cli_args.cache_sampling_noise))}-"
                    f"{_schedule_mode_name(bool(cli_args.use_sampling_scheduler))}"
                ),
                'use_ema': bool(cli_args.use_ema_model),
            },
            'prior_rows': prior_rows,
            'contraction_rows': contraction_rows,
        }
        with open(osp.join(out_dir, save_name), 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2)
        logger.info('saved probe results to %s', osp.join(out_dir, save_name))
    destroy_dist()


if __name__ == '__main__':
    main(cli_args)
