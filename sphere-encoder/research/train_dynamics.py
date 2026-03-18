from __future__ import annotations

import json
import logging
import os
import os.path as osp
import shutil
from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist
import torch_fidelity
from torchvision import datasets

from research.compat import (
    ambient_curvature_deg,
    prepare_cond_labels,
    sample_latent_trajectory,
    step_angles_deg,
)
from sphere.utils import save_image, save_tensors_to_images, vector_compute_angle

logger = logging.getLogger(__name__)


def gather_numpy(local_arr: np.ndarray) -> np.ndarray:
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if world_size == 1:
        return local_arr
    gather_list = [None for _ in range(world_size)]
    dist.all_gather_object(gather_list, local_arr)
    arrays = [np.asarray(arr) for arr in gather_list if arr is not None]
    return np.concatenate(arrays, axis=0) if arrays else local_arr


def sigma_max_from_angle(angle_deg: float) -> float:
    return float(np.tan(np.deg2rad(angle_deg)))


def make_fixed_noise(
    model,
    *,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    rng_devices = [device] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=rng_devices):
        torch.manual_seed(seed)
        return torch.randn(batch_size, *model.latent_shape[1:], device=device)


def _closest(values: list[float], target: float) -> float:
    if not values:
        return target
    return min(values, key=lambda x: abs(x - target))


def _rows_to_map(rows: list[dict], key_name: str) -> dict[float, dict]:
    return {float(row[key_name]): row for row in rows}


def _metric_suffix(value: float) -> str:
    text = f'{value:g}'
    return text.replace('-', 'm').replace('.', 'p')


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


@torch.no_grad()
def build_train_latent_bank(
    model,
    *,
    loader,
    device: torch.device,
    num_data_samples: int,
) -> np.ndarray:
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


@torch.no_grad()
def probe_prior(
    model,
    *,
    batch_size: int,
    device: torch.device,
    num_prior_samples: int,
    forward_steps: list[int],
    taus_deg: list[float],
    cfg: float,
    cfg_position: str,
    use_sampling_scheduler: bool,
    cache_sampling_noise: bool,
    seed: int,
    train_latent_bank: np.ndarray | None = None,
) -> list[dict]:
    rows = []
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    total = num_prior_samples
    assert total % (world_size * batch_size) == 0, (
        'num_prior_samples must divide world_size*batch_size_per_rank'
    )
    num_batches = total // (world_size * batch_size)
    rng_devices = [device] if device.type == 'cuda' else []
    local_rank = dist.get_rank() if dist.is_initialized() else 0

    for fwd in forward_steps:
        terminal_angles = []
        curvatures = []
        path_lengths = []
        initial_latents = []
        terminal_latents = []
        with torch.random.fork_rng(devices=rng_devices):
            torch.manual_seed(seed + 1000 * local_rank + fwd)
            for _ in range(num_batches):
                zs, _ = sample_latent_trajectory(
                    model=model,
                    batch_size=batch_size,
                    device=device,
                    forward_steps=fwd,
                    cfg=cfg,
                    cfg_position=cfg_position,
                    use_sampling_scheduler=use_sampling_scheduler,
                    cache_sampling_noise=cache_sampling_noise,
                )
                step_angles = step_angles_deg(zs)
                initial_latents.append(zs[0].flatten(1).cpu().numpy())
                terminal = step_angles[-1].flatten().cpu().numpy()
                terminal_angles.append(terminal)
                terminal_latents.append(zs[-1].flatten(1).cpu().numpy())
                curvatures.append(ambient_curvature_deg(zs).flatten().cpu().numpy())
                path_lengths.append(
                    np.stack(
                        [sa.flatten().cpu().numpy() for sa in step_angles], axis=1
                    ).sum(axis=1)
                )

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

        for tau in taus_deg:
            row = {
                'forward_steps': int(fwd),
                'tau_deg': float(tau),
                'terminal_capture_mass': float((terminal_angles <= tau).mean()),
                'curvature_mean_deg': float(curvatures.mean()),
                'path_length_mean_deg': float(path_lengths.mean()),
                'cache_sampling_noise': bool(cache_sampling_noise),
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
def probe_contraction(
    model,
    *,
    loader,
    device: torch.device,
    num_data_samples: int,
    contraction_noise_scalers: list[float],
    taus_deg: list[float],
) -> list[dict]:
    rows = []
    for noise_scaler in contraction_noise_scalers:
        seen = 0
        ratios = []
        angle_ins = []
        angle_outs = []
        for batch in loader:
            imgs, labels = batch[0].to(device), batch[1].to(device)
            y = labels if model.num_classes > 0 else None
            z_clean = model.encoder(imgs, y)
            v = model.spherify(z_clean, sampling=False)
            u = model.spherify(
                z_clean,
                sampling=True,
                noise_scaler=noise_scaler,
                cache_noise=False,
            )
            x_noisy = model.decoder(u, y)
            z_ret = model.encoder(x_noisy, y)
            v_ret = model.spherify(z_ret, sampling=False)
            angle_in = vector_compute_angle(u, v).flatten().cpu().numpy()
            angle_out = vector_compute_angle(v_ret, v).flatten().cpu().numpy()
            ratios.append(angle_out / np.clip(angle_in, 1e-6, None))
            angle_ins.append(angle_in)
            angle_outs.append(angle_out)
            seen += imgs.shape[0]
            if seen >= num_data_samples:
                break
        ratios = gather_numpy(np.concatenate(ratios))
        angle_ins = gather_numpy(np.concatenate(angle_ins))
        angle_outs = gather_numpy(np.concatenate(angle_outs))
        for tau in taus_deg:
            rows.append(
                {
                    'noise_scaler': float(noise_scaler),
                    'tau_deg': float(tau),
                    'kappa_mean': float(ratios.mean()),
                    'kappa_std': float(ratios.std()),
                    'angle_in_mean_deg': float(angle_ins.mean()),
                    'angle_out_mean_deg': float(angle_outs.mean()),
                    'capture_mass': float((angle_outs <= tau).mean()),
                }
            )
    return rows


def summarize_theory_metrics(
    *,
    prior_shared_rows: list[dict],
    prior_independent_rows: list[dict],
    contraction_rows: list[dict],
    target_tau_deg: float,
    target_noise_scaler: float,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    available_taus = [float(row['tau_deg']) for row in prior_shared_rows]
    tau = _closest(available_taus, target_tau_deg)

    for row in prior_shared_rows:
        fwd = int(row['forward_steps'])
        tau_suffix = _metric_suffix(float(row['tau_deg']))
        metrics[f'Theory/Terminal_CDF/M{fwd}_tau{tau_suffix}_deg'] = float(
            row.get('terminal_capture_mass', 0.0)
        )
        metrics[f'Theory/NN_Manifold/Capture/M{fwd}_eps{tau_suffix}_deg'] = float(
            row.get('capture_mass', 0.0)
        )
        if float(row['tau_deg']) != tau:
            continue
        metrics[f'Theory/Basin_Mass/M{fwd}'] = float(row.get('capture_mass', 0.0))
        metrics[f'Theory/Terminal_Angle/M{fwd}_deg'] = float(
            row['terminal_angle_mean_deg']
        )
        metrics[f'Theory/Terminal_Angle/M{fwd}_median_deg'] = float(
            row.get('terminal_angle_median_deg', row['terminal_angle_mean_deg'])
        )
        metrics[f'Theory/Terminal_Angle/M{fwd}_p25_deg'] = float(
            row.get('terminal_angle_p25_deg', row['terminal_angle_mean_deg'])
        )
        metrics[f'Theory/Terminal_Angle/M{fwd}_p75_deg'] = float(
            row.get('terminal_angle_p75_deg', row['terminal_angle_mean_deg'])
        )
        metrics[f'Theory/Path_Curvature/shared/M{fwd}_deg'] = float(
            row['curvature_mean_deg']
        )
        metrics[f'Theory/Path_Length/shared/M{fwd}_deg'] = float(
            row['path_length_mean_deg']
        )
        if 'nn_angle_before_mean_deg' in row:
            metrics[f'Theory/NN_Manifold/Before/M{fwd}_deg'] = float(
                row['nn_angle_before_mean_deg']
            )
            metrics[f'Theory/NN_Manifold/After/M{fwd}_deg'] = float(
                row['nn_angle_after_mean_deg']
            )
            metrics[f'Theory/NN_Manifold/Improvement/M{fwd}_deg'] = float(
                row['nn_angle_improvement_mean_deg']
            )
            metrics[f'Theory/NN_Manifold/ImprovementMedian/M{fwd}_deg'] = float(
                row.get(
                    'nn_angle_improvement_median_deg',
                    row['nn_angle_improvement_mean_deg'],
                )
            )
            metrics[f'Theory/NN_Manifold/ImprovedMass/M{fwd}'] = float(
                row.get('nn_improved_mass', 0.0)
            )

    for row in prior_independent_rows:
        if float(row['tau_deg']) != tau:
            continue
        fwd = int(row['forward_steps'])
        metrics[f'Theory/Path_Curvature/independent/M{fwd}_deg'] = float(
            row['curvature_mean_deg']
        )
        metrics[f'Theory/Path_Length/independent/M{fwd}_deg'] = float(
            row['path_length_mean_deg']
        )
        shared_key = f'Theory/Path_Curvature/shared/M{fwd}_deg'
        if shared_key in metrics:
            metrics[f'Theory/Path_Curvature/delta/M{fwd}_deg'] = float(
                row['curvature_mean_deg'] - metrics[shared_key]
            )

    available_noise = [float(row['noise_scaler']) for row in contraction_rows]
    noise_scaler = _closest(available_noise, target_noise_scaler)
    noise_rows = [
        row
        for row in contraction_rows
        if float(row['noise_scaler']) == noise_scaler and float(row['tau_deg']) == tau
    ]
    if noise_rows:
        row = noise_rows[0]
        metrics['Theory/Local_Contraction/kappa'] = float(row['kappa_mean'])
        metrics['Theory/Local_Contraction/angle_in_deg'] = float(
            row['angle_in_mean_deg']
        )
        metrics['Theory/Local_Contraction/angle_out_deg'] = float(
            row['angle_out_mean_deg']
        )
        metrics['Theory/Local_Contraction/capture_mass'] = float(row['capture_mass'])

    for row in contraction_rows:
        ns = float(row['noise_scaler'])
        if float(row['tau_deg']) != tau:
            continue
        metrics[f'Theory/Local_Contraction/kappa_ns{ns:.2f}'] = float(
            row['kappa_mean']
        )

    metrics['Theory/tau_deg'] = float(tau)
    metrics['Theory/noise_scaler'] = float(noise_scaler)
    return metrics


@torch.no_grad()
def run_theory_probe(
    model,
    *,
    loader,
    device: torch.device,
    batch_size_per_rank: int,
    num_prior_samples: int,
    num_data_samples: int,
    forward_steps: list[int],
    taus_deg: list[float],
    contraction_noise_scalers: list[float],
    target_tau_deg: float,
    target_noise_scaler: float,
    cfg: float,
    cfg_position: str,
    use_sampling_scheduler: bool,
    seed: int,
) -> tuple[dict[str, float], dict]:
    train_latent_bank = build_train_latent_bank(
        model,
        loader=loader,
        device=device,
        num_data_samples=num_data_samples,
    )
    prior_shared_rows = probe_prior(
        model,
        batch_size=batch_size_per_rank,
        device=device,
        num_prior_samples=num_prior_samples,
        forward_steps=forward_steps,
        taus_deg=taus_deg,
        cfg=cfg,
        cfg_position=cfg_position,
        use_sampling_scheduler=use_sampling_scheduler,
        cache_sampling_noise=True,
        seed=seed,
        train_latent_bank=train_latent_bank,
    )
    prior_independent_rows = probe_prior(
        model,
        batch_size=batch_size_per_rank,
        device=device,
        num_prior_samples=num_prior_samples,
        forward_steps=forward_steps,
        taus_deg=taus_deg,
        cfg=cfg,
        cfg_position=cfg_position,
        use_sampling_scheduler=use_sampling_scheduler,
        cache_sampling_noise=False,
        seed=seed,
        train_latent_bank=train_latent_bank,
    )
    contraction_rows = probe_contraction(
        model,
        loader=loader,
        device=device,
        num_data_samples=num_data_samples,
        contraction_noise_scalers=contraction_noise_scalers,
        taus_deg=taus_deg,
    )
    metrics = summarize_theory_metrics(
        prior_shared_rows=prior_shared_rows,
        prior_independent_rows=prior_independent_rows,
        contraction_rows=contraction_rows,
        target_tau_deg=target_tau_deg,
        target_noise_scaler=target_noise_scaler,
    )
    payload = {
        'prior_shared_rows': prior_shared_rows,
        'prior_independent_rows': prior_independent_rows,
        'contraction_rows': contraction_rows,
    }
    return metrics, payload


@torch.no_grad()
def compute_reconstruction_metrics(
    model,
    *,
    loader,
    device: torch.device,
    num_samples: int,
    ptdtype: torch.dtype,
) -> dict[str, float]:
    if num_samples <= 0:
        return {}
    total_sq = torch.zeros((), device=device)
    total_abs = torch.zeros((), device=device)
    total_numel = torch.zeros((), device=device)
    seen = 0
    autocast_ctx = (
        torch.autocast(device_type='cuda', dtype=ptdtype)
        if device.type == 'cuda'
        else nullcontext()
    )
    for batch in loader:
        imgs = batch[0].to(device, non_blocking=True)
        labels = batch[1].to(device, non_blocking=True)
        y = labels if model.num_classes > 0 else None
        with autocast_ctx:
            recons = model.reconstruct(imgs, y, sampling=False)
        target = imgs * 0.5 + 0.5
        diff = (recons.float() - target.float()).contiguous()
        total_sq += diff.square().sum()
        total_abs += diff.abs().sum()
        total_numel += torch.tensor(diff.numel(), device=device, dtype=diff.dtype)
        seen += imgs.shape[0]
        if seen >= num_samples:
            break

    if dist.is_initialized():
        dist.all_reduce(total_sq, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_abs, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_numel, op=dist.ReduceOp.SUM)

    return {
        'Eval/Recon_MSE': float((total_sq / total_numel).item()),
        'Eval/Recon_L1': float((total_abs / total_numel).item()),
    }


def _save_snapshot_grid(
    *,
    gen_imgs_dir: str,
    snapshot_img_path: str,
    num_snapshot_samples: int,
) -> str | None:
    imgs = sorted(
        osp.join(gen_imgs_dir, name)
        for name in os.listdir(gen_imgs_dir)
        if name.endswith('.png')
    )
    if not imgs:
        return None
    sample_paths = imgs[: min(len(imgs), num_snapshot_samples)]
    sample_imgs = [datasets.folder.pil_loader(img) for img in sample_paths]
    tensors = [torch.from_numpy(np.array(img)) for img in sample_imgs]
    grid_tensor = torch.stack(tensors, dim=0).permute(0, 3, 1, 2) / 255.0
    save_tensors_to_images(
        grid_tensor,
        path=snapshot_img_path,
        nrow=max(8, int(num_snapshot_samples / 128 * 8)),
        max_nimgs=num_snapshot_samples,
    )
    return snapshot_img_path


@torch.no_grad()
def run_lightweight_generation_eval(
    model,
    *,
    exp_dir: str,
    image_size: int,
    num_classes: int,
    device: torch.device,
    ptdtype: torch.dtype,
    num_eval_samples: int,
    batch_size_per_rank: int,
    forward_steps: list[int],
    cfg: float,
    cfg_position: str,
    use_sampling_scheduler: bool,
    cache_sampling_noise: bool,
    fid_stats_file_path: str,
    epoch: int,
    step: int,
    seed: int,
    num_snapshot_samples: int,
    cleanup: bool = True,
) -> tuple[dict[str, float], dict]:
    if num_eval_samples <= 0:
        return {}, {'snapshots': {}}

    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    assert num_eval_samples % (world_size * batch_size_per_rank) == 0, (
        'num_eval_samples must divide world_size*batch_size_per_rank'
    )
    num_batches = num_eval_samples // (world_size * batch_size_per_rank)
    eval_root = osp.join(exp_dir, 'research', 'light_eval')
    metrics: dict[str, float] = {}
    payload = {'snapshots': {}, 'runs': []}

    class_ids = None
    if num_classes > 0:
        assert num_eval_samples % num_classes == 0
        num_per_class = num_eval_samples // num_classes
        class_ids = np.arange(0, num_classes).repeat(num_per_class)

    autocast_ctx = (
        torch.autocast(device_type='cuda', dtype=ptdtype)
        if device.type == 'cuda'
        else nullcontext()
    )

    for forward_step in forward_steps:
        save_dir = osp.join(
            eval_root, f'ep{epoch:04d}_step{step:07d}_fwd{forward_step:02d}'
        )
        gen_imgs_dir = osp.join(save_dir, 'gens')
        if rank == 0 and osp.exists(save_dir):
            shutil.rmtree(save_dir, ignore_errors=True)
        if dist.is_initialized():
            dist.barrier()
        os.makedirs(gen_imgs_dir, exist_ok=True)

        rng_devices = [device] if device.type == 'cuda' else []
        for batch_idx in range(num_batches):
            start_idx = (
                batch_idx * batch_size_per_rank * world_size
                + local_rank * batch_size_per_rank
            )
            end_idx = start_idx + batch_size_per_rank
            y = None
            if class_ids is not None:
                y = torch.tensor(class_ids[start_idx:end_idx], device=device).long()
            with torch.random.fork_rng(devices=rng_devices), autocast_ctx:
                torch.manual_seed(seed + 1000 * rank + batch_idx + 100 * forward_step)
                _, outs = model.generate(
                    batch_size=batch_size_per_rank,
                    y=y,
                    cfg=cfg,
                    cfg_position=cfg_position,
                    forward_steps=forward_step,
                    use_sampling_scheduler=use_sampling_scheduler,
                    cache_sampling_noise=cache_sampling_noise,
                    device=device,
                )
            save_image(
                x=outs,
                batch_idx=batch_idx,
                ddp_rank=rank,
                save_dir=gen_imgs_dir,
                force_image_size=image_size,
            )
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        if dist.is_initialized():
            dist.barrier()

        if rank == 0:
            result = torch_fidelity.calculate_metrics(
                input1=gen_imgs_dir,
                input2=None,
                fid_statistics_file=fid_stats_file_path,
                cuda=device.type == 'cuda',
                isc=True,
                fid=True,
                kid=False,
                prc=False,
                verbose=False,
            )
            metrics[f'Eval/FID_{forward_step}step'] = float(
                result['frechet_inception_distance']
            )
            metrics[f'Eval/ISC_mean_{forward_step}step'] = float(
                result['inception_score_mean']
            )
            metrics[f'Eval/ISC_std_{forward_step}step'] = float(
                result['inception_score_std']
            )
            snapshot_path = _save_snapshot_grid(
                gen_imgs_dir=gen_imgs_dir,
                snapshot_img_path=osp.join(save_dir, 'snapshot.png'),
                num_snapshot_samples=num_snapshot_samples,
            )
            payload['snapshots'][str(forward_step)] = snapshot_path
            payload['runs'].append(
                {
                    'forward_steps': int(forward_step),
                    'save_dir': save_dir,
                    'snapshot_path': snapshot_path,
                    'metrics': {
                        'fid': metrics[f'Eval/FID_{forward_step}step'],
                        'isc_mean': metrics[f'Eval/ISC_mean_{forward_step}step'],
                        'isc_std': metrics[f'Eval/ISC_std_{forward_step}step'],
                    },
                }
            )

        if dist.is_initialized():
            dist.barrier()
        if cleanup and rank == 0:
            shutil.rmtree(gen_imgs_dir, ignore_errors=True)
        if dist.is_initialized():
            dist.barrier()

    return metrics, payload


def save_training_metrics_payload(
    *,
    exp_dir: str,
    epoch: int,
    step: int,
    payload: dict,
) -> str:
    out_dir = osp.join(exp_dir, 'research', 'training_dynamics')
    os.makedirs(out_dir, exist_ok=True)
    out_path = osp.join(out_dir, f'ep{epoch:04d}_step{step:07d}.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    logger.info('saved training dynamics payload to %s', out_path)
    return out_path
