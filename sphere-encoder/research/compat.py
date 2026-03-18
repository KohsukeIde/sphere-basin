from __future__ import annotations

import datetime
import glob
import json
import logging
import os
import os.path as osp
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms

from sphere.ema import SimpleEMA
from sphere.loader import ListDataset, center_crop_arr, create_dataset
from sphere.model import G
from sphere.utils import load_ckpt, vector_compute_angle

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def init_dist(device_type: str = 'cuda') -> tuple[int, int, int, torch.device]:
    if 'RANK' not in os.environ:
        os.environ.setdefault('RANK', '0')
        os.environ.setdefault('LOCAL_RANK', '0')
        os.environ.setdefault('WORLD_SIZE', '1')
        os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
        os.environ.setdefault('MASTER_PORT', '29500')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = torch.device(f'{device_type}:{ddp_local_rank}' if device_type == 'cuda' else 'cpu')
    if device_type == 'cuda':
        torch.cuda.set_device(device)
        backend = 'nccl'
    else:
        backend = 'gloo'
    if not dist.is_initialized():
        kwargs = {
            'backend': backend,
            'timeout': datetime.timedelta(hours=2),
        }
        if device_type == 'cuda':
            kwargs['device_id'] = device
        dist.init_process_group(**kwargs)
    return ddp_rank, ddp_local_rank, ddp_world_size, device


def destroy_dist() -> None:
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def find_exp_dir(dev_dir: str, job_dir: str) -> str:
    cands = [
        osp.join(dev_dir, 'experiments', job_dir),
        osp.join(dev_dir, 'jobs', job_dir),
    ]
    for cand in cands:
        if osp.exists(cand):
            return cand
    raise FileNotFoundError(f'job_dir not found in experiments/jobs: {job_dir}')


def load_job_args(dev_dir: str, job_dir: str, cli_overrides: dict) -> tuple[SimpleNamespace, str]:
    exp_dir = find_exp_dir(dev_dir, job_dir)
    cfg_path = osp.join(exp_dir, 'cfg.json')
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg_args = json.load(f)
    cfg_args.update({k: v for k, v in cli_overrides.items() if v is not None})
    args = SimpleNamespace(**cfg_args)
    return args, exp_dir


def build_model(args, device, use_ema_model: bool = False, compile_model: bool = False):
    model = G(
        input_size=args.image_size,
        patch_size=args.patch_size,
        vit_enc_model_size=args.vit_enc_model_size,
        vit_dec_model_size=args.vit_dec_model_size,
        token_channels=args.token_channels,
        num_classes=args.num_classes if args.cond_generator else 0,
        halve_model_size=args.halve_model_size,
        spherify_model=args.spherify_model,
        pixel_head_type=args.pixel_head_type,
        in_context_size=args.in_context_size,
        noise_sigma_max_angle=args.noise_sigma_max_angle,
        vit_enc_latent_mlp_mixer_depth=args.vit_enc_latent_mlp_mixer_depth,
        vit_dec_latent_mlp_mixer_depth=args.vit_dec_latent_mlp_mixer_depth,
        affine_latent_mlp_mixer=args.affine_latent_mlp_mixer,
    )
    model.to(device=device)
    ema_model = SimpleEMA(model)

    ckpt_dir = osp.join(find_exp_dir(args.dev_dir, args.job_dir), 'ckpt')
    ckpts = sorted(glob.glob(osp.join(ckpt_dir, '*.pth')))
    if len(ckpts) == 0:
        raise FileNotFoundError(f'no checkpoints found in {ckpt_dir}')
    load_from = ckpts[-1] if getattr(args, 'ckpt_fname', None) is None else osp.join(ckpt_dir, args.ckpt_fname)
    load_ckpt(
        model,
        ckpt_path=load_from,
        ema_model=ema_model,
        strict=True,
        override_model_with_ema=use_ema_model,
        verbose=True,
    )
    if compile_model:
        model = torch.compile(model)
    model.eval().requires_grad_(False)
    return model, osp.basename(load_from).replace('.pth', '')


def get_dataset_cls(dataset_name: str):
    if dataset_name == 'cifar-10':
        return datasets.CIFAR10
    if dataset_name == 'cifar-100':
        return datasets.CIFAR100
    return ListDataset


def build_analysis_loader(args, ddp_rank: int, ddp_world_size: int, split: str = 'train', max_samples: int = 4096, batch_size_per_rank: int = 64, num_workers: int = 4):
    dataset_cls = get_dataset_cls(args.dataset_name)
    dataset_root = osp.join(args.dev_dir, args.data_dir, args.dataset_name)
    transform = transforms.Compose([
        partial(center_crop_arr, image_size=args.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    ds = create_dataset(
        dataset_cls,
        root=dataset_root,
        split=split,
        concat_train_val_splits=getattr(args, 'concat_train_val_splits', False),
        download=True,
        transform=transform,
        max_samples=max_samples,
        load_from_zip=getattr(args, 'load_from_zip', False),
    )
    sampler = DistributedSampler(ds, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=False)
    loader = DataLoader(
        ds,
        batch_size=batch_size_per_rank,
        sampler=sampler,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=False,
        persistent_workers=True if num_workers > 0 else False,
    )
    return loader


@torch.no_grad()
def prepare_cond_labels(model, batch_size: int, device: torch.device, fixed_class: int | None = None):
    if model.num_classes <= 0:
        return None, None
    if fixed_class is None:
        y = torch.randint(0, model.decoder.num_classes, (batch_size,), device=device)
    else:
        y = torch.full((batch_size,), fixed_class, device=device, dtype=torch.long)
    y_uncond = torch.full_like(y, model.num_classes)
    return y, y_uncond


@torch.no_grad()
def apply_cfg_on_decode(model, z, y, y_uncond, cfg: float, cfg_position: str):
    x = model.decoder(z, y)
    if cfg > 1.0 and cfg_position in ['dec', 'combo']:
        cfg_eff = cfg ** 0.5 if cfg_position == 'combo' else cfg
        x_uncond = model.decoder(z, y_uncond)
        x = torch.lerp(x_uncond, x, cfg_eff).clamp_(-1, 1)
    return x


@torch.no_grad()
def apply_cfg_on_encode(model, x, y, y_uncond, cfg: float, cfg_position: str):
    z = model.encoder(x, y)
    if cfg > 1.0 and cfg_position in ['enc', 'combo']:
        cfg_eff = cfg ** 0.5 if cfg_position == 'combo' else cfg
        z_uncond = model.encoder(x, y_uncond)
        z = torch.lerp(z_uncond, z, cfg_eff)
    return z


@torch.no_grad()
def sample_latent_trajectory(model, batch_size: int, device: torch.device, forward_steps: int, cfg: float = 1.0, cfg_position: str = 'combo', use_sampling_scheduler: bool = False, cache_sampling_noise: bool = False, fixed_class: int | None = None):
    model.cached_noise = None
    e = torch.randn(batch_size, *model.latent_shape[1:], device=device)
    y, y_uncond = prepare_cond_labels(model, batch_size, device, fixed_class=fixed_class)
    z = model.spherify(e, sampling=False)
    zs = [z.detach().float().cpu()]
    x = apply_cfg_on_decode(model, z, y, y_uncond, cfg, cfg_position)
    xs = [torch.clamp(x * 0.5 + 0.5, 0, 1).detach().float().cpu()]
    for step in range(max(0, forward_steps - 1)):
        z = apply_cfg_on_encode(model, x, y, y_uncond, cfg, cfg_position)
        if use_sampling_scheduler:
            T = forward_steps
            t = step + 1
            r = 1.0 - t / T
        else:
            r = 1.0
        z = model.spherify(z, sampling=True, noise_scaler=r, cache_noise=cache_sampling_noise)
        zs.append(z.detach().float().cpu())
        x = apply_cfg_on_decode(model, z, y, y_uncond, cfg, cfg_position)
        xs.append(torch.clamp(x * 0.5 + 0.5, 0, 1).detach().float().cpu())
    # one extra re-encode to measure self-consistency basin capture
    z_terminal = apply_cfg_on_encode(model, x, y, y_uncond, cfg, cfg_position)
    z_terminal = model.spherify(z_terminal, sampling=False)
    zs.append(z_terminal.detach().float().cpu())
    return zs, xs


@torch.no_grad()
def ambient_curvature_deg(zs: list[torch.Tensor]) -> torch.Tensor:
    if len(zs) < 3:
        return torch.zeros(zs[0].shape[0], 1)
    vals = []
    for i in range(1, len(zs) - 1):
        d1 = (zs[i] - zs[i - 1]).flatten(1)
        d2 = (zs[i + 1] - zs[i]).flatten(1)
        vals.append(vector_compute_angle(d1, d2).cpu())
    return torch.cat(vals, dim=1).mean(dim=1, keepdim=True)


@torch.no_grad()
def step_angles_deg(zs: list[torch.Tensor]) -> list[torch.Tensor]:
    vals = []
    for i in range(len(zs) - 1):
        vals.append(vector_compute_angle(zs[i], zs[i + 1]).cpu())
    return vals
