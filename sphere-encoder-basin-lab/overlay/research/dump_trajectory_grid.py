from __future__ import annotations

import argparse
import logging
import os
import os.path as osp

import torch

from cli_utils import str2bool
from research.compat import build_model, destroy_dist, init_dist, load_job_args, sample_latent_trajectory
from sphere.utils import save_tensors_to_images

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='Dump image grids across sampling steps.')
parser.add_argument('--dev_dir', type=str, default='workspace')
parser.add_argument('--job_dir', type=str, required=True)
parser.add_argument('--ckpt_fname', type=str, default=None)
parser.add_argument('--num_samples', type=int, default=16)
parser.add_argument('--forward_steps', type=int, default=4)
parser.add_argument('--cfg', type=float, default=1.0)
parser.add_argument('--cfg_position', type=str, default='combo')
parser.add_argument('--use_sampling_scheduler', type=str2bool, default=False)
parser.add_argument('--cache_sampling_noise', type=str2bool, default=True)
parser.add_argument('--use_ema_model', type=str2bool, default=False)
parser.add_argument('--compile_model', type=str2bool, default=False)
cli_args = parser.parse_args()


def main(args) -> None:
    ddp_rank, _, _, device = init_dist('cuda' if torch.cuda.is_available() else 'cpu')
    merged_args, exp_dir = load_job_args(args.dev_dir, args.job_dir, vars(args))
    model, ckpt_epoch = build_model(merged_args, device=device, use_ema_model=args.use_ema_model, compile_model=args.compile_model)
    zs, xs = sample_latent_trajectory(
        model=model,
        batch_size=args.num_samples,
        device=device,
        forward_steps=args.forward_steps,
        cfg=args.cfg,
        cfg_position=args.cfg_position,
        use_sampling_scheduler=args.use_sampling_scheduler,
        cache_sampling_noise=args.cache_sampling_noise,
    )
    if ddp_rank == 0:
        out_dir = osp.join(exp_dir, 'research')
        os.makedirs(out_dir, exist_ok=True)
        out_path = osp.join(
            out_dir,
            f'trajectory_ckpt={ckpt_epoch}_steps={args.forward_steps}_cfg={args.cfg}-{args.cfg_position}.png',
        )
        save_tensors_to_images(xs, path=out_path, nrow=min(args.num_samples, 8), max_nimgs=args.num_samples)
        logger.info('saved trajectory grid to %s', out_path)
    destroy_dist()


if __name__ == '__main__':
    main(cli_args)
