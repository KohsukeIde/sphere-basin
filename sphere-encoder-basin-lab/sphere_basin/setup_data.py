from __future__ import annotations

import argparse
from pathlib import Path

from .paths import ensure_workspace_compat, resolve_dev_dir


def _download_cifar(workspace: Path, dataset_name: str) -> None:
    try:
        from torchvision import datasets
    except ModuleNotFoundError as exc:
        raise SystemExit(
            'torchvision is not installed. Run `bash scripts/setup_env.sh` '
            'after installing a matching torch/torchvision build.'
        ) from exc

    dataset_root = workspace / 'datasets' / dataset_name
    dataset_root.mkdir(parents=True, exist_ok=True)

    dataset_cls = {
        'cifar-10': datasets.CIFAR10,
        'cifar-100': datasets.CIFAR100,
    }[dataset_name]

    print(f'>>> download {dataset_name} train split -> {dataset_root}')
    dataset_cls(root=str(dataset_root), train=True, download=True)
    print(f'>>> download {dataset_name} test split -> {dataset_root}')
    dataset_cls(root=str(dataset_root), train=False, download=True)


def _download_fid_stats(workspace: Path, dataset_name: str, image_size: int, repo_id: str) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError as exc:
        raise SystemExit(
            'huggingface_hub is not installed. Run `bash scripts/setup_env.sh` first.'
        ) from exc

    stats_name = f'fid_stats_extr_{dataset_name}_{image_size}px.npz'
    target = workspace / 'fid_stats' / stats_name
    if target.exists():
        print(f'>>> reuse existing fid stats: {target}')
        return

    print(f'>>> download fid stats: {stats_name}')
    snapshot_download(
        repo_id=repo_id,
        repo_type='dataset',
        local_dir=str(workspace),
        allow_patterns=[f'fid_stats/{stats_name}'],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description='Prepare research datasets and FID artifacts.')
    parser.add_argument('--workspace', type=str, default=None)
    parser.add_argument('--dataset-name', type=str, default='cifar-10', choices=['cifar-10', 'cifar-100'])
    parser.add_argument('--image-size', type=int, default=32)
    parser.add_argument('--skip-dataset-download', action='store_true')
    parser.add_argument('--skip-fid-stats-download', action='store_true')
    parser.add_argument('--fid-repo-id', type=str, default='kaiyuyue/sphere-encoder-fid-artifacts')
    args = parser.parse_args()

    workspace = ensure_workspace_compat(resolve_dev_dir(args.workspace))
    (workspace / 'datasets').mkdir(parents=True, exist_ok=True)
    (workspace / 'fid_stats').mkdir(parents=True, exist_ok=True)

    if not args.skip_dataset_download:
        _download_cifar(workspace, args.dataset_name)

    if not args.skip_fid_stats_download:
        _download_fid_stats(
            workspace=workspace,
            dataset_name=args.dataset_name,
            image_size=args.image_size,
            repo_id=args.fid_repo_id,
        )

    print(f'>>> research data ready under: {workspace}')


if __name__ == '__main__':
    main()
