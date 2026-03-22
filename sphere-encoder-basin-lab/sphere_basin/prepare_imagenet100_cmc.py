from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np

from .paths import ensure_workspace_compat, project_root


IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def _resolve_local_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (project_root() / path).resolve()


def _read_class_list(path: Path) -> list[str]:
    classes = [
        line.strip()
        for line in path.read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]
    if len(classes) != len(set(classes)):
        raise ValueError(f'duplicate synsets in class list: {path}')
    if len(classes) != 100:
        raise ValueError(f'expected 100 synsets, got {len(classes)} from {path}')
    return classes


def _iter_images(path: Path) -> Iterable[Path]:
    for child in sorted(path.iterdir()):
        if child.is_file() and child.suffix.lower() in IMAGE_EXTS:
            yield child


def _collect_split_records(source_root: Path, split: str, classes: list[str]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    split_root = source_root / split
    if not split_root.exists():
        raise FileNotFoundError(f'missing split root: {split_root}')
    for class_id, synset in enumerate(classes):
        class_root = split_root / synset
        if not class_root.exists():
            raise FileNotFoundError(f'missing synset directory: {class_root}')
        imgs = list(_iter_images(class_root))
        if not imgs:
            raise FileNotFoundError(f'no images found under: {class_root}')
        for img_path in imgs:
            records.append(
                {
                    'class_id': class_id,
                    'class_name': synset,
                    'image_path': str(img_path.resolve()),
                    'is_absolute_path': True,
                }
            )
    return records


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + '\n')


def _ensure_ref_images(source_root: Path, classes: list[str], ref_dir: Path) -> None:
    images_root = ref_dir / 'images'
    images_root.mkdir(parents=True, exist_ok=True)
    for synset in classes:
        target = source_root / 'train' / synset
        link = images_root / synset
        if link.is_symlink() or link.exists():
            if link.is_symlink() and link.resolve() == target.resolve():
                continue
            if link.is_dir() and not link.is_symlink():
                raise FileExistsError(f'ref image path exists and is a real directory: {link}')
            link.unlink()
        os.symlink(target.resolve(), link)


def _compute_fid_stats(
    *,
    ref_images_dir: Path,
    image_size: int,
    out_path: Path,
    batch_size: int,
    cuda: bool,
) -> None:
    from torch_fidelity.metric_fid import fid_input_id_to_statistics
    from torch_fidelity.utils import create_feature_extractor, resolve_feature_layer_for_metric

    feat_extractor_name = 'inception-v3-compat'
    feat_layer = resolve_feature_layer_for_metric('fid', feature_extractor=feat_extractor_name)
    feat_extractor = create_feature_extractor(
        feat_extractor_name,
        [feat_layer],
        cuda=cuda,
        verbose=True,
    )
    stats = fid_input_id_to_statistics(
        1,
        feat_extractor,
        feat_layer,
        input1=str(ref_images_dir),
        cuda=cuda,
        batch_size=batch_size,
        cache=False,
        save_cpu_ram=True,
        verbose=True,
        samples_find_deep=True,
        samples_resize_and_crop=image_size,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, mu=stats['mu'], sigma=stats['sigma'])


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Prepare ImageNet-100 (CMC split) manifests and FID artifacts from a local ImageNet-1k root.'
    )
    parser.add_argument('--source-root', type=str, required=True)
    parser.add_argument('--dev-dir', type=str, default='workspace')
    parser.add_argument('--dataset-name', type=str, default='imagenet-100')
    parser.add_argument('--image-size', type=int, default=160)
    parser.add_argument(
        '--class-list-path',
        type=str,
        default=str(project_root() / 'references' / 'imagenet100_cmc.txt'),
    )
    parser.add_argument('--fid-stats-mode', type=str, default='extr')
    parser.add_argument('--fid-stats-batch-size', type=int, default=64)
    parser.add_argument('--fid-stats-cuda', action='store_true')
    parser.add_argument('--skip-fid-stats', action='store_true')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    source_root = Path(args.source_root).expanduser().resolve()
    workspace_root = ensure_workspace_compat(args.dev_dir)
    class_list_path = _resolve_local_path(args.class_list_path)
    classes = _read_class_list(class_list_path)

    dataset_root = workspace_root / 'datasets' / args.dataset_name
    dataset_root.mkdir(parents=True, exist_ok=True)

    train_rows = _collect_split_records(source_root, 'train', classes)
    val_rows = _collect_split_records(source_root, 'val', classes)

    train_json = dataset_root / 'train.json'
    val_json = dataset_root / 'val.json'
    if args.force or not train_json.exists():
        _write_jsonl(train_json, train_rows)
    if args.force or not val_json.exists():
        _write_jsonl(val_json, val_rows)

    meta = {
        'dataset_name': args.dataset_name,
        'source_root': str(source_root),
        'class_list_path': str(class_list_path),
        'num_classes': len(classes),
        'train_samples': len(train_rows),
        'val_samples': len(val_rows),
        'image_size': args.image_size,
        'fid_stats_mode': args.fid_stats_mode,
    }
    (dataset_root / 'meta.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')

    ref_root = workspace_root / 'fid_refs' / f'ref_images_{args.dataset_name}_{args.image_size}px'
    _ensure_ref_images(source_root, classes, ref_root)

    fid_stats_path = (
        workspace_root
        / 'fid_stats'
        / f'fid_stats_{args.fid_stats_mode}_{args.dataset_name}_{args.image_size}px.npz'
    )
    if not args.skip_fid_stats and (args.force or not fid_stats_path.exists()):
        _compute_fid_stats(
            ref_images_dir=ref_root / 'images',
            image_size=args.image_size,
            out_path=fid_stats_path,
            batch_size=args.fid_stats_batch_size,
            cuda=bool(args.fid_stats_cuda),
        )

    summary = {
        'dataset_root': str(dataset_root),
        'ref_root': str(ref_root),
        'fid_stats_path': str(fid_stats_path),
        **meta,
    }
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
