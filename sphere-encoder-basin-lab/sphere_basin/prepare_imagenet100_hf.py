from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm import tqdm

from .paths import ensure_workspace_compat, project_root
from .prepare_imagenet100_cmc import (
    _compute_fid_stats,
    _ensure_ref_images,
    _read_class_list,
    _write_jsonl,
)


def _resolve_local_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (project_root() / path).resolve()


def _extract_image_bytes(value: Any) -> bytes:
    if isinstance(value, dict):
        raw = value.get('bytes')
        if raw is None:
            raise ValueError(
                'HF image record only contains a path, not bytes. '
                'This prep path requires the parquet image bytes.'
            )
        return bytes(raw)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    raise TypeError(f'unsupported HF image payload type: {type(value).__name__}')


def _suffix_for_image(raw: bytes) -> str:
    with Image.open(io.BytesIO(raw)) as img:
        fmt = (img.format or '').upper()
    if fmt in {'JPEG', 'JPG'}:
        return '.JPEG'
    if fmt == 'PNG':
        return '.png'
    if fmt == 'WEBP':
        return '.webp'
    return '.png'


def _download_snapshot(repo_id: str, revision: str | None, download_dir: Path) -> tuple[Path, str | None]:
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ModuleNotFoundError as exc:
        raise SystemExit('huggingface_hub is required for HF ImageNet-100 prep.') from exc

    download_dir.mkdir(parents=True, exist_ok=True)
    info = HfApi().dataset_info(repo_id, revision=revision)
    snapshot = snapshot_download(
        repo_id=repo_id,
        repo_type='dataset',
        revision=revision,
        local_dir=str(download_dir),
        allow_patterns=['README.md', 'scripts/classes.py', 'data/*.parquet'],
    )
    return Path(snapshot).resolve(), getattr(info, 'sha', None)


def _iter_parquet_batches(path: Path, batch_size: int):
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise SystemExit(
            'pyarrow is required to unpack clane9/imagenet-100 parquet shards. '
            'Install it with `python -m pip install pyarrow`.'
        ) from exc

    parquet = pq.ParquetFile(path)
    yield from parquet.iter_batches(batch_size=batch_size, columns=['image', 'label'])


def _export_split(
    *,
    parquet_files: list[Path],
    split: str,
    source_root: Path,
    classes: list[str],
    force_images: bool,
    batch_size: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    counts = {idx: 0 for idx in range(len(classes))}
    split_root = source_root / split
    split_root.mkdir(parents=True, exist_ok=True)

    for class_id, synset in enumerate(classes):
        (split_root / synset).mkdir(parents=True, exist_ok=True)

    for parquet_path in parquet_files:
        with tqdm(desc=f'export {split} {parquet_path.name}', unit='img') as pbar:
            for batch in _iter_parquet_batches(parquet_path, batch_size=batch_size):
                image_idx = batch.schema.get_field_index('image')
                label_idx = batch.schema.get_field_index('label')
                images = batch.column(image_idx).to_pylist()
                labels = batch.column(label_idx).to_pylist()
                for image_value, label_value in zip(images, labels, strict=True):
                    class_id = int(label_value)
                    if class_id < 0 or class_id >= len(classes):
                        raise ValueError(f'out-of-range label {class_id} in {parquet_path}')
                    synset = classes[class_id]
                    raw = _extract_image_bytes(image_value)
                    suffix = _suffix_for_image(raw)
                    image_name = f'{synset}_{counts[class_id]:06d}{suffix}'
                    image_path = split_root / synset / image_name
                    if force_images or not image_path.exists() or image_path.stat().st_size == 0:
                        image_path.write_bytes(raw)
                    rows.append(
                        {
                            'class_id': class_id,
                            'class_name': synset,
                            'image_path': str(image_path.resolve()),
                            'is_absolute_path': True,
                        }
                    )
                    counts[class_id] += 1
                    pbar.update(1)

    return rows


def _missing_count(rows: list[dict[str, object]]) -> int:
    return sum(1 for row in rows if not Path(str(row['image_path'])).exists())


def _stats_meta_path(fid_stats_path: Path) -> Path:
    return fid_stats_path.with_name(fid_stats_path.name + '.meta.json')


def _should_recompute_fid_stats(fid_stats_path: Path, expected_meta: dict[str, object], force: bool) -> bool:
    if force or not fid_stats_path.exists():
        return True
    meta_path = _stats_meta_path(fid_stats_path)
    if not meta_path.exists():
        return True
    try:
        found = json.loads(meta_path.read_text(encoding='utf-8'))
    except json.JSONDecodeError:
        return True
    return any(found.get(k) != v for k, v in expected_meta.items())


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Download clane9/imagenet-100 from Hugging Face and export Sphere Encoder manifests.'
    )
    parser.add_argument('--repo-id', type=str, default='clane9/imagenet-100')
    parser.add_argument('--revision', type=str, default=None)
    parser.add_argument('--download-dir', type=str, default='workspace/downloads/imagenet-100-hf')
    parser.add_argument('--source-root', type=str, default='workspace/datasets/imagenet-100/images')
    parser.add_argument('--dev-dir', type=str, default='workspace')
    parser.add_argument('--dataset-name', type=str, default='imagenet-100')
    parser.add_argument('--image-size', type=int, default=160)
    parser.add_argument('--class-list-path', type=str, default='references/imagenet100_cmc.txt')
    parser.add_argument('--fid-stats-mode', type=str, default='extr')
    parser.add_argument('--fid-stats-batch-size', type=int, default=64)
    parser.add_argument('--fid-stats-cuda', action='store_true')
    parser.add_argument('--skip-fid-stats', action='store_true')
    parser.add_argument('--force', action='store_true', help='rewrite manifests/meta even if they already exist')
    parser.add_argument('--force-images', action='store_true')
    parser.add_argument('--force-fid-stats', action='store_true')
    parser.add_argument('--export-batch-size', type=int, default=512)
    parser.add_argument('--expected-train-samples', type=int, default=126689)
    parser.add_argument('--expected-val-samples', type=int, default=5000)
    args = parser.parse_args()

    workspace_root = ensure_workspace_compat(args.dev_dir)
    class_list_path = _resolve_local_path(args.class_list_path)
    classes = _read_class_list(class_list_path)
    download_dir = _resolve_local_path(args.download_dir)
    source_root = _resolve_local_path(args.source_root)

    snapshot_root, snapshot_sha = _download_snapshot(args.repo_id, args.revision, download_dir)
    data_root = snapshot_root / 'data'
    train_files = sorted(data_root.glob('train-*.parquet'))
    val_files = sorted(data_root.glob('validation-*.parquet'))
    if len(train_files) != 17:
        raise FileNotFoundError(f'expected 17 train parquet shards under {data_root}, got {len(train_files)}')
    if len(val_files) != 1:
        raise FileNotFoundError(f'expected 1 validation parquet shard under {data_root}, got {len(val_files)}')

    source_root.mkdir(parents=True, exist_ok=True)
    train_rows = _export_split(
        parquet_files=train_files,
        split='train',
        source_root=source_root,
        classes=classes,
        force_images=args.force_images,
        batch_size=args.export_batch_size,
    )
    val_rows = _export_split(
        parquet_files=val_files,
        split='val',
        source_root=source_root,
        classes=classes,
        force_images=args.force_images,
        batch_size=args.export_batch_size,
    )

    if len(train_rows) != args.expected_train_samples:
        raise ValueError(f'expected {args.expected_train_samples} train rows, got {len(train_rows)}')
    if len(val_rows) != args.expected_val_samples:
        raise ValueError(f'expected {args.expected_val_samples} val rows, got {len(val_rows)}')
    train_missing = _missing_count(train_rows)
    val_missing = _missing_count(val_rows)
    if train_missing or val_missing:
        raise FileNotFoundError(f'exported manifest has missing images: train={train_missing}, val={val_missing}')

    dataset_root = workspace_root / 'datasets' / args.dataset_name
    dataset_root.mkdir(parents=True, exist_ok=True)
    train_json = dataset_root / 'train.json'
    val_json = dataset_root / 'val.json'
    if args.force or not train_json.exists():
        _write_jsonl(train_json, train_rows)
    if args.force or not val_json.exists():
        _write_jsonl(val_json, val_rows)

    meta = {
        'dataset_name': args.dataset_name,
        'source_type': 'huggingface_parquet',
        'hf_repo_id': args.repo_id,
        'hf_revision': args.revision,
        'hf_snapshot_sha': snapshot_sha,
        'download_dir': str(download_dir),
        'source_root': str(source_root),
        'class_list_path': str(class_list_path),
        'num_classes': len(classes),
        'train_samples': len(train_rows),
        'val_samples': len(val_rows),
        'train_missing': train_missing,
        'val_missing': val_missing,
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
    fid_meta = {
        'source_type': 'huggingface_parquet',
        'hf_repo_id': args.repo_id,
        'hf_snapshot_sha': snapshot_sha,
        'dataset_name': args.dataset_name,
        'image_size': args.image_size,
        'train_samples': len(train_rows),
    }
    if not args.skip_fid_stats and _should_recompute_fid_stats(
        fid_stats_path,
        fid_meta,
        force=args.force_fid_stats,
    ):
        _compute_fid_stats(
            ref_images_dir=ref_root / 'images',
            image_size=args.image_size,
            out_path=fid_stats_path,
            batch_size=args.fid_stats_batch_size,
            cuda=bool(args.fid_stats_cuda),
        )
        _stats_meta_path(fid_stats_path).write_text(json.dumps(fid_meta, indent=2), encoding='utf-8')

    summary = {
        'dataset_root': str(dataset_root),
        'source_root': str(source_root),
        'download_dir': str(download_dir),
        'ref_root': str(ref_root),
        'fid_stats_path': str(fid_stats_path),
        **meta,
    }
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
