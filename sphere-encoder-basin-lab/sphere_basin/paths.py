from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_root() -> Path:
    return PROJECT_ROOT


def default_workspace_root() -> Path:
    return PROJECT_ROOT / 'workspace'


def resolve_dev_dir(dev_dir: str | Path | None = None) -> Path:
    if dev_dir is None:
        return default_workspace_root()
    path = Path(dev_dir).expanduser()
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def ensure_workspace_compat(dev_dir: str | Path | None = None) -> Path:
    workspace_root = resolve_dev_dir(dev_dir)
    exp_dir = workspace_root / 'experiments'
    jobs_dir = workspace_root / 'jobs'
    workspace_root.mkdir(parents=True, exist_ok=True)
    exp_dir.mkdir(parents=True, exist_ok=True)
    if not jobs_dir.exists():
        jobs_dir.symlink_to('experiments')
    return workspace_root


def find_job_dir(workspace_root: str | Path, job_dir: str) -> Path:
    workspace_root = resolve_dev_dir(workspace_root)
    cands = [
        workspace_root / 'experiments' / job_dir,
        workspace_root / 'jobs' / job_dir,
    ]
    for cand in cands:
        if cand.exists():
            return cand
    raise FileNotFoundError(f'job_dir not found under experiments/jobs: {job_dir}')


def canonical_job_name(train_args: dict) -> str:
    return (
        f"sphere-{train_args['vit_enc_model_size']}"
        f"-{train_args['vit_dec_model_size']}"
        f"-{train_args['dataset_name']}"
        f"-{train_args['image_size']}px"
    )


def slugify(text: str) -> str:
    out = []
    for ch in text:
        if ch.isalnum() or ch in ['-', '_']:
            out.append(ch)
        else:
            out.append('-')
    return ''.join(out).strip('-')
