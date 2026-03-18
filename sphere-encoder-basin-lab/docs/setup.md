# Setup

## Goal

Before running any alpha sweep, prepare:

- a Python environment with the official Sphere Encoder runtime deps
- a research-only workspace under `sphere-encoder-basin-lab/workspace`
- CIFAR-10 data under `workspace/datasets/cifar-10`
- FID stats under `workspace/fid_stats/fid_stats_extr_cifar-10_32px.npz`

## Recommended order

1. Create the environment.

```bash
bash scripts/setup_env.sh
source .venv/bin/activate
```

If you are fine with the default pip wheels, use `INSTALL_TORCH=1 bash scripts/setup_env.sh`.

If your machine needs a specific CUDA wheel, install `torch` and `torchvision` manually first, then rerun the script.

2. Install the overlay into the official repo.

```bash
export SPHERE_REPO=/absolute/path/to/sphere-encoder
bash scripts/install_overlay.sh "$SPHERE_REPO"
```

3. Initialize the research workspace and fetch CIFAR-10 + FID stats.

```bash
bash scripts/bootstrap_cifar_research.sh
```

## Resulting layout

```text
sphere-encoder-basin-lab/
├── workspace/
│   ├── datasets/
│   │   └── cifar-10/
│   ├── experiments/
│   ├── evaluation/
│   ├── fid_stats/
│   │   └── fid_stats_extr_cifar-10_32px.npz
│   ├── jobs -> experiments
│   └── research_summary/
```

## Notes

- For CIFAR-10 and CIFAR-100, dataset download is automatic through `torchvision`.
- For gFID / rFID, the official evaluator still needs the `.npz` stats file; `bootstrap_cifar_research.sh` fetches it from the Hugging Face artifact repo referenced by the official README.
- Non-CIFAR datasets still need manual dataset-list preparation in the official format.
