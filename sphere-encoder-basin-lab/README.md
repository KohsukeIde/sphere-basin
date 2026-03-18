# Sphere Encoder Basin Lab

Research overlay for the official [`facebookresearch/sphere-encoder`](https://github.com/facebookresearch/sphere-encoder) repo.

This scaffold is aimed at the NeurIPS-style "why does it generate?" project we discussed:

- alpha / angle phase transition
- loss ablations
- projector probes on the sphere
- sampling dynamics probes
- lightweight aggregation and plotting

It does **not** vendor the official Sphere Encoder code. Instead, it installs a small `research/` overlay into your cloned Sphere Encoder repo and drives training / evaluation from the outside.

By default, all research outputs go into `sphere-encoder-basin-lab/workspace/`, not the official repo's `workspace/`.

## What is inside

- `configs/`: experiment presets for CIFAR-10 pilot / main sweeps and loss ablations.
- `scripts/`: shell entrypoints.
- `sphere_basin/`: light orchestration, parsing, aggregation, and plotting utilities.
- `overlay/research/`: drop-in scripts that run **inside** the official Sphere Encoder repo.
- `workspace/`: default research-only artifact root created on demand.

## Quick start

1. Point this toolkit to your cloned Sphere Encoder repo.

```bash
export SPHERE_REPO=/absolute/path/to/sphere-encoder
```

2. Build the Python environment.

```bash
bash scripts/setup_env.sh
source .venv/bin/activate
```

If you are fine with the default pip wheels, use `INSTALL_TORCH=1 bash scripts/setup_env.sh`. If you need a specific CUDA wheel, install `torch` and `torchvision` manually first, then rerun `setup_env.sh`.

3. Install the overlay scripts into the official repo.

```bash
bash scripts/install_overlay.sh "$SPHERE_REPO"
```

4. Initialize a research-only workspace and fetch CIFAR-10 + FID stats.

```bash
bash scripts/bootstrap_cifar_research.sh
```

This creates `./workspace/` in this repo, reuses `datasets`, `fid_stats`, and `fid_refs` from `$SPHERE_REPO/workspace/` when available, downloads CIFAR-10 through `torchvision`, and fetches `fid_stats_extr_cifar-10_32px.npz`.

5. Run a pilot alpha sweep.

```bash
bash scripts/train_alpha_sweep.sh configs/cifar_alpha_pilot.yaml
```

Or run the whole E1 pipeline in one command:

```bash
bash scripts/run_e1_phase.sh configs/cifar_alpha_pilot.yaml
```

The provided CIFAR configs use both local GPUs by default. Training jobs are spread across `gpu_groups`, and `run_e1_phase.sh` also parallelizes eval + probe across the same groups.

6. Run official FID eval on one trained job.

```bash
bash scripts/official_eval.sh \
  --job-dir sphere-base-base-cifar-10-32px-a85-pilot \
  --forward-steps 1 4 \
  --cfg-min 1.0 \
  --cfg-max 1.0
```

7. Run projector / basin probes.

```bash
bash scripts/probe_basin_metrics.sh \
  --job-dir sphere-base-base-cifar-10-32px-a85-pilot \
  --forward-steps 1 4 \
  --num-prior-samples 4096 \
  --num-data-samples 4096
```

8. Aggregate and plot.

```bash
python3 -m sphere_basin.aggregate
python3 -m sphere_basin.plot_phase_diagram \
  --csv "./workspace/research_summary/summary_generation.csv"
```

For a fast end-to-end sanity check before the real pilot:

```bash
bash scripts/run_e1_phase.sh configs/cifar_alpha_smoke.yaml
```

## Why the wrapper renames jobs

The official training code names experiments using only model size, dataset, and image resolution. A direct alpha sweep would overwrite runs. This toolkit launches training, then renames the created job directory to a unique name such as:

```text
sphere-base-base-cifar-10-32px-a85-pilot
```

Each renamed job also gets a `research/manifest.json` with the sweep metadata.

## Notes

- The official repo stays mostly untouched. The only in-place addition is `research/` for probe scripts.
- Relative `dev_dir` values in the configs are resolved against this repo, so `dev_dir: workspace` means `sphere-encoder-basin-lab/workspace`.
- If you explicitly want to use the official repo's artifact root, pass an absolute path such as `--dev-dir "$SPHERE_REPO/workspace"`.
- The wrapper creates a `jobs -> experiments` symlink in the research workspace because the official code mixes both paths.
- The projector probes use the official `G.encoder`, `G.decoder`, and `G.spherify` methods directly, so they stay close to the actual sampling dynamics.
- The provided configs are tuned for *small-scale mechanistic experiments*, not for reproducing the paper’s full-scale numbers.
- A more explicit bootstrap guide is in [docs/setup.md](/home/cvrt/Desktop/dev/sphere-encoder-basin-lab/docs/setup.md).
