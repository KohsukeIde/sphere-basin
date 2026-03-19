# Canonical Probe Matrix

This workflow rebuilds a single post-hoc evaluation regime for the current CIFAR pilot checkpoints.

The default matrix is:

- jobs:
  - `sphere-small-small-cifar-10-32px-a80-pilot`: `ep0024`, `ep0049`, `ep0074`, `ep0099`, `ep0124`
  - `sphere-small-small-cifar-10-32px-a83-pilot`: `ep0024`
  - `sphere-small-small-cifar-10-32px-a85-pilot`: `ep0024`
- regimes:
  - `independent-fixed`
  - `shared-fixed`

Run the canonical reprobe batch:

```bash
export SPHERE_REPO=/absolute/path/to/sphere-encoder
bash scripts/install_overlay.sh "$SPHERE_REPO"
bash scripts/run_canonical_probe_matrix.sh \
  --config configs/canonical_probe_matrix.yaml \
  --force
```

Build the clean tables:

```bash
bash scripts/build_canonical_master.sh \
  --config configs/canonical_probe_matrix.yaml
```

Outputs are written to `workspace/research_summary/`:

- `canonical_probe_matrix_manifest.json`
- `canonical_probe_manifest.csv`
- `canonical_prior_long.csv`
- `canonical_contraction_long.csv`
- `canonical_master.csv`
- `canonical_phase.csv`

`canonical_master.csv` is the main table. Each row is one
`(job_dir, ckpt_epoch, regime_name, forward_steps, tau_deg)` entry with:

- job metadata: `alpha`, dataset, model sizes, checkpoint
- probe metadata: `noise_mode`, `schedule_mode`, `regime_name`
- prior metrics: terminal CDF, NN manifold capture, NN improvement, curvature
- generation metrics: `fid`, `isc_mean`, `isc_std`, `recon_mse`
- contraction metrics: `kappa_mean` and `contraction_capture_mass` at the target noise scaler

`canonical_phase.csv` is a filtered export for the current phase plot:

- checkpoint: `ep0024`
- regime: `independent-fixed`
- forward steps: `4`
- tau: `60`
