# Paper skeleton

Working title:

> Why Sphere Encoders Generate: Basin Capture and Local Retraction on the Sphere

Core claims:

1. `noise_sigma_max_angle` controls prior-side basin mass.
2. `lat_con_loss_weight` controls local contraction back to clean latents.
3. shared fixed noise reduces trajectory curvature and improves few-step refinement.

Repository support for those claims:

- alpha sweep: `configs/cifar_alpha_pilot.yaml`, `configs/cifar_alpha_main.yaml`
- loss ablation: `configs/cifar_loss_ablation.yaml`
- prior-side projector probe: `research/probe_projector.py`
- training/eval aggregation: `sphere_basin.aggregate`
- phase diagrams: `sphere_basin.plot_phase_diagram`
