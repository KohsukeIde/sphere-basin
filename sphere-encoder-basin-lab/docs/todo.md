# TODO

# Week 0
- [ ] create Python environment
- [ ] install overlay into official repo
- [ ] initialize research workspace
- [ ] fetch CIFAR-10 and FID stats

## Week 1
- [ ] run CIFAR-10 pilot alpha sweep `{80,83,85,87}`
- [ ] sanity-check `probe_projector.py` on one checkpoint

## Week 2
- [ ] main alpha sweep `{80,83,84,85,86,88}`
- [ ] run official eval for `forward_steps={1,4}`
- [ ] aggregate phase diagram `alpha -> {FID, capture mass}`

## Week 3
- [ ] run loss ablations at `alpha=85`
- [ ] run shared-vs-independent / fixed-vs-scheduler sampling probes
- [ ] plot contraction metrics vs ablation

## Week 4
- [ ] optional second dataset
- [ ] finalize figures and tables
- [ ] freeze csv/json artifacts for paper writing
