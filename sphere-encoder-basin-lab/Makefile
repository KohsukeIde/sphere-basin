CONFIG ?= configs/cifar_alpha_pilot.yaml
WORKSPACE ?= $(CURDIR)/workspace

.PHONY: deps env install workspace data alpha loss aggregate plot

deps:
	python3 -m pip install -r requirements.txt

env:
	bash scripts/setup_env.sh

data:
	bash scripts/bootstrap_cifar_research.sh "$(WORKSPACE)"

install:
	bash scripts/install_overlay.sh "$(SPHERE_REPO)"

workspace:
	bash scripts/init_research_workspace.sh "$(WORKSPACE)"

alpha:
	bash scripts/train_alpha_sweep.sh $(CONFIG)

loss:
	bash scripts/train_loss_ablation.sh configs/cifar_loss_ablation.yaml

aggregate:
	python3 -m sphere_basin.aggregate --workspace "$(WORKSPACE)"

plot:
	python3 -m sphere_basin.plot_phase_diagram --csv "$(WORKSPACE)/research_summary/summary_generation.csv"
