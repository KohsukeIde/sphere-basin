#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${1:-$ROOT_DIR/.venv}"

python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel

if [[ "${INSTALL_TORCH:-0}" == "1" ]]; then
  python -m pip install torch torchvision
else
  echo "skip torch/torchvision install; set INSTALL_TORCH=1 to install default wheels"
  echo "if you need a CUDA-specific build, install torch/torchvision manually before training"
fi

python -m pip install -r "$ROOT_DIR/requirements.txt"

python - <<'PY'
mods = ['torch', 'torchvision', 'fvcore', 'wandb', 'torch_fidelity', 'huggingface_hub', 'pandas', 'matplotlib', 'yaml', 'tabulate', 'rich', 'requests']
missing = []
for mod in mods:
    try:
        __import__(mod)
    except Exception:
        missing.append(mod)

if missing:
    print('missing modules:', ', '.join(missing))
else:
    print('environment looks ready')
PY

echo "activate with: source \"$VENV_DIR/bin/activate\""
