#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY_BIN="python3"
if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PY_BIN="$ROOT_DIR/.venv/bin/python"
fi

CONFIG_PATH="${1:-configs/cifar_alpha_pilot.yaml}"

cd "$ROOT_DIR"
"$PY_BIN" -m sphere_basin.launch run-e1 --config "$CONFIG_PATH" --sphere-repo "${SPHERE_REPO:?set SPHERE_REPO}" "${@:2}"
