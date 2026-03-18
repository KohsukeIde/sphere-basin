#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_DIR="${1:-$ROOT_DIR/workspace}"
PY_BIN="python3"
if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PY_BIN="$ROOT_DIR/.venv/bin/python"
fi

bash "$ROOT_DIR/scripts/init_research_workspace.sh" "$WORKSPACE_DIR"
cd "$ROOT_DIR"
"$PY_BIN" -m sphere_basin.setup_data --workspace "$WORKSPACE_DIR" --dataset-name cifar-10 --image-size 32
