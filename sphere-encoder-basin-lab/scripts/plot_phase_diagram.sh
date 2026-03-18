#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY_BIN="python3"
if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PY_BIN="$ROOT_DIR/.venv/bin/python"
fi
CSV_PATH="${1:-$ROOT_DIR/workspace/research_summary/summary_generation.csv}"
cd "$ROOT_DIR"
"$PY_BIN" -m sphere_basin.plot_phase_diagram --csv "$CSV_PATH"
