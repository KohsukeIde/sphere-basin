#!/usr/bin/env bash
set -euo pipefail

TARGET_REPO="${1:-${SPHERE_REPO:-}}"
if [[ -z "$TARGET_REPO" ]]; then
  echo "usage: bash scripts/install_overlay.sh /path/to/sphere-encoder"
  exit 1
fi

if [[ ! -d "$TARGET_REPO" ]]; then
  echo "target repo not found: $TARGET_REPO"
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$TARGET_REPO/research"
cp -f "$ROOT_DIR/overlay/research/"*.py "$TARGET_REPO/research/"
cp -f "$ROOT_DIR/overlay/research/__init__.py" "$TARGET_REPO/research/"

echo "overlay installed into: $TARGET_REPO/research"
