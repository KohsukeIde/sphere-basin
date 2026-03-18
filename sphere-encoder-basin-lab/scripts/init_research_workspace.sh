#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_DIR="${1:-$ROOT_DIR/workspace}"

mkdir -p \
  "$WORKSPACE_DIR/experiments" \
  "$WORKSPACE_DIR/evaluation" \
  "$WORKSPACE_DIR/visualization" \
  "$WORKSPACE_DIR/interpolation" \
  "$WORKSPACE_DIR/image_editing" \
  "$WORKSPACE_DIR/research_summary"

if [[ ! -e "$WORKSPACE_DIR/jobs" ]]; then
  ln -s experiments "$WORKSPACE_DIR/jobs"
fi

for name in datasets fid_stats fid_refs; do
  dst="$WORKSPACE_DIR/$name"
  if [[ -e "$dst" ]]; then
    continue
  fi
  if [[ -n "${SPHERE_REPO:-}" && -e "$SPHERE_REPO/workspace/$name" ]]; then
    ln -s "$SPHERE_REPO/workspace/$name" "$dst"
  else
    mkdir -p "$dst"
  fi
done

echo "research workspace ready: $WORKSPACE_DIR"
