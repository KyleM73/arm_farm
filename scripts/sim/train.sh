#!/usr/bin/env bash
# Train an mjlab task with PPO. Linux + CUDA only (CPU torch on macOS is
# unworkably slow). Outputs land under outputs/sim/<timestamp>-<task>/.
#
# Examples:
#   ./scripts/sim/train.sh
#   ./scripts/sim/train.sh Cube --env.scene.num-envs 4096
#   ./scripts/sim/train.sh Cube-Depth --agent.max-iterations 3000
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -f "$REPO_ROOT/.env" ]]; then set -a; source "$REPO_ROOT/.env"; set +a; fi
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$REPO_ROOT/data}"
# Headless OpenGL for the offscreen renderer (--video). Must be set before
# mujoco is imported; mjlab's own MUJOCO_GL=egl runs too late.
export MUJOCO_GL="${MUJOCO_GL:-egl}"

TASK="${1:-Cube}"
shift || true

OUT_DIR="$REPO_ROOT/outputs/sim/$(date +%Y%m%dT%H%M%S)-$TASK"
mkdir -p "$OUT_DIR"

uv run --extra sim train "$TASK" \
  --output_dir "$OUT_DIR" \
  "$@"
