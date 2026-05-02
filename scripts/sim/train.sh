#!/usr/bin/env bash
# Train an mjlab task with PPO. Linux + NVIDIA CUDA only; on macOS this will
# fall back to CPU torch and run unworkably slowly. Outputs land under
# outputs/sim/<timestamp>-<task>/.
#
# Examples:
#   ./scripts/sim/train.sh                                            # Cube
#   ./scripts/sim/train.sh Cube --env.scene.num-envs 4096
#   ./scripts/sim/train.sh Cube-Depth --agent.max-iterations 3000
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -f "$REPO_ROOT/.env" ]]; then set -a; source "$REPO_ROOT/.env"; set +a; fi
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$REPO_ROOT/data}"

TASK="${1:-Cube}"
shift || true

OUT_DIR="$REPO_ROOT/outputs/sim/$(date +%Y%m%dT%H%M%S)-$TASK"
mkdir -p "$OUT_DIR"

uv run --extra sim train "$TASK" \
  --output_dir "$OUT_DIR" \
  "$@"
