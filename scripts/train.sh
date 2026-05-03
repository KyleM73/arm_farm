#!/usr/bin/env bash
# Train a behavior cloning policy on a recorded local dataset.
# Usage: ./scripts/train.sh <act|diffusion|smolvla|pi> <dataset_name> [extra flags...]
# Example: ./scripts/train.sh act so101_pickplace --batch_size=8 --steps=100000
#
# Picks the uv extra by policy type so heavy deps are only pulled when used.
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <policy_type: act|diffusion|smolvla|pi> <dataset_name> [extra flags...]" >&2
  exit 2
fi

policy="$1"
dataset_name="$2"
shift 2

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$REPO_ROOT/.env" ]]; then set -a; source "$REPO_ROOT/.env"; set +a; fi
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$REPO_ROOT/data}"

: "${HF_USER:=arm_farm}"

case "$policy" in
  act)             extra="act" ;;
  diffusion)       extra="diffusion" ;;
  smolvla)         extra="smolvla" ;;
  pi|pi0)          extra="pi"; policy="pi0" ;;
  *)               echo "unknown policy: $policy" >&2; exit 2 ;;
esac

run_id="${policy}_$(date +%Y%m%d_%H%M%S)"
output_dir="$REPO_ROOT/outputs/train/$run_id"

uv run --extra "$extra" lerobot-train \
  --policy.type="$policy" \
  --dataset.repo_id="$HF_USER/$dataset_name" \
  --output_dir="$output_dir" \
  --job_name="$run_id" \
  --wandb.enable="${WANDB_ENABLE:-true}" \
  --wandb.project="${WANDB_PROJECT:-arm_farm}" \
  "$@"
