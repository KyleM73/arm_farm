#!/usr/bin/env bash
# Calibrate the SO-ARM101 follower and leader. Run once per arm; the
# resulting calibration is keyed by `id` and reused on every subsequent run.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$REPO_ROOT/.env" ]]; then set -a; source "$REPO_ROOT/.env"; set +a; fi
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$REPO_ROOT/data}"

: "${ARM_FARM_FOLLOWER_PORT:?source .env first}"
: "${ARM_FARM_LEADER_PORT:?source .env first}"
: "${ARM_FARM_FOLLOWER_ID:=follower_01}"
: "${ARM_FARM_LEADER_ID:=leader_01}"

echo "==> Calibrating follower (id=$ARM_FARM_FOLLOWER_ID, port=$ARM_FARM_FOLLOWER_PORT)"
uv run lerobot-calibrate \
  --robot.type=so101_follower \
  --robot.port="$ARM_FARM_FOLLOWER_PORT" \
  --robot.id="$ARM_FARM_FOLLOWER_ID"

echo "==> Calibrating leader (id=$ARM_FARM_LEADER_ID, port=$ARM_FARM_LEADER_PORT)"
uv run lerobot-calibrate \
  --teleop.type=so101_leader \
  --teleop.port="$ARM_FARM_LEADER_PORT" \
  --teleop.id="$ARM_FARM_LEADER_ID"
