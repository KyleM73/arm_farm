#!/usr/bin/env bash
# Drive the follower with the leader and view live cameras. Use this to
# sanity-check ports, IDs, calibration, and camera framing before recording.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$REPO_ROOT/data}"

: "${ARM_FARM_FOLLOWER_PORT:?source .env first}"
: "${ARM_FARM_LEADER_PORT:?source .env first}"
: "${ARM_FARM_WRIST_CAM:=0}"
: "${ARM_FARM_FRONT_CAM:=1}"
: "${ARM_FARM_FOLLOWER_ID:=follower_01}"
: "${ARM_FARM_LEADER_ID:=leader_01}"

cameras="{wrist:{type:opencv,index_or_path:$ARM_FARM_WRIST_CAM,width:640,height:480,fps:30},"
cameras+="front:{type:opencv,index_or_path:$ARM_FARM_FRONT_CAM,width:640,height:480,fps:30}}"

uv run lerobot-teleoperate \
  --robot.type=so101_follower \
  --robot.port="$ARM_FARM_FOLLOWER_PORT" \
  --robot.id="$ARM_FARM_FOLLOWER_ID" \
  --robot.cameras="$cameras" \
  --teleop.type=so101_leader \
  --teleop.port="$ARM_FARM_LEADER_PORT" \
  --teleop.id="$ARM_FARM_LEADER_ID" \
  --display_data=true
