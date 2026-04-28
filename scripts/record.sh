#!/usr/bin/env bash
# Record a teleoperated dataset.
# Usage: ./scripts/record.sh "<task prompt>" [dataset_name] [num_episodes]
# Datasets stay local — push_to_hub is forced off.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 \"<task prompt>\" [dataset_name=so101_default] [num_episodes=50]" >&2
  exit 2
fi

task="$1"
dataset_name="${2:-so101_default}"
num_episodes="${3:-50}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$REPO_ROOT/data}"

: "${ARM_FARM_FOLLOWER_PORT:?source .env first}"
: "${ARM_FARM_LEADER_PORT:?source .env first}"
: "${ARM_FARM_WRIST_CAM:=0}"
: "${ARM_FARM_FRONT_CAM:=1}"
: "${ARM_FARM_FOLLOWER_ID:=follower_01}"
: "${ARM_FARM_LEADER_ID:=leader_01}"
: "${HF_USER:=arm_farm}"

cameras="{wrist:{type:opencv,index_or_path:$ARM_FARM_WRIST_CAM,width:640,height:480,fps:30},"
cameras+="front:{type:opencv,index_or_path:$ARM_FARM_FRONT_CAM,width:640,height:480,fps:30}}"

uv run lerobot-record \
  --robot.type=so101_follower \
  --robot.port="$ARM_FARM_FOLLOWER_PORT" \
  --robot.id="$ARM_FARM_FOLLOWER_ID" \
  --robot.cameras="$cameras" \
  --teleop.type=so101_leader \
  --teleop.port="$ARM_FARM_LEADER_PORT" \
  --teleop.id="$ARM_FARM_LEADER_ID" \
  --dataset.repo_id="$HF_USER/$dataset_name" \
  --dataset.single_task="$task" \
  --dataset.num_episodes="$num_episodes" \
  --dataset.push_to_hub=false \
  --display_data=true
