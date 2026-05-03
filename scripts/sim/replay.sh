#!/usr/bin/env bash
# Replay a recorded SO-ARM101 dataset inside mjlab via the MujocoSO101
# robot wrapper. Viewer backend selectable via ARM_FARM_SIM_VIEWER:
# `viser` (default, web URL printed), `native` (OpenGL window — macOS
# re-execs under mjpython for the Cocoa main loop), or `none` (headless).
#
# Examples:
#   ./scripts/sim/replay.sh --dataset.repo_id=arm_farm/pickplace --dataset.episode=0
#   ARM_FARM_SIM_VIEWER=native ./scripts/sim/replay.sh --dataset.repo_id=arm_farm/pickplace
#   ARM_FARM_SIM_VIEWER=none   ./scripts/sim/replay.sh --dataset.repo_id=arm_farm/pickplace
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -f "$REPO_ROOT/.env" ]]; then set -a; source "$REPO_ROOT/.env"; set +a; fi
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$REPO_ROOT/data}"

VIEWER="${ARM_FARM_SIM_VIEWER:-viser}"

REPLAY_ARGS=(
  --robot.type=mujoco_so101
  --robot.id="${ARM_FARM_FOLLOWER_ID:-follower_01}"
  --robot.viewer="$VIEWER"
  --dataset.push_to_hub=false
)

# macOS + native viewer: re-exec under mjpython for the Cocoa main loop;
# DYLD_FALLBACK_LIBRARY_PATH points at the uv-managed interpreter's LIBDIR
# so mjpython's dlopen of libpython succeeds.
if [[ "$(uname -s)" == "Darwin" && "$VIEWER" == "native" ]]; then
  PYLIB="$(uv run --extra sim python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')"
  export DYLD_FALLBACK_LIBRARY_PATH="$PYLIB${DYLD_FALLBACK_LIBRARY_PATH:+:$DYLD_FALLBACK_LIBRARY_PATH}"
  exec uv run --extra sim mjpython -m lerobot.scripts.lerobot_replay \
    "${REPLAY_ARGS[@]}" "$@"
fi
exec uv run --extra sim lerobot-replay "${REPLAY_ARGS[@]}" "$@"
