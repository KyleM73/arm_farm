#!/usr/bin/env bash
# Replay a recorded SO-ARM101 dataset inside mjlab via the MujocoSO101 robot
# wrapper. The viewer backend is selectable via ARM_FARM_SIM_VIEWER:
#   viser (default) — web viewer; viser prints the URL on startup.
#   native          — MuJoCo OpenGL window (needs a display; macOS routes
#                     through `mjpython` automatically).
#   none            — no viewer (headless replay-to-disk).
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

# macOS + native viewer: re-exec under mjpython so launch_passive can host
# the Cocoa main loop. uv-managed Pythons keep libpython outside the venv,
# so DYLD_FALLBACK_LIBRARY_PATH has to point at the interpreter's LIBDIR
# for mjpython's dlopen to succeed. Other (viewer, platform) combinations
# run under regular python.
if [[ "$(uname -s)" == "Darwin" && "$VIEWER" == "native" ]]; then
  PYLIB="$(uv run --extra sim python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')"
  export DYLD_FALLBACK_LIBRARY_PATH="$PYLIB${DYLD_FALLBACK_LIBRARY_PATH:+:$DYLD_FALLBACK_LIBRARY_PATH}"
  exec uv run --extra sim mjpython -m lerobot.scripts.lerobot_replay \
    "${REPLAY_ARGS[@]}" "$@"
fi
exec uv run --extra sim lerobot-replay "${REPLAY_ARGS[@]}" "$@"
