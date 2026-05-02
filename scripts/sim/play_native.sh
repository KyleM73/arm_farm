#!/usr/bin/env bash
# Open a registered task's scene under native `mujoco.mj_step` + mjviser.
# Bypasses mjwarp entirely for ~10x faster Mac CPU playback. Useful for
# inspecting MJCF / home pose / ee_site placement without paying the
# kernel-compile + dispatch cost mjwarp imposes on Apple silicon.
#
# Action source defaults to `zero` (arm settles into its keyframe). Pass
# `--action=random` or `--action=sine` to drive the actuators with a
# scripted source. Pass `--checkpoint <path.onnx>` to run a trained
# mjlab policy — the ONNX file mjlab auto-exports alongside each
# checkpoint embeds the obs/action metadata this script reads to wire
# inference up correctly.
#
# Examples:
#   ./scripts/sim/play_native.sh                            # Cube task, zero actions
#   ./scripts/sim/play_native.sh --task=Play                # idle scene, zero actions
#   ./scripts/sim/play_native.sh --task=Cube --action=sine  # sinusoidal joint sweep
#   ./scripts/sim/play_native.sh \
#       --task=Cube \
#       --checkpoint=outputs/sim/20260501T123456-Cube/20260501T123456-Cube.onnx
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -f "$REPO_ROOT/.env" ]]; then set -a; source "$REPO_ROOT/.env"; set +a; fi
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$REPO_ROOT/data}"

exec uv run --extra sim arm-farm-play-native "$@"
