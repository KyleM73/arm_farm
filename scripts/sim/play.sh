#!/usr/bin/env bash
# Play (or roll out a checkpoint of) an mjlab task. Default task is `Cube`;
# pass another task ID as the first arg. Any extra arguments are forwarded
# to mjlab's `play` CLI verbatim.
#
# Defaults to the viser web viewer (no display / mjpython dance needed);
# override with `--viewer=native` for the OpenGL window. macOS + native
# routes through `mjpython` with DYLD_FALLBACK_LIBRARY_PATH pointed at
# uv's libpython so launch_passive can host the Cocoa main loop.
#
# Examples:
#   ./scripts/sim/play.sh                              # Cube via viser (web URL printed)
#   ./scripts/sim/play.sh Play --agent zero            # idle scene, zero actions
#   ./scripts/sim/play.sh Cube --viewer=native         # native MuJoCo window
#   ./scripts/sim/play.sh Cube --wandb-run-path <run>  # roll out a trained checkpoint
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -f "$REPO_ROOT/.env" ]]; then set -a; source "$REPO_ROOT/.env"; set +a; fi
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$REPO_ROOT/data}"

TASK="${1:-Cube}"
shift || true

# Detect a user-supplied --viewer (either `--viewer=X` or `--viewer X`);
# if absent, default to viser.
USER_VIEWER=""
prev=""
for arg in "$@"; do
  if [[ "$prev" == "--viewer" ]]; then
    USER_VIEWER="$arg"
  elif [[ "$arg" == --viewer=* ]]; then
    USER_VIEWER="${arg#--viewer=}"
  fi
  prev="$arg"
done
VIEWER="${USER_VIEWER:-viser}"
if [[ -z "$USER_VIEWER" ]]; then
  set -- "$@" "--viewer=$VIEWER"
fi

if [[ "$(uname -s)" == "Darwin" && "$VIEWER" == "native" ]]; then
  PYLIB="$(uv run --extra sim python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')"
  export DYLD_FALLBACK_LIBRARY_PATH="$PYLIB${DYLD_FALLBACK_LIBRARY_PATH:+:$DYLD_FALLBACK_LIBRARY_PATH}"
  exec uv run --extra sim mjpython -m mjlab.scripts.play "$TASK" "$@"
fi
exec uv run --extra sim play "$TASK" "$@"
