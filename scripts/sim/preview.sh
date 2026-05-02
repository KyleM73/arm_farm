#!/usr/bin/env bash
# Open the SO-ARM101 sim in the native MuJoCo viewer with no policy and zero
# actions. The robot drops into its home pose and the viewer stays open. Use
# this as a smoke check after `uv sync --extra sim`.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec "$REPO_ROOT/scripts/sim/play.sh" Play --agent zero "$@"
