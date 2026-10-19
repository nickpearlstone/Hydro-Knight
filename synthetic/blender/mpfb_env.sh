#!/usr/bin/env bash
# Run a command with Blender's user dirs pointed at the repo-local MPFB install (synthetic/mpfb/bl),
# so the extension and the CC0 asset pack load without touching the global Blender config.
#   synthetic/blender/mpfb_env.sh synthetic/blender-bin/Blender.app/Contents/MacOS/Blender -b -P ...
set -euo pipefail
B="$(cd "$(dirname "$0")/.." && pwd)/mpfb/bl"
export BLENDER_USER_CONFIG="$B/config" BLENDER_USER_SCRIPTS="$B/scripts"
export BLENDER_USER_EXTENSIONS="$B/ext" BLENDER_USER_DATAFILES="$B/datafiles"
exec "$@"
