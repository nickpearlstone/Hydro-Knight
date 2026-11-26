#!/usr/bin/env bash
set -euo pipefail
B="$(cd "$(dirname "$0")/.." && pwd)/mpfb/bl"
export BLENDER_USER_CONFIG="$B/config" BLENDER_USER_SCRIPTS="$B/scripts"
export BLENDER_USER_EXTENSIONS="$B/ext" BLENDER_USER_DATAFILES="$B/datafiles"
exec "$@"
