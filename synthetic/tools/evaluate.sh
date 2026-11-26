#!/usr/bin/env bash
set -euo pipefail
D=$1
P=${2:-'{}'}
export PYTORCH_ENABLE_MPS_FALLBACK=1
uv run python synthetic/tools/post.py "$D" --params "$P"
for f in "$D"/rgbv_*.png; do
  uv run python synthetic/tools/stats.py raw_local/pose_bench/6f5c67826a76_720.png "$f" 2>/dev/null
done
rm -f "$D"/rgbv_*.dets.npz
uv run --with transformers python synthetic/tools/score.py synth "$D" --prefix rgbv 2>&1 | grep -E "^==|@"
uv run --with transformers --with scikit-learn python synthetic/tools/realism.py "$D" --prefix rgbv 2>&1 | grep -E "^real|^rgbv"
