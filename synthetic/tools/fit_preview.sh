#!/usr/bin/env bash
# Quick camera-fit check: exact (unvaried) low-quality render of a config, with its lens applied,
# stacked above the config's real reference frame.  synthetic/tools/fit_preview.sh ref_dc93 [samples] [res]
set -euo pipefail
NAME=$1
OUT=synthetic/renders/fit/$NAME
rm -rf "$OUT"
synthetic/blender-bin/Blender.app/Contents/MacOS/Blender -b -P synthetic/blender/scene.py -- \
  --config "synthetic/configs/$NAME.json" --out "$OUT" --exact --samples "${2:-8}" --res "${3:-960}" \
  2>&1 | grep -E "wrote frame|Traceback|Error" || true
uv run python - "$NAME" "$OUT" <<'PY'
import json, sys
import cv2, numpy as np
sys.path.insert(0, "synthetic/tools")
import post
name, out = sys.argv[1], sys.argv[2]
cfg = json.load(open(f"synthetic/configs/{name}.json"))
img = cv2.imread(f"{out}/rgb_0000.png")
k = cfg["camera"].get("lens_k", 0.0)
if k > 0:
    mx, my = post.fisheye_maps(img.shape[1], img.shape[0], k)
    img = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
real = cv2.imread(f"raw_local/pose_bench/{cfg['reference']}.png")
cv2.imwrite(f"{out}/fit.jpg", np.vstack([cv2.resize(real, (960, 540)), cv2.resize(img, (960, 540))]), [cv2.IMWRITE_JPEG_QUALITY, 82])
print("fit image", f"{out}/fit.jpg")
PY
