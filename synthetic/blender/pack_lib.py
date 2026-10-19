"""Make a self-contained copy of a human library: textures shrunk to <= --max px and packed into the .blend.

People are at most ~120 px tall in the renders, so 1024 px textures lose nothing visible and keep the
file small enough to upload (e.g. to a Kaggle dataset).
    Blender -b synthetic/mpfb/humans.blend -P synthetic/blender/pack_lib.py -- --out synthetic/mpfb/humans_packed.blend
"""

import argparse
import os
import sys

import bpy

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--max", type=int, default=1024)
args = ap.parse_args(sys.argv[sys.argv.index("--") + 1 :])
n = 0
for img in bpy.data.images:
    if img.source != "FILE" or not img.filepath:
        continue
    w, h = img.size
    if max(w, h) > args.max:
        k = args.max / max(w, h)
        img.scale(max(1, int(w * k)), max(1, int(h * k)))
    img.pack()
    n += 1
bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(args.out), compress=True)
print("packed", n, "images ->", args.out)
