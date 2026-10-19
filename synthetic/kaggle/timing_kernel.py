"""Kaggle GPU job: render the synthetic wave-pool scene with Linux Blender and time it.

Input dataset (npstone/hk-synth): scene.py, humans.py, post.py, ref_6f5c.json, humans_packed.blend.
Run 1 = the b13 setup (seed 2, 4-frame moving-water clip, 32 samples, 1920 px); run 2 = one plain
frame (seed 3, no clip); then post.py on run 1. Writes timings.json + frames to /kaggle/working.
"""

import glob
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

W = "/kaggle/working"
BL_URL = (
    "https://download.blender.org/release/Blender4.2/blender-4.2.1-linux-x64.tar.xz"
)


def sh(cmd, log=None):
    print("$", cmd, flush=True)
    t = time.time()
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    out = r.stdout + r.stderr
    if log:
        with open(log, "w") as f:
            f.write(out)
    print(out[-3000:], flush=True)
    return out, time.time() - t, r.returncode


timings = {}
src = os.path.dirname(glob.glob("/kaggle/input/**/scene.py", recursive=True)[0])
work = f"{W}/synth"
shutil.copytree(src, work, dirs_exist_ok=True)
cfg = json.loads(Path(f"{work}/ref_6f5c.json").read_text())
cfg["crowd"]["human_lib"] = f"{src}/humans_packed.blend"
Path(f"{work}/ref_kaggle.json").write_text(json.dumps(cfg, indent=1))

timings["gpu"] = sh("nvidia-smi --query-gpu=name,memory.total --format=csv")[0]
sh(
    "apt-get -qq install -y libxi6 libxkbcommon0 libxxf86vm1 libxfixes3 libxrender1 libgl1 libsm6 > /dev/null 2>&1 || true"
)
_, t, _ = sh(f"cd /tmp && wget -q {BL_URL} && tar xf blender-4.2.1-linux-x64.tar.xz")
timings["blender_download_s"] = round(t, 1)
B = "/tmp/blender-4.2.1-linux-x64/blender"

runs = {
    "b13_clip4_seed2": f"--seed 2 --clip 4 --samples 32 --res 1920 --out {W}/b13k",
    "single_seed3": f"--seed 3 --clip 1 --samples 32 --res 1920 --out {W}/single",
}
for name, extra in runs.items():
    out, t, rc = sh(
        f"cd {work} && {B} -b -P scene.py -- --config ref_kaggle.json {extra}",
        log=f"{W}/{name}.log",
    )
    timings[name] = {
        "wall_s": round(t, 1),
        "returncode": rc,
        "devices": re.findall(r"DEVICES .*", out)[:1],
        "build_s": [float(x) for x in re.findall(r"TIMING build ([0-9.]+)s", out)],
        "render_s": [float(x) for x in re.findall(r"TIMING render ([0-9.]+)s", out)],
    }

_, t, rc = sh(
    f"pip -q install opencv-python-headless pandas pyarrow > /dev/null 2>&1; cd {work} && python post.py {W}/b13k"
)
timings["post_s"] = round(t, 1)
Path(f"{W}/timings.json").write_text(json.dumps(timings, indent=1))
print(json.dumps(timings, indent=1))
shutil.rmtree(work, ignore_errors=True)
