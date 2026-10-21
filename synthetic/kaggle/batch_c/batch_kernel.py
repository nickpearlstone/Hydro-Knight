"""Kaggle GPU job: render a varied batch of labelled wave-pool frames (one notebook = one PLAN entry).

Each config renders FRAMES frames with its "vary" block (camera, crowd), a 4-frame moving-water clip
per frame, then post.py (camera look + lens + video codec) and export_yolo.py (person + head boxes).
Two workers run side by side, one per T4 (CUDA_VISIBLE_DEVICES), each taking every second config.
A worker stops starting new frames after MAX_HOURS so the session never hits Kaggle's 12 h limit and
loses its output. Inputs: the private dataset npstone/hk-synth.
"""

import glob
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

PLAN = "c"  # set per notebook folder by synthetic/kaggle/make_batch.py
ALL = ["ref_6f5c", "ref_dc93", "ref_9992", "ref_1d50", "ref_644c", "ref_8c83"]
CONFIGS = {"a": ALL[:3], "b": ALL[3:], "c": ALL}[PLAN]
FRAMES = {"a": 50, "b": 50, "c": 25}[
    PLAN
]  # c: all six cameras in one notebook (half-size batch)
MAX_HOURS = 9.5  # leaves ~1 h for post + export before Kaggle's 12 h limit
W = "/kaggle/working"
BL_URL = (
    "https://download.blender.org/release/Blender4.2/blender-4.2.1-linux-x64.tar.xz"
)
T0 = time.time()


def sh(cmd, env=None):
    print("$", cmd, flush=True)
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env)
    print((r.stdout + r.stderr)[-1500:], flush=True)
    return r.returncode


src = os.path.dirname(glob.glob("/kaggle/input/**/scene.py", recursive=True)[0])
if (
    "--exact" not in Path(f"{src}/scene.py").read_text()
):  # stale dataset version mounted
    raise SystemExit(
        f"stale scene.py in {src}: re-version the hk-synth dataset and re-run"
    )
work = f"{W}/synth"
shutil.copytree(src, work, dirs_exist_ok=True)
for name in CONFIGS:
    cfg = json.loads(Path(f"{work}/{name}.json").read_text())
    cfg["crowd"]["human_lib"] = f"{src}/humans_packed.blend"
    Path(f"{work}/{name}.json").write_text(json.dumps(cfg, indent=1))

sh(
    "apt-get -qq install -y libxi6 libxkbcommon0 libxxf86vm1 libxfixes3 libxrender1 libgl1 libsm6 > /dev/null 2>&1 || true"
)
sh(f"cd /tmp && wget -q {BL_URL} && tar xf blender-4.2.1-linux-x64.tar.xz")
sh("pip -q install opencv-python-headless scipy > /dev/null 2>&1 || true")
B = "/tmp/blender-4.2.1-linux-x64/blender"

# one shell script per GPU; GPU g renders frames i with i % 2 == g of every config, one Blender run per
# frame (so the time guard can stop between frames)
procs = []
for gpu in (0, 1):
    lines = ["set -u"]
    for name in CONFIGS:
        seed0 = 1000 * (ALL.index(name) + 1) + {"a": 0, "b": 50000, "c": 90000}[PLAN]
        out = f"{W}/batch_{PLAN}/{name}/gpu{gpu}"  # own folder: frame numbers come from file counts
        lines.append(f"mkdir -p {out}")
        lines.append(
            f"for i in $(seq {gpu} 2 {FRAMES - 1}); do "
            f"if [ $(( $(date +%s) - {int(T0)} )) -gt {int(MAX_HOURS * 3600)} ]; then echo TIME_GUARD; break; fi; "
            f"{B} -b -P {work}/scene.py -- --config {work}/{name}.json --out {out} --seed $(( {seed0} + i )) "
            f"--clip 4 --samples 32 --res 1920 --backend CUDA >> {W}/log_{name}.txt 2>&1; done"
        )
    script = f"{W}/worker{gpu}.sh"
    Path(script).write_text("\n".join(lines) + "\n")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    procs.append(subprocess.Popen(["bash", script], env=env))
for p in procs:
    p.wait()

# post (look + lens + codec), boxes, then drop the clip neighbours and compress masks
summary = {"plan": PLAN, "configs": {}, "hours": None}
for name, gpu in [(n, g) for n in CONFIGS for g in (0, 1)]:
    out = f"{W}/batch_{PLAN}/{name}/gpu{gpu}"
    sh(
        f"cd {work} && python post.py {out} --seed {CONFIGS.index(name)} && python export_yolo.py {out}"
    )
    for f in glob.glob(f"{out}/clip_*.png"):
        os.remove(f)
    for f in glob.glob(f"{out}/*seg_*.npy"):
        np.savez_compressed(f[:-4] + ".npz", m=np.load(f))
        os.remove(f)
    summary["configs"][f"{name}/gpu{gpu}"] = len(glob.glob(f"{out}/labels_*.json"))
summary["hours"] = round((time.time() - T0) / 3600, 2)
Path(f"{W}/batch_{PLAN}/summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
shutil.rmtree(work, ignore_errors=True)
