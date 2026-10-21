"""Kaggle GPU job: fine-tune the head detectors on the synthetic tiles (private dataset hk-synth-tiles).

RUNS lists the jobs (one per GPU); see the comment there for what each version trained. Weights + training curves land in /kaggle/working/train.
Scoring on real frames happens on the Mac afterwards (synthetic/train/score_heads.py); no real
footage is uploaded.
"""

import glob
import os
import subprocess
from pathlib import Path

W = "/kaggle/working"
src = os.path.dirname(glob.glob("/kaggle/input/**/data.yaml", recursive=True)[0])
code = os.path.dirname(glob.glob("/kaggle/input/**/train_rtdetr.py", recursive=True)[0])
tiles = f"{W}/tiles"
subprocess.run(f"cp -r {src} {tiles}", shell=True, check=True)
Path(f"{tiles}/data.yaml").write_text(
    f"path: {tiles}\ntrain: images/train\nval: images/val\nnames:\n  0: head\n"
)
subprocess.run(
    "pip -q install ultralytics 'transformers>=4.50' > /dev/null 2>&1",
    shell=True,
    check=False,
)


def run(cmd, gpu, log):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    return subprocess.Popen(f"{cmd} > {W}/{log} 2>&1", shell=True, env=env)


# RUNS: which jobs to start, one per GPU. v1 (10-20) = RT-DETR frozen then 0.1x on GPU 0 + YOLO on GPU 1;
# v2 = RT-DETR with scale jitter, warm-up and EMA for 100 epochs, frozen vs 0.1x backbone side by side.
RUNS = [
    (
        f"python {code}/train_rtdetr.py {tiles} --epochs 100 --batch 8 --out {W}/train/rtdetr_v2_frozen",
        0,
        "log_rtdetr_frozen.txt",
    ),
    (
        f"python {code}/train_rtdetr.py {tiles} --epochs 100 --batch 8 --backbone-lr-scale 0.1 --out {W}/train/rtdetr_v2_full",
        1,
        "log_rtdetr_full.txt",
    ),
]
procs = [run(cmd, gpu, log) for cmd, gpu, log in RUNS]
for p in procs:
    p.wait()
subprocess.run(f"rm -rf {tiles}", shell=True, check=False)
for f in sorted(glob.glob(f"{W}/log_*.txt")):
    print("=====", f)
    print(Path(f).read_text()[-3000:])
