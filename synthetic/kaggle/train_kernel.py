"""Kaggle GPU job: fine-tune the head detectors on the synthetic tiles (private dataset hk-synth-tiles).

GPU 0: RT-DETRv2-R18, backbone frozen, then a second run with the backbone at 0.1x learning rate.
GPU 1: YOLO11n baseline (backbone frozen). Weights + training curves land in /kaggle/working/train.
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


rt = run(
    f"python {code}/train_rtdetr.py {tiles} --epochs 40 --batch 8 --out {W}/train/rtdetr_frozen && "
    f"python {code}/train_rtdetr.py {tiles} --epochs 40 --batch 8 --backbone-lr-scale 0.1 --out {W}/train/rtdetr_full",
    0,
    "log_rtdetr.txt",
)
yo = run(
    f"python {code}/train_yolo.py {tiles}/data.yaml --epochs 60 --batch 16 --device 0 --project {W}/train",
    1,
    "log_yolo.txt",
)
rt.wait()
yo.wait()
subprocess.run(f"rm -rf {tiles}", shell=True, check=False)
for f in sorted(glob.glob(f"{W}/log_*.txt")):
    print("=====", f)
    print(Path(f).read_text()[-3000:])
