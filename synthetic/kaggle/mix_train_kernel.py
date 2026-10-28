"""Kaggle GPU job: fine-tune the head detectors on real tiles (SAM 3 labels, cut 0.40) + synthetic tiles (private dataset hk-tiles-mix).

RUNS lists the jobs (one per GPU); see the comment there for what each version trained. Weights + training curves land in /kaggle/working/train.
Scoring on real frames happens on the Mac afterwards (synthetic/train/score_heads.py); no real
footage is uploaded.
"""

import glob
import os
import subprocess
import time
from pathlib import Path

W = "/kaggle/working"
T0 = time.time()
src = os.path.dirname(glob.glob("/kaggle/input/**/data.yaml", recursive=True)[0])
code = os.path.dirname(glob.glob("/kaggle/input/**/train_rtdetr.py", recursive=True)[0])
for name, flag in [
    ("train_rtdetr.py", "--save-every"),
    ("train_yolo.py", "--save-period"),
]:
    if flag not in Path(f"{code}/{name}").read_text():
        raise SystemExit(f"stale {name} in {code}: re-version the tiles dataset")
tiles = f"{W}/tiles"
subprocess.run(f"cp -r {src} {tiles}", shell=True, check=True)
Path(f"{tiles}/data.yaml").write_text(
    f"path: {tiles}\ntrain: images/train\nval: images/val\nnames:\n  0: head\n"
)
subprocess.run(
    "pip -q install ultralytics 'transformers==5.21.0' > /dev/null 2>&1",
    shell=True,
    check=False,
)
# stop here if the pin didn't take: 5.16.1's RT-DETR loss leaves the last decoder layer untrained
subprocess.run(
    "python -c 'import transformers; assert transformers.__version__ == \"5.21.0\", transformers.__version__'",
    shell=True,
    check=True,
)


def run(cmd, gpu, log):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    return subprocess.Popen(f"{cmd} > {W}/{log} 2>&1", shell=True, env=env)


# v7 (10-27): v6's RT-DETR seeds again with transformers pinned to 5.21.0. Under 5.16.1 (what Kaggle
# installed for v1-v6) the last decoder layer's classifier never learns (top score ~0.06 vs ~0.99 on
# earlier layers), so every RT-DETR before v7 was crippled. YOLO is unaffected and not rerun.
# v6 (10-26): real + synthetic. Per GPU: two RT-DETR seeds (20-epoch schedule from COCO, weights every
# 2 epochs; ~7.5 min/epoch on 8.8k tiles), then one YOLO11n seed (30 epochs, weights every 5).
# Seed and checkpoint are picked on Nick's 12 validation frames on the Mac, never on the test frames.
START_BY, HARD_STOP = 9 * 3600, 11.5 * 3600
RTDETR = (
    "python {code}/train_rtdetr.py {tiles} --epochs 20 --save-every 2 --batch 8 "
    "--seed {seed} --out {W}/train/rtdetr_v7_mix_seed{seed}"
)
YOLO = (
    "python {code}/train_yolo.py {tiles}/data.yaml --epochs 30 --batch 16 --device 0 "
    "--save-period 5 --seed {seed} --project {W}/train --name yolo_v6_mix_seed{seed}"
)
RUNS = []
for gpu in (0, 1):
    jobs = [(RTDETR, 1 + gpu), (RTDETR, 3 + gpu)]
    lines = []
    for tmpl, seed in jobs:
        job = tmpl.format(code=code, tiles=tiles, W=W, seed=seed)
        log = f"{W}/log_{tmpl.split('/')[1].split('.')[0]}_seed{seed}.txt"
        lines.append(
            f"el=$(( $(date +%s) - {int(T0)} )); "
            f'if [ $el -gt {START_BY} ]; then echo "TIME_GUARD before {log}"; exit; fi; '
            f"timeout $(( {int(HARD_STOP)} - el )) {job} > {log} 2>&1"
        )
    script = f"{W}/worker{gpu}.sh"
    Path(script).write_text("\n".join(lines) + "\n")
    RUNS.append((f"bash {script}", gpu, f"log_worker{gpu}.txt"))
procs = [run(cmd, gpu, log) for cmd, gpu, log in RUNS]
for p in procs:
    p.wait()
subprocess.run(f"rm -rf {tiles}", shell=True, check=False)
for f in sorted(glob.glob(f"{W}/log_*.txt")):
    print("=====", f)
    print(Path(f).read_text()[-3000:])
