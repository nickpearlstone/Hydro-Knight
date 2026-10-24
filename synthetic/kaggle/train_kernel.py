"""Kaggle GPU job: fine-tune the head detectors on the synthetic tiles (private dataset hk-synth-tiles).

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
if "--save-every" not in Path(f"{code}/train_rtdetr.py").read_text():
    raise SystemExit(f"stale train_rtdetr.py in {code}: re-version hk-synth-tiles")
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
# v2 = RT-DETR with scale jitter, warm-up and EMA for 100 epochs, frozen vs 0.1x backbone side by side;
# v3 (10-23) = seed spread of v2 frozen: the same 100-epoch schedule cut at 30 epochs, weights kept every
# 5 epochs, seeds 1-8, four after another per GPU (~21 min each). Run with ~2 h of GPU quota left, so a
# seed only starts if it can finish before START_BY, and `timeout` ends a slow one at HARD_STOP.
START_BY, HARD_STOP = 85 * 60, 110 * 60
RUNS = []
for gpu in (0, 1):
    lines = []
    for seed in range(1 + gpu, 9, 2):
        lines.append(
            f"el=$(( $(date +%s) - {int(T0)} )); "
            f'if [ $el -gt {START_BY} ]; then echo "TIME_GUARD before seed {seed}"; exit; fi; '
            f"timeout $(( {HARD_STOP} - el )) python {code}/train_rtdetr.py {tiles} --epochs 100 --stop-after 30 "
            f"--save-every 5 --batch 8 --seed {seed} --out {W}/train/rtdetr_v3_seed{seed} "
            f"> {W}/log_rtdetr_seed{seed}.txt 2>&1"
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
