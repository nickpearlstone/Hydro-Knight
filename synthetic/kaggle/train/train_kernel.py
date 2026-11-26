"""Kaggle GPU job: fine-tune the head detectors on the synthetic tiles (private dataset hk-synth-tiles).

RUNS lists the jobs (one per GPU); weights + training curves land in /kaggle/working/train.
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
    "pip -q install ultralytics 'transformers>=4.50' > /dev/null 2>&1",
    shell=True,
    check=False,
)
# CAUTION[train-coco-prefetch]: fetch COCO weights once before the parallel YOLO workers start
subprocess.run(
    f"cd {W} && python -c \"from ultralytics import YOLO; YOLO('yolo11n.pt')\"",
    shell=True,
    check=False,
)


def run(cmd, gpu, log):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    return subprocess.Popen(f"{cmd} > {W}/{log} 2>&1", shell=True, env=env)


# CAUTION[train-start-by-hard-stop]: a seed starts only before START_BY; timeout ends it at HARD_STOP (Kaggle limit 12 h)
START_BY, HARD_STOP = 9 * 3600, 11 * 3600
JOB, LOG = (
    "python {code}/train_rtdetr.py {tiles} --epochs 100 --stop-after 30 --save-every 5 --batch 8 "
    "--seed {seed} --out {W}/train/rtdetr_v5_cd_seed{seed}",
    "log_rtdetr_cd_seed{seed}.txt",
)
RUNS = []
for gpu in (0, 1):
    lines = []
    for seed in range(1 + gpu, 9, 2):
        job = JOB.format(code=code, tiles=tiles, W=W, seed=seed)
        lines.append(
            f"el=$(( $(date +%s) - {int(T0)} )); "
            f'if [ $el -gt {START_BY} ]; then echo "TIME_GUARD before seed {seed}"; exit; fi; '
            f"timeout $(( {HARD_STOP} - el )) {job} > {W}/{LOG.format(seed=seed)} 2>&1"
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
