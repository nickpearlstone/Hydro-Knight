# Training in Colab — dataset + notebook recipe

GPU-side recipe: extract poses (slow, needs GPU), then train the Rung 3 TCN
autoencoder and evaluate against the labeled distress clips. The repo clone
brings the **code + manifest**; the **videos** come from your Google Drive
(they're gitignored, never in the repo).

---

## 1. The dataset (what's on disk → training roles)

| Source | Clips | Role | Notes |
|---|---|---|---|
| Wavepool rescue clips | 66 (all have event windows) | **Distress eval** = frames *inside* event windows; **Normal train** = frames *outside* (the lead-ins) | Pose-rich; the reusability design — one clip serves both |
| Swim relay + AMI Pool Cam | 2 | **Normal train** | Adds competitive + recreational variety |
| Jupiter Reef Club | 4 chunks (~9.5h) | **Normal train** (optional) | Sparse + huge; sample frames or skip for a first run |

**Split logic:**
- **Train** the autoencoders on **normal** windows only (rescue lead-ins + the 2 normal clips [+ reef]).
- **Evaluate** by scoring **held-out normal** vs **distress** (event-window) windows — does reconstruction error separate them?

**Transfer to Colab:** zip `raw_local/*.mp4` + commit the manifest (already in repo),
upload the zip to Google Drive (e.g. `MyDrive/hydroknight/videos.zip`, ~17 GB).
For a faster first run, upload just the **66 rescue clips** (~few GB) — they alone
give both normal and distress.

---

## 2. Notebook cells

The flow is: **setup → restore data → (extract only if needed) → one report
command**. The old manual train/eval cells are gone — `scripts/eval_report.py`
does windowing, training/loading, metrics, and figures in one call.

### Cell 1 — setup (repo + deps + Drive), idempotent
```python
!git clone -q https://github.com/DistancedTurtle/Hydro-Knight.git 2>/dev/null || (cd Hydro-Knight && git pull -q)
%cd Hydro-Knight
# Colab ships a GPU build of torch; install the rest explicitly so it isn't clobbered.
!pip -q install ultralytics trackers supervision opencv-python pandas pyarrow scikit-learn matplotlib
# editable-install just the local package (no deps -> leaves Colab's torch intact)
!pip -q install -e . --no-deps
from google.colab import drive; drive.mount("/content/drive")
DRIVE = "/content/drive/MyDrive/hydroknight"
```

### Cell 2 — restore data from Drive
```python
from pathlib import Path
# keypoints: skips re-extraction entirely if you saved them last session
!mkdir -p data && cp -r {DRIVE}/keypoints data/ 2>/dev/null || echo "no saved keypoints -> run Cell 3"
# videos: needed for extraction AND for true per-clip fps in the report
!mkdir -p raw_local
!unzip -q -n {DRIVE}/videos.zip -d raw_local
RAW = Path("raw_local"); KP = Path("data/keypoints")
print("videos:", len(list(RAW.glob("*.mp4"))), "| keypoint parquets:", len(list(KP.glob("*.parquet"))))
```

### Cell 3 — extract poses (SKIP if Cell 2 restored keypoints)
```python
import warnings; warnings.simplefilter("ignore")
import cv2
from hydro_knight.ingest.manifest import Manifest, Label
from hydro_knight.preprocess.extract_pose import extract, extract_tiled

def _readable(p):                      # skip missing / 0-byte / unreadable clips
    c = cv2.VideoCapture(str(p)); ok = c.isOpened() and c.get(cv2.CAP_PROP_FRAME_COUNT) > 0; c.release()
    return ok

recs = Manifest(Path("data/manifests/pool_footage.jsonl")).load()
clips = [r for r in recs
         if (RAW / f"{r.clip_id}.mp4").exists()
         and "[HOLD" not in r.notes               # skip held (indoor/out-of-scope) clips
         and _readable(RAW / f"{r.clip_id}.mp4")]  # skip dead/0-byte (e.g. evicted reef) clips
print(f"{len(clips)} clips to extract")

import torch
assert torch.cuda.is_available(), "No GPU! Runtime > Change runtime type > T4 GPU"
print("GPU:", torch.cuda.get_device_name(0))

KP = Path("data/keypoints"); KP.mkdir(parents=True, exist_ok=True)
for i, r in enumerate(clips, 1):
    out = KP / f"{r.clip_id}.parquet"
    if out.exists():
        continue
    # imgsz=640 is the speedup (~4x vs 1280). Do NOT cap frames: many distress
    # events occur >25s in, so each clip must be extracted in full to include
    # them. (Raise to imgsz=1280 / swap to extract_tiled for the recall upgrade.)
    extract(RAW / f"{r.clip_id}.mp4", out, imgsz=640, device=0)
    print(f"[{i}/{len(clips)}] {r.clip_id} done")
```
> `extract()` is the fast whole-frame path (~1 inference/frame); `extract_tiled()`
> is SAHI (~7×, max recall) — use it only for a later recall run. The `_readable`
> guard skips evicted/0-byte clips so the loop never stalls on them. **Save the
> keypoints to Drive right after** so future sessions skip this cell:
> ```python
> !cp -r data/keypoints {DRIVE}/
> ```

### Cell 4 — the report (pick ONE mode)

**Mode A — saved weights** (score the existing checkpoint, e.g. the 0.539 baseline):
```python
!python scripts/eval_report.py \
    --keypoints data/keypoints --manifest data/manifests/pool_footage.jsonl \
    --videos raw_local --ckpt {DRIVE}/tcn_ae.pt --out runs/tcn_baseline
```

**Mode B — fresh train** (new model; holds out 20% of normal windows; saves weights):
```python
!python scripts/eval_report.py \
    --keypoints data/keypoints --manifest data/manifests/pool_footage.jsonl \
    --videos raw_local --train-fresh --epochs 300 \
    --save-ckpt {DRIVE}/tcn_fresh.pt --out runs/tcn_fresh
```

What one run produces in `runs/<name>/`: `summary.md` (headline numbers:
**per-event recall**, ROC-AUC/PR-AUC, error percentiles), the figures
(loss curve, normal-vs-distress error overlap, ROC/PR, **recall vs
false-alarms-per-hour**, latency, per-event catch/miss board, pose coverage
inside events, track lengths), plus `sweep.parquet` / `event_catches.parquet`.

Notes that keep the numbers honest:
- `--videos raw_local` reads each clip's **true fps** — event windows are in
  seconds, so latency/overlap need it (a 60 fps clip at an assumed 30 would
  double every latency).
- In Mode B, window-level ROC/PR/percentiles use **only held-out** normals —
  never windows the model trained on. Per-event recall uses everything.
- Mode A's window metrics have no holdout information (the old checkpoint's
  training split isn't recorded), so its ROC skews slightly optimistic — its
  per-event recall/latency are the numbers to trust most.

### Cell 5 — view results inline + save the run to Drive
```python
from IPython.display import Image, Markdown, display
from pathlib import Path
RUN = Path("runs/tcn_baseline")            # or runs/tcn_fresh
display(Markdown((RUN / "summary.md").read_text()))
for png in sorted(RUN.glob("*.png")):
    display(Image(str(png)))
!cp -r {RUN} {DRIVE}/{RUN.name}_$(date +%Y%m%d)   # archive the run to Drive
```

---

## 3. Honest expectations & next steps

- **First number won't be SOTA.** Train/eval drawn partly from the same clips
  risks clip-specific cues; strengthen by training normal on reef/AMI and the 2
  normal clips, evaluating distress only on rescue events.
- **Tracking churn** still affects window quality — fewer, cleaner long tracks help.
- If TCN underperforms, **STG-NF** is the SOTA upgrade (lightweight, pose-specific).
- Rung 4 (per-swimmer resurface state machine) consumes these scores + the track ids.
