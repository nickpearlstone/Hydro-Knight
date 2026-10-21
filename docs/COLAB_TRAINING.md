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
command**. The old manual train/eval cells are gone — `hydroknight legacy eval-report`
does windowing, training/loading, metrics, and figures in one call.

### Cell 1 — setup (repo + deps + Drive), idempotent
```python
!git clone -q https://github.com/nickpearlstone/Hydro-Knight.git 2>/dev/null || (cd Hydro-Knight && git pull -q)
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
# keypoints: skips extraction entirely if you built them last session
!mkdir -p data && cp -r {DRIVE}/keypoints_tiled data/keypoints 2>/dev/null || echo "no saved keypoints -> run Cell 3"
# videos: needed for extraction
!mkdir -p raw_local
!unzip -q -n {DRIVE}/videos.zip -d raw_local
# Colab's OpenCV cannot decode AV1. The 10 AV1 clips were re-encoded to H.264 with
# hydroknight transcode-av1 (frame-exact) and uploaded to videos_h264/; they replace the zip's copies.
!cp -f {DRIVE}/videos_h264/*.mp4 raw_local/
!for f in raw_local/*.mp4; do [ "$(ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of csv=p=0 "$f")" = av1 ] && echo "STILL AV1: $f"; done
RAW = Path("raw_local"); KP = Path("data/keypoints")
print("videos:", len(list(RAW.glob("*.mp4"))), "| keypoint parquets:", len(list(KP.glob("*.parquet"))))
```

### Cell 3 — extract raw detections (SKIP if Cell 2 restored keypoints)
```python
import torch
assert torch.cuda.is_available(), "No GPU! Runtime > Change runtime type > T4 GPU"
print("GPU:", torch.cuda.get_device_name(0))
# Writes straight to Drive in 1000-frame chunks. After a disconnect, rerun this
# same cell: finished clips are skipped and the interrupted one resumes.
!hydroknight extract --device 0 --out {DRIVE}/detections
```
> This is the only GPU step. It runs YOLO11n-pose over 480px tiles at imgsz 1280
> plus a whole-frame pass, and saves every raw detection before any merging or
> tracking (see `preprocess/extract_pose.py` for the folder format). Labeled clips
> run first; unlabeled reef footage runs last. Every clip is extracted in full:
> many distress events start more than 25 s in.

### Cell 3b — merge + track into keypoint files (CPU, minutes)
```python
!hydroknight tracks --detections {DRIVE}/detections --out data/keypoints --overwrite
!cp -r data/keypoints {DRIVE}/keypoints_tiled   # so Cell 2 can restore them next time
```
> Rerun this cell any time the merge or tracker settings change; no GPU or video
> needed. `--legacy` reproduces the original merge + tracker for comparison.

### Cell 4 — the report (pick ONE mode)

**Mode A — saved weights** (score the existing checkpoint, e.g. the 0.539 baseline):
```python
!hydroknight legacy eval-report \
    --keypoints data/keypoints --manifest data/manifests/pool_footage.jsonl \
    --videos raw_local --ckpt {DRIVE}/tcn_ae.pt --out runs/tcn_baseline
```

**Mode B — fresh train** (new model; holds out 20% of normal windows; saves weights):
```python
!hydroknight legacy eval-report \
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
- Each clip's **true fps** comes from the manifest (`hydroknight backfill-fps`);
  event windows are in seconds, so latency/overlap need it (a 60 fps clip at an
  assumed 30 would double every latency).
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

---

## 3. One-cell extraction run (low confidence floor, all active clips)

Re-extracts every active clip (no `[HOLD` in its notes) with the low-floor settings:
detections down to confidence 0.05, at most 30 frames per second, stopping 5 s after
guard contact (clips with no contact yet run to the end). A 6-clip pilot showed this
raises the share of victim clicks covered by a 2 s track from 26% to about 60%
(details in PR #30). Expect about 10 hours of T4 time for 66 clips, usually over 2 or 3
sessions.

Needs the low-floor extraction code on `main` (PR #30) before it runs.

**On the Mac:** build the video zip (gitignored folder) and upload it to Google Drive as
`MyDrive/hydroknight/videos_active.zip`:

```bash
uv run hydroknight zip-videos
```

**In Colab:** Runtime > Change runtime type > **T4 GPU**, paste this one cell, run it. It
is safe to rerun after a disconnect: finished clips are skipped and an interrupted one
resumes from its last 1000-frame chunk (progress is saved to Drive as it goes). Keep the
tab open; an idle tab is disconnected after about 90 minutes.

```python
RUN = "c05"                            # output folder prefix in Drive
import os, torch
assert torch.cuda.is_available(), "No GPU: Runtime > Change runtime type > T4 GPU"
from google.colab import drive; drive.mount("/content/drive")
D = "/content/drive/MyDrive/hydroknight"
os.chdir("/content")
if not os.path.exists("Hydro-Knight"):
    !git clone -q https://github.com/nickpearlstone/Hydro-Knight.git
os.chdir("/content/Hydro-Knight"); !git pull -q
!pip -q install ultralytics trackers supervision opencv-python pandas pyarrow scikit-learn tqdm
!pip -q install -e . --no-deps
!mkdir -p raw_local && unzip -q -n {D}/videos_active.zip -d raw_local
!hydroknight extract --device 0 --conf 0.05 --target-fps 30 --stop-after-contact 5 --out {D}/{RUN}_detections
!hydroknight tracks --detections {D}/{RUN}_detections --out {D}/{RUN}_keypoints --same-crop-conf 0.25 --activation 0.25 --high-conf 0.25 --lost-track-seconds 2
!cd {D} && zip -q -r -0 {RUN}_results.zip {RUN}_detections {RUN}_keypoints
print("DONE: download MyDrive/hydroknight/" + RUN + "_results.zip")
```

**Back on the Mac:** download `MyDrive/hydroknight/c05_results.zip` into the repo's
`scratch/colab_upload/` folder and unpack it into `data/detections_c05/` and
`data/keypoints_c05/` (both gitignored). The current `data/detections/` and
`data/keypoints_tiled/` stay as they are for comparison. Tracker settings can be rerun
on the Mac in minutes with `hydroknight tracks`; only the detections need the GPU.
