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

### Cell 1 — setup (GPU + repo + deps)
```python
import torch; print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE — set Runtime>GPU")
!git clone https://github.com/DistancedTurtle/Hydro-Knight.git
%cd Hydro-Knight
# Colab ships a GPU build of torch; install the rest explicitly so it isn't clobbered.
!pip -q install ultralytics trackers supervision opencv-python pandas pyarrow scikit-learn
# editable-install just the local package (no deps -> leaves Colab's torch intact),
# which makes `hydro_knight` importable without any sys.path hack.
!pip -q install -e . --no-deps
```

### Cell 2 — mount videos from Drive
```python
from google.colab import drive; drive.mount("/content/drive")
!mkdir -p raw_local
!unzip -q -o /content/drive/MyDrive/hydroknight/videos.zip -d raw_local
from pathlib import Path
RAW = Path("raw_local")
print("videos found:", len(list(RAW.glob("*.mp4"))))
```

### Cell 3 — extract poses (the slow GPU step)
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
> guard skips evicted/0-byte clips so the loop never stalls on them. Save
> `data/keypoints/` back to Drive so you don't re-extract.

### Cell 4 — build the labeled, windowed dataset
```python
import cv2, numpy as np, pandas as pd
from hydro_knight.features.windows import make_windows

WINDOW = 32
def clip_fps(cid):
    c = cv2.VideoCapture(str(RAW / f"{cid}.mp4")); f = c.get(cv2.CAP_PROP_FPS); c.release()
    return f or 25.0

normal_win, distress_win = [], []
for r in clips:
    pq = KP / f"{r.clip_id}.parquet"
    if not pq.exists():
        continue
    fps = clip_fps(r.clip_id)
    events = [(e["start"], e["end"]) for e in r.events]          # source seconds
    W, info = make_windows(pd.read_parquet(pq), window=WINDOW, stride=8)
    for win, (tid, start_frame) in zip(W, info):
        t0, t1 = start_frame / fps, (start_frame + WINDOW) / fps  # window's time span
        is_distress = any(s < t1 and e > t0 for s, e in events)   # overlaps an event?
        (distress_win if is_distress else normal_win).append(win)

normal_win = np.array(normal_win, np.float32)
distress_win = np.array(distress_win, np.float32)
print("normal windows:", len(normal_win), "| distress windows:", len(distress_win))
```

### Cell 5 — train (Rung 3 TCN)
```python
from hydro_knight.models.tcn_autoencoder import train_tcn, reconstruction_error

# hold out 20% of NORMAL as the negative test set; train on the rest
idx = np.random.RandomState(0).permutation(len(normal_win)); split = int(0.8 * len(normal_win))
train_n, test_n = normal_win[idx[:split]], normal_win[idx[split:]]

model, scaler = train_tcn(train_n, epochs=300)   # more epochs/data than the local demo
print("trained on", len(train_n), "normal windows")
```

### Cell 6 — evaluate (does error separate normal vs distress?)
```python
from sklearn.metrics import roc_auc_score, average_precision_score

err_normal = reconstruction_error(model, test_n, scaler)
err_distress = reconstruction_error(model, distress_win, scaler)

scores = np.concatenate([err_normal, err_distress])
labels = np.concatenate([np.zeros(len(err_normal)), np.ones(len(err_distress))])
print(f"normal mean err   : {err_normal.mean():.3f}")
print(f"distress mean err : {err_distress.mean():.3f}")
print(f"ROC-AUC           : {roc_auc_score(labels, scores):.3f}")   # 0.5 = chance, 1.0 = perfect
print(f"PR-AUC            : {average_precision_score(labels, scores):.3f}")
```
> ROC-AUC is the headline number: can the reconstruction error tell distress
> windows from normal ones? Recall-favoring threshold selection (drowning >
> false alarm) is the next step from the PR curve.

### Cell 7 — save the model back to Drive
```python
import torch
torch.save({"model": model.state_dict(), "scaler": scaler},
           "/content/drive/MyDrive/hydroknight/tcn_ae.pt")
```

---

## 3. Honest expectations & next steps

- **First number won't be SOTA.** Train/eval drawn partly from the same clips
  risks clip-specific cues; strengthen by training normal on reef/AMI and the 2
  normal clips, evaluating distress only on rescue events.
- **Tracking churn** still affects window quality — fewer, cleaner long tracks help.
- If TCN underperforms, **STG-NF** is the SOTA upgrade (lightweight, pose-specific).
- Rung 4 (per-swimmer resurface state machine) consumes these scores + the track ids.
