# Hydro-Knight

A pose-based **anomaly-detection** system for swimming-pool safety. It watches
pool footage, tracks individual swimmers, and flags potential drowning events —
not by classifying drowning types, but by learning what *normal* swimming looks
like and flagging deviations (the right framing when real positives are rare,
and it biases toward recall: a missed drowning costs more than a false alarm).

Designed with a former lifeguard's domain knowledge baked in: real drownings are
usually **silent** — no flailing, no shouting; someone slips under and doesn't
come back up. So the system's primary signal isn't dramatic motion, it's
*absence*: a tracked swimmer whose keypoints vanish and don't return.

```
 video (manifest-driven, never committed)
   │
   ▼
 YOLO-pose @ ≥1280px  ──  SAHI tiling for distant swimmers
   │
   ▼
 ByteTrack  →  per-swimmer keypoint time-series (Parquet)
   │
   ├─▶ normalized pose windows → temporal autoencoder → anomaly score   (flailing)
   └─▶ per-track state machine over confidence + position  [next build] (silent sink, bobbing)
```

See [PLAN.md](PLAN.md) for the full architecture, decisions of record, and
build progression.

---

## What drowning actually looks like (the five signatures)

| Signature | What the camera sees | Detection lane |
|---|---|---|
| Silent sink | Track vanishes mid-pool, never resurfaces | State-machine timer |
| Bobbing in place | Repeated submerge/resurface cycles, zero net displacement | State-machine rules |
| Passive face-down | Motionless prone pose beyond normal duration | Pose + duration logic |
| Aggressive flailing | Erratic, high-magnitude limb motion | Autoencoder reconstruction error |
| Effort without progress | Swimming hard, going nowhere (e.g. against a current) | Displacement features |

The split matters: only *flailing* is something a reconstruction-error model
natively catches. The rest are explicit temporal/spatial rules over the track
data — a lesson the first training run taught empirically (below).

---

## Engineering findings (each one tested, not assumed)

### 1. Pose backend: YOLO beats MediaPipe — but resolution was the real lever

The entire architecture rests on one assumption: that an off-the-shelf pose
estimator can find swimmers in real pool footage. Tested directly — YOLO-pose
(Ultralytics) and MediaPipe on identical frames from real clips.

> Figures are **deliberately pixelated** — backgrounds blurred to
> unrecognizability, only extracted skeletons drawn crisply, consistent with
> the project's no-identifiable-footage policy.

On crowded scenes MediaPipe found 0–1 swimmers per frame (it's effectively
single-person); YOLO found many:

![YOLO vs MediaPipe](docs/images/pose_model.png)

The bigger surprise: YOLO's default 640px input shrinks distant swimmers below
detectability. The same frame at 1280px recovered 3–6× more swimmers — a
near-free config change that mattered more than model choice:

![YOLO 640 vs 1280](docs/images/pose_resolution.png)

| Scene | YOLO @640 | YOLO @1280 | MediaPipe |
|---|---|---|---|
| Crowded wavepool | 2–4 | **7–13** | 0–1 |
| Sparse resort pool | 0–1 | 0–1 | 0 |

Reproduce: `uv run python scripts/make_pose_figure.py` (spike harness:
[`pose_spike.py`](src/hydro_knight/preprocess/pose_spike.py)).

### 2. Distant swimmers: SAHI tiling + decoupled ByteTrack

Whole-frame inference downscales far swimmers below detectability. **SAHI** —
slice into overlapping tiles, run YOLO on each *upscaled* tile, map back,
NMS-merge — took recall from ~13 to ~50 swimmers/frame on a crowded clip.

YOLO's built-in tracker can't consume merged tiled detections, so tracking is
decoupled: a naive IoU tracker churned badly (53 tracks, 11 single-frame);
**ByteTrack** cut that to 15 stable tracks with **0 flicker** — the identity
stability the not-surfacing signal depends on.

### 3. The first training run failed at exactly chance — and the diagnosis reshaped the project

First real run (TCN autoencoder over pose windows, 66 labeled distress events):
**ROC-AUC 0.539** — a coin flip. The reconstruction-error distributions for
normal vs distress windows were *identical through the 90th percentile*:

| percentile | normal | distress |
|---|---|---|
| p50 | 0.105 | 0.111 |
| p90 | 0.322 | 0.323 |
| p99 | 1.414 | 1.680 |

Digging into *why* (in code, not by guessing) surfaced a representation
problem, not a tuning problem:

- **The submersion never enters the dataset.** When a swimmer goes under,
  keypoint confidence collapses; normalization drops those low-confidence
  frames and windowing stitches over the gap. The model is asked to find
  drownings in data from which the drowning itself has been filtered out.
- **Face-down floating is anti-signal** — a motionless pose is trivially easy
  to reconstruct, so it scores as *extra normal*.
- **Hip-centered normalization discards displacement** — bobbing-in-place and
  effort-without-progress are invisible when every pose is re-centered on the
  swimmer's own hips.

The autoencoder's honest lane is flailing. Everything else needs the track
data the features were throwing away — hence the next build: a **per-track
state machine** (track loss mid-pool → timer → alert, with edge-of-frame
suppression), evaluated by per-event recall and *latency* against the 66
labeled events. No retraining required to detect the most lethal case.

---

## Setup

Managed with [uv](https://docs.astral.sh/uv/). From the repo root:

```bash
uv sync                                  # build the env from pyproject.toml + uv.lock
uv run python -c "import hydro_knight"   # editable install — no PYTHONPATH needed
uv run python -m pytest                  # run the unit tests
```

`uv sync --extra spike` additionally installs MediaPipe (only needed to
reproduce the backend comparison). Heavy pose extraction and training run on
Colab GPU — recipe in [docs/COLAB_TRAINING.md](docs/COLAB_TRAINING.md).

## Data & ethics

The repo commits **manifests** (source URLs + metadata + typed event
annotations, as diff-friendly JSONL), never video: public visibility isn't a
redistribution license, and pool footage shows identifiable people — often
minors. Labels are **outcome-based** (guards intervened → positive), avoiding
ambiguous mid-event judgment calls. Full rationale in
[docs/DATA_SOURCING.md](docs/DATA_SOURCING.md).

## License

[AGPL-3.0](LICENSE). The pose backend ([Ultralytics YOLO](https://github.com/ultralytics/ultralytics))
is AGPL-3.0, and a public work that builds on AGPL code inherits that license.
Non-commercial learning/portfolio project; if the AGPL dependency were ever
swapped out, the license could be revisited.
