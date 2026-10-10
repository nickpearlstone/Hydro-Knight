# Engineering findings

Detailed writeups of the experiments summarized in the [README](../README.md), all run
on real pool footage.

> Figures are **deliberately pixelated** so no one can be recognized. Only the
> extracted skeletons are drawn clearly.

## 1. Pose model and resolution

YOLO-pose (Ultralytics) and MediaPipe were run on the same frames. MediaPipe is built
for one person at a time and found 0 to 1 swimmers per crowded frame. YOLO found many:

![YOLO vs MediaPipe](images/pose_model.png)

At its default 640px input, YOLO misses distant swimmers. Running at 1280px found 3 to
6 times more, a bigger effect than the choice of model:

![YOLO 640 vs 1280](images/pose_resolution.png)

| Scene | YOLO @640 | YOLO @1280 | MediaPipe |
|---|---|---|---|
| Crowded wave pool | 2–4 | **7–13** | 0–1 |
| Sparse resort pool | 0–1 | 0–1 | 0 |

**Decision:** YOLO-pose at 1280px or higher. Confirmed against hand-counted swimmers
in [section 4](#4-pose-model-benchmark).

Reproduce: `uv sync --extra spike`, then `uv run python scripts/make_pose_figure.py`.

## 2. Distant swimmers: tiling and tracking

SAHI tiling splits each frame into overlapping tiles, runs YOLO on each, and merges
the results. On a crowded clip it raised detections from about 13 to about 50 swimmers
per frame, at about 7 times the cost.

YOLO's built-in tracker can't use merged tile output, so ByteTrack runs as a separate
step:

| Tracker | Tracks on test clip | Single-frame tracks |
|---|---|---|
| Simple IoU matching | 53 | 11 |
| **ByteTrack** | **15** | **0** |

That test covered a short stretch of one clip. Over 22 full clips, the median track
lasts about 1 second, so keeping one ID through a submersion is still unsolved.

Code: [`tiled_pose.py`](../src/hydro_knight/preprocess/tiled_pose.py),
[`build_tracks.py`](../src/hydro_knight/preprocess/build_tracks.py).

## 3. First training run (ROC-AUC 0.539) and a broken evaluation

A TCN autoencoder was trained on normal 32-frame pose windows and scored by
reconstruction error. It reached ROC-AUC 0.539 and PR-AUC 0.522. Error distributions
for 3,821 normal and 3,872 distress windows were nearly identical:

| Percentile | Normal | Distress |
|---|---|---|
| p50 | 0.105 | 0.111 |
| p90 | 0.322 | 0.323 |
| p99 | 1.414 | 1.680 |

**The evaluation could not measure the model.** Rescues were labeled as time ranges,
and an event counted as caught if any swimmer was flagged during it. With about 21
swimmers in frame per rescue:

- A random-number detector scored ROC-AUC 0.497 and caught 23 of 23 events.
- A perfect detector of the drowning swimmer could reach only about 0.52.
- Post-rescue footage counted as normal, and held-out windows overlapped training ones.

The 0.539 (and a later 0.537 with velocity features) therefore says nothing about the
model.

**Known feature limits**, found in the code but not tested by this run:

- Poses with a low-confidence hip or shoulder are dropped
  ([`normalize.py`](../src/hydro_knight/features/normalize.py)), which removes the
  frames where a swimmer goes under.
- A still, face-down body is easy to reconstruct, so it scores as normal.
- Poses are centered on the hips, which hides movement through the water.
- The pose model itself scored a submerged victim at about 0.02 confidence.

**Changes:** evaluation is being rebuilt to score only the victim's track, from
drowning onset to lifeguard contact, with a random baseline in every report. Victim
positions and contact times are now labeled, and clip trims are stored separately
from the download window. The autoencoder is limited to flailing; the other
signatures move to per-track rules.

## 4. Pose model benchmark

Every person on 10 frames was clicked by hand (8 rescues at drowning onset, 2 wide
shots of a resort pool): 736 in the water, 26 out of it. Each model ran on the whole
frame at 1280px and with the production tiling from section 2.

A box counts as finding a swimmer if the swimmer's click is inside it. Extra boxes on
the same person are duplicates; boxes with no click are false positives (mostly posts,
railings, and the lane rope). Speed is the median per frame on an 8 GB M1 GPU.

**At the production confidence cutoff (0.25):**

| Model | Found, tiled | Duplicates | False pos | s/frame, tiled | Found, whole frame |
|---|---|---|---|---|---|
| **YOLO11n** | **219 (30%)** | 6 | 30 | **1.1** | 26 (4%) |
| YOLO11s | 202 (27%) | 14 | 30 | 2.1 | 21 |
| YOLO11m | 206 (28%) | 18 | 32 | 5.3 | 28 |
| YOLO26n | 125 (17%) | 5 | 20 | 1.1 | 18 |
| YOLO26s | 105 (14%) | 7 | 13 | 2.2 | 17 |
| YOLO26m | 75 (10%) | 7 | 17 | 6.1 | 15 |
| MediaPipe | 44 (6%) | 26 | 42 | 0.5 | 8 |

**At equal false positives.** YOLO26 assigns lower confidence to the same people, so
each model was also scored at the cutoff giving the same false-positive count. YOLO26
was tested with and without its optional NMS step.

| False positives | YOLO11n | YOLO26n | YOLO26n + NMS |
|---|---|---|---|
| ~20 | **146** found | 125 | 113 |
| ~30 | **203** | 188 | 159 |
| ~50 | **293** | 237 | 220 |

**Takeaways**

- Tiling is required: whole-frame inference found under 5% of swimmers.
- YOLO11n stays. It found the most swimmers at every false-positive level; larger
  YOLO11 models were 2 to 5 times slower with no gain. YOLO26n matched its speed on the
  M1 (GPU and CPU) but found fewer swimmers. Edge-device speed is untested.
- Lowering the cutoff to 0.15 raised YOLO11n from 30% to 40% found, for about 2 more
  false positives per frame, but duplicates rose from 6 to 25.
- Even at a 0.05 cutoff, 37% of swimmers were never found. In the frames inspected,
  most were small, distant, or low in the water. Next: a head-focused detector and a
  per-camera distance cutoff.

Reproduce: `uv run python scripts/benchmark_pose.py run` (and `sweep`). Labels:
[`pose_bench_gt.json`](../data/annotations/pose_bench_gt.json).
