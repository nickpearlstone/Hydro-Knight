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

## 5. Second model (v2): above chance at ranking victims, no catches

**Bottom line.** At false-alarm rates a lifeguard could work with, the v2 model catches
no drownings on held-out clips, and neither does a random scorer. It does rank the
victim as more unusual than most other swimmers, which is above chance, but the
signal is far too weak to raise an alarm.

**Setup.** 46 labeled rescue clips (51 labeled, 5 left out because a review found the
wrong victim clicked). Each swimmer's track becomes a 30 fps series of 58 features per
step: pose relative to the box (scaled by box height), all 17 keypoint confidences,
box confidence, a detected flag, box shape and size, and box speed (capped at 3 box
heights per second). The model is a dilated TCN autoencoder (small, 35k parameters)
trained on 3 s windows of non-victim swimmers only. Clips are split whole into train
and test; each of 3 seeds sets both the split and the initialization. Tracks of at
least 1 s are scored.

Victim tracks are removed from the normal data and scored separately: tracks under a
victim click, plus "path fragments" (tracks that stay within one box height of the line
between clicks for at least 0.3 s, because the tracker splits one swimmer into short
pieces).

**Detection.** A catch is an alarm on the victim's own track between drowning onset
and guard contact. Every alarm episode on any other track is a false alarm. Caught
rescues out of 25 held-out rescues (8 + 11 + 6 over the 3 seeds), at the best
threshold that stays within each false-alarm budget:

| False alarms per hour, at most | 120 | 240 | 1,000 | 2,000 | 5,000 |
|---|---|---|---|---|---|
| v2 model | 0 | 0 | 2 | 9 | 13 |
| Random scores | 0 | 0 | 1 | 2 | 3 |

The model beats random only at thousands of false alarms per hour. False alarms are
counted on the rescue clips themselves (about 3 to 5 minutes of footage per seed),
since there is no long normal footage yet.

**Ranking.** Two numbers measure whether victim moments score higher than everyone
else's, regardless of any alarm threshold:

- ROC-AUC between victim windows (onset to contact) and all other windows: **0.83**
  (seeds 0.88 / 0.74 / 0.86), random 0.50.
- Per-clip rank: the share of a clip's other windows scoring below the victim's best
  window. Median **0.97**, random 0.90. Random is high because it takes the best of
  several victim windows.

A high rank is still far from an alarm. In a clip ranked 0.975, the victim's best
moment was outscored by 157 moments from 25 other swimmers. Victims peak at about 0.8
to 0.9, and the alarm threshold for 120 false alarms per hour sits at 2 to 4.

**Lifeguards counted as victims.** Watching scores drawn on the video showed that in
the best-ranked clip, the "victim" tracks were the lifeguard filming in the
foreground: boxes about 950 px tall among 50 px swimmers. A box that contains the
victim click always won, and the path-fragment radius grows with box size. The matcher
now rejects any box more than 3 times taller than the swimmers near the click, either
at the click or as its track's median over its whole life. Before the fix, real victim
tracks were at most 2.4 times their neighbors and the wrong ones were 4 to 17 times.
The fix removed lifeguard or merged boxes from 6 clips.

| Run (small, 3 seeds) | AUC | Rescues caught at 240 FA/h or fewer |
|---|---|---|
| Before the fix | 0.846 | 1 of 25 (the lifeguard) |
| Same models, fixed matcher | 0.829 | 0 of 25 |
| Retrained with the fixed matcher | 0.829 | 0 of 25 |

**Comparing settings (before the matcher fix).** These runs compare feature and size
choices by AUC, which is fine for choosing between settings, though the absolute
values are about 0.02 too high.

Box speed is mostly tracker jitter (99th percentile about 7 box heights per second,
maximum above 100). Medium model:

| Box speed | AUC, seeds 0 / 1 / 2 | Mean |
|---|---|---|
| Raw | 0.914 / 0.749 / 0.881 | 0.848 |
| Capped at 3 per second | 0.931 / 0.746 / 0.875 | 0.851 |
| Removed | 0.887 / 0.726 / 0.854 | 0.822 |

Model size, box speed capped:

| Size | Parameters | AUC, seeds 0 / 1 / 2 | Mean AUC | Held-out normal error (median) |
|---|---|---|---|---|
| Small | 35k | 0.914 / 0.774 / 0.871 | 0.853 | 0.276 |
| Medium | 122k | 0.931 / 0.746 / 0.875 | 0.851 | 0.225 |
| Large | 450k | 0.904 / 0.757 / 0.891 | 0.850 | 0.180 |

**Takeaways**

- No catches at usable false-alarm rates. Detection results lead from now on, with
  ranking numbers as a secondary check.
- Watching scores on video is part of evaluation. It found the lifeguard problem, and
  that in several of the clips watched the victim has no detection box for much of the
  drowning, and no model built on pose tracks can score a swimmer it never sees.
- Larger models reconstruct normal swimming better without ranking victims any better.
- The test split moves AUC from 0.74 to 0.93, far more than any setting. Each split has
  only 6 to 11 test clips, so single numbers are not precise.

Reproduce: `uv run python scripts/train_v2.py --sizes small --seeds 0 1 2 --rates capped
--min-track-s 1.0 --detection --device mps --out runs/v2_ckpt_after`. Model weights are
saved per seed as `model_small_s<seed>.pt`; `--load-dir` re-scores a saved run.
