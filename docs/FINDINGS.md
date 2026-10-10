# Engineering findings

Detailed writeups of the experiments summarized in the [README](../README.md). Each
one was tested on real pool footage.

> Figures are **deliberately pixelated**. Backgrounds are blurred until people can't
> be recognized, and only the extracted skeletons are drawn clearly. This follows
> the project's rule against publishing identifiable footage.

## 1. Choosing a pose model: YOLO vs MediaPipe, and resolution

The whole design depends on an off-the-shelf pose model being able to find swimmers
in real pool footage, so this was tested first. YOLO-pose (Ultralytics) and MediaPipe
were run on the same frames from real clips.

On crowded scenes, MediaPipe found 0 to 1 swimmers per frame because it is built to
track one person at a time. YOLO found many:

![YOLO vs MediaPipe](images/pose_model.png)

YOLO's default 640px input shrinks distant swimmers until they can't be detected.
Running the same frames at 1280px found 3 to 6 times more swimmers. This one config
change had a bigger effect than the choice of model:

![YOLO 640 vs 1280](images/pose_resolution.png)

| Scene | YOLO @640 | YOLO @1280 | MediaPipe |
|---|---|---|---|
| Crowded wave pool | 2–4 | **7–13** | 0–1 |
| Sparse resort pool | 0–1 | 0–1 | 0 |

**Decision:** YOLO-pose at an input size of at least 1280px. A later benchmark against
hand-counted swimmers ([section 4](#4-pose-model-benchmark-against-hand-counted-swimmers))
confirmed YOLO11n over larger YOLO11 sizes, YOLO26, and MediaPipe.

Reproduce the figures with `uv sync --extra spike`, then
`uv run python scripts/make_pose_figure.py`.

## 2. Distant swimmers: SAHI tiling and ByteTrack

Even at 1280px, whole-frame inference misses swimmers far from the camera. SAHI
splits the frame into overlapping tiles, runs YOLO on each tile at a larger scale,
maps the detections back to the full frame, and merges duplicates. On a crowded clip
this raised detections from about 13 to about 50 swimmers per frame. The tradeoff is
speed: tiled extraction runs about 7 times slower.

YOLO's built-in tracker only works on its own single-pass detections, so it can't
use the merged tile output. Tracking runs as a separate step instead:

| Tracker | Tracks on test clip | Single-frame tracks |
|---|---|---|
| Simple IoU matching | 53 | 11 |
| **ByteTrack** | **15** | **0** |

Stable IDs matter because the planned "didn't resurface" rule needs to know that the
swimmer who went under is the same person across frames.

That test covered a short stretch of one clip. Across whole clips, tracks are still
short: on 22 extracted clips the median track lasted about 1 second, with about 430
tracks per minute of footage. Keeping one ID through a submersion is still an open
problem.

Code: [`tiled_pose.py`](../src/hydro_knight/preprocess/tiled_pose.py) (tiling and merge)
and [`build_tracks.py`](../src/hydro_knight/preprocess/build_tracks.py) (tracking).

## 3. First training run: ROC-AUC 0.539, and why the number can't be trusted

The first full run trained a TCN autoencoder on Colab over 32-frame pose windows.
It trained only on normal windows: the footage before each rescue plus two normal
clips. Some normal windows were held out for evaluation. The goal was for distress
windows to have higher reconstruction error than normal ones.

**Result:** ROC-AUC 0.539 and PR-AUC 0.522, close to random guessing. Across 3,821
held-out normal windows and 3,872 distress windows, the error distributions were
nearly identical up to the 90th percentile:

| Percentile | Normal | Distress |
|---|---|---|
| p50 | 0.105 | 0.111 |
| p90 | 0.322 | 0.323 |
| p99 | 1.414 | 1.680 |

### The evaluation couldn't measure the model

I first blamed the input features. An audit in September 2026, on the 22 clips
extracted with tiling, found a more basic problem: the evaluation itself can't tell a
good detector from a random one.

Each rescue was labeled as a time range, and an event counted as caught if *any*
swimmer was flagged during it. About 21 swimmers are in frame during a typical rescue,
and about 127 tracks pass through each event window, so almost any detector gets
credit.

- A detector that outputs random numbers scored ROC-AUC 0.497 and caught 23 of 23
  events (at a 99th-percentile threshold, with 417 false alarms per hour).
- A perfect detector of the drowning swimmer could reach only about 0.52, because most
  of the windows inside each event belong to other swimmers.
- Footage after the save and end cards was counted as normal, and held-out normal
  windows overlapped training windows.

So 0.539 says nothing about the model or its features. A later run with velocity
features (0.537) has the same problem.

### Feature problems that still stand

Reading the feature code turned up three problems. They are real, but this run could
not show whether they caused the score.

- **Submersion is filtered out.** When a swimmer goes under, keypoint confidence
  drops. [`normalize.py`](../src/hydro_knight/features/normalize.py) drops any pose
  where a reference joint has low confidence, and windowing joins the frames that
  remain.
- **Face-down floating looks normal.** A still body is easy to reconstruct, so it
  gets a low error score.
- **Movement through the water is removed.** Every pose is re-centered on the
  swimmer's hips, so bobbing in place looks like regular swimming.

A fourth limit is in the pose model itself: on a real rescue frame, the submerged
victim scored about 0.02 detection confidence in every tile, with every model tried.

### What changed because of it

- **Victim labels.** The labeler now records when the lifeguard makes contact and
  where the victim is (clicked positions, matched to tracks later). Evaluation will
  score only the victim's track, from drowning onset to guard contact, skip trimmed
  footage, and report a random baseline next to every result.
- **Trims are stored separately** from the part of the source video that was
  downloaded, so trimming a clip can never shift its labels.
- **The autoencoder's role narrowed** to flailing, the one signature that shows up in
  pose shape and motion alone. It may end up as one input feature to another detector.
- **The next detector is rule-based** and works on each swimmer's track: detection
  confidence and position over time.

## 4. Pose model benchmark against hand-counted swimmers

Sections 1 and 2 compared models by how many swimmers they found, with no check on
whether those were real people. In October 2026 I hand-counted every person on 10
frames and scored each model against those counts.

**Setup**

- **Frames:** 8 wave-pool rescues at the moment of drowning onset, plus 2 wide shots of
  a resort pool. 736 people in the water and 26 out of it, each marked with one click.
- **Models:** YOLO11-pose (n, s, m), YOLO26-pose (n, s, m), and MediaPipe Pose.
- **Two ways of running each model:** the whole frame at 1280px, and the production
  tiling from section 2 (480px tiles with 25% overlap, plus a whole-frame pass, merged).
- **Scoring:** a box finds a swimmer if that swimmer's click falls inside it, one box
  per swimmer. A second box on the same person is a duplicate. A box with no click
  under it is a false positive; most were posts, railings, and the lane rope. Boxes on
  people out of the water are not scored.
- **Speed:** median time per frame on an 8 GB M1 MacBook GPU, with tiles sent to the
  model 4 at a time.

**Results at the production confidence cutoff (0.25):**

| Model | Found (tiled) | Duplicates | False pos | Tiled s/frame | Found (whole frame) |
|---|---|---|---|---|---|
| **YOLO11n** | **219 (30%)** | 6 | 30 | **1.1** | 26 (4%) |
| YOLO11s | 202 (27%) | 14 | 30 | 2.1 | 21 |
| YOLO11m | 206 (28%) | 18 | 32 | 5.3 | 28 |
| YOLO26n | 125 (17%) | 5 | 20 | 1.1 | 18 |
| YOLO26s | 105 (14%) | 7 | 13 | 2.2 | 17 |
| YOLO26m | 75 (10%) | 7 | 17 | 6.1 | 15 |
| MediaPipe | 44 (6%) | 26 | 42 | 0.5 | 8 |

**Comparing at equal false positives.** YOLO26 gives the same people lower confidence,
so a shared cutoff of 0.25 is unfair to it. Running each model once at a cutoff of
0.05 and scoring every cutoff up to 0.5 lets the models be compared at the same number
of false positives. YOLO26 also normally skips the duplicate-removal step (NMS); it was
tested both ways.

| False positives | YOLO11n | YOLO26n | YOLO26n with NMS |
|---|---|---|---|
| ~20 | 146 found, 3 duplicates | 125 found, 5 duplicates | 113 found, 4 duplicates |
| ~30 | 203 found, 3 duplicates | 188 found, 12 duplicates | 159 found, 11 duplicates |
| ~50 | 293 found, 25 duplicates | 237 found, 30 duplicates | 220 found, 21 duplicates |

On the Mac's CPU, the two nano models ran at 4.0 s (YOLO11n) and 4.2 s (YOLO26n) per
tiled frame.

**What it shows**

- **Tiling is required.** On the whole frame, every model found under 5% of swimmers.
- **YOLO11n stays.** It found the most swimmers at every false-positive level, with
  equal or fewer duplicates in most comparisons, and the larger YOLO11 sizes were 2 to 5 times slower without
  finding more. YOLO26 found fewer swimmers with or without NMS, and its speed was tied
  on this hardware. Its advantage on edge devices such as a Jetson is untested here.
- **The 0.25 cutoff costs swimmers.** At 0.15, YOLO11n found 40% instead of 30%, for
  about 2 more false positives per frame. Duplicates rose from 6 to 25, so a lower
  cutoff would need a better merge step.
- **Many swimmers are never detected.** Even at a cutoff of 0.05, tiled YOLO11n missed
  37% of them. In the frames I inspected, most of the missed swimmers were small, far
  from the camera, or low in the water. This is the
  reason to try a head-focused detector and to limit each camera to the zone it can
  actually see.

Reproduce with `uv run python scripts/benchmark_pose.py run` and `... sweep`. The
clicks are in [`data/annotations/pose_bench_gt.json`](../data/annotations/pose_bench_gt.json);
the frames themselves come from local video and are never committed.
