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

**Decision:** YOLO-pose at an input size of at least 1280px.

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

Code: [`tiled_pose.py`](../src/hydro_knight/preprocess/tiled_pose.py) (tiling and merge)
and [`build_tracks.py`](../src/hydro_knight/preprocess/build_tracks.py) (tracking).

## 3. First training run: ROC-AUC 0.539

The first full run trained a TCN autoencoder on Colab over 32-frame pose windows.
It trained only on normal windows: the footage before each rescue plus two normal
clips. Some normal windows were held out for evaluation. The goal was for distress windows to have higher reconstruction
error than normal ones.

**Result:** ROC-AUC 0.539 and PR-AUC 0.522, close to random guessing. Across 3,821
held-out normal windows and 3,872 distress windows, the error distributions were
nearly identical up to the 90th percentile:

| Percentile | Normal | Distress |
|---|---|---|
| p50 | 0.105 | 0.111 |
| p90 | 0.322 | 0.323 |
| p99 | 1.414 | 1.680 |

The only separation was in a thin tail of a few windows, which isn't enough to build
a detector on.

### Why it failed

Reading through the feature code turned up three problems with the input data.
Tuning the model or training longer can't fix any of them.

- **Submersion is filtered out.** When a swimmer goes under, keypoint confidence
  drops. [`normalize.py`](../src/hydro_knight/features/normalize.py) drops any pose
  where a reference joint has low confidence, and windowing joins the frames that
  remain. The distress windows that reach the model are mostly the parts where the
  swimmer is still visible and looks normal.
- **Face-down floating looks normal.** A still body is easy to reconstruct, so it
  gets a low error score.
- **Movement through the water is removed.** Every pose is re-centered on the
  swimmer's hips. Bobbing in place and swimming without making progress look the same
  as regular swimming.

### What changed because of it

- The autoencoder's role narrowed to flailing, which is the one signature that shows
  up in pose shape and motion alone.
- Velocity features were added (34 to 70 values per frame) to give the model motion
  information. A controlled comparison against the pose-only version is in progress.
- The next detector is rule-based and works on each swimmer's track: detection
  confidence and position over time. Its first rule targets swimmers who go under in
  the middle of the pool and aren't picked up again within a time limit.
- An evaluation harness was built so the autoencoder and the rule-based detector are
  scored the same way: per-event recall, time to detection, and false alarms per hour.

This is an early baseline. The feature pipeline, labels, and detectors are all still
being improved.
