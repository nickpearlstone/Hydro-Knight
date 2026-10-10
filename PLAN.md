# Hydro-Knight: Project Plan

The roadmap and decisions of record. For the project overview see the
[README](README.md); for experiment writeups see [docs/FINDINGS.md](docs/FINDINGS.md).

*Last updated 2026-10-10.*

## Current status

**Built**
- **Data:** JSONL manifest of source clips, metadata-only collection, local download,
  and a browser labeling app with two tabs. *Rescue timeline* marks onset, guard
  contact, saved, trims, and clicked victim positions. *Swimmer count* marks every
  person on still frames for the pose benchmark. Trims (`trim_start`/`trim_end`) are
  stored separately from the downloaded window (`start_sec`/`end_sec`), so trimming
  never shifts labels.
- **Pose extraction, two steps.** `scripts/extract_dataset.py` (GPU) runs YOLO11n-pose
  over 480px tiles at imgsz 1280 plus a whole-frame pass and saves every raw detection,
  un-merged and un-tracked, in resumable chunks with per-frame records and provenance.
  `scripts/build_tracks.py` (CPU) merges tile duplicates and runs ByteTrack into the
  keypoint files. Merge and tracker changes never need another GPU pass.
- **Features:** hip-centered, torso-scaled poses (34 values) plus keypoint and body
  velocity, for 70 values per frame, in 32-frame sliding windows per track.
- **Model:** TCN autoencoder over the windows.
- **Evaluation:** a detector-agnostic harness. Any detector produces the same detections
  table and gets the same report: per-event recall, detection latency, false alarms per
  hour, ROC/PR, and pose coverage inside each event.
- **Pose benchmark:** `scripts/benchmark_pose.py` scores pose models against
  hand-counted swimmers (found, missed, duplicates, false positives, speed) and compares
  them at equal false positives.
- **Tooling:** MLflow tracking (SQLite), CI running pytest, ruff lint, and ruff format.

**Dataset:** 74 clips (66 rescue, 4 normal, 4 unlabeled), 57 extracted with the tiled
pipeline. Each rescue clip has one labeled event window. Many events start more than 25 seconds in, so clips are always
extracted in full. Clips marked `[HOLD` in their notes are excluded from training.

## The first evaluation was broken

The first TCN autoencoder run scored ROC-AUC 0.539. A September 2026 audit showed the
evaluation can't separate a good detector from a random one: events are time ranges
and any swimmer's detection counts as a catch, so a random detector scored 0.497 and
caught 23 of 23 events, and a perfect victim detector could reach only about 0.52.
The 0.539 says nothing about the features or the model. Full writeup in
[docs/FINDINGS.md](docs/FINDINGS.md#3-first-training-run-roc-auc-0539-and-why-the-number-cant-be-trusted).

**Decisions:** evaluation is rebuilt around the victim's track before any more model
work. The autoencoder is kept for flailing only and may end up as one input feature.
Everything else moves to explicit per-track rules.

## Five distress signatures

| Signature | What the camera sees | Detection |
|---|---|---|
| Silent sink | Goes under mid-pool, never resurfaces | Plan A rule |
| Bobbing in place | Repeated submerge/resurface, no net displacement | Plan A rule |
| Passive face-down | Still, prone body beyond normal duration | Pose + duration rule |
| Flailing | Erratic, high-magnitude limb motion | Autoencoder |
| Effort without progress | Swimming hard, not moving through the water | Displacement features |

## Plan A: per-track rule detector (Rung 4, `detect/`)

No ML and no training. Rules run over each track's `box_conf` and centroid timeline.

**Rule 1, silent sink.** A track is lost, or its `box_conf` collapses, in the pool
interior. Start a timer. Alert if no track re-appears within radius R within N seconds.
Suppress when the loss happens at a frame edge or the track was heading out of frame.

**Rule 2, bobbing.** K submerge/resurface cycles with near-zero net displacement.

**Evaluation:** per-event recall, latency, and false alarms per hour, scored on the
victim's track from onset to guard contact (needs the victim labels).

Rule 1 has to match a *new* track to the lost one, because the tracker gives a swimmer a
new ID after a short submersion (see Known issues).

## Next steps

1. **Label the 66 rescue clips:** guard contact time, victim click at onset, victim
   click at the last visible moment. Everything below depends on these labels.
2. **Fix the evaluation:** match victim clicks to tracks by position and time, score
   only the victim's track from onset to guard contact, skip trimmed footage, and
   report a random baseline with every result.
3. **Finish extracting the labeled clips** (11 left).
4. **Go/no-go:** is the victim detected and tracked before going under? If usually not
   at these camera distances, this footage can't support the approach.
5. **Plan A, Rule 1,** starting from the strict disappearance rule (a track of 2 s or
   more vanishes mid-pool with no detection in that spot for 10 s).
6. **Head-focused detection with a distance cutoff.** The pose benchmark found that
   even the best setup misses most distant swimmers, and a lifeguard can't judge a
   swimmer who is too far away either. Plan: label heads on the benchmark frames, test
   an off-the-shelf head detector against YOLO11n, pick a per-camera distance cutoff
   from the measured recall, and track heads as the primary object with pose attached
   when the swimmer is close enough. Victims beyond the cutoff are reported as out of
   zone, never dropped. A real deployment would use several cameras, each covering one
   zone, like lifeguard sections.
7. **Small supervised model on per-track features.** Victim labels make every other
   swimmer a negative example. Hold out whole clips, and use only pre-contact footage.
8. **Scenario generation for Plan A (backlog).** Build distress scenarios as data
   rather than video: scripted track timelines (confidence and position over time) for
   the five signatures, plus scripted events injected into *real* extracted tracks so
   the pose statistics stay real and the timing is exactly known. Uses: tuning Plan A's
   thresholds and timers, measuring detection latency, and measuring false alarms over
   real normal footage. Never a substitute for recall measured on real rescues.
   Synthetic *video* was set aside: short clips break tracking, generated swimmers are
   easier to detect than real submerged ones, and hours of footage would be needed.

## Known issues

- **Most distant swimmers are never detected.** Against 736 hand-counted swimmers,
  tiled YOLO11n found 30% at the 0.25 cutoff and 40% at 0.15; even at 0.05 it missed
  37%. Whole-frame inference found under 5%.
- **Tracks are short.** On 22 extracted clips the median track lasted about 1 second.
- **Submerged victims are nearly invisible to pose models.** On a real rescue frame the
  arms-up victim scored about 0.02 in every tile and nothing whole-frame, with YOLO11 and
  YOLO26 alike. Plan A has to rely on the victim being tracked before going under.
- **Tracking fixes, not yet run on the full dataset.** The original merge and tracker
  had three problems, now fixed in `build_tracks.py` (`--legacy` reproduces them):
  half-body boxes at tile seams survived the IoU-only merge; lost tracks were deleted
  after a fixed 30 frames (0.5 s at 60 fps); and new tracks could only start at 0.6
  confidence, so the 0.3 activation setting did nothing. Defaults now merge on IoU or
  intersection-over-smaller across crops, keep lost tracks 1.0 s at the clip's fps, and
  start tracks at 0.5. On 90 frames of one rescue clip that gave 19 tracks of 45+ frames
  vs 15, at the cost of more short fragments (7 vs 5).
- **Track IDs still change after longer submersions.** A 1.0 s buffer does not cover a
  real submersion, so Rule 1 must re-associate new tracks with lost ones.
- **Time-scoped labels.** Event labels cover every swimmer in frame, including the
  lifeguard, which is why the first evaluation could not measure anything. The fix is
  victim labels (Next steps 1 and 2).

## Open questions

- **Confidence cutoff:** extraction keeps detections at 0.25 and up. Lowering it to
  0.15 found a third more swimmers in the benchmark but needs a re-extraction and a
  better duplicate merge. Decide together with the head detector.
- **Distance cutoff:** how far from the camera a swimmer can be and still be judged,
  measured as recall against apparent head size.

- **Submersion threshold:** how long under water should trigger an alert, given normal
  breath-holding?
- **Occlusion:** keeping identity when swimmers overlap in a crowded pool. BoT-SORT or
  OC-SORT with re-ID is the upgrade path.
- **Partial-pose retention:** `normalize.py` drops a whole pose if any of the four
  reference joints (hips and shoulders) is weak. This deletes exactly the
  partial-visibility cases that matter, such as shoulders up and hips sinking. Options:
  keep partial poses with a visibility channel, fall back to shoulder-based
  normalization, or normalize by the bounding box.
- **Frame rate is not normalized in the features.** `windows.py` measures velocity per
  *frame* and windows are 32 *frames*, but the dataset mixes 24, 30, and 60 fps: the same
  swimming speed gives half the per-frame motion at 60 fps, and a window covers 0.53 s
  there versus 1.07 s at 30 fps. Fix at the feature layer (resample tracks to a canonical
  fps, or express velocity per second and windows in seconds), not at extraction, which
  correctly stores native frame rates. A 30 fps edge deployment would add a third variant.
- **Scene cuts:** multi-angle videos break tracking. Extraction now records a per-frame
  scene-change score, so clips can be split into single-camera segments without
  re-reading the video.
- **Normal data volume:** how much normal swimming the autoencoder needs. This can't be
  answered until the features carry the signal.

## Decisions of record

- **Anomaly detection over classification.** Real positives are too rare. Favor recall.
- **Pose backend:** YOLO11n-pose at imgsz 1280 or higher, with tiling. It beat
  MediaPipe, and resolution was the biggest recall lever. The October 2026 benchmark
  against hand-counted swimmers confirmed it: YOLO11n found more swimmers than YOLO11
  s/m (which were 2 to 5 times slower) and YOLO26 n/s/m (with or without NMS) at equal
  false positives, and speed was tied with YOLO26n on an M1.
- **Save raw detections before merging or tracking,** so only YOLO needs the GPU and
  everything after it is rerunnable.
- **SAHI-style tiling** for distant swimmers. Only helps when imgsz is larger than the
  tile size.
- **ByteTrack, run separately from YOLO,** because YOLO's tracker can't consume merged
  tile detections.
- **ML scores anomalies; explicit logic handles temporal rules.**
- **Manifest-driven data.** No video in git, for copyright and privacy.
- **Labels are outcome-based.** A lifeguard intervention marks a positive.
- **Compute split:** Mac for ingest, annotation, and dev. Colab GPU for extraction and
  training. The hand-off is the manifest plus keypoint Parquet files.

## Later, once features and data are fixed

- **Transfer learning** from a pretrained skeleton-action model (ST-GCN, PoseC3D).
  More data-efficient at this scale. Needs a one-class head and a COCO-17 to NTU-25
  joint mapping.
- **Fine-tuning YOLO-pose on aquatic frames** to improve keypoints on submerged and
  prone swimmers. Needs labeled swimmer keypoints.
- **JEPA-style masked prediction** in embedding space. Avoids rewarding still poses,
  but likely too data-hungry for about 74 clips.
