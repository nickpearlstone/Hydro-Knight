# Hydro-Knight: Project Plan

The roadmap and decisions of record. For the project overview see the
[README](README.md); for experiment writeups see [docs/FINDINGS.md](docs/FINDINGS.md).

*Last updated 2026-09-14.*

## Current status

**Built**
- **Data:** JSONL manifest of source clips, metadata-only collection, local download,
  and a tkinter annotation app for marking distress events by time.
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
- **Tooling:** MLflow tracking (SQLite), CI running pytest, ruff lint, and ruff format.

**Dataset:** 74 clips (66 rescue, 4 normal, 4 unlabeled). Each rescue clip has one
labeled event window. Many events start more than 25 seconds in, so clips are always
extracted in full. Clips marked `[HOLD` in their notes are excluded from training.

## First result: ROC-AUC 0.539

The first Colab training run of the TCN autoencoder scored at chance. The cause is the
feature representation, so tuning the model will not fix it:

1. **Submersion is filtered out.** Low-confidence frames are dropped and windows are
   stitched over the gap.
2. **Face-down floating looks normal.** A still pose is easy to reconstruct.
3. **Displacement is removed.** Hip-centering hides movement through the water.

Full writeup in [docs/FINDINGS.md](docs/FINDINGS.md#3-first-training-run-roc-auc-0539).

**Decision:** the autoencoder is kept for flailing only. Everything else moves to
explicit per-track rules.

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

**Evaluation:** per-event recall and latency on the existing labeled events, using the
eval harness. No new annotation is needed.

Rule 1 has to match a *new* track to the lost one, because the tracker gives a swimmer a
new ID after a short submersion (see Known issues).

## Next steps

1. **Re-extract the dataset** with the new tiled pipeline: a pilot on about 5 clips
   (including one that came back empty), then all clips, labeled first.
2. **Clean velocity A/B.** The `--no-velocity` switch and per-clip manifest fps are in
   place; the matched 34 vs 70 value run on the new keypoints is next.
3. **Plan A, Rule 1.**
4. **Tune tracking** on the re-extracted data (start threshold, lost-track seconds).
5. **Feature fixes:** keep partial poses (below) and label windows by the victim's
   track instead of by time.

## Known issues

- **Dataset quality.** The current keypoints were extracted whole-frame at imgsz 640,
  the setting the resolution test showed finds far fewer swimmers. 10 rescue clips have
  no keypoints even though their videos are readable and swimmers are detectable, and
  about 10 of 56 measurable events have under 50% pose coverage. Roughly 30% of
  positives are compromised until re-extraction.
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
- **Time-scoped labels.** During a rescue, the lifeguard's track is also labeled
  distress, which inflates autoencoder eval. Plan A is not affected.

## Open questions

- **Submersion threshold:** how long under water should trigger an alert, given normal
  breath-holding?
- **Occlusion:** keeping identity when swimmers overlap in a crowded pool. BoT-SORT or
  OC-SORT with re-ID is the upgrade path.
- **Partial-pose retention:** `normalize.py` drops a whole pose if any of the four
  reference joints (hips and shoulders) is weak. This deletes exactly the
  partial-visibility cases that matter, such as shoulders up and hips sinking. Options:
  keep partial poses with a visibility channel, fall back to shoulder-based
  normalization, or normalize by the bounding box.
- **Scene cuts:** multi-angle videos break tracking. Extraction now records a per-frame
  scene-change score, so clips can be split into single-camera segments without
  re-reading the video.
- **Normal data volume:** how much normal swimming the autoencoder needs. This can't be
  answered until the features carry the signal.

## Decisions of record

- **Anomaly detection over classification.** Real positives are too rare. Favor recall.
- **Pose backend:** YOLO11n-pose at imgsz 1280 or higher. It beat MediaPipe, and
  resolution was the biggest recall lever. YOLO26-pose (n/s/m) was tested on 10 rescue
  frames in September 2026: despite better COCO scores it found about half as many
  swimmers and scored the same people lower, so YOLO11n stays.
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
