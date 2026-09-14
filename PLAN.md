# Hydro-Knight: Project Plan

The roadmap and decisions of record. For the project overview see the
[README](README.md); for experiment writeups see [docs/FINDINGS.md](docs/FINDINGS.md).

*Last updated 2026-09-14.*

## Current status

**Built**
- **Data:** JSONL manifest of source clips, metadata-only collection, local download,
  and a tkinter annotation app for marking distress events by time.
- **Pose extraction:** YOLO11n-pose. Two paths: `extract()` (whole frame, fast) and
  `extract_tiled()` (SAHI-style tiling + ByteTrack, higher recall, about 7x slower).
- **Features:** hip-centered, torso-scaled poses (34 values) plus keypoint and body
  velocity, for 70 values per frame, in 32-frame sliding windows per track.
- **Model:** TCN autoencoder over the windows.
- **Evaluation:** a detector-agnostic harness. Any detector produces the same detections
  table and gets the same report: per-event recall, detection latency, false alarms per
  hour, ROC/PR, and pose coverage inside each event.
- **Tooling:** MLflow tracking (SQLite), CI running pytest, ruff lint, and ruff format.

**Dataset:** 72 clips (66 rescue, 2 normal, 4 unlabeled). Each rescue clip has one
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

1. **Clean velocity A/B.** A `--no-velocity` switch so a 34-value baseline runs under
   the same conditions as the 70-value model. *(In progress.)*
2. **Correct frame rates.** Store each clip's fps in the manifest so event times map to
   the right frames. Without it, eval assumes 30 fps, but 53 clips are 60 fps.
   *(In progress.)*
3. **Plan A, Rule 1.**
4. **Re-extract the dataset** at imgsz 1280 or with tiling, starting with the 10 rescue
   clips that came back empty.
5. **Feature fixes:** keep partial poses (below) and label windows by the victim's
   track instead of by time.

## Known issues

- **Dataset quality.** The current keypoints were extracted with `extract()` at
  imgsz 640, the setting the resolution test showed finds far fewer swimmers. 10 rescue
  clips have no keypoints, and about 10 of 56 measurable events have under 50% pose
  coverage. Roughly 30% of positives are compromised.
- **Track IDs die after 30 frames.** Both trackers delete a lost track after 30 frames
  and neither uses the clip's fps. At 60 fps, a swimmer under water for more than
  0.5 s comes back with a new ID.
- **Activation threshold has no effect in `extract_tiled()`.** New tracks only start
  from detections at or above `high_conf_det_threshold` (default 0.6), so
  `track_activation_threshold=0.3` never applies. Detections between 0.25 and 0.6 can
  extend a track but never start one.
- **Seam duplicates in the tile merge.** A swimmer cut by a tile edge yields a partial
  box. IoU against the full box stays under 0.5, so both survive and can spawn an extra
  track. Fix: merge on intersection over the smaller box.
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
- **Scene cuts:** multi-angle videos break tracking. Split clips into single-camera
  segments before extraction.
- **Normal data volume:** how much normal swimming the autoencoder needs. This can't be
  answered until the features carry the signal.

## Decisions of record

- **Anomaly detection over classification.** Real positives are too rare. Favor recall.
- **Pose backend:** YOLO-pose at imgsz 1280 or higher. It beat MediaPipe, and
  resolution was the biggest recall lever.
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
