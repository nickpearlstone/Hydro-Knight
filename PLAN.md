# Aqua-Anomaly: Project Plan

## What this is

A pose-based anomaly detection system for swimming pool safety. It watches pool footage, tracks individual swimmers, and flags potential drowning events. Purpose: genuine learning project and recruiting portfolio piece. Not a commercial product, but it should actually work.

> **Naming note:** the repo/project is **Hydro-Knight** and the importable Python package is **`hydro_knight`** (renamed from the original `aqua_anomaly` on 2026-06-21). `import hydro_knight`.

---

## Current status *(updated 2026-06-21)*

What is actually built in the repo (the rung markers further down describe the *design*; this is the *progress*):

- **Rung 1 — done.** Pose extraction (`preprocess/extract_pose.py`), the YOLO-vs-MediaPipe spike (`pose_spike.py`), SAHI tiling (`tiled_pose.py`). Backend decision **banked: YOLO-pose @ ≥1280px** (see README for the empirical writeup).
- **Rung 2 — done, then retired (2026-07-08).** Pose normalization (`features/normalize.py`) survives; the single-pose MLP autoencoder + `rung2_demo.py` were **removed post-pivot** — the AE's one real lane (flailing) is temporal, which a per-frame model can't see. The Rung 3 TCN is the only autoencoder now. (History in git if ever needed.)
- **Rung 2.5 — done.** ByteTrack over SAHI-merged detections (`preprocess/tracking.py`); 15 stable tracks, 0 flicker on the test clip.
- **Rung 3 — done.** TCN autoencoder over pose-sequence windows (`features/windows.py` + `models/tcn_autoencoder.py`), with `scripts/rung3_demo.py`.
- **Annotation tooling — done** (`annotate/annotator.py`).
- **Dataset:** 72 clips registered in `data/manifests/pool_footage.jsonl` (66 `distress`, 2 `normal`, 4 `unlabeled`); each distress clip has one time-window event, all generically tagged `distress` (no `submerged`/`face_down` sub-typing). Events are from "Spot the Drowning" wavepool-rescue footage; many occur >25 s in (so full-clip extraction is required — see the Colab `max_frames` lesson in `docs/COLAB_TRAINING.md`).
- **First Colab TCN run — done, and it's the pivotal result.** ROC-AUC **0.539 ≈ chance.** Diagnosed (in code, not guessed) as a **representation problem, not tuning.** Full writeup in the next section. This redirected the project: the next build is **Plan A — the not-surfacing state machine (Rung 4)**, not more model tuning.
- **Eval harness — built (2026-07-08, `eval/`).** Detector-agnostic: every detector (TCN now, Plan A state machine later) reduces to one neutral detections table, so both land on the same per-event recall/latency/false-alarm report and can be compared directly. Includes the data-health views (pose coverage inside events — the "did the filter eat the drowning?" number), recall-vs-FA/hour operating curve, and per-event catch board. Run: `scripts/eval_report.py`. Recall-first by design; accuracy appears nowhere.
- **Not started:** Plan A (Rung 4 state machine, `detect/`) — the decided next step; the displacement-feature upgrade; the **partial-pose retention** fix in `normalize.py` (see Open questions). These three converge — all are "stop discarding signal in the feature layer."

---

## First results & the representation pivot *(2026-06-20/21)*

**The number:** first Colab TCN-autoencoder run scored **ROC-AUC 0.539** (PR-AUC 0.522) — essentially a coin flip. (ROC-AUC = "pick a random distress window and a random normal window; what's the probability the model scores the distress one as more anomalous?" 0.5 = guessing.)

**The diagnostic** (reconstruction-error percentiles, 3821 held-out normal vs 3872 distress windows):

| pct | normal | distress |
|---|---|---|
| p50 | 0.105 | 0.111 |
| p90 | 0.322 | 0.323 |
| p99 | 1.414 | 1.680 |

The distributions are **identical through p90.** The separated *means* (0.194 vs 0.345) were purely a thin p99 tail — a handful of windows, not usable signal. So: **not a tuning problem** (more epochs / threshold / STG-NF can't move a number capped by what's in the feature vector) — a **representation problem.**

**Why, by signal** (this is the key insight — see [[representation-displacement-gap]]):
1. **Not-surfacing (silent sink — the case that matters most):** *structurally absent.* When a swimmer submerges, keypoint confidence collapses, `normalize.py:39` drops those frames, and `windows.py` stitches over the gap. The submersion never enters the dataset; the 3,872 distress windows that survive are the *visible* (normal-looking) portions. The AE literally cannot see the drowning.
2. **Passive face-down:** *anti-signal.* A motionless prone pose is trivial to reconstruct → **low** error → scores as normal.
3. **Flailing:** *weakly present* — the only signal reconstruction error natively catches; it's the p99 tail.

This matches PLAN's original split: *"ML scores anomalies; logic handles explicit temporal rules."* The AE's real lane is **flailing**; the other signals need explicit track/pose logic.

### Five distress signatures (the lifeguard expanded the original three)

See [[distress-signal-taxonomy]]: **silent sink, passive face-down, flailing, bobbing-in-place, effort-without-progress.** The last two are new and both hinge on **net displacement through the scene** — the exact thing normalization throws away. Bobbing defeats a lone submersion timer (each resurface resets the clock); "effort without progress" (incl. fighting the wavepool current) is invisible because the AE sees only hip-centered *shape*, not motion through the water.

### Decided direction

- **Plan A (next) — not-surfacing state machine (Rung 4).** A *family* of per-track rules over the `box_conf` + centroid timeline, no ML/training. **v1 silent-sink rule:** track loss / `box_conf` collapse in the pool **interior** → start timer → ALERT if no track re-associates within radius R within N seconds; **suppress** if the loss is at a frame edge (left frame) or velocity heads out. Built so the **bobbing rule** (#4: K cycles + near-zero net displacement + deep zone) drops in as rule 2. Eval = **per-event recall + latency** (how *early* it fired) on the existing 66 events — no annotation pass needed.
- **Plan B (later) — add velocity/displacement to the AE features**, sharpening flailing and enabling #5.
- **Labeling fix (for B, not A):** Cell 4 labels windows by *time* overlap, so during a rescue the **lifeguard's track** gets stamped distress → inflates the AE eval. Fix = **track-scoped (victim), not time-scoped** labeling. Do **not** trim the rescue period — real victim distress continues during the save. A is naturally robust (it keys on the victim submerging; the guard's track stays confident and moving), so A needs no re-annotation.

---

## Problem framing

### Binary anomaly detection, not classification

Started as multi-class (first aid, regular save, spinal save, etc.) — scrapped early. Mechanism of injury isn't reliably visible from camera footage. Narrowed to: **is someone in distress or not.**

Anomaly detection is the right framing because:
- Real drowning events are extremely rare → severe class imbalance in any classification setup
- Anomaly detection is purpose-built for sparse positives: train on abundant "normal" data, flag deviations
- Detection threshold should bias **toward recall over precision** — missed drownings are more dangerous than false alarms, but alarm fatigue is a real concern to monitor

### Three target signals

| Signal | Mechanism | Detection approach |
|---|---|---|
| Not surfacing | Tracked swimmer hasn't reappeared past a time threshold | Tracking + timer logic |
| Face down too long | Passive, motionless prone position beyond normal duration | Pose angle over time |
| Aggressive flailing | Erratic, high-magnitude limb motion inconsistent with normal strokes | Motion variance / reconstruction error |

**Silent drowning is the critical case.** The most common real pattern is no flailing, no shouting — a person goes limp and sinks. The model needs temporal memory, not single-frame classification. Keypoints vanishing and not returning is the primary signal.

### False positive baseline

All three signals fire on healthy swimmers (breath-holding, dead man's float, excited kids). The correct baseline is **human guard performance**, not a perfect detector. A former lifeguard can't reliably distinguish these in real time either. An ML model processing continuous footage may catch patterns guards miss.

---

## Architecture

### Input representation: pose-primary

Run MediaPipe pose estimation as a preprocessing step, converting raw video into time-series of keypoint coordinates per tracked person. This converts "video anomaly detection" into "multivariate time series anomaly detection."

**Why this works:**
- When keypoint confidence collapses (person submerges), that collapse is itself a feature — the non-surfacing signal expressed natively in the representation
- Pose keypoints are naturally invariant to lighting, camera angle, and weather

**Known limitation to validate early:** MediaPipe was trained on above-water, upright humans. It degrades on prone swimmers, underwater partial occlusion, and crowded pools. The confidence-collapse-as-feature argument depends on MediaPipe failing *predictably* when someone submerges. **Validate this empirically before Rung 3.**

### Feature engineering (to decide before training)

Raw keypoint coordinates are not the best representation. Choices matter:
- **Raw coordinates** — simplest, but view-dependent
- **Joint angles** — more view-invariant, better for pose classification
- **Velocities / accelerations** — captures flailing better than position
- **Normalized-to-torso coordinates** — removes camera angle dependence

Decision: use joint angles + per-joint velocity as primary features. Revisit if autoencoder reconstruction quality is poor.

---

## Build progression

### Rung 1 — Pose extraction preprocessing *(done — see Current status)*
Turn video into keypoint time series → per-frame keypoint Parquet files, one file per clip, one row per frame per detected person.

**Pose backend: YOLO-pose primary, MediaPipe fallback — decide via a head-to-head spike.** The pose estimator is a *swappable* preprocessing choice: both backends output the same downstream representation (per-person, per-frame keypoints + confidence), so features/models/state-machine don't care which produced them. Pick empirically.

- **YOLO-pose (Ultralytics YOLOv8/v11-pose) — leading candidate.** Natively **multi-person** (one pass finds every swimmer — pools are crowded; classic MediaPipe Pose is single-person). Ships **built-in tracking** (ByteTrack/BoT-SORT via `model.track()`), which gives per-swimmer IDs for free and **may subsume Rung 2.5 entirely**. One actively-maintained pipeline (detect + pose + track). 17 COCO keypoints (2D). License: AGPL-3.0 (fine for non-commercial portfolio; note obligations if it ever goes commercial).
- **MediaPipe — fallback.** 33 keypoints + a rough monocular 3D estimate; CPU-optimized; Apache-2.0. But single-person by default (needs detector+crop per swimmer for crowded scenes) and a less stable Python API.
- **Both share the key risk:** trained on land-based upright humans → both degrade on prone/submerged swimmers. The spike validates this at the same time.

**The Rung 1 spike (do first):** run YOLO-pose *and* MediaPipe on the same clips (a meet, the reef, a rescue) and compare which actually detects swimmers — especially small, distant, and submerging ones. That single experiment picks the backend and de-risks the aquatic-domain assumption the whole architecture rests on.

**Scene-cut detection (required, runs before pose extraction).** The model's true unit is a *continuous single-camera shot*, not a whole video. A camera-angle cut breaks tracking identity, injects a fake instantaneous pose jump into the temporal signal, and makes a single `camera_view` label dishonest. So preprocess must split each video into single-camera segments before pose extraction (e.g. PySceneDetect content detector, or frame-difference thresholding). Each segment becomes its own continuous unit with its own keypoint timeline.

*Annotation guidance until the cut detector exists:* annotate continuous-shot clips normally. For multi-angle clips, set `camera_view = unknown` and still mark events — events are timestamp-based so each will fall into the correct segment after the split. Per-segment `camera_view` gets assigned/verified in a quick re-review pass once splitting is in place. Don't delete multi-angle clips just for having cuts; the splitting is deferred, not the data.

### Rung 2 — Simple autoencoder on keypoint sequences *(built, validated, then removed 2026-07-08)*
Train to reconstruct normal swimming. High reconstruction error = anomaly. Get this working end-to-end before anything fancier. Validates the representation before adding temporal complexity.

*Outcome:* it served exactly that purpose — proved the reconstruction-error mechanism end-to-end — and was then deleted once the TCN (Rung 3) superseded it. Post-pivot, the AE's only real lane is flailing, which is inherently temporal; a single-frame model can never see it, so keeping the MLP was dead weight.

### Rung 2.5 — Lightweight tracking *(done — ByteTrack, decoupled from YOLO)*
**Resolution:** because tracking runs over *SAHI-merged* tiled detections, YOLO's built-in `model.track()` can't be used (it only tracks its own single-pass detections). Tracking is therefore decoupled: **ByteTrack** (model-agnostic, via the `trackers` package) runs over the merged detections — 15 stable tracks, 0 flicker on the test clip vs. 53 churning tracks from a naive IoU tracker. BoT-SORT/OC-SORT with re-ID is the future option if re-identifying a swimmer *after* a submersion gap becomes the bottleneck.

ByteTrack or DeepSORT for bounding-box-level identity continuity. Originally planned as Rung 4, but moved up because:
- "Not resurfaced in N seconds" is arguably the most important signal
- It requires identity continuity **across a submersion gap**
- Tracking in pool scenes is harder than pedestrian scenes (bodies submerging, overlapping wakes, oblique cameras)
- Better to discover tracking failures early than after the autoencoder is trained

### Rung 3 — Temporal structure
LSTM autoencoder or temporal CNN so the model reasons over windows of motion, not single frames. This is where flailing and non-surfacing become learnable patterns rather than one-frame snapshots.

### Rung 4 — Per-swimmer state machine
Attaches explicit states to tracked identities:
```
at_surface → submerged → resurfaced   (normal)
at_surface → submerged → [timer] → ALERT   (not-surfacing anomaly)
```
ML scores anomalies; logic handles explicit temporal rules. Combines the autoencoder scores with the state machine to produce alerts.

### Compute strategy
- **Local (MacBook):** ingestion, annotation, dev/iteration, small spikes. CPU/MPS only.
- **Colab / Kaggle (GPU available):** the heavy lifting — batch pose extraction over the full clip set (YOLO-pose runs much faster on GPU), and model training (Rungs 2–4). This removes YOLO's "wants a GPU" downside entirely.
- **Hand-off between them is the manifest + keypoint Parquet files**, not raw video: extract poses on GPU, commit the lightweight keypoint files, train from those. Keeps the heavy data off the laptop and off the repo.

---

## Data strategy

### Manifest-driven dataset (no video in repo)

The repo commits manifests (JSONL files of source URLs + metadata + annotations) and code, never raw video. Anyone reproducing the work re-downloads from the manifest. Mirrors how Kinetics and AVA distribute data.

**Why:**
- Copyright: public visibility ≠ redistribution rights
- Consent/privacy: pool footage shows identifiable people, often minors
- Reproducibility: manifests are tiny and diff-friendly; video dumps aren't

### Collection

- `yt-dlp` in metadata-only mode to search for varied pool footage
- Default search terms bias toward outdoor-daytime normal pool activity
- Download to a gitignored local directory for processing only
- Condition metadata (camera view, setting, time of day, weather) is auto-filled as search-intent guess and must be verified in a manual review pass

### Positive (anomaly) examples — in priority order

1. Consented research datasets (e.g. Figshare underwater drowning dataset)
2. Self-recorded simulations with consenting participants
3. Proxy footage (breath-hold freediving, lifeguard training drills)

### Annotation schema

Per-clip fields: `clip_id`, `source_url`, `platform`, `start_sec`, `end_sec`, `camera_view`, `setting`, `time_of_day`, `weather`, `label`, `notes`, provenance fields.

Labels: `normal`, `review`, `distress`, `submerged`, `face_down`, `unlabeled`

**Outcome-based retrospective labeling:** if they surface → negative; if guards intervene or someone is pulled out → positive. The footage tells you the answer after the fact. No ambiguous mid-event judgment calls.

---

## Labeling tooling — build this before the annotation pass

**This is a likely blocker.** Fast annotation of clip-level labels on video is genuinely painful without dedicated tooling. Doing it manually with a video player and a spreadsheet will be far slower than expected and invite label noise.

### Minimum viable annotator

A script that:
1. Reads the manifest and finds unlabeled or `review` clips
2. Steps through clips, showing sampled frames (e.g. 1fps) in sequence
3. Waits for a keypress: `n` (normal), `d` (distress), `s` (submerged), `f` (face_down), `r` (review/skip)
4. Writes the label back to the manifest JSONL

This is a half-day build that pays for itself within the first annotation session. **Build before starting the annotation pass, not after.**

---

## Scope

**First working version:** outdoor daytime pools only. Broad generalization (weather, night, indoor) is the aspiration the dataset collection is oriented toward, but the first validated model is scoped to this narrower slice.

---

## Repository structure

```
src/hydro_knight/
├── ingest/
│   ├── manifest.py     — ClipRecord dataclass, controlled-vocabulary enums,
│   │                     deterministic clip_id hashing, Manifest JSONL reader/writer
│   │                     with append-safe de-duplication
│   ├── collect.py      — yt-dlp metadata-only search → manifest registration
│   └── download.py     — local-only video download (gitignored output)
├── annotate/           — labeling tooling (build before annotation pass)
├── preprocess/         — video standardization, pose extraction pipeline
├── features/           — keypoint → time series feature engineering
├── models/             — autoencoder, LSTM, anomaly scorers
├── detect/             — inference pipeline, state machine, alerting
└── utils/              — shared helpers

data/
├── manifests/          — committed JSONL manifests
└── annotations/        — committed annotation files

raw_local/              — gitignored, local video only
docs/
└── DATA_SOURCING.md    — manifest approach and ethics
```

---

## Repo hygiene & standards backlog *(assessed 2026-06-21)*

The committed tree is clean (no video/weights/venv ever tracked — keep it that way).

**Done (2026-06-21 cleanup pass):**
- ✅ Package renamed `aqua_anomaly` → **`hydro_knight`** to match the repo (imports, README, pyproject, docs all updated).
- ✅ `.gitignore`: added `.DS_Store` and `scratch/`; deleted loose `.DS_Store` files.
- ✅ Pruned the unreferenced 113 MB `yolo11x-pose.pt`; kept `yolo11n` (code default) + `yolo26n` (spike). Weights stay at repo root by Ultralytics' auto-download convention (all `*.pt` are gitignored).
- ✅ `raw_local/` is now **source video only** — scratch artifacts (viz PNGs, test parquets, `tracking.mp4`, `pose_spike_out/`) moved to gitignored `scratch/`. (`pose_landmarker.task` stays — code references `raw_local/pose_landmarker.task`.)
- ✅ **`tests/`** added (16 tests): clip_id hashing + manifest append-dedup, normalize invariances, windowing. `[tool.pytest.ini_options]` in pyproject. Run with **`uv run python -m pytest`** (the `uv run pytest` console-script shim doesn't resolve here).
- ✅ **LICENSE** = AGPL-3.0 (matches the Ultralytics dependency); declared in `pyproject.toml` (`license = "AGPL-3.0-only"`) + a README License section.
- ✅ Doc/reality drift: created `data/annotations/.gitkeep` and `docs/DATA_SOURCING.md` (both were named in the repo-structure block but didn't exist).

**Still TODO (deliberately deferred — to be done hands-on as a learning exercise):**
- **CI** — `.github/workflows/ci.yml` running `uv sync` → `ruff` → `pytest` on push.
- **Ruff** — add a `[tool.ruff]` block to `pyproject.toml` (`.ruff_cache/` is already ignored).

---

## Collaboration rules (Claude must follow these)

- **Multi-file changes are fine.** Modify as many files in one response as the task needs. *(Relaxed 2026-07-04; was one-file-per-response.)*
- **Ask clarifying questions before starting.** Keep asking until ~95% confident the request can be accomplished as actually intended. Don't guess at ambiguous intent — surface the ambiguity and ask.
- **Explain every change in plain English.** After writing or editing any line of code, describe what it does as if the reader has never seen that line before — what problem it solves, what the individual pieces mean, and why it was written that way. No assumed context.

---

## Open questions

- What submersion duration threshold triggers the not-surfacing alert? Needs empirical tuning against normal breath-hold behavior.
- How to handle crowded pools where multiple swimmers overlap? Tracking identity across occlusion is unsolved.
- What's the minimum normal-swim training data volume for a useful autoencoder? Still open — the first real run's failure was representational, not data-volume, so this can't be answered until the feature layer carries the signal (Plan B).
- **Partial-pose retention — don't discard a whole pose on one weak reference joint.** `features/normalize.py` returns `None` (drops the entire pose) if *any* of the four reference joints (both hips + both shoulders) is below `min_ref_conf`, throwing away even the high-confidence keypoints on that pose. This is biased to delete exactly the drowning-relevant partial-visibility cases — e.g. shoulders out of the water while the hips are submerged, which is itself a vertical-sinking distress posture. It's the same information-loss mechanism as the submersion blindness (low-confidence frames dropped + windows stitched over the gap). Options to revisit: (a) keep partial poses + a per-keypoint confidence/visibility channel (the same fix as the submersion presence-channel — serves double duty); (b) a fallback reference-frame hierarchy (normalize against shoulder-center/width when the hips are unreliable); (c) bounding-box normalization (use the always-present detection box instead of body joints — more robust, less anatomically clean). Converges with the displacement / presence-channel work, so worth doing together.
