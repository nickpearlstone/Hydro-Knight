# CLAUDE.md — working guide for this repo

Hydro-Knight is a pose-based **anomaly-detection** system for swimming-pool
safety: it turns pool footage into per-swimmer keypoint time-series, then flags
potential drowning events. Learning project + recruiting portfolio piece — not
commercial, but it must actually work.

> **Naming:** the repo/project is **Hydro-Knight**; the importable package is
> **`hydro_knight`** (renamed from `aqua_anomaly` on 2026-06-21).

---

## Collaboration rules (non-negotiable)

- **One file per response.** If a change naturally spans multiple files, stop
  and ask which to tackle first.
- **Explain every change in plain English** — what problem it solves, what the
  pieces mean, why it was written that way. No assumed context.
- The user is a **former lifeguard** and the domain authority. Architecture
  decisions are made turn-by-turn with them; don't bank a decision unilaterally.
- **Never commit:** raw video (`raw_local/`), keypoint parquets
  (`data/keypoints/`), model weights (`*.pt`), scratch outputs (`scratch/`),
  cookies files. Footage shows identifiable people, often minors — figures for
  the README must be pixelated (see `scripts/make_pose_figure.py`).

## Commands

```bash
uv sync                        # build env from pyproject.toml + uv.lock
uv sync --extra spike          # + MediaPipe (only for the backend-comparison spike)
uv run python -m pytest        # run tests — NOTE: `uv run pytest` shim does NOT resolve here
uv run python scripts/rung2_demo.py   # any script/module runs via `uv run python …`
```

- Python ≥3.11, src-layout, hatchling; `uv sync` installs `hydro_knight` editable.
- License is **AGPL-3.0** (inherited from Ultralytics) — keep it in mind if
  swapping dependencies.
- Local Mac = ingest/annotate/dev only (CPU/MPS). Heavy pose extraction and
  training run on **Colab GPU** (`docs/COLAB_TRAINING.md`); hand-off between
  them is the manifest + keypoint Parquet files, never raw video.

## Document map

| File | What it holds | Committed? |
|---|---|---|
| `PLAN.md` | **Decisions of record**: current status, the ROC-AUC 0.539 representation pivot, five distress signatures, Plan A/B, rung design, data strategy, open questions | yes |
| `README.md` | Public-facing: project summary + empirical engineering writeups (YOLO vs MediaPipe, resolution, SAHI + ByteTrack) with anonymized figures | yes |
| `docs/DATA_SOURCING.md` | Manifest-driven data approach, ethics/privacy rationale, outcome-based labeling | yes |
| `docs/COLAB_TRAINING.md` | GPU-side recipe: extraction + train/eval notebook cells, dataset split logic | yes |
| `DEV_WALKTHROUGH.md` | Personal file-by-file walkthrough notes | **gitignored** — and stale (predates the `hydro_knight` rename; trust the code over it) |
| `CLAUDE.md` | This file — entry point and working rules | yes |

## Where the project stands (2026-07)

Rungs 1–3 are built and mechanism-validated; the **first real Colab TCN run
scored ROC-AUC 0.539 ≈ chance**, diagnosed as a **representation problem, not
tuning**: normalization discards displacement, and low-confidence (submerged)
frames are dropped then stitched over — so the drowning literally never enters
the dataset. Full writeup in PLAN.md → "First results & the representation pivot".

**Next build = Plan A: the not-surfacing state machine (Rung 4, `detect/`)** —
per-track rules over `box_conf` + centroid timelines, no ML. Converging future
fixes: displacement features for the AE (Plan B), partial-pose retention in
`normalize.py`, track-scoped (not time-scoped) event labeling.

Dataset: 72 clips in `data/manifests/pool_footage.jsonl` (66 distress / 2
normal / 4 unlabeled), events as time windows; many events start >25 s in, so
**always extract full clips — never cap frames**. Clips with `[HOLD` in `notes`
are excluded from training.

## Repository layout (actual, verified 2026-07-03)

```
src/hydro_knight/
├── ingest/
│   ├── manifest.py        ClipRecord + enums, clip_id hashing, JSONL Manifest
│   │                      (append-dedup, atomic rewrite) — the spine of everything
│   ├── collect.py         yt-dlp metadata-only search/channel pull → manifest
│   ├── download.py        manifest → raw_local/<clip_id>.mp4 (+ .done markers)
│   ├── register_local.py  local recordings → manifest + raw_local/
│   └── blocklist.py       rejected URLs (data/blocklist.txt) so collect never re-adds
├── annotate/
│   └── annotator.py       tkinter labeling GUI: trim, typed event windows, labels
├── preprocess/
│   ├── extract_pose.py    video → keypoint Parquet; extract() = fast whole-frame,
│   │                      extract_tiled() = SAHI + ByteTrack (high recall, ~7× slower)
│   ├── tiled_pose.py      SAHI tiling: overlapping tiles, upscaled inference, NMS-merge
│   ├── tracking.py        SimpleTracker (greedy IoU) — superseded by ByteTrack, kept as fallback
│   └── pose_spike.py      multi-backend comparison harness (YOLO variants vs MediaPipe)
├── features/
│   ├── normalize.py       hip-centered, torso-scaled 34-dim pose vectors
│   │                      (⚠ drops any pose with one weak reference joint — known bias)
│   └── windows.py         per-track sliding windows → (N, 32, 34) sequences
└── models/
    ├── autoencoder.py     Rung 2: 34→12→34 MLP AE, per-pose reconstruction error
    └── tcn_autoencoder.py Rung 3: Conv1d temporal AE over windows

scripts/       rung2_demo.py, rung3_demo.py (mechanism checks), make_pose_figure.py
tests/         16 tests: manifest hashing/dedup, normalize invariances, windowing
data/          manifests/ (committed JSONL), annotations/, blocklist.txt;
               keypoints/ is gitignored (regenerable)
docs/          DATA_SOURCING.md, COLAB_TRAINING.md, images/ (anonymized figures)
raw_local/     gitignored source video (+ .done markers, pose_landmarker.task)
scratch/       gitignored spike outputs (viz PNGs, test parquets)
*.pt           YOLO weights auto-download to repo root (Ultralytics convention), gitignored
```

Keypoint Parquet schema: `frame, track_id, box_conf` + `x0,y0,c0 … x16,y16,c16`
(54 cols, one row per swimmer per frame). Keypoint confidences carry the
submersion signal.

## Decisions banked (don't relitigate without new evidence)

- **Anomaly detection, not classification** — real positives too rare; bias recall over precision.
- **Pose backend = YOLO-pose (yolo11n) at imgsz ≥ 1280** — beat MediaPipe empirically; resolution was the biggest recall lever.
- **SAHI tiling** for distant swimmers (recall ~13 → ~50/frame; only helps when imgsz > tile).
- **Tracker = ByteTrack** via `trackers`, decoupled from YOLO (built-in tracker can't consume merged tiled detections). BoT-SORT/OC-SORT re-ID is the upgrade path for post-submersion re-identification.
- **ML scores anomalies; explicit logic handles temporal rules.** Post-pivot: the AE's real lane is flailing; silent-sink / bobbing / face-down need per-track rule logic (Plan A).
- **Manifest-driven data, no video in git** — copyright + privacy.

## Deferred / open (see PLAN.md "Open questions" for detail)

- CI (`uv sync` → ruff → pytest) and a `[tool.ruff]` block — deliberately
  deferred as a hands-on learning exercise; don't just add them.
- Scene-cut detection before pose extraction (multi-angle videos break tracking).
- Submersion-duration threshold, crowded-pool occlusion, partial-pose retention.
