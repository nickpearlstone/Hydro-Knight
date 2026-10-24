# Synthetic realism loop (2026-10-19)

Goal: one Blender frame that the detectors treat like a real wave-pool frame, then scale up
on Kaggle. Reference: real frame `6f5c67826a76` #720 (227 hand-counted swimmers in the water).
Code: `synthetic/` (scene + config + tools); renders are gitignored.

## How a frame is judged

Same rules for the real frame and the render (`synthetic/tools/evaluate.sh`):

1. **Detector level** (`score.py`): YOLO11n-pose and RT-DETRv2-R18, production tiling.
   Swimmers found and false boxes at conf 0.05 and 0.25, by size band.
2. **Feature level** (`realism.py`): DINOv2-small features of 64 px crops around each person.
   KID (lower = closer) and a real-vs-render classifier AUC (0.5 = can't tell apart).
   Yardstick = real vs real (left half vs right half of the same frame): KID 12 (11–20), AUC 0.65.
3. **Image statistics** (`stats.py`): diagnostics only (water colour, noise, sharpness, blocking).

## Real targets

| | found @0.05 | false @0.05 | found @0.25 | false @0.25 |
|---|---|---|---|---|
| YOLO11n-pose | 65% | 25 | 23% | 3 |
| RT-DETRv2 | 89% | 289 | 70% | 44 |

## Batches (one 1920×1080 frame each, 32 samples, ~4 min on the M1)

| Batch | Change | YOLO @0.05 / @0.25 | RT-DETR @0.05 / @0.25 | RT false @0.05 / @0.25 | KID | AUC |
|---|---|---|---|---|---|---|
| v2 (10-17) | old spike frames | – | – | – | 78–84 | 1.00 |
| b01 | v3 scene: reference camera, fan-shaped pool, teal tubes, people in tubes, real x264 chain | 34% / 10% | 67% / 30% | 129 / 13 | 67 | 1.00 |
| b02 | deck-overlap bug fixed, murkier water, bigger adults | 45% / 21% | 78% / 45% | 141 / 21 | 76 | 1.00 |
| b02 + tuned post | darker, greener, softer, lower bitrate with motion | 34% / 11% | 70% / 39% | 306 / 47 | 33 | 1.00 |
| b03 | camera refit (70° FOV, from near/far size ratio), crowd clustered far, swimmers sunk | 43% / 14% | 85% / 48% | 264 / 30 | 34 | 1.00 |
| b04 | bulkier bodies (build ×1.15–1.45), treading arms under water, milkier water, fewer tubes mid-pool | 50% / 18% | 87% / 50% | 262 / 29 | 23 | 0.99 |
| b05 | **MPFB humans** (CC0, 70-human library, painted swimwear, matched skins, dark wet hair) | 41% / 15% | 86% / 56% | 386 / 55 | 27 | 1.00 |
| b06 | floor caustics, sharper ripples, glossy translucent tubes | 40% / 15% | 87% / 57% | 436 / 61 | 24 | 0.99 |
| b07 | denoiser off, 64 samples (test) | 45% / 14% | 84% / 58% | 500 / 95 | 28 | 1.00 |
| b08 | diagnosis fixes: clear water, glints, darker hair/clothes, clustered crowd, black point, lower bitrate | 33% / 12% | 81% / 53% | 330 / 51 | 28 | 0.99 |
| b09 seed 1 | b08 with people back higher in the water, lighter water | 44% / 13% | 81% / 53% | 368 / 55 | 23 | 1.00 |
| b09 seed 2 | same config, new seed | 36% / 9% | 86% / 54% | 333 / 38 | **19.5** | 1.00 |
| b10 seed 1 | **geometry refit 2**: pool 30 m (not 45), camera pitch 25°; 4-frame moving-water clips into the encoder | **64% / 21%** | **89%** / 58% | 262 / 26 | 25 | 1.00 |
| b10 seed 2 | same config, new seed | **68% / 21%** | **87%** / 57% | 340 / 40 | 23 | 0.99 |
| b11 (2 seeds) | varied stances (hips, one arm up, forearms on water, holding), body twist + head turn; grain 0.45 | 61–62% / 23–25% | 90–91% / 53–56% | 274–467 / 44–46 | 28–30 | 0.99 |
| b12 (2 seeds) | b11 with treading forearms kept low | 62–66% / 21–22% | 86–93% / 54% | 336–481 / 55–61 | 27–28 | 0.99 |
| b13 (2 seeds) | beach starts at 75% of the pool (mid-pool = deep water, heads only), warped caustics, light aqua tubes, less skin gloss, 270 people | 50% / 14–15% | 90–94% / **67–68%** | 346–401 / 60–74 | 25–28 | 1.00 |
| b13 @ 3.5 Mbps, wobble 1.5 | post only | 67–73% / 26–31% | 93–95% / 69–70% | 303–354 / 47–66 | 35–46 | 1.00 |
| b13 @ 5.8 Mbps, wobble 1.0 | post only (the clip's nominal bitrate) | 70–73% / 31–35% | 91–95% / 68–69% | 270–329 / 48–49 | 54–65 | 1.00 |
| **b13 @ 2.5 Mbps, wobble 2.5** | post only; **new default** | **62–64% / 22–23%** | **90–96% / 65–70%** | 359–370 / 51–60 | 27–29 | 1.00 |

## What we learned

- **Camera geometry first.** The real pool is a fan (narrow at the wave machine, wide at the
  beach) and the camera sits on the side deck. The field of view came from the ratio of the
  near-tube size to the far-person size (~7×), giving about 70° horizontal, not GoPro-wide 94°.
  With the wrong FOV the far swimmers were half their real size and the detectors missed them.
- **The video chain matters most at feature level.** Encoding a single still gives an I-frame
  that's far too clean. A 16-frame clip with shake and a ~3 px wobble, run through x264 twice
  (camera 40 Mbps, then YouTube-like 2.5 Mbps with B-frames) and keeping a mid-GOP frame, cut KID
  from 76 to 33 with no re-render. The real clip is H.264, 60 fps, ~5.8 Mbps.
- **The far end of a wave pool is the beach.** People there stand waist-deep, so the far crowd is
  mostly torsos, not heads.
- **Remaining gap = the people themselves.** The classifier still separates every render crop
  from real ones (AUC 1.0) and YOLO-pose finds too few, which points to the procedural
  "doll" humans (thin limbs, flat skin, stiff poses).
- **Realistic humans helped the detectors more than the features.** MPFB bodies (real proportions,
  photo skins, hair) raised RT-DETR's confident finds, but the crop classifier still separates
  every render from real (AUC ~0.99).
- **Things that did not move the classifier:** matching the water's Lab colour exactly, switching
  the denoiser off, and darker hair/clothing. So the cue is structural (how water, tubes and
  bodies look at 64 px), not colour or denoiser texture.
- **b09 seed 2 is the first frame whose KID (19.5) sits inside the real-vs-real range (11–20).**
  Seed 1 of the same config is 23, so the frame-to-frame spread is about ±3.

- **The YOLO gap was geometry again, not appearance.** A breakdown of b09 misses showed the far beach
  crowd (half of all swimmers, heads <= 8 px) was found 19-35% of the time. Real people at the same
  image row were twice as tall in pixels: the real beach is ~22 m from the camera, ours was ~40 m.
  Shortening the pool to 30 m and pitching the camera to 25° brought YOLO11n-pose to the real
  numbers (64-68% vs 65% at 0.05, 21% vs 23% at 0.25).
- **Grain (Nick's suggestion, 10-19).** Our flat-water grain measured 0.12-0.34 vs 0.42 real, so the
  renders were too clean. Grain added after the codec fixes that statistic at strength ~0.45; it did
  not move KID or the classifier (coarse grain made them slightly worse), so it is on at 0.45.
- **Moving water through the encoder** did not raise blockiness (0.98-1.07 vs 1.16 real); the real
  clip's blocking probably comes from much larger motion (camera pan, crowd motion) than ours.

- **More varied poses made it worse.** The stance library (b11, b12) raised KID from 23–25 to 27–30
  and roughly doubled YOLO's false boxes (45–48 vs 25 real), with recall unchanged. Likely partial-body
  boxes from raised arms and twisted torsos. The config is back on the original pose library;
  `humans_v3.blend` is kept for later, hand-checked pose work with Nick.

**Best setup now = b10 config + grain 0.45.** Detector level matches the real frame:
- YOLO11n-pose finds 62–68% vs 65% at conf 0.05, and 21% vs 23% at 0.25.
- RT-DETR finds 87–89% vs 89% at conf 0.05.
- False boxes at conf 0.05 are within about 1.3× of real for both detectors.

Two gaps remain: RT-DETR at conf 0.25 finds 57% vs 70%, and the crop classifier is still
at AUC ~0.99.

- **Bitrate trades detector match against feature distance.** Higher bitrate makes people easier
  (recall above real) and moves DINOv2 features further from real (KID 35–65). 2.5 Mbps with a
  2.5 px wobble matches all four real recall numbers within a few points; KID 27–29.
- **The crop classifier sees renders as renders regardless.** Restricting crops to the same image
  rows did not lower AUC (0.97–0.998), so it is appearance, not where people sit in the frame.

## Research round 3 (10-19): is AUC ~0.99 beatable, and does it matter?

- No published synthetic person/crowd dataset (GCC, JTA, MOTSynth, PeopleSansPeople, BEDLAM,
  SwimXYZ) reports a feature-classifier or KID score; they are judged only by downstream results.
  A frozen-feature probe separating renders at ~0.99 is likely normal even for datasets that
  transfer well. Distribution metrics predict transfer only weakly (SADGE 2026 preprint; Wang et
  al. WACV 2024: the harmful gap is content, not artefacts).
- What transfers: synthetic pretraining then fine-tuning on a little real data (PeopleSansPeople
  few-shot keypoint AP 44 vs 6 real-only; GCC; Nowruzi 2019: fine-tune beats mixing); diversity over
  photorealism; content statistics (density, layout) matched to the target.
- If appearance is pushed further, the best-supported tool is **CUT / FastCUT** unpaired
  translation (BSD licence) trained on native-resolution crops of real pool frames vs renders;
  ParGANDA (MOTSynth→MOT17/20) gained 3–6 AP from translation. EPE (MIT, uses G-buffers) is the
  heavier fallback. Diffusion img2img is risky at 30 px (the VAE alone may erase swimmers).

**Plateau reached on render-side realism.** Detector-level statistics match the real frame; the
feature probe has not moved in ~8 batches despite water, humans, grain, denoiser, compression
and pose changes.

## Where it stood after b09

**Close at the detector level for RT-DETR:**
- At conf 0.05, RT-DETR finds 81–86% with 330–370 false boxes. Real: 89% with 289.
- The YOLO11n-pose false-box rate matches.

**Still off:**
- YOLO11n-pose finds fewer people than in the real frame: 36–44% vs 65% at conf 0.05, and
  9–13% vs 23% at 0.25. Its 45–60 px band is weakest, so pose evidence on small far bodies is
  still too weak.
- RT-DETR at conf 0.25 finds 53% vs 70%.
- The crop classifier (AUC ~0.99) is not beaten. Real-vs-real is 0.65.

**Not yet "looks the same to the model."** The geometry, colour, false-alarm rate and
generic-feature distance (KID) are close; per-person appearance is not.

## Next (in order, updated after research round 3)

1. **More reference cameras, not more polish on one.** Write 3–5 more configs matched to other
   wave-pool clips (camera fit from near-tube vs far-person size, crowd density), so a batch is
   diverse in content: the factor the literature says transfers.
2. **Kaggle throughput test.** Port `synthetic/` (Linux Blender + `synthetic/mpfb/`), render the
   b13 config with a 4-frame clip on a T4, measure seconds per frame.
3. **Downstream test (the one that decides it).** Render a few hundred frames; pretrain a head or
   person detector on them; fine-tune on the real labelled frames; evaluate on held-out clips.
   Arms: COCO-only fine-tune, synthetic-pretrain fine-tune, synthetic only. Real-data learning
   curve (10 / 25 / 50 / 150 frames), synthetic size sweep (100 / 300 / 1000), several seeds.
   Keep KID/AUC per variant as a logged diagnostic and see whether they track AP.
4. **Only if 3 shows an appearance gap:** CUT/FastCUT translation trained on native-resolution
   crops (real pool frames vs renders), checked with the detector match before/after, by size band.

## Kaggle port (10-19, in progress)

- `synthetic/kaggle/timing_kernel.py` + `kernel-metadata.json` (private, T4, internet on): downloads
  Linux Blender 4.2.1, renders the b13 config (seed 2, 4-frame clip) and a plain frame (seed 3),
  runs post, writes `timings.json`. Inputs come from the private dataset `npstone/hk-synth`
  (scene.py, humans.py, post.py, ref_6f5c.json, `humans_packed.blend`: textures shrunk to 1024 px
  and packed, 94 MB, made by `synthetic/blender/pack_lib.py`).
- **Mac M1 baseline** (packed library, same command): build 22 s, ~80 s per 1920 px frame at
  32 samples, so ~5.8 min for one labelled frame with a 4-frame water clip (~1.7 min without a clip).
- **Kaggle 2x T4 (OptiX, both GPUs used):** ~27 s per 1920 px frame at 32 samples (3x the M1),
  scene build 36-47 s, segmentation pass ~9 s, Blender download 26 s. One labelled frame with a
  4-frame water clip = 158 s wall; a plain labelled frame = 86 s wall. At that rate 30 GPU-hours buys
  roughly 650-700 clip frames or ~1,250 plain frames (more if several frames share one Blender
  session and scene build).
- The background ("Save & Run All") job queued for ~3 h before running; interactive sessions start
  faster.
- **Kaggle vs Mac, same seed:** labels and segmentation identical (286 people, 0 px difference,
  100% seg agreement). Detector recall matches (YOLO 59% / 19-22%, RT-DETR 90-92% / 70%), but the
  OptiX render has darker water (Lab L 125 vs 139), fainter floor caustics and lighter tube shadows,
  and RT-DETR fires ~30% more false boxes (411 vs 308). **Calibrate exposure/water on Kaggle itself**
  (score one Kaggle frame against the real reference) before rendering a large batch.

### Backend calibration (10-19)

Same frame (seed 2, t = 0) rendered three ways:

| Backend | water L | frame L | floor shadows / bright floor line |
|---|---|---|---|
| CPU (Cycles reference, M1) | 148.0 | 183.5 | yes |
| Metal (M1 GPU, where the look was tuned) | 156.8 | 189.9 | yes |
| OptiX (Kaggle T4) | 147.4 | 171.2 | mostly missing |

- Metal renders this scene about 6% brighter than the CPU reference; OptiX matches the CPU on water
  brightness but loses most shadows cast through the water onto the floor.
- **Brightness fix:** `post.py` now picks its gain by the backend recorded in each frame's labels
  (Metal 0.90, CPU/OptiX/CUDA 0.96). On the Kaggle OptiX frame this gives YOLO 60% / 23% (25 / 5
  false), RT-DETR 93% / 68% (355 / 50 false), close to the Mac frame and the real targets.
- **Shadow fix: render with CUDA, not OptiX.** Same frame on Kaggle with CUDA: mean difference from
  the CPU reference 3.2 levels (OptiX 18.9, Metal 7.8), floor shadows and the bright floor line back,
  water L 151.9 vs CPU 148.0. Same speed on 2x T4 (32.5 s vs 34.2 s). `scene.py` now tries CUDA first.
- **Kaggle CUDA frame vs the real frame:** YOLO 64% / 22% (real 65% / 23%), false 32 / 9 (25 / 3);
  RT-DETR 91% / 70% (real 89% / 70%), false 309 / 44 (289 / 44). KID 27, classifier AUC 0.995.
  The closest detector-level match so far, so Kaggle batches can start from this setup.

## Varied batch (10-19 evening): six reference cameras, ~300 frames

Nick's call: render ~300 varied frames now; drowning sequences later. The September rule that
synthetic data is only for tuning rules is dropped: realistic synthetic sequences are fair game for
training and evaluation later.

- **Six reference cameras**, each fitted against its own hand-counted real frame with
  `synthetic/tools/fit_preview.sh`: 6f5c (left deck, the original), dc93 (beach end looking back at
  the wave-machine wall, deep blue), 1d50 (low, across a field of blue tubes, hazy), 9992 (right
  lifeguard deck, fish-eye), 644c (high left stand, fish-eye), 8c83 (left deck, deep blue, fish-eye).
  bffdeb793010 was skipped (same view as 6f5c), and the Jupiter Reef frames (on hold, not wave pools).
- **New in the pipeline:**
  - fish-eye lens in post (`camera.lens_k`), with labels and masks moved to match, approved by
    Nick on a preview;
  - a head-only mask pass (vertices within 1.3 head radii of the head centre), so head boxes cover
    only the visible part;
  - `export_yolo.py` (person = 0, head = 1, plus a JSON sidecar with id, pose and submersion);
  - a per-frame `vary` block (camera position, aim, field of view, crowd size);
  - crowd `y_min` / `y_max`, `tube_colors`, `trees_behind`.
- **Per-reference check** (one exact frame each; detectors vs that reference's real frame):

| Reference | Real: YOLO @0.05 / @0.25, RT-DETR @0.05 / @0.25 | Render: same |
|---|---|---|
| dc93 | 82% / 27%, 100% / 100% | 91% / 48%, 100% / 93% (people slightly too close; camera moved back 2 m after) |
| 1d50 | 66% / 40%, 100% / 92% | 80% / 26%, 100% / 89% |
| 9992 | 42% / 20%, 96% / 94% | 56% / 20%, 100% / 93% |
| 644c | 76% / 31%, 99% / 92% | 75% / 35%, 100% / 86% |
| 8c83 | 55% / 20%, 100% / 100% | 53% / 25%, 100% / 93% |

- **Kaggle batch:**
  - two private notebooks, `hk-synth-batch-a` (6f5c, dc93, 9992) and `-b` (1d50, 644c, 8c83), each
    3 configs × 50 frames;
  - 4-frame water clips, CUDA, one Blender worker per T4;
  - the render guard stops at 9.5 h so post and export finish inside Kaggle's 12 h limit.

## First head-detector test (10-20): trained on synthetic only, scored on real frames

- **Data:**
  - batch c re-rendered after fixing the deck clutter, which had put people and lifeguards in
    the water in 297 of the first 300 frames;
  - 150 frames (6 cameras × 25) → 810 training tiles of 640 px (17,313 head boxes) and 90
    validation tiles from held-out frames;
  - the first 300-frame batch is set aside because of that bug.
- **Training:** Kaggle 2× T4, private notebook `hk-synth-train`, about 1 h.
  - **YOLO11n:** COCO weights, backbone frozen (10 layers), 60 epochs. Synthetic validation
    mAP50 0.91.
  - **RT-DETRv2-R18:** COCO weights, 40 epochs. Frozen backbone: best validation loss at epoch 24.
    Backbone at 0.1× learning rate: best at epoch 7, then overfit.
- **Test:** 10 real hand-counted frames (736 swimmers), 640 px tiles, "fair" head rule, compared
  at matched false boxes (`synthetic/train/score_heads.py`):

| Model | Found @ 20 false | @ 50 | @ 100 |
|---|---|---|---|
| Off-the-shelf `11n-head` (real crowd photos, AGPL) | 10.2% | 14.5% | 19.2% |
| **YOLO11n, synthetic only** | 10.1% | **22.0%** | **36.8%** |
| RT-DETRv2, backbone frozen | 4.1% | 11.4% | 22.0% |
| RT-DETRv2, backbone 0.1× | 2.7% | 5.3% | 13.6% |

- **Reading:**
  - A detector trained only on renders, never on a real pool frame, finds about twice as many
    real swimmers as an off-the-shelf head detector at 100 false boxes, and matches it at 20.
    On the busiest frame (6f5c) it finds 89 of 227 with 24 false boxes; most misses are tiny
    far heads.
  - RT-DETR underperformed. Likely causes are a recipe that's too short (the authors fine-tune
    for 72+ epochs), many tiny objects per tile, and 150 frames being too few for a
    transformer decoder.
  - These are one run each, no seeds yet.
- **Next:**
  1. Label about 40 real frames with head boxes and run the real comparison: COCO → real vs
     COCO → synthetic → real.
  2. A longer RT-DETR recipe (more epochs, unfrozen backbone with warm-up) on the full batch.
  3. Several seeds per arm.
  4. Render the other 150 frames next week.

### In range, and a better RT-DETR recipe (10-20)

- **In range** (`score_heads.py --in-range`, the 10-16 rule): only swimmers inside the clip's
  hand-drawn zone whose expected body height there is at least 60 px count (151 of 736).
  Boxes centred outside the zone are ignored. Far swimmers are out of scope, as Nick
  pointed out.
- **RT-DETR recipe v2:** scale and crop jitter (0.5–1.5×), 2-epoch learning-rate warm-up, EMA of
  the weights (0.999), 100 epochs; backbone frozen vs 0.1× (about 1.2 GPU h on 2× T4).

| Model | In range @ 20 / 50 / 100 false | All frames @ 20 / 50 / 100 |
|---|---|---|
| Off-the-shelf `11n-head` | 31.8 / 39.7 / 39.7% | 10.2 / 14.5 / 19.2% |
| YOLO11n, synthetic only | 35.1 / 56.3 / 60.9% | 10.1 / 22.0 / 36.8% |
| RT-DETRv2 recipe v1, frozen | 15.9 / 30.5 / 49.7% | 4.1 / 11.4 / 22.0% |
| **RT-DETRv2 recipe v2, frozen, best (epoch 21)** | **39.1 / 55.6 / 75.5%** | 6.1 / 16.7 / 25.3% |
| RT-DETRv2 recipe v2, frozen, last (epoch 100) | 8.6 / 16.6 / 18.5% | 1.9 / 3.7 / 6.4% |
| RT-DETRv2 recipe v2, backbone 0.1×, best (epoch 15) | 13.9 / 31.1 / 49.0% | 0.0 / 6.0 / 10.2% |

- **Reading:**
  - With augmentation, warm-up and EMA, RT-DETR (Apache) matches or beats the synthetic-trained
    YOLO in range. Far, out-of-range heads still favour YOLO. *(Corrected 10-23: this was one
    lucky run; see the seed spread below.)*
  - **Training longer on synthetic data hurts real-world results**: the same run collapses from
    75.5% to 18.5% between epoch 21 and epoch 100. The model drifts toward the synthetic look.
    Checkpoint choice matters more than any other knob here. It needs a real validation signal
    (a few real labelled frames), or a fixed early stop, rather than luck with synthetic
    validation loss.
  - Unfreezing the backbone hurts in both recipes; keep it frozen at this data size.
  - RT-DETR's scores stay below 0.25, so its operating threshold must be set on real validation
    data.
  - Still one test set of 151 in-range swimmers (one swimmer = 0.7 points) and one run each.

### Seed spread (10-23): the 75.5% was a lucky run

- **Run:** recipe v2, backbone frozen, seeds 1–8. Each seed trains the first 30 epochs of the same
  100-epoch schedule, so up to epoch 29 it follows the 10-20 run except for the random start. The
  weights are kept every 5 epochs, plus `best` (lowest synthetic validation loss) and `last`.
  Kaggle `hk-synth-train` version 3, 2× T4, about 1.5 GPU h. New trainer flags: `--seed`,
  `--stop-after`, `--save-every`.
- **Test:** as above, in range only (151 swimmers), all 56 checkpoints. Re-scoring the 10-20
  checkpoint first gave the same 39.1 / 55.6 / 75.5%.

Mean ± standard deviation over the 8 seeds, with the range at 100 false boxes:

| Checkpoint | In range @ 20 false | @ 50 | @ 100 |
|---|---|---|---|
| epoch 5 | 8.4 ± 9.5 | 16.1 ± 15.5 | 24.2 ± 20.4 (1.3–51.0) |
| epoch 10 | 15.8 ± 11.7 | 30.6 ± 17.6 | 41.5 ± 21.3 (0.7–62.9) |
| epoch 15 | 21.8 ± 10.2 | 38.7 ± 15.9 | 49.6 ± 19.7 (4.0–66.2) |
| epoch 20 | 25.8 ± 11.2 | 39.5 ± 15.1 | 52.8 ± 16.7 (15.9–66.9) |
| epoch 25 | 25.3 ± 11.5 | 41.4 ± 14.2 | 53.1 ± 15.7 (24.5–68.2) |
| last (epoch 29) | 24.7 ± 11.6 | 40.5 ± 15.7 | 52.3 ± 16.1 (25.8–66.9) |
| **best (synthetic val loss)** | **24.8 ± 9.6** | **40.4 ± 13.9** | **53.6 ± 15.4 (17.9–64.2)** |
| *10-20 run, best (epoch 21)* | *39.1* | *55.6* | *75.5* |

Per seed, in range at 100 false boxes (%):

| Seed | ep 5 | ep 10 | ep 15 | ep 20 | ep 25 | last | best (its epoch) |
|---|---|---|---|---|---|---|---|
| 1 | 1.3 | 19.2 | 44.4 | 55.6 | 60.3 | 64.9 | 57.6 (19) |
| 2 | 4.0 | 53.6 | 66.2 | 66.9 | 68.2 | 65.6 | 62.9 (18) |
| 3 | 3.3 | 0.7 | 4.0 | 15.9 | 24.5 | 25.8 | 17.9 (22) |
| 4 | 51.0 | 62.9 | 58.3 | 61.6 | 66.2 | 66.9 | 64.2 (21) |
| 5 | 14.6 | 37.7 | 62.3 | 62.3 | 56.3 | 53.6 | 60.3 (22) |
| 6 | 41.7 | 51.7 | 49.0 | 43.0 | 35.8 | 32.5 | 49.0 (15) |
| 7 | 39.1 | 55.6 | 57.6 | 63.6 | 64.9 | 63.6 | 63.6 (19) |
| 8 | 38.4 | 51.0 | 55.0 | 53.6 | 49.0 | 45.7 | 53.6 (20) |

- **Reading:**
  - The 10-20 result is higher than all 56 checkpoints here (the highest is 68.2%). This recipe
    typically finds 53.6 ± 15.4% of in-range swimmers at 100 false boxes (median of the 8 `best`
    checkpoints: 59.0%).
  - RT-DETR vs the synthetic-only YOLO (60.9%, one run): no clear winner. YOLO needs seeds too
    before either is called better.
  - The random seed matters as much as the epoch. Across seeds the spread is about 15 points,
    roughly four times the test set's own noise (a binomial standard deviation of about 4 points
    at 151 swimmers). Seed 3 fails at every epoch.
  - No collapse within 30 epochs: the mean is flat from epoch 20 to 29 (52–53%). A fixed stop at
    epoch 20–25 does as well as the synthetic-validation-loss pick, which always lands in epochs
    15–22. The drop to 18.5% seen on 10-20 comes later than epoch 30.
  - Synthetic validation loss can't tell a good seed from a bad one: seed 3's best validation
    loss (5.50) sits inside the good seeds' range (5.33–5.54). Picking a seed or checkpoint needs real labelled
    frames, kept separate from these 10 test frames.
  - Caveat: Kaggle installs the newest `transformers` on each run, so this run may use a newer
    version than 10-20. A library effect can't be ruled out, but 75.5% sitting above the whole
    spread is what a lucky draw looks like.
- **Next:**
  1. A few real validation frames (not the 10 test frames) to choose the seed, the checkpoint
     and the operating threshold.
  2. Seeds for the synthetic-only YOLO before comparing the two detectors again.
  3. From now on, report head-detector numbers as mean ± standard deviation over seeds.

### YOLO seed spread (10-24): synthetic-only YOLO is steadier than RT-DETR

- **Run:** the 10-20 YOLO11n recipe (COCO weights, backbone frozen, 60 epochs, patience 15),
  seeds 1–8, weights every 10 epochs plus `best` (synthetic validation mAP) and `last`.
  Kaggle `hk-synth-train` version 4, about 40 min on 2× T4. New trainer flags: `--seed`,
  `--save-period`. Re-scoring the 10-20 weights first gave the same 35.1 / 56.3 / 60.9%.
- **Test:** as above, in range (151 swimmers), all 56 checkpoints.

Mean ± standard deviation over 8 seeds (range at 100 false boxes):

| Checkpoint | In range @ 20 false | @ 50 | @ 100 |
|---|---|---|---|
| epoch 10 | 35.3 ± 4.6 | 57.3 ± 2.7 | 61.4 ± 2.4 (58.9–66.2) |
| epoch 20 | 39.2 ± 4.8 | 61.5 ± 2.4 | 64.6 ± 2.3 (61.6–67.5) |
| epoch 30 | 38.5 ± 5.4 | 58.2 ± 3.3 | 61.9 ± 2.3 (58.3–64.9) |
| epoch 40 | 41.4 ± 6.1 | 59.9 ± 3.0 | 62.8 ± 2.1 (60.3–65.6) |
| epoch 50 | 42.2 ± 3.8 | 59.0 ± 2.0 | 62.3 ± 2.2 (58.9–66.2) |
| last | 42.3 ± 3.2 | 59.0 ± 2.0 | 62.1 ± 1.8 (59.6–64.9) |
| **YOLO best** | **42.1 ± 5.5** | **59.1 ± 1.7** | **62.1 ± 1.9 (59.6–64.2)** |
| *RT-DETR best (seed spread above)* | *24.8 ± 9.6* | *40.4 ± 13.9* | *53.6 ± 15.4 (17.9–64.2)* |

- **Reading:**
  - Trained on synthetic frames only, YOLO11n finds more in-range swimmers than RT-DETRv2 at
    every false-box budget, and its spread across seeds is about 2 points against
    RT-DETR's 15. No YOLO seed failed.
  - Recall is reached by epoch 10 and stays flat to epoch 60, so checkpoint choice barely
    matters for YOLO here.
  - This is a data point only, not a model choice (Nick, 10-24). Both detectors will be
    compared trained on a mix of real and synthetic data, which is the setting that counts.
    RT-DETR (Apache) stays the commercial path; YOLO is AGPL.
