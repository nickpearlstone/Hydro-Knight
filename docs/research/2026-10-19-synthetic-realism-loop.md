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
- **Shadow fix (pending):** `scene.py --backend CUDA` on Kaggle, compared against the CPU reference.
