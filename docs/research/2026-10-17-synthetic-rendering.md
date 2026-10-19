# Rendered training data: who has done it, and which engine

Short research session, 2026-10-17, after Nick asked to look into game-engine data (see the
synthetic-data part of section 6 in `2026-10-17-research-sweep.md`). It reopens the
2026-09-20 decision to set synthetic *video* aside, because the bottleneck has since moved to
**detection**: the submerged victim scored 0.02 box confidence, and YOLO11n tiled misses 37%
of hand-counted swimmers even at confidence 0.05.

## 1. Prior work

**No one has published rendered swimmers in a pool that were shown to help on real pool
footage.** The closest work:

| Work | Engine | What it is | Real-world evidence |
|---|---|---|---|
| [SwimXYZ](https://arxiv.org/html/2310.04360v1) (2023) | Unity, custom water (~50 parameters: colour, depth, refraction, reflection, waves; 60 to 80 splash particles per stroke; wet-skin effect) | 3.4 M frames of single competitive swimmers with 2D/3D joints, five camera views; 240 SMPL stroke motions | ViTPose fine-tuned on 5k synthetic frames improved PCK on **synthetic** test images (PCK5 46 → 71). On real web images: "seem satisfactory", **no numbers** |
| [SynBASe](https://www.dlsi.ua.es/~jgallego/datasets/synbase/) (Martinez-Esteso et al., 2025, [EAAI](https://www.sciencedirect.com/science/article/pii/S0952197624017445)) | Unreal Engine 4 | 1,295 drone-view images, 3,415 bodies in open sea (1 to 12 per image), plus fog, rain, low light, sunset glare | Paper reports synthetic + real helps at-sea detection; synthetic-trained model also produced extra false positives in the water |
| [Progressive fine-tuning of YOLOv9 for maritime SAR](https://www.researchgate.net/publication/389446908_Optimizing_Object_Detection_for_Maritime_Search_and_Rescue_Progressive_Fine-Tuning_of_YOLOv9_with_Real_and_Synthetic_Data) (2025) | (synthetic sea data) | YOLOv9t, mixing ratios | Synthetic pretraining + 10% real beat real-only by 13.7%; best mix was 70% real. (Search-result summary; the PDF didn't parse, numbers unverified) |
| [SeaDronesSee](https://arxiv.org/pdf/2105.01922) (WACV 2022) | real | drone benchmark, swimmers in open water | Names the hard parts: **reflections, shadows, waves and foam**, the same as our glare and wave-pool swell |
| [PeopleSansPeople](https://huggingface.co/papers/2112.09290) (Unity, 2021) | Unity | synthetic humans for person detection/keypoints | Synthetic pretraining: keypoint AP 6.4 → 44.4 with **few** real labels, only +1.5 with plenty of real data |

What this says for us:
- **Synthetic data helps most when real labels are scarce**, which is our situation (one
  labeler, 10 hand-counted frames, 57 rescues).
- **The standard recipe is pretrain on synthetic, then fine-tune on real** (or mix, with real
  in the majority). Synthetic-only models transfer poorly and tend to add false positives
  in the water.
- **Nobody rendered crowded wave pools, tubes, or heads partly under water.** We'd be first,
  so there's no proven recipe, and also a real portfolio contribution if it works.

## 2. Engines on our hardware (8 GB M1)

| | Blender + BlenderProc | Unreal Engine 5 | Unity Perception |
|---|---|---|---|
| Status | BlenderProc 2.8 (Oct 2024, repo active to Jan 2026), pins Blender 4.2.1, **ships a macOS arm64 build** ([installer](https://github.com/DLR-RM/BlenderProc/blob/main/blenderproc/python/utility/InstallUtility.py)) | Native Apple Silicon editor since 5.1, but reports say it needed a source build, Nanite unstable, Lumen contested on M1 ([gamefromscratch](https://gamefromscratch.com/running-unreal-engine-5-1-on-apple-m1-m2-hardware/), [AppleGamingWiki](https://www.applegamingwiki.com/wiki/Unreal_Engine_5)) | **Discontinued**: README says no longer supported by Unity ([repo](https://github.com/Unity-Technologies/com.unity.perception)) |
| Memory | fine for small scenes | 8 GB is Epic's minimum, 16 GB recommended | n/a |
| Labels | built in: boxes, segmentation masks, depth, per-object ids, all from Python | add-ons: [UnrealCV](https://astro.paperswithcode.com/paper/unrealcv-connecting-computer-vision-to-unreal) / UnrealZoo (UE 5.4), [3D bounding-box plugin](https://github.com/Plasma-Lab/Unreal-3D-Bounding-Box-Plugin) | was built in |
| Water | Ocean modifier + Cycles path tracing (true refraction, caustics); slow per frame | Water plugin + Lumen; real-time, prettiest swells and splash, best humans (MetaHumans) | custom (SwimXYZ built its own) |
| Scripting | plain Python, fits this repo | Blueprints / C++, Python editor scripting | C# |
| Render speed | Cycles on M1 GPU ~1.2 to 1.7× CPU ([Blender](https://www.cgchannel.com/?p=134471)); can also render headless on Colab | real-time once the scene runs | n/a |
| License | BlenderProc GPL-3.0 (a tool; the images are ours) | free under Epic's EULA | Apache-2.0 |

## 3. Recommendation

**Blender + BlenderProc first.** It runs on the machine we have, it's Python, and every
label we'd want (head box, how much of the head is under the waterline, which person, the
exact frame) falls out for free. Path-traced refraction matters for heads at the waterline.
Unreal is the stronger engine for crowds and splash, and a good later step if the Blender
spike shows the idea works but the realism isn't enough. It also needs a bigger machine.

**Ordered plan:**
1. **Realism spike (about an afternoon).** One wave-pool scene from a raised camera: water
   with swell, a few tubes, 5 to 10 simple human figures with heads at random depths (fully
   out down to 80% under). Render around 200 frames at 1080p with head boxes and a
   "fraction submerged" value.
2. **Measure before generating more.** Run the current YOLO11n tiled pipeline on them and
   compare its **recall versus head size** with the real curve (FINDINGS §4). If synthetic
   heads are found much more easily than real ones of the same size, the render is too clean.
   Fix glare, swell and blur first.
3. **Then the real test.** Generate a few thousand randomized frames, pretrain or mix into the
   head-detector fine-tune, and judge **only** on the 736 hand-counted real swimmers at
   matched false boxes. Keep it if recall goes up.
4. **Later:** short scripted scenes for tracker stress tests (submerge and resurface with
   exact timing). Distress-behaviour animation stays last and is never used as evidence of
   real-world recall.

**Copy-paste is the cheap baseline to beat:** pasting real head crops (with a band of
surrounding water) into real pool frames ([Kisantal et al.](https://arxiv.org/pdf/1902.07296)).
If rendering can't beat that on the real benchmark, it isn't worth the effort.

## 4. Spike log (2026-10-17)

Everything lives in `scratch/synth/` (gitignored). Plain Blender 4.2.1 (arm64) runs the
scripts headless; BlenderProc was dropped because macOS blocks writes inside the Blender
app bundle, where BlenderProc installs its Python packages.

| File | What it does |
|---|---|
| `wavepool.py` | v1 scene: water box with swell + ripple bump, pool, deck, tubes, primitive "mannequin" people, sky; several camera views per scene; writes `rgb_*.png`, `seg_*.npy` (person id per pixel), `labels_*.json` |
| `wavepool_v2.py` + `humans.py` | v2: procedural humans, see-through vinyl tubes, deck clutter, crest foam and swimmer splash |
| `post.py` | "video look": softness, saturation, sensor noise, 4:2:0 chroma, JPEG blocking; geometry never moves, so labels stay exact |
| `score_synth.py` | runs the production detector (YOLO11n-pose, tiled, imgsz 1280) and scores with the same exclusive match rule and size bands as the real 736-swimmer benchmark |
| `mosaic.py` | quick contact sheets |

**How labels are made.** Each person's head centre is known in 3D and projected into the
image. A second render replaces every material with a flat colour that encodes the person's
id (red = id mod 256, green = id div 256; 1 sample, no dithering, no filter blur), with the
water made opaque black, so a pixel belongs to a person only where they are visible above
the water. Ground truth = people with at least 6 visible pixels. "Fraction of head under
water" comes from the head's height against the analytic wave height at its position.

**Results so far**

1. *v1 mannequins, 100 frames, 7,400 visible people.* Recall per size band matched the real
   curve closely in 30–80 px (e.g. 45–60 px: 75% vs 71% real at conf 0.05), but partly for
   the wrong reason: the detector missed large, clear mannequins because they don't look
   human, and false boxes were ~2 per frame vs ~9 real (no deck clutter). The useful new
   number: recall falls from 61% (head fully out) to 43% (30–60% under) to 25% (60–100%
   under) at conf 0.05.
2. *v2 procedural humans, clutter, splash, 2 frames, 153 visible swimmers* (small sample,
   read as direction only). The detector now finds near, clear people reliably, and misses
   the far, flat-swimming, splash-covered and low-in-the-water ones, the same pattern as real
   footage. Clean renders became *easier* than real at conf 0.25 (60–80 px: 84% vs 36% real);
   the video-look pass pulls that back toward real (63%), and false boxes rose to ~6 per frame
   (real ~9) thanks to deck people, posts and floats.

   | Size band | Real (0.05 / 0.25) | v2 clean | v2 + video look |
   |---|---|---|---|
   | 45–60 px | 71% / 25% | 71% / 44% | 64% / 31% |
   | 60–80 px | 85% / 36% | 95% / 84% | 79% / 63% |

   Next realism gaps: the post-process strength should be tuned against real frames, and
   the 60–80 px band is still too easy at 0.25 (real swimmers there are often half-hidden by
   waves and other people).

**What made the biggest visual difference, in order:** human silhouettes (Skin-modifier
bodies with shoulders, neck, elbows, hands; heads with nose, ears, brows; wet glossy skin
and hair); murky turquoise water (absorption + light scatter volume); see-through vinyl
tubes (thin shell, transmission 1, alpha 0.55); splash as fine low spray + lacy foam ring
(big droplets read as popcorn, a dense mist volume reads as a white dome); hiding the water
box's side faces behind the pool walls.

**Cost.** About 20 s per 1080p frame at 12 samples on the M1 GPU for v1; v2 at 128 samples
is roughly 3–4 minutes per frame plus about a minute to build each scene. Training-scale
data (thousands of frames) belongs on a Colab GPU or overnight runs at low samples, since
the denoiser and the video-look pass hide most of the noise.

## 5. Making it a pipeline: what to automate

Ordered by value for the head-detector goal.

1. **One config per scene, sampled from distributions.** Pool size, swell height and period,
   water murk, sun angle, crowd size, pose mix, splash rate, camera height/FOV/tilt and
   post-process strength all become ranges in one YAML. Each frame stores its config, so
   any result can be traced to the settings that made it (and bad settings can be cut).
2. **Calibrate the distributions to real footage automatically.** Fit the camera height and
   tilt per real clip from the zone polygon and the perspective size fit we already compute;
   match crowd density to the hand-counted frames; match water colour by comparing colour
   histograms of real and rendered water. Then sample renders around real cameras, not
   arbitrary ones.
3. **A realism gate before training.** For each batch, run the detector and compare its
   recall-by-size and false-box rate with the real benchmark (this spike's comparison, as a
   script). A batch that's far too easy, too hard, or has the wrong false-positive rate is
   flagged before it pollutes training.
4. **Head boxes for free.** Export a tight box around each visible head (head-only segmentation
   pass), plus the fraction submerged, in YOLO format; that's exactly what the head-detector
   fine-tune needs, including the half-submerged heads nobody can label by hand.
5. **Hard cases on purpose.** Over-sample what real data lacks: heads 30–90% under water,
   distressed poses (head back, arms pressing down), occlusion by tubes and other swimmers,
   glare. Keep a ratio cap so the model still sees mostly ordinary swimmers.
6. **Short clips, not just frames.** Animate the swell, bob heads with the wave, and script a
   victim who slips under and resurfaces with exact timing. That tests tracking and the ghost
   stage with perfect ground truth (the "tracker stress test" job).
7. **Render farm.** The scripts already run headless; a Colab cell (Linux Blender + CUDA)
   can render a few hundred frames per session to Drive while the Mac does other work.
8. **Domain-gap reducers that cost nothing at render time:** the video-look pass, random
   crops and rescales, and optionally an image-to-image style model trained on real pool
   frames (later; the cheap steps first).

**Asset upgrade path** if procedural humans stop being enough: MakeHuman / MPFB (code GPL,
core assets CC0, swimwear packs CC-BY, so commercially usable with a credit line). Avoid SMPL
bodies: their licence is non-commercial.
