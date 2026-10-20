# Research sweep 2: architecture, synthetic data, Jetson, labels (2026-10-17)

Four parallel research agents, about 25 minutes each, each given the project context. Their
sources are linked inline. Licences marked **verified** were checked against the repository's
LICENSE file or the GitHub API on 2026-10-17; everything else is as the agents reported it.

## The five findings that change plans

1. **Decide the licence path first.** Ultralytics treats any shipped product built on its
   code *or on weights you trained with it* as AGPL, unless you buy an Enterprise licence
   (quote only) ([ultralytics.com/license](https://www.ultralytics.com/license)). NVIDIA's
   DeepStream licence forbids making it subject to a FOSS licence, so AGPL code inside a
   DeepStream app is a direct conflict ([DeepStream EULA](https://developer.nvidia.com/deepstream-eula)).
   Permissive detectors that run on Jetson exist (table below).
2. **There is no commercially clean pretrained head detector.** CrowdHuman, CroHD/HT21 and
   the public head checkpoints trained on them are non-commercial
   ([CrowdHuman](https://www.crowdhuman.org/download.html),
   [HT21](https://motchallenge.net/data/Head_Tracking_21/)). Fine-tune from COCO weights on our
   own pool frames; use the head datasets only for research comparisons.
3. **Our false-alarm budget is far looser than the market, and we can't even measure it.**
   Lynxight (Fluidra) reports 2 to 3 alerts per pool per *day*, one YMCA reports 3 to 4
   ([Lynxight](https://www.lynxight.com/content/what-should-fitness-centres-ask-ai-drowning-detection-vendors)).
   Our working budget of 10 to 20 per *hour* is about 100x looser. And with 0.27 h of normal
   footage, even zero false alarms only bounds the rate at about 11 per hour (rule of three:
   3.0 / hours). **Hours of normal footage are the binding constraint**, ahead of any model work.
4. **For the head detector, copy-paste beats rendering on expected value.** Pasting real head
   and swimmer crops into real wave-pool frames keeps the real camera, glare, compression and
   water. Reported gains: about +10 AP at 10% of COCO and +5 AP on all of it
   ([Simple Copy-Paste](https://ar5iv.labs.arxiv.org/html/2012.07177)); +7.1% relative on small
   objects ([Kisantal](https://arxiv.org/abs/1902.07296)). Renders help as *pretraining then
   real fine-tuning*, typically 5 to 13% relative when real data is scarce. After fine-tuning,
   graphics quality stopped mattering (DeepGTAV low 45.1 vs high 45.0); matching camera angle
   and time of day did matter ([DeepGTAV](https://arxiv.org/pdf/2112.12252)).
5. **On the Nano, TensorRT is mandatory and Python overhead is the real bottleneck.**
   - **GPU time per 640 image** (YOLOv8n): PyTorch 20.9 ms, TensorRT FP16 4.4 ms, INT8 3.5 ms
     ([benchmarks](https://github.com/hokwangchoi/jetson-orin-nano-benchmarks)).
   - **End to end in Python** was about 20 fps. Pre- plus post-processing cost about 10 to 13 ms
     per call ([NVIDIA forum](https://forums.developer.nvidia.com/t/low-performance-jetson-orin-nano-super/318862)).
   - **Our pipeline (estimate, scaled by pixel count):** 8 zone tiles plus a whole frame at 1280
     is about 40 to 45 ms of GPU per frame, comfortably fast enough at 5 to 10 fps. That holds
     only if the tiles run as **one batched call**.

## Model architecture

**Detectors**

| Detector | Licence | Note for us |
|---|---|---|
| YOLO11n / YOLO26n (Ultralytics) | AGPL-3.0 or paid | YOLO26 adds STAL (more positive labels for small objects); Nano Super TensorRT FP16 4.6 ms at 640 ([docs](https://docs.ultralytics.com/guides/nvidia-jetson/)) |
| RF-DETR Nano / Small / Medium | Apache-2.0 **verified** | 48.4 AP at 384 px; about 30M params; low native resolution needs tiling ([repo](https://github.com/roboflow/rf-detr)) |
| RF-DETR Atto / Femto / Pico / XL | **PML 1.0** (verified in README) | avoid |
| D-FINE | Apache-2.0 **verified** | strong small DETR; Nano speed not found |
| DEIM (v1) | Apache-2.0 **verified** (LICENSE text) | |
| **DEIMv2** | **non-commercial** **verified** | explicitly bans commercial hardware; avoid |
| RT-DETR / RT-DETRv2 (lyuwenyu) | Apache-2.0 **verified** | |
| YOLOX | Apache-2.0 **verified** | proven on Jetson |
| RTMPose / RTMO (MMPose) | Apache-2.0 **verified** | permissive pose option |
| YOLOv7 / YOLOv9 | GPL-3.0 | avoid for a closed product |
| YOLOv10 | AGPL-3.0 | avoid |

- **A stride-4 "P2" head is the best-supported single change for tiny objects.** It raised
  AP_small 0.047 → 0.070 (+49%) in one VisDrone A/B, but cost speed on the Nano (41 vs 31 ms)
  ([HF](https://huggingface.co/LibreYOLO/LibreYOLO9P2s-visdrone),
  [paper](https://arxiv.org/pdf/2602.13378)).
- **For a fixed camera, a static per-zone tile plan** (tiles sized to the expected head pixels,
  dropping tiles where heads are already large) gets most of what dynamic tiling offers.
  DeepStream's `nvdspreprocess` takes static ROIs per camera and batches them into one tensor
  ([docs](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvdspreprocess.html)).
  The cross-tile merge stays our own code.

**Tracking through submersion**

- **`roboflow/trackers`** (Apache-2.0 **verified**, already in use) ships ByteTrack, OC-SORT,
  BoT-SORT and C-BIoU, so swapping trackers is cheap. DanceTrack HOTA: ByteTrack 53.3, OC-SORT
  54.1, C-BIoU 56.7, BoT-SORT 57.8 ([repo](https://github.com/roboflow/trackers)).
- **Hybrid-SORT** (MIT **verified**) adds detection-confidence and box-height cues, which fits
  a head whose confidence falls as it goes under. DanceTrack HOTA 62.2 vs OC-SORT 54.6
  ([repo](https://github.com/ymzis69/HybridSORT)).
- **Appearance re-ID is weak on 10–30 px heads.** The practical answer is our own ghost state:
  freeze the position, widen the search radius over time, and re-associate by location plus
  head size (agent inference; no paper tests submersion).
- **StrongSORT's AFLink and GSI run offline.** Use them to build labels and features, never as
  the live mechanism ([paper](https://arxiv.org/abs/2202.13514)).
- **BoxMOT is AGPL.** Use its benchmark, don't ship its code.

**Scoring with few positives**

- **Gradient-boosted trees:** trees beat deep nets on tabular data of this size
  ([Grinsztajn](https://arxiv.org/abs/2207.08815)).
- **Reiss–Hoshen-style k-NN** on a few interpretable per-head attributes
  ([paper](https://arxiv.org/abs/2212.00789)).
- **MiniRocket** as a cheap time-series baseline ([paper](https://arxiv.org/abs/2012.08791)).
- **TabPFN v2,** pinned to v2: later versions have non-commercial weights (unverified).
- **No published drowning work** combines real rescues, track-level labels and whole-clip
  splits. Ours would be novel.

## Synthetic data and Blender

- **Synthetic-only training fails for people in water.** Maritime SAR human class: mAP50 0.012.
  Synthetic pretraining helps with little real data (0.329 → 0.371 at 10% real), and the best
  result used 70% real plus synthetic pretraining (0.734 vs 0.718)
  ([SciTePress](https://scitepress.org/Papers/2025/131127/131127.pdf)).
- **Diversity and viewpoint matching matter more than photorealism.** Synthetic pools far from
  the real data, or with low pose diversity, hurt; gains flatten after about 500 to 1,000
  synthetic images ([Archangel](https://arxiv.org/html/2405.15203)).
- **Generative restyling is optional and risky.** Stock ControlNet restyling trained a *worse*
  detector than CycleGAN; segmentation-conditioned ALDM did best (mAP50 0.633 vs 0.460 raw sim)
  ([ALDM](https://arxiv.org/html/2403.11459v1)). Model licences:
  - FLUX.1-dev: **non-commercial, and bans surveillance uses.** Avoid.
  - FLUX.1-schnell: Apache-2.0.
  - SDXL: OpenRAIL++-M, commercial OK with use restrictions.
- **Diffusion inpainting into real frames:** mixed results, and box accuracy can be too poor
  to train on ([2501.08142](https://arxiv.org/pdf/2501.08142)).
- **Assets and licences:**
  - MakeHuman core assets: CC0, selling models allowed
    ([FAQ](https://static.makehumancommunity.org/mpfb/faq/can_i_sell_models.html)).
  - Reallusion Character Creator: its EULA *excludes* AI training and synthetic data.
  - Mixamo: treat as off-limits for ML.
  - SMPL / SwimXYZ bodies: non-commercial.
- **Throughput:** BlenderProc reports about 3,000 images per hour with low samples plus
  denoising. Our 3 to 6 minutes per frame is about 100x slower; use low samples, the denoiser
  and Linux GPUs.
- **Rendering is clearly worth it for tracker stress tests** (scripted submerge and resurface
  with exact ground truth), where labels matter more than realism.

## Jetson Orin Nano

- **JetPack:** JetPack 7.2.x now supports Orin (Ubuntu 24.04, CUDA 13, DeepStream 9.1; flash
  from USB, Super mode by default)
  ([NVIDIA](https://developer.nvidia.com/embedded/jetpack/downloads/archive-7.2)). Ultralytics
  validates JetPack 6.1/6.2 and recommends a native install for TensorRT export; **use JetPack
  6.2 for now.**
- **Power:** `sudo nvpmodel -m 2` (MAXN_SUPER) plus `sudo jetson_clocks`; monitor with
  `jtop`. Real draw was about 9 to 13 W.
- **Video decode:** hardware decode handles 11× 1080p30. Use GStreamer `nvv4l2decoder`, not
  `cv2.VideoCapture`.
- **uv pitfalls:** pin the Jetson torch wheel in `[tool.uv.sources]` (plain PyPI torch doesn't
  use the GPU); build the venv with `--system-site-packages` for TensorRT; keep a separate lock
  or platform markers; run from NVMe.
- **Extraction on the Nano in PyTorch:** an estimated 0.4 to 0.8 s per frame for the current
  17-pass recipe, roughly T4-class. **Benchmark 50 frames before committing.**
- **Commercial pool systems:**
  - Lynxight: software on existing overhead CCTV, 2+ angles, 2 to 3 alerts per pool per day.
  - Poseidon, SwimEye, AngelEye, Coral Manta: mostly underwater cameras, alarm 5 to 15 s after
    a body is motionless. None publishes its compute.

## Labels and evaluation

- **Pre-label heads, then correct by click.**
  - SAM 3 with a text prompt finds every matching instance
    ([paper](https://arxiv.org/html/2511.16719v2)). Its custom licence appears commercially
    usable (agent reading, not legal advice).
  - Open-vocabulary detectors collapse on small, unusual targets: OWLv2 F1 27.6% on aerial
    imagery. Cutting the vocabulary to about 3 classes helped 15x, so prompt only "head"
    ([2601.22164](https://arxiv.org/abs/2601.22164)).
  - Extreme clicking makes a box in 7 s versus 34.5 s for drawing one
    ([paper](https://arxiv.org/pdf/1708.02750)).
- **Turn victim clicks into dense tracks.**
  - Run SAM 2.1 (Apache-2.0 **verified**) or BootsTAPIR/TAPNext (Apache-2.0 **verified**)
    forward and backward between clicks. Keep a span only where both directions agree; you
    review the rest.
  - SAM 2's own limitations list crowded scenes and long occlusion
    ([repo](https://github.com/facebookresearch/sam2)).
  - CoTracker3 is CC BY-NC (**verified**): research only.
- **Evaluation protocol.**
  - Never report pooled frame-level AUC: a 2026 UCF-Crime audit shows it rewards telling video
    sources apart ([2608.11985](https://arxiv.org/abs/2608.11985)).
  - Use event-level scoring with an onset tolerance, as seizure detection does
    ([SzCORE/timescoring](https://github.com/esl-epfl/timescoring)).
  - Put exact (Garwood) intervals on false alarms per hour, bootstrap by clip, and plot latency
    against false alarms per hour.
- **Onset definition.**
  - Store onset as an interval spanning our label and Z5cc's (median gap 2.4 s), score latency
    against both, and report a "silent submersion" subset separately.
  - A video study of 52 drowning people found visible distress in only 11%. 5 of 24
    disappeared with no visible behaviour, backward "water-milling" was the most common sign,
    and nobody waved or shouted ([PMC7559254](https://pmc.ncbi.nlm.nih.gov/articles/PMC7559254/)).
- **Facility data agreements:** use FineBio-style terms. We keep the IP in models and labels
  for commercial use; they keep the raw video and can demand deletion. Offer on-site
  processing and retention limits
  ([FineBio](https://finebio.s3.abci.ai/FineBio_License_Agreement.pdf)).
- **Live pool webcams** draw data-protection objections when minors are recorded
  ([bluewin](https://www.bluewin.ch/en/news/switzerland/camera-streams-bathers-live-on-the-internet-data-protectionists-outraged-li.2696077)).
  A CC BY label on YouTube is a claim, not clearance.

## Ranked recommendations

1. **Normal footage first** (facility outreach with FineBio-style terms). Without hours of it,
   no alarm rate can be validated. Also revisit the 10 to 20 per hour budget against the market
   norm of a few per pool per day.
2. **Lock the evaluation protocol now:**
   - event-level scoring with onset tolerance
   - Garwood intervals on false alarms per hour
   - bootstrap by clip, within-clip ranking
   - random baseline
   - latency versus false alarms per hour
   - onset stored as an interval
3. **Licence decision** before building more on Ultralytics. The research baseline can stay
   YOLO11n. Benchmark D-FINE-N/S, RT-DETRv2, YOLOX-nano/tiny and RF-DETR N/S (all Apache) on
   the 736-swimmer frames and the victim clicks, at matched false boxes. Or get an Ultralytics
   Enterprise quote if switching costs too much recall.
4. **Head detector:**
   - Fine-tune from COCO weights on pool frames that SAM 3 pre-labels and you correct by click.
   - Add **copy-paste augmentation** (half-submerged heads via water-line masks).
   - Try a P2 head.
   - Use a static per-zone tile plan.
5. **Tracking:** A/B the trackers already in `roboflow/trackers`, then add a Hybrid-SORT-style
   confidence cue and an explicit ghost state. Measure fragmentation around labelled
   submersions.
6. **Scorer:** gradient-boosted trees, k-NN on per-head attributes, and MiniRocket, with
   whole-clip holdout.
7. **Synthetic:**
   - Renders become pretraining only, plus tracker stress-test clips.
   - Match camera and head size to the real cameras rather than chasing photorealism.
   - Composite rendered people onto real background plates as the middle path.
   - Cut samples and denoise.
8. **Jetson:**
   - JetPack 6.2, TensorRT FP16, one batched engine for the zone tiles.
   - 640 whole frame instead of 1280.
   - 5 fps first.
   - INT8 last.
   - Benchmark the Nano for extraction on 50 frames before splitting clips.

## Couldn't verify

- No measured pose-model or 1280-input numbers on the Nano (tile figures are pixel-scaled
  estimates).
- ISO 20380's exact requirement: "80% detection, alarm within 2 s" comes from a secondary source.
- Full text of the SynBASe transfer results; the swimmer copy-paste preprint (403).
- Ultralytics Enterprise pricing.
- The TabPFN licence change.
- The full DeepStream EULA.
- No paper evaluates rendered head detection in pools: our own A/B is the only evidence there.
