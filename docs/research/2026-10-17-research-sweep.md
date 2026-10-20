# Research sweep, 2026-10-17

Started 12:28. Every claim has a source link, and numbers are the sources' own.
Anything not verified past an abstract or snippet is marked *(unverified)*.
"Fit" says how directly a finding applies to Hydro-Knight: high, medium, low.

## Contents
1. Detection: small heads and swimmers
2. Tracking through occlusion and disappearance
3. Drowning behavior science and lifeguard scanning
4. Temporal and anomaly models
5. Convolution padding, edges and missing frames
6. Data: datasets, video sources, live cams, synthetic
7. Standards, commercial systems and how they are evaluated
8. Labeling efficiency
9. Compute and attention (scanning)
10. Camera geometry and distance
11. Ranked recommendations

---

## 7. Standards, commercial systems and how they are evaluated

### ISO 20380:2017, the standard for exactly this kind of system (fit: high)
"Public swimming pools: computer vision systems for the detection of drowning accidents:
safety requirements and test methods." The free preview
([iTeh sample PDF](https://cdn.standards.iteh.ai/samples/67857/66a010c70cd940dc963def72fa5773eb/ISO-20380-2017.pdf))
contains the full performance clauses:
- **Scope:** public pools of 150 m² or more. Systems detect "a solid mass, without trajectory,
  lying at the pool basin bottom", so the standard is written for the *late* stage (a body on
  the bottom), not surface distress.
- **Alarm set off time ≤ 15 s**, fixed by the vendor, shown on screen, not changeable by staff.
- **Detection rate ≥ 80 %** in the standard's detection test (section 5.3, a test person or
  mannequin placed at measured positions).
- **False alarms ≤ 5 per day per pool basin**, averaged over 30 days of public opening hours,
  outside special activities (aqua-aerobics, play structures). At 10 to 12 open hours a day
  that is **about 0.4 to 0.5 false alarms per hour**, roughly 20 to 50 times stricter than
  the 10 to 20 per hour Nick named for a surface-distress assistant.
- **Alarm must carry the position to ±2 m**, reach staff two ways (one wired), and be logged
  with images.
- **Temporary "non-detection zones"** that staff draw for drills, auto-expiring. This is the
  same idea as our coverage zones, used the other way round.
- **Non-detection test:** a competent swimmer stands still at the centre of each quadrant of
  the shallow (≤ 1.5 m) area for 35 to 40 s and must *not* trigger an alarm. A useful
  ready-made negative test: a still, upright, head-up swimmer is normal.
- **Self-monitoring:** the system must tell staff immediately when its detection performance
  degrades (for example glare or turbidity), and show the level of degradation.

What it means for us: it gives an outside reference for the false-alarm budget. A surface
distress detector is a different (harder, earlier) task than ISO 20380's bottom detection,
so 10 to 20 per hour is a reasonable start, but it shows where a deployable system has to
end up. "Report the degradation" is also cheap to copy: a per-camera "how many swimmers I
can currently see" health signal.

### Lifeguard detection performance (fit: medium, gives a human baseline)
From the [WIRH research summary](https://www.thewirh.com/blog/detection-performance), which
cites the original studies *(numbers as summarized there)*:
- Lifeguards detected 67 % of drowning targets in an occlusion task (Laxton et al. 2022),
  77 % in another (Laxton et al. 2021). Response times were about 3.5 to 4.2 s after a
  drowning appeared in video clips.
- 78 % of lifeguards spotted a simulated manikin casualty within 30 s
  (Hunsucker & Davison 2013), with average scan cycles of 19 to 24 s.
- Detection slows as swimmer numbers rise (several studies). Crowding is the human weak
  spot too, which is where an assistant earns its keep.
- [Lanagan-Leitzel & Moore 2010](https://scholarworks.bgsu.edu/ijare/vol4/iss3/4)
  (eye tracking): lifeguards' scanning was often not measurably better than briefly
  trained novices.

### Poseidon (underwater, commercial)
No independent false-alarm evaluation found. A vendor-linked case study says the alarm fires
after **10 s of immobility on the bottom**, chosen because shorter windows alarmed too often
([Matrox case study](https://imaging.matrox.com/en/media/case-studies/tools-pools)). Pilots
used mannequins.

## 2. Tracking through occlusion and disappearance

### Trackers we already have installed (fit: high, zero new dependencies)
Our `trackers` library (v2.4.0 locally) already ships `BoTSORTTracker`, `OCSORTTracker` and
`SORTTracker` next to the ByteTrack we use. Roboflow's own comparison
([trackers comparison page](https://trackers.roboflow.com/latest/trackers/comparison/)),
HOTA scores with default settings:

| Tracker | MOT17 | SportsMOT | DanceTrack | Note |
|---|---|---|---|---|
| ByteTrack (ours) | 60.1 | 73.0 | 53.3 | two-stage, keeps low-confidence boxes for matching |
| OC-SORT | 61.9 | 71.7 | 54.1 | "observation-centric re-update": repairs the motion model after a gap, built for occlusion and erratic motion |
| BoT-SORT | 63.7 | 73.8 | 57.8 | adds camera-motion compensation: fits our handheld / panning clips |
| McByte | 64.1 | 76.5 | 67.2 | adds propagated segmentation masks (SAM/Cutie); best identity keeping; newer library versions only |

DanceTrack (people in similar clothes, erratic motion, constant crossing) is the closest
public benchmark to a crowded wave pool, and it is where the gap between trackers is
largest (53 to 67 HOTA). Suggested experiment: rerun `build_tracks` with OC-SORT and BoT-SORT
on the pilot detections and compare victim coverage. That's offline and cheap, since our raw
detections are saved.

### SAM 2 family: follow one object through occlusion (fit: high for labeling, medium for detection)
- **SAM 2** (Meta) tracks a clicked object through a video with a memory of past frames, and
  handles objects disappearing and reappearing ([Ultralytics SAM2 docs](https://docs.ultralytics.com/models/sam2)).
- **SAMURAI** adds a Kalman-filter motion model and picks memory frames by quality: zero-shot,
  better under occlusion and fast motion ([arXiv 2411.11922](https://arxiv.org/html/2411.11922v2)).
  SAMITE reports SAMURAI still stores partly occluded frames and drifts
  ([arXiv 2507.21732](https://arxiv.org/html/2507.21732v1)).
- **SAM2MOT** does multi-object tracking *by segmentation*: a detector only seeds new
  objects, SAM 2 carries them, and a cross-object module handles occlusion. Zero-shot,
  state of the art on DanceTrack (+2.1 HOTA, +4.5 IDF1) ([arXiv 2504.04519](https://arxiv.org/html/2504.04519v4), AAAI).

Uses for us:
1. **Labeling:** click the victim once at onset and let SAM 2 / SAMURAI propagate a mask to
   guard contact. That gives a dense victim path (every frame) instead of our sparse clicks,
   and a measure of how long the victim is actually visible vs submerged. Nick would only
   correct the frames where it drifts.
2. **Detection:** a mask tracker keeps hold of a half-submerged head that a detector loses
   frame to frame. Heavy (SAM 2 is a large model), so it fits as an offline teacher or
   for re-checking candidates, not as the live detector on a Jetson.

### Point tracking (fit: low to medium)
CoTracker3 (Meta/Oxford) tracks arbitrary points through occlusion and long videos, online
or offline ([ICCV 2025 paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Karaev_CoTracker3_Simpler_and_Better_Point_Tracking_by_Pseudo-Labelling_Real_Videos_ICCV_2025_paper.pdf)).
Could track a head point between detections, but re-detection after full disappearance is
a known weakness of point trackers ([TAPNext++](https://www.alphaxiv.org/abs/2604.10582v1)),
and full disappearance is exactly our case.

### Swimmer-specific tracking
Little recent work. [Woinoski et al. 2020](https://arxiv.org/abs/2001.04433) built a
swimmer detection/tracking dataset from competition video. [Jacquelin et al. (MLSA21)](https://dtai.cs.kuleuven.be/events/MLSA21/papers/MLSA21_paper_jacquelin.pdf)
trained a small U-Net swimmer detector across distances, angles and reflections. Eng et al.'s
pool work handled splash occlusion with a Markov Random Field (via
[this survey](https://arxiv.org/pdf/1901.03577)). Nobody publishes a "ghost track" for
submerged swimmers. The closest is the Lynxight patent's submersion probability per frame
accumulated over a window (section 7).

## 3. Drowning behavior science and lifeguard scanning

### The instinctive drowning response (Pia) (fit: high, defines the target)
Described by Frank Pia from film of real drownings (1974 film and paper, 2006 USCG *On
Scene* article), summarized by [Wikipedia](https://en.wikipedia.org/wiki/Instinctive_drowning_response)
and in a [2011 WBUR interview](https://www.wbur.org/hereandnow/2011/06/29/drowning-pools-swimming):
- vertical body, no supporting kick
- head tilted back, mouth at or just below the surface, rising and sinking
- arms pressing down and out to the sides (lateral), not waving
- cannot call out: breathing takes priority over speech
- lasts about **20 to 60 s** of surface struggle before submersion, if nobody intervenes
- often happens with people close by who don't notice

This matches all three of Nick's "obvious tells" (stops moving forward, head under repeatedly
and erratically, arms flapping at the sides) and adds two measurable ones: **verticality**
(an upright body or head-and-shoulders box, against the horizontal shape of swimming)
and **mouth-level bobbing** (the head's top edge going up and down by a few pixels at the
waterline). The 20 to 60 s window sets the latency budget: an alert has to land well inside
it. That supports scoring catches from onset to guard contact.

Pia and colleagues' 2022 review ([Langendorfer, Pia & Beale-Tawfeeq, IJARE](https://scholarworks.bgsu.edu/ijare/vol13/iss4/8))
frames lifeguard recognition with signal detection theory: hits vs false alarms, trained
discrimination. That is the same framing as our detector evaluation.

### What real rescues look like: Fielding's lifeguard rescue reports (fit: high)
[UNC Charlotte rescue reporting system, 2009 to 2012](https://aphcs.charlotte.edu/wp-content/uploads/sites/1309/2023/04/Rescue-Data-2009-2012-Pool-and-W-Park.pdf)
(146 water park and 885 pool responses, guards' own reports):
- **Wave pools: 26 % of water park responses** (slides 27 %, lazy river 14 %, play features 12 %).
  An earlier 2009 sample put wave pools at 38 % ([2009 data](https://aphcs.charlotte.edu/wp-content/uploads/sites/1309/2023/04/Rescue-Data-2009.pdf)).
- **Water depth at water parks: 68 % in 3 to 5 ft.** Victims are mostly in standing-depth
  water, where head-and-shoulders views and upright bodies are the norm.
- **Recognized by sight: 92 to 93 %**, by sound 16 to 22 %. **"Profiled high risk before
  distress": 11 to 12 %.** Guards flag likely victims before trouble starts (small kids
  alone, weak swimmers, kids drifting past a depth line).
- **Activity before rescue (water parks):** wading 25 %, swimming 19 %, floating with
  equipment 10 % (tubes), entry or jumping 16 %. Free-text: "walked until it was too deep",
  "followed bigger kids past the lifeline", "bobbing".
- **Above-water visibility:** limited by patrons 8 % and glare 4 % at water parks: crowding
  is the main occluder.
- **Time of day:** water park rescues cluster from 2 to 5 pm (75 % of responses).

Ideas this gives us:
1. A **risk prior** feature: small apparent size (child) plus position (deeper zone, near a
   depth drop), the way guards profile swimmers.
2. **"Walked into deep water"** as a pattern: a track moving steadily toward deeper water
   then stalling. It needs a depth map per zone, which could be one polygon drawn per camera
   in the Zones tab.
3. **Tube-related events:** 10 % of water park rescues involve floating equipment, and our
   clips show tube flips (`34978d070280`'s victim fell off a flipped tube). A "person
   separates from a tube" event is detectable from boxes.

### Lifeguard scanning and human performance (fit: medium)
See section 7 for numbers. In short: guards' scan cycles are about 20 s; detection drops
as crowds grow; trained and briefly trained observers often scan similarly. An assistant
that never tires and doesn't slow down with crowd size targets exactly the human weak spot.

## 6. Data: datasets, video sources, live cams, synthetic

### Our main source is used up (fit: high)
The "Wavepool Lifeguard Rescue - Spot the Drowning" series comes from the **Lifeguard
Rescue** YouTube channel ([channel](https://www.youtube.com/channel/UCnERyC7dwJwTvEyzYz6uxHw)),
posted for staff training by Whirlin' Waters Adventure Park, North Charleston, SC
([2015 coverage](https://abc7.com/lifeguard-how-to-tell-if-someone-is-drowning-save-victim-pool-safety/794799/)).
A metadata-only listing (2026-10-17) shows **75 uploads: 66 are in our manifest and the
other 9 are on our blocklist** (rescues 7, 19, 29, 30, 36, 67, a top-5 compilation, a
seizure and a "difficult guest" clip). Nothing new is left on that channel.
More on the source ([Aquatics International, 2016 "Best Training" award](https://www.aquaticsintl.com/?p=11008);
[ABC 2015](https://abc7.com/lifeguard-how-to-tell-if-someone-is-drowning-save-victim-pool-safety/794799/)):
the park is county-owned, with up to 3,500 visitors on peak days, 100+ lifeguards (about
30 on duty), and **150 to 175 documented rescues per summer**, more than one a day. The
park says every video is real and unstaged, and new guards train by watching rescue
footage and raising a hand when they'd respond.

### Unscreened candidate videos (fit: medium, need Nick's eye)
Found by title search; none are in the manifest, the blocklist, or the harvest queue
(`data/harvest/candidates.jsonl`, 294 screened, 293 rejected). Queue with
`uv run python scripts/harvest.py add URL`. Titles only; nothing downloaded.

Wave pools:
- "3 Wavepool rescues" (C Gibby, 2:18): https://www.youtube.com/watch?v=QrxvbDP4boU
- "Wave Pool and lifeguard saving a man" (1:49): https://www.youtube.com/watch?v=8NWlA5XWPYo
- "Lifeguard saves unconscious 8-year-old from wave pool at Hawaiian Falls" (news, 1:21):
  https://www.youtube.com/watch?v=fz8zpE89cQQ

Pools with fixed CCTV-style cameras (out of wave-pool scope, but the right camera type):
- "五歲男童泳池溺水掙扎 周圍多人竟無察覺" (SETN; a 5-year-old struggling while nobody
  around notices: the textbook instinctive drowning response): https://www.youtube.com/watch?v=5Pd9PFz8baQ
- "游泳健將溺水3分鐘 泳池救生員沒發現" (CTi; drowned 3 minutes unseen by the guard):
  https://www.youtube.com/watch?v=mwZnvK01Uyg
- "泳池兒童溺水案例" (compilation of child pool drowning cases, 3:40):
  https://www.youtube.com/watch?v=onLDySCkRnw
- "誰小孩? 5歲童泳池溺水 熱心男跳水搶救" (TVBS): https://www.youtube.com/watch?v=lH8N9HZcG8I
- "好奇寶寶墮池 溺水拚命掙扎" (on.cc): https://www.youtube.com/watch?v=SkmjCAKhKlA
- "Hero lifeguard's amazing rescue caught on CCTV" (two uploads):
  https://www.youtube.com/watch?v=HNXXlMnPddc, https://www.youtube.com/watch?v=-s6DGKE_6kM
- "Boy Nearly Drowns in Vista Waterpark": https://www.youtube.com/watch?v=smm0v9exNZY
- "CAUGHT ON CAMERA drowning young girl save, lifeguard in swimming pool":
  https://www.youtube.com/watch?v=IWC3VGebaj4

News clips are often edited (cuts, zooms, overlays), so many will fail the harvest rules.
Searching in Chinese (泳池 溺水 监控 = "pool drowning surveillance") surfaces far more
fixed-camera pool footage than English searches do. **Bilibili** has much more of it
(searchable through yt-dlp's `bilisearch`, though it rate-limits quickly), but it is
outside the harvest allowlist (YouTube, Dailymotion, Vimeo). Adding it would be a
deliberate scope decision.

### Z5cc/drowning-detection: someone already labeled this channel (fit: high)
[github.com/Z5cc/drowning-detection](https://github.com/Z5cc/drowning-detection) (David
Nicklaser, 2024). Built a **spatio-temporal drowning dataset from 82 YouTube videos, mostly
the same Lifeguard Rescue channel**: 111 snippets of about 25 s, cropped to 224×224, with a
box around every person (about 3 per snippet) labeled *swim* or *drown* on every 6th frame
(CVAT, interpolated between). Model: YOWOv2 (YOLO with a 3D-convolution branch over the last
32 frames) plus a convolutional LSTM ("2Dseq"), which beat plain YOWOv2 and ran 3x faster at
the tiny size.

Their key lesson: **splashing was less of a problem than expected. The hard cases were
"descending and ascending" (people bobbing for fun) and "crawling" (swimming strokes)
vs drowning.** That matches Nick's tells ("head under repeatedly, more erratic") and says
bobbing *regularity* is the feature to get right.

What's useful:
1. **25 of their 81 videos are not in our manifest or blocklist**
   ([datalist.csv](https://github.com/Z5cc/drowning-detection/blob/main/dataset_lifeguard/datalist.csv)).
   Titles (metadata only):
   - real drownings: "Afogamento real criança em piscina" (hrbv1QxV0cU), "Drowning four year
     old in swimming class" (wJaE9V4N81U), "saves kids on vacation from drowning"
     (9uJazpj9xPw, 1080p, 8.5 min), "Can YOU spot the drowning child?" (The Sun,
     gY0ASobuVok, 1080p), "Drowning" (FwXH9N5JrE4, 240p)
   - **hard negatives** (thrashing that is not drowning): water polo (hSPOqCbpq1c full
     NCAA final, 79 min 1080p; vPunQYcer10), pool play-fights and dunking games
     ("ahogadillas", "aguadillas", "Briga na piscina", "Combat de TITANS", "ATAQUE EN
     PISCINA"), aquatic judo, "Parc Aquatique" (_wMF6AER-L4), "piscina koral de oropesa"
     (tKLV08BXRLU)
   - unavailable: zuZIfy4aBEY (age-gated), -A-eXmSQYO0, lnxwi0_41L0, 9daL_swXKCs
2. **Hard negatives are a category we have zero of.** Every non-victim in our data is
   ordinary swimming. A model that has never seen dunking or water polo will flag them.
   These belong in the *test* set as a false-alarm stress test, even if not in training.
3. **An independent second labeling of ~54 shared videos.** Their drown boxes can be
   checked against our victim clicks to catch labeling mistakes (ours or theirs).
4. **License: none stated** (all rights reserved by default). Labels can't be copied into
   our AGPL repo without the author's permission. Worth a GitHub issue asking him: he
   explicitly offers to provide the videos on request.

## 4. Temporal and anomaly models

### Simplest strong anomaly baseline: attributes plus density (fit: high, keeps Nick's anomaly bet alive)
[Reiss & Hoshen, "Attribute-based representations for accurate and interpretable video
anomaly detection" (TMLR 2025)](https://cris.huji.ac.il/en/publications/an-attribute-based-method-for-video-anomaly-detection/):
represent each detected person by **velocity and pose**, score anomalies with a
**k-nearest-neighbour density** over normal examples, no neural network training. This
reaches state of the art on ShanghaiTech; adding a pretrained deep feature gives 99.1 / 93.7 /
85.9 AUROC on Ped2 / Avenue / ShanghaiTech *(abstract numbers)*.

For us: once each head track has a measurement vector (time under, forward movement in
head-widths, bobbing irregularity, side splash, verticality), the anomaly model can be
**k-NN distance to normal swimmers' vectors**. Cheap, explainable ("flagged because: under
4 times in 10 s, no forward movement"), and trainable on tiny normal data. It's a strong
first anomaly scorer before any network.

### STG-NF (already our stated priority)
[Hirschorn & Avidan, ICCV 2023](https://ar5iv.labs.arxiv.org/html/2211.10946): normalizing
flow on pose graphs, about 1K parameters, 85.9 AUC ShanghaiTech / 79.2 UBnormal. A 2025
active-learning paper still uses it as the lightweight baseline
([ALFred, arXiv 2508.09058](https://arxiv.org/pdf/2508.09058)). Caveat stays the same: it
consumes keypoints, and our keypoints at 40 to 60 px are mostly noise. It only makes sense
on near-field tracks or on the head-measurement vectors, not raw COCO joints.

### Spatio-temporal action detectors (fit: medium)
[YOWOv2 (Yang & Dai 2023)](https://arxiv.org/pdf/2302.06848): a YOLO frame branch plus a 3D
convolution clip branch, detecting "who is doing what" per frame in real time (87.0 frame
mAP on UCF101-24 at more than 20 FPS). Z5cc's drowning project (section 6) built on it.
This is the end-to-end alternative to our "detect, track, then score the track" pipeline:
it learns appearance and motion of drowning directly from pixels. It needs boxes labeled
*drown* on many frames, which our victim clicks plus SAM 2 propagation could provide.

### Handling missing frames properly (fit: high)
Our v2 features hold the last coordinates when a swimmer isn't detected and add a
"detected" flag. Two well-established better options:
- **GRU-D** ([Che et al., Scientific Reports 2018](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5904216/)):
  feeds the **mask** (seen or not) and the **time since last seen** into the model, and
  decays stale values toward a default. Its core finding is that *missingness itself is
  informative*. For us that's literal: a head that stops being detected *is* the
  submersion signal.
- **Partial (masked) convolutions** ([Liu et al., NVIDIA 2018](https://arxiv.org/pdf/1811.11718)):
  treat missing steps as holes; each convolution only sums over observed steps and
  re-weights by how many it saw. A drop-in replacement for convolutions over sequences
  with gaps, the time-axis version of image inpainting.
Either way: give the model "how long since this head was last visible" as an explicit
feature. It is probably the single most useful submersion feature, and it's free.

### Motion and splash, without keypoints (fit: high)
- **Periodicity separates water from people.** [Chan 2013 (Machine Vision and Applications)](https://unpaywall.org/10.1007%2Fs00138-012-0419-3)
  computes dense optical flow plus a **per-pixel periodogram** over a short window and
  sorts pixels into *stationary/random*, *ripple* and *swimming*, detecting swimmers much
  better than intensity alone. For a wave pool this is a strong hint: **the wave machine is
  periodic** (a fixed wave period), and so are swimming strokes, while flailing and
  bobbing-in-distress are irregular. Spectral features of the motion around a head
  (energy at the wave frequency vs everything else) could cancel the waves out.
- **Splash shows up as white:** one swimming study saw a clear rise in the red channel
  (whitening) at splashes, after subtracting a background image, but the surface made it
  noisy ([Polimi, Sensors 2023](https://re.public.polimi.it/bitstream/11311/1233457/1/sensors-23-02364-v2_LowRes.pdf)).
  So splash works best **relative to the water around the head**, as already planned.
- **Two-class head detector: "above water" vs "under water".** A swim-analysis group
  trained YOLOv3 on head boxes with two classes, head over the water and head under the
  water, and used the switching to read stroke timing (48.1 mAP; splash and reflections
  were the main errors)
  ([SPIE 2022](https://electronicimaging.spiedigitallibrary.org/conference-proceedings-of-spie/12177/1217713/Swimmer-position-estimation-and-stroke-analysis-based-on-head-recognition/10.1117/12.2626170.full)).
  **That is exactly the submersion signal, per frame.** When we label heads for the
  fine-tune, adding an "at or under the surface" class costs one extra keypress per head
  and gives the bobbing and submersion features directly.

### Vision-language models as a second opinion (fit: medium, later)
Frozen VLMs can score and *explain* video anomalies: LAVAD (training-free, captions then an
LLM scores), Holmes-VAD (fine-tuned), VERA (frozen VLM with learned guiding questions; 86.6 /
88.3 AUC on UCF-Crime / XD-Violence) ([VERA, arXiv 2412.01095](https://arxiv.org/pdf/2412.01095)).
Too slow and heavy to watch every swimmer, but a plausible **second-stage verifier**: the
cheap detector raises a candidate, and a VLM looks at a 5 s crop and answers "is this
person in distress or playing?". That's one way to push false alarms from tens per hour
toward ISO 20380's well under one per hour. Footage privacy matters here: a local model,
not a cloud API, for real pool video.

## 1. Detection: small heads and swimmers

### Train from the clicks we already have: point-supervised head localization (fit: high)
[P2PNet (Song et al., ICCV 2021)](https://openaccess.thecvf.com/content/ICCV2021/papers/Song_Rethinking_Counting_and_Localization_in_Crowds_A_Purely_Point-Based_Framework_ICCV_2021_paper.pdf),
code at [TencentYoutuResearch/CrowdCounting-P2PNet](https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet):
trains directly on **one point per head**, with no boxes, and predicts head points, matched
to the labels one-to-one with the Hungarian algorithm. Crowd counting's own datasets are
labeled exactly like our Swimmer count tab. A follow-up (P2PNeXt, 2025) swaps in a ConvNeXt
backbone. For us: our count-tab clicks and victim clicks are already point labels. A
point-based head localizer avoids the "derive a box from a dot" step in the fine-tune plan,
and its natural metric (point matching within a radius) is the one our benchmark already
uses. Its license needs checking before reuse.

### Head-tracking data from an elevated view (fit: medium, pretraining)
[CroHD / MOTChallenge Head Tracking 21](https://motchallenge.net/data/Head_Tracking_21)
([paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Sundararaman_Tracking_Pedestrian_Heads_in_Dense_Crowd_CVPR_2021_paper.pdf)):
**2.28 million head boxes** in 11,463 Full-HD frames, 25 fps, **elevated cameras**, dense
crowds (Shibuya crossing). The closest public match to "many small heads seen from above
at a slant", with tracking IDs. A good pretraining set for a head detector or tracker
before fine-tuning on pool frames. [A 2025 tiny-head-tracking paper](https://arxiv.org/abs/2505.22677)
adds a cross-domain loss and small-receptive-field detection for tiny heads on CroHD and
CrowdHuman (no numbers in the abstract).

### Detector families worth knowing (fit: medium)
- **RF-DETR (Roboflow):** a transformer detector, **Apache 2.0** for the core sizes (XL and
  2XL are under a separate license). RF-DETR-L reaches 56.5 COCO AP at 6.8 ms on a T4; on
  RF100-VL (100 varied fine-tuning sets) RF-DETR-B scores 86.7 mAP50 vs YOLO11m's 84.9
  ([rfdetr docs](https://rfdetr.roboflow.com/latest/), [comparison](https://blog.roboflow.com/rf-detr-vs-alternatives/)).
  Two reasons to care: it fine-tunes well on small custom sets, and its license is looser
  than Ultralytics' AGPL. A SeaDronesSee (people in open water) fine-tune exists on Hugging
  Face ([dronefreak/seadronessee-rfdetr-medium](https://huggingface.co/dronefreak/seadronessee-rfdetr-medium)).
- **Open-vocabulary detectors (Grounding DINO, YOLO-World) as label teachers:** prompt
  "head" or "person in water" and auto-label frames offline; a YOLO student trained on
  Grounding DINO labels lost nothing in one study, and mixing auto plus manual labels
  helped most ([MDPI Applied Sciences 2024](https://www.mdpi.com/2076-3417/14/6/2232);
  [DART pipeline](https://arxiv.org/pdf/2407.09174)). Slow, so offline only. No source
  reports their accuracy on dense crowds, so measure on our count frames first.
- **TinyPerson** ([Scale Match, WACV 2020](https://arxiv.org/pdf/1912.10664)): the benchmark
  for people under 20 px. Its "scale match" trick (rescale pretraining images so object
  sizes match the target data) is a cheap idea for pretraining on CroHD or CrowdHuman
  before our pool frames.

### People-in-water detection (fit: medium)
[SeaDronesSee (WACV 2022)](https://arxiv.org/abs/2105.01922): 54,000 drone frames, 400,000
boxes of swimmers and floaters in open water, with RGB plus near-infrared (swimmers stand
out in NIR). Different viewpoint (drone, top-down), but the same "small head in waves and
glare" problem; detectors fine-tuned on it are on Hugging Face (YOLOv8 and RF-DETR, AGPL /
Apache). A cheap extra candidate for the head bake-off.

## 8. Labeling efficiency

- **SAM 2 propagation for victim paths** (section 2): click once, propagate, correct drift.
  CVAT's SAM 2 tracker is **enterprise-only** (the community edition has keyframe
  interpolation, claimed 5 to 10x faster than frame-by-frame)
  ([CVAT docs](https://docs.cvat.ai/docs/enterprise/segment-anything-2-tracker)), so the
  cheaper route is running SAM 2 ourselves (Ultralytics ships it) behind a "propagate
  victim" button in our own labeler.
- **Cost of SAM 2 on our Mac:** SAM 2.1 Tiny (75 MB) takes about **1.6 s per point
  prompt** on an M1 with MPS for single images ([blog, 2026](https://ice-ice-bear.github.io/ko/posts/2026-04-08-sam2-mac/));
  no published video-propagation speed for M1. Guessing ~1 s per frame, a 15 s onset-to-contact
  span at 30 fps is ~7 to 8 minutes per victim on the Mac: fine as a background job, or
  minutes on Colab. (Meta has since released SAM 3, compared in the same post.)
- **Pre-label, then correct:** the detector proposes, the human fixes. Standard practice,
  and it's the plan for the head fine-tune. A YOLO student trained on Grounding DINO
  auto-labels matched manual labels in one study, and auto plus manual was best
  ([MDPI 2024](https://www.mdpi.com/2076-3417/14/6/2232)).
- **Pick frames by uncertainty plus diversity, not at random:** video frames are highly
  redundant. Uncertainty-based selection reported about 55 % time savings and about 30 %
  data savings ([Active Learning for Deep Object Detection](https://arxiv.org/pdf/1809.09875);
  [uncertainty-aware AL for new domains](https://arxiv.org/pdf/2303.12760)). There's no
  universal label count: start with a seed set, add rounds of the frames the model is
  least sure about, stop when the validation curve flattens. For the pilot: label the 40
  frames the current head model is *most unsure* about, not 40 random ones.
- **Point labels instead of boxes** (section 1, P2PNet): one click per head, no box. That's
  what the count tab already does.
- **"Above water / under water" as a second head class** (section 4): one extra keypress per
  head, gives submersion directly.

## 10. Camera geometry and distance

- **Why the size cutoff works:** on a flat surface (the water), a person's image height
  grows roughly linearly with how far below the horizon they appear, scaled by camera
  height. From people's sizes and positions alone you can recover the horizon and the
  camera height ([size constancy paper, arXiv 1509.08147](https://arxiv.org/pdf/1509.08147);
  [CameraTransform, arXiv 1712.07438](https://arxiv.org/pdf/1712.07438)). Our per-frame
  "size vs height in the image" fit is a simple version of that. Fitting *linear* height
  in y, instead of our log-linear fit, matches the geometry better and would give each
  camera's horizon and an approximate metres-per-pixel scale for free.
- **Perspective maps:** crowd counting predicts, per pixel, how many pixels equal one metre,
  and uses that to weight scales ([PACNN, CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/papers/Shi_Revisiting_Perspective_Information_for_Efficient_Crowd_Counting_CVPR_2019_paper.pdf)).
  With a pixels-per-metre map per camera, the distance cutoff can be stated in metres, and
  "forward movement" can be measured in metres instead of head-widths.
- **Warp the image so every head is the same size** (fit: high, compute idea): "height
  reverse perspective transformation" enlarges far heads and shrinks near ones as a
  preprocessing step, so one detector scale fits all
  ([Optoelectronics Letters 2021](https://www.researching.cn/ArticlePdf/m00050/2021/17/6/OEL-2021-06-0367.pdf)).
  For a fixed pool camera this could replace uniform tiling: warp the zone polygon to a
  roughly top-down view (one homography per camera, from 4 pool corners) and tile that.
  Far water gets more pixels, near water fewer, and the deck isn't processed at all. It
  turns the zone, the size cutoff and the tiling into one step.

## 9. Compute and attention (scanning)

### RL that decides where to zoom already exists (fit: high, this is the RL back-burner project)
[Dynamic Zoom-in Network (Gao et al., CVPR 2018, Google/UMD)](https://openaccess.thecvf.com/content_cvpr_2018/papers/Gao_Dynamic_Zoom-In_Network_CVPR_2018_paper.pdf):
detect on a downsampled frame first, then a **Q-learning network picks which regions to
re-run at high resolution**, rewarded for accuracy gained minus pixels processed. A second
network predicts the accuracy gain of zooming into each region. **Over 50 % fewer pixels
processed with no accuracy drop** on Caltech Pedestrians, and about 70 % fewer pixels and
over 50 % less time on high-resolution YFCC images. This is a published template for the
lifeguard-scanning RL project: same action (which tile to look at closely), same reward
shape (catches minus compute). Our saved per-tile detections let us replay it offline.

### Non-RL ways to pick crops (fit: medium)
- **ClusDet** ([ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/html/Yang_Clustered_Object_Detection_in_Aerial_Images_ICCV_2019_paper.html)):
  predicts where objects cluster and their scale, then detects only in those chips.
- **DMNet** ([arXiv 2004.05520](https://arxiv.org/pdf/2004.05520)): a density map chooses
  the crops, with no extra network to train.
- **Learning to Zoom** ([Recasens et al., ECCV 2018](https://cdfg.mit.edu/assets/files/learning_to_zoom.pdf)):
  a differentiable saliency sampler that gives more resolution to salient regions, falling
  back to uniform when unsure.
For a pool, the cheapest version is fixed: the zone polygon plus the perspective warp
(section 10) decide resolution once per camera. Adaptive cropping matters later for live
compute.

### Edge hardware reference points (fit: low for now)
No direct YOLO11n-on-Orin-Nano benchmark found. Reference: YOLOv8n/s run at about **100
FPS FP16** on an Orin Nano at 640 px (inference only), and INT8 gave little or no speedup
with worse accuracy there ([Qengineering](https://github.com/Qengineering/YoloV8-TensorRT-Jetson_Nano);
[NVIDIA forum](https://forums.developer.nvidia.com/t/jetson-orin-nano-fp16-int8-performance/326723)).
Our tiled pipeline is 17 passes per frame at 1280 px, so live tiling at 30 fps is far out
of reach on one Orin Nano. At 5 frames per second on the zone only, it is roughly within
reach. Worth measuring with `model.benchmark()` on the device before planning hardware.

### Lynxight (overhead cameras, the closest commercial analogue) (fit: high)
Vendor claims, labeled as such:
- standard overhead security cameras, 10 to 12 supported models, **every patch of water seen
  from at least two angles** to beat occlusion and glare
  ([crowded-water page](https://www.lynxight.com/content/how-does-ai-drowning-detection-cope-with-crowded-water))
- tracks each swimmer individually; flags surface distress (including the instinctive
  drowning response) before submersion, as well as submersion
- alert to a **smartwatch with a snapshot and the location**; the lifeguard decides
- **2 to 3 alerts per pool per day** on average (a YMCA customer reports 3 to 4)
- named false-alert sources: **shadows, reflections, floats, lane ropes**; resting
  motionless at a wall is handled as normal
- 1,000+ pools in 16 countries; Fluidra quotes "95 % effectiveness"; an operator (BlueFit)
  says guards in a test found less than half of what the system did. No independent study found.
- EU funding summary: networked cameras, central unit, smartwatch alerts
  ([CORDIS](https://cordis.europa.eu/article/id/418293-machine-vision-and-analytics-suite-boosts-safety-standards-and-prevents-drowning)).

Three things to copy:
1. **The alert is a snapshot plus location on a watch.** Our "alert" output should be a
   cropped image and a pool position, not a score.
2. **Two cameras per patch of water** is how a commercial system handles crowds. Our
   single-camera clips are a harder version of the problem than real installs.
3. **Their alert budget is about 3 per pool per day**, far stricter than 10 to 20 per hour.
   Again, that's where a mature system lands.

### AngelEye, SwimEye, Sentag, SenSwim
AngelEye (underwater cameras plus overhead) says it was **independently tested against ISO
20380**. Australia adopted **AS ISO 20380:2022**. Operators are advised to ask whether a
system is independently tested against the standard, not just whether it can track
swimmers ([Australasian Leisure Management](https://www.ausleisure.com.au/news/safety-privacy-and-the-what-aquatic-operators-need-to-know-about-ai-drowning-detection)).
SwimEye: camera-based; Sentag: a wristband wearable; SenSwim: camera-only (via
[CB Insights](https://www.cbinsights.com/company/angeleye/alternatives-competitors)). A 2024
Sensors review rates image processing and sensors as the most effective families, with
image processing the costliest and most complex
([PMC10820385](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10820385/)).

### Privacy norms for real deployments (fit: medium, matters for the portfolio story)
- Imperial College's Lynxight install deletes footage after 7 days unless needed for an
  incident.
- Operators are advised to tell patrons that AI analyses their position (generic "CCTV in
  use" signs may not be enough), keep ownership of data, know whether footage trains the
  vendor's models, and prefer on-site processing.
- Drowning detection **locates** swimmers and doesn't need to **identify** them (no face
  recognition). That's a design principle worth stating in the README.

### Synthetic data (fit: medium for detection, low for behavior)
- **Copy-paste of small objects:** oversample images with small objects and paste extra
  copies; +7.1 % relative on small-object detection on COCO
  ([Kisantal et al., arXiv 1902.07296](https://arxiv.org/pdf/1902.07296)). Simple Copy-Paste
  (Ghiasi et al.) is the general version. Video variants paste instances consistently across
  frames ([SDI-Paste, arXiv 2410.13565](https://arxiv.org/abs/2410.13565v1)). Caveat: pasted
  edges look unnatural and can hurt small models
  ([arXiv 2604.19999](https://arxiv.org/pdf/2604.19999)). For heads in water, the waterline
  edge matters, so paste head crops *with* a band of surrounding water.
- **Drowning from purely synthetic images:** an IEEE Access 2026 paper trained a drowning
  classifier only on prompt-generated plus GAN images and reports about 94 % on real test
  images ([GNU ScholarWorks](https://scholarworks.gnu.ac.kr/item/70c007a5-4eeb-497c-a59f-58fbbaf1c653)).
  The test set was a public Roboflow image set, the same single-image, staged-footage trap
  as the 2023 to 2026 YOLO papers, so treat it as weak evidence.
- **Game engines:** Unreal Engine 4 with domain randomization trained a working real-world
  detector ([Applied Sciences 2022](https://doaj.org/article/47370d84f7aa4a19b125b9eaa6e38354));
  Unity's Perception package does the same. Nobody has published rendered swimmers in
  water that transfer to real pools (refraction, splash and wet hair are hard).
- **Synthetic swimming motion:** [SwimXYZ](https://arxiv.org/pdf/2310.04360): 3.4 M
  synthetic frames with 2D/3D joints and 240 SMPL swimming motion sequences, on
  [Zenodo](https://zenodo.org/record/8399376). Competitive strokes only, but the SMPL motions
  could seed track-level scenario generation (our backlog item) for *normal* swimming
  kinematics.
- **Track-level scenario generation** (already in our backlog) stays the cheapest synthetic
  route for behavior: inject bobbing / stall / disappearance patterns into real normal head
  tracks. It's only honest for testing the rule logic, never as evidence of real-world recall.

### Hours of normal wave-pool footage: Divertiland livestream archives (fit: high, our biggest gap)
The **Divertiland Parks** YouTube channel has archived water-park livestreams of about
**11.8 hours each**. One, ["Divertiland Park Live Stream" (varQEogbBLU)](https://www.youtube.com/watch?v=varQEogbBLU)
(2016-08-12, 1280×720, 25 fps, 42,605 s), shows on its thumbnail a **fixed elevated camera
over a crowded wave pool with tubes**. That's the same kind of view as our rescue clips.
Further archives of the same length (content unchecked; their thumbnails are the channel
default): vGtoqadb-mw, M5YXv6M77hk (55 min), Nxos619PWzI, f4UTrpsjk3U, A9j6qtT1rUY,
gHoeCigVoNY.

Why it matters: the false-alarm budget (10 to 20 per hour, or ISO 20380's well under 1 per
hour) can't be measured on 3 to 5 minutes of rescue highlights. **One of these archives
alone is about 150 times the normal footage we score today.** A couple of hours sampled
across a day (morning, afternoon crowd peak) would give the first honest false-alarm-per-hour
number, and a large pool of normal head tracks for any anomaly model.

Caveats: these are public broadcasts by the park, similar in kind to the AMI resort cam
already in the manifest (on hold). They still show identifiable people, so the same rules
apply (local only, never committed, no figures without pixelation). 720p at that distance
means small heads, so the zone and size cutoff will discard much of the far water. That's
fine for false alarms inside the range, which is the number that matters. Needs Nick's
call on scope (a different park and camera than the rescues).

Other leads, not checked: "Splashzone WaterPark Webcam" (The Surfers View, YrnuniZOjp8),
"Avalanche Bay Indoor Waterpark" live cam (qfHSDPMyWt4), "Live Pool Cam" (lw0UHzKdHqo, 2 min),
"Thunder Bay wave pool - Full Session" (The Pool Channel, TuZs1dx4vgo, 7.5 min), and a
Russian indoor wave-pool webcam at Aquamir, Novosibirsk, 1080p, 10:00 to 21:00
([webcams.org.ru listing](https://webcams.org.ru/ob.htm)).

### Other drowning datasets (fit: low to medium)
- **ODIV** ([J. Building Design & Env. 2026](https://www.sciexplor.com/jbde/articles/jbde.2026.0039)):
  203 **real** open-water clips (101 drowning, 102 not) from social media and news, with
  clip-level labels. A **TimeSformer video transformer with multiple-instance learning**
  (MIL: a video is a bag of short clips; its score is the mean of the top 20 % of clip
  scores; only video-level labels needed) reached AUC 0.975 on 22 test clips, with **no
  person detection or tracking at all**. Pool-trained YOLO drowning detectors scored only
  0.68 to 0.72 on this footage. Not released (data "from the corresponding author"). Clip
  classification ("does this video contain a drowning?") is much easier than localizing
  the victim in time and space, and no false alarms per hour are reported. But **MIL over
  space and time** is a useful idea for us: treat each zone tile × 2 s as an instance and
  each clip as a bag, train from "a drowning happens in this clip between onset and
  contact", and let the model find where. It's a detection-free fallback if the head
  pipeline stalls.
- **Underwater sets:** [Figshare underwater drowning dataset](https://figshare.com/articles/dataset/Underwater_Drowning_Detection_Dataset/29497235)
  (5,613 annotated underwater images, swimming / struggling / drowning, YOLO format,
  simulated in a pool), and a 69,512-frame multi-swimmer underwater set
  ([ICIAP 2025](https://dl.acm.org/doi/10.1007/978-3-032-10192-1_48)). Different camera
  (underwater), staged.
- **Roboflow Universe:** many small CC BY 4.0 drowning sets
  ([poolwatchv8](https://universe.roboflow.com/poolwatchv8/drowning-detection-76qli/dataset/),
  [Drowning_1](https://universe.roboflow.com/drowning-detection-zb55m/drowning_1), and others).
  One, ["Early drowning detection" (Pool safety)](https://universe.roboflow.com/pool-safety/early-drowning-detection/dataset/),
  has instinctive-drowning-response classes ("Head tilted back", "Mouth open", "distressed
  swimmer"). Mostly single images, often staged; useful at most as a sanity check or for
  pretraining a head-state classifier.
- **The staged-data warning, repeated by several papers:** the 2023 to 2026 YOLO drowning
  papers (94 to 99 % accuracy) mostly use simulated drownings by lifeguards and volunteers
  ([YOLOv11 study, 1,519 images, all simulated](https://www.researchgate.net/publication/394500994_Intelligent_eyes_on_water_YOLOv11-based_real-time_drowning_detection_system)).
  ODIV's authors argue staged data "fails to reproduce the dynamics and visual ambiguity of
  genuine incidents". Our real rescues are the rarer and more valuable kind.

### How to score an early-warning detector (fit: high)
- **AMOC curve** (activity monitoring operating characteristic, Fawcett & Provost): plots
  **expected time to detection against false-alarm rate**, the early-warning replacement
  for ROC, which ignores timing. Misses count as a delay equal to the event length.
  Surveillance work reports the **partial area up to a small false-alarm rate**, where the
  budget actually lives ([overview, PMC2815453](https://pmc.ncbi.nlm.nih.gov/articles/PMC2815453);
  [G-AMOC, Neill](https://www.cs.nyu.edu/~neill/papers/gamoc.pdf)). Our recall-vs-FA/h sweep
  plus median time-to-alarm is most of an AMOC curve already. Plotting delay (with misses
  = onset-to-contact duration) against FA/h, and reporting the area up to 20 FA/h, would
  turn it into one number per model that rewards both catching and catching early.
- **Expected Warning Time**, a generalization: how much earlier than the human the system
  alerts. Our labels already have the human's time (guard contact), so **"seconds before
  the guard reached the victim"** is a direct, honest headline metric for an assistant.
- **NAB scoring** (Numenta): one reward per event window, decaying the later the first
  alarm comes; every false alarm costs. Profiles that weight false positives or misses
  differently ([summary](https://blog.dominodatalab.com/numenta-anomaly-benchmark)).
  Criticized as domain-specific ([Precision and Recall for Time Series](https://arxiv.org/pdf/1803.03639)),
  but the "earlier is worth more" shape fits drowning (20 to 60 s window).

### Sensing submersion directly (fit: medium; hardware note for the future)
- **Thermal (long-wave infrared) cameras see only what is above water:** water absorbs
  LWIR strongly, so a thermal camera shows heads and shoulders and nothing below the
  surface. That gives a clean above/below signal that visible cameras (reflections,
  refraction) struggle with ([patent US 10803724, via USPTO](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/10803724)).
  A thermal-head-detection study reached 88.6 % detection
  ([WIRH summary](https://www.thewirh.com/blog/dds-research)). It's the strongest hardware
  answer to "is the head under?", and it also hides faces (a privacy plus).
- **Waterline geometry:** treat the still pool surface as a plane (a homography from the
  pool edges) and time how long a tracked head stays below it
  ([patent 12039851](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/12039851)).
  A cruder variant uses the head-area to region ratio against a threshold. In a wave pool
  the surface isn't still, so this only works with the wave motion modeled out.
- **Treading water is the classic false alarm** for vision systems
  ([Lu & Tan, IEEE TCSVT 2004](https://www.etasr.com/index.php/ETASR/article/view/7804);
  WIRH summary). Treading water is upright, in place, head up, the same posture as the
  instinctive drowning response minus the bobbing below the surface. So **"head goes under
  repeatedly"** is the feature that separates them, not verticality or lack of progress
  alone. ISO 20380's non-detection test (a person standing still for 35 to 40 s must not
  alarm) encodes the same lesson.

### Lynxight patent details, second half (fit: medium)
From the rest of [US 11769387](https://patents.google.com/patent/US11769387):
- cameras mounted **above 1.80 m**, viewing angle to the farthest water point **at most 70°**
  (steeper is better: less glare, less occlusion)
- motionless = centre of mass within **10, 20 or 50 cm** over the window; alert when
  underwater time and motionless time overlap by a set amount; "relatively motionless and
  not submerged for over 30 s" is its own flagged case
- body orientation: projected-length / true-height ratio **above ~0.8 = horizontal, below
  ~0.5 = vertical**, using the largest length seen during tracking as the true height
- **face-down is treated as higher risk** and can trigger a sooner alert
- submersion is a per-frame probability, combined across frames, and an "above surface"
  frame can be explained away as a detection error or as drowning-like bobbing
- calibration maps image vectors to real lengths at representative points, then
  interpolates, which is the pixels-per-metre map idea from section 10

### Wave timing, for "cancel the waves out" (fit: medium)
- Recreational wave pools typically run **timed cycles**, for example **5 minutes on, 5 off**
  ([Schlitterbahn](https://www.schlitterbahn.com/new-braunfels/rides/the-beach-wave-pool)).
- Wave period depends on the pool: children's wave pools around **2 s per wave**
  ([WhiteWater West](https://www.whitewaterwest.com/products/children-wave-pool/)); surf
  pools 6 to 90 s between waves (a Wavegarden patent).
Ideas: (1) a per-clip **"waves on / off" state** from global motion periodicity is a cheap
scene feature. Check whether our rescues and false alarms cluster in wave-on periods; Nick
would know whether real drownings do. (2) The wave period gives the frequency to subtract
from each head's vertical motion, leaving the irregular bobbing that separates distress
from riding the waves.

### Use several frames to find faint heads (fit: high)
A single frame shows a half-submerged head as a few ambiguous pixels; several frames show
something that keeps reappearing in the same place. Video object detectors exploit this:
- **YOLOV** ([code](https://github.com/YuHengsss/YOLOV)): a one-stage detector (YOLOX) that
  aggregates the top proposals across sampled frames with attention. Cheap enough for
  near-real-time.
- **Corsel et al., "Exploiting Temporal Context for Tiny Object Detection"**
  ([WACV workshops 2023](https://openaccess.thecvf.com/content/WACV2023W/RWS/html/Corsel_Exploiting_Temporal_Context_for_Tiny_Object_Detection_WACVW_2023_paper.html)):
  YOLOv5 fed a short stack of frames at once, for tiny objects from **static surveillance
  cameras**. That's our setting.
- **Temporal-YOLOv8** ([Sensors 2024](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11598073/)):
  video context plus small-object augmentation, mAP 0.465 to 0.839 on their in-house set
  (the two effects aren't separated).
The cheapest version for us needs no new model: **feed the detector a 3-frame stack**
(previous, current, next frame as the three colour channels of a grayscale image, or the
difference images), the "temporal shift" trick for static cameras. A half-visible head that
pops up and down becomes a strong, consistent signal against static water. Worth a spike
after the head fine-tune.

### Learn from unlabeled footage (fit: medium)
We have ~240,000 unlabeled frames. Teacher-student self-training labels them automatically:
- **PseudoProp** ([arXiv 2203.05983](https://arxiv.org/pdf/2203.05983)) and **SSVOD**
  ([WACV 2024](https://openaccess.thecvf.com/content/WACV2024/papers/Mahmud_SSVOD_Semi-Supervised_Video_Object_Detection_With_Sparse_Annotations_WACV_2024_paper.pdf)):
  pseudo-labels for unlabeled frames made more reliable by checking they agree with
  neighbouring frames (motion or optical flow).
- With very few labels, **weighting each pseudo-label's loss by its confidence** worked best
  ([Vandeghen thesis summary](https://www.montefiore.uliege.be/cms/c_12902778/en/learning-with-unlabeled-data)).
Plan fit: after the 40-frame head pilot, run the fine-tuned model on all frames, keep
detections that persist across neighbouring frames as pseudo-labels, and retrain. It's one
round of self-training before asking for more hand labels.

### SwimID: CCTV swimming-incident dataset with tracks (fit: high, request access)
[github.com/cv416/SwimID](https://github.com/cv416/SwimID), "Detecting Swimming Incidents in
Videos: Dataset and Approach" (Northwest A&F University, repo Aug 2026, paper under review).
**Fixed CCTV cameras in indoor pools**, realistic swimming, incident classes **struggling,
floating, sinking**, with **dense per-frame boxes, action labels and tracking IDs**.
Folder names suggest NVR channel recordings across many cameras and days. Access: sign a
release agreement and send it in, as the repo's README describes; academic research only. Code
(spatio-temporal action detection) comes after acceptance.

Why it matters: it's the only dataset found that has the right camera type, real (or
realistic) incidents, *and* per-person tracks. It would give a second, independent test
set from a different pool type, which is the strongest guard against overfitting to one
wave pool's camera. Indoor lap or leisure pool, not a wave pool. Worth requesting now:
the agreement is free for academic use, and approval can take weeks.

### zseng0912/Drowning-Detection-System (shared by Nick) (fit: low to medium)
[github.com/zseng0912/Drowning-Detection-System](https://github.com/zseng0912/Drowning-Detection-System)
(2025, 6 stars, no license file). An end-to-end demo: FastAPI backend, React dashboard,
webcam or upload input.
- **Detection:** a single-frame **YOLO11s** with a "drowning" class (plus swimming),
  trained through Roboflow / Ultralytics Hub, one model for above water and one for
  underwater (weights in the repo, 19 MB each). The above-water notebook reports **mAP50
  0.88 for "drowning"** on its own validation set (756 images, 434 drowning instances).
- **Underwater enhancement:** FUnIE-GAN (TensorFlow) cleans up murky underwater frames
  before detection.
- **Tracking:** DeepSORT (appearance re-ID plus Kalman).
- **Alarm rule:** sound an alarm every **5 consecutive frames** containing any "drowning"
  box (about 0.2 s at 25 fps).
- Extras: nearest-emergency-services lookup, login, test tabs.

How it compares: it's the single-image approach from section 1's 2023 to 2026 papers:
appearance of one frame, staged or Roboflow data, no time-to-alarm, no false alarms per
hour, no distance handling. "5 consecutive frames" is very short against Poseidon's 10 s
and ISO 20380's 15 s, so it would alarm on any brief drowning-looking pose (dunking, a tube
flip). The useful parts for us:
1. **Its weights are a free external baseline.** Running its above-water model on our
   in-range rescue clips, scored with our victim-track harness and false alarms per hour,
   would show concretely how a single-frame "drowning" classifier fares on real footage.
   That's a strong portfolio comparison. No license is stated, so use it only for local
   evaluation and don't redistribute.
2. **The dashboard pattern** (live feed, alarm, snapshot) is close to Lynxight's
   smartwatch alert and a reasonable demo target later.
3. Its own "future work" lists **head/partial-body keypoints because only parts are
   visible above water**, the same conclusion we reached.

### Submersion-timer designs others built (fit: medium, design patterns)
- **SALMON** (U. Calgary capstone 2023, sponsor Au-Zone;
  [poster](https://schulich.ucalgary.ca/sites/default/files/teams/4/2023%20Capstone%20Projects/52404-Swimming%20Alert%20and%20Monitoring%20Platform%20SALMON%20ENG%2024%20Booth%2051.pdf)):
  detect, then a separate **submerged / above-water classifier** (527 submerged vs 1,240
  above-water crops), tracked per swimmer, with a **four-stage colour escalation by time
  underwater: blue (above), green (briefly under), yellow (under a while), red (potential
  drowning)**, thresholds set by the operator. Underwater cameras. A clean UI pattern for an
  overlay or dashboard, and the same "classifier for submerged" idea as the two-class head
  detector.
- **Multi-camera head fusion** ([patent US 11756306](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11756306)):
  heads and bodies detected in several cameras, fused into one pool position, tracked with
  Hungarian matching, and an alarm *countdown* state when a drowning condition starts.
- **Head-under timer** ([patent US 12039851](https://patents.google.com/patent/US12039851B2)):
  times how long the head is below the surface against a threshold; head state from either
  a head-area / region ratio or a learned "is the nose/mouth above water" classifier. It
  explicitly lists the **lost-head case** but describes no mechanism. Nobody published one,
  so our ghost-track stage would be genuinely new work.
- A student project used a CNN head detector plus a Kalman filter in space-time
  ([Society for Science abstract](https://abstracts.societyforscience.org/Home/PrintPdf/17792)).

### PoolWatch, and why public drowning benchmarks overstate accuracy (fit: medium)
[sanju908-awsome/drowning_detection_system](https://github.com/sanju908-awsome/drowning_detection_system)
("PoolWatch", Oct 2026, no license): YOLO detection, DeepSORT tracks, optional MediaPipe
pose, **per-track motion features into a Random Forest** (SAFE / WARNING / EMERGENCY),
trained on public temporal CVAT annotations (very likely Z5cc's) and split by source video.
Architecturally close to our plan (tracks, then motion features, then a small classifier),
and candid that nothing is validated.

Its [dataset audit](https://github.com/sanju908-awsome/drowning_detection_system/blob/main/docs/dataset_report.md)
of the most-used Roboflow set, *"Drowning Detection And Prevention In Swimming Pools" v3*
(8,925 images, CC BY 4.0), found:
- **37 groups of byte-identical images shared across train, validation and test.** The test
  split is not independent, so mAPs reported on it partly measure memorization.
- Seven classes, three of them duplicates differing only by capitalization ("Drowning" vs
  "drowning", 1,620 vs 4,547 boxes).
- No source-video IDs, so near-duplicate frames from the same video can't be kept apart.
That's concrete support for treating the 90 to 99 % figures in single-image drowning papers
with suspicion, and for our own rule of splitting by whole clip.

### The latency budget: drowning timeline and the 10/20 standard (fit: high)
- **Surface struggle:** about 20 to 60 s (Pia; instinctive drowning response).
- **After submersion:** unconsciousness usually within 1 to 3 minutes (a journal review:
  "within 3 minutes of total submersion" in the animal studies reviewed
  ([Edmonds, SPUMS J.](https://www.dhmjournal.com/images/IndividArticles/28March/Edmonds_SPUMSJ.28.1.2-9.pdf)));
  irreversible brain damage believed beyond about 3 to 7 minutes in warm water
  ([Britannica](https://www.britannica.com/science/drowning)). Children's timelines weren't
  found separately.
- **The lifeguard 10/20 standard:** detect a swimmer in trouble within **10 s**, reach them
  within **20 s** (Ellis & Associates, 1991; [WIRH history](https://www.thewirh.com/blog/10-20)).
  In practice only about 42 % of scans finish under 15 s, averaging 22 s
  (Hunsucker & Davison 2013).
- **Commercial / standard alarm delays** once a body is on the bottom: 10 s (Poseidon),
  ≤ 15 s (ISO 20380).

Targets for us: an alert inside the surface struggle (**well under 20 to 60 s after
onset**) beats the ISO-style bottom detectors by definition. "Within 10 s of onset" would
match the human 10/20 standard and is a meaningful, explainable goal. Our current window is
3 s plus 1.5 s of right context (4.5 s per alarm), already inside that. Time-to-alarm
should be reported against these three references (10 s, 20 to 60 s, guard contact).

### The review Nick shared: "Advances and Challenges in Automated Drowning Detection and Prevention Systems" (fit: medium, context)
[MDPI *Information* 15(11):721, 2024](https://www.mdpi.com/2078-2489/15/11/721) (open access;
the ResearchGate table link points to its Table 2). By the same UAE group as Shatnawi et al.
2023. Its **Table 2, "CV-based drowning detection approaches"**:

| Approach | What it is | Claimed | Stated limits |
|---|---|---|---|
| Body-part coordinates + DL | 16 **underwater** cameras capture key body parts, DL classifies, Arduino alarm | identifies drowning-specific body movements | many cameras, costly |
| IoT + transfer learning | motion sensor triggers a photo, a CNN classifies, phone alert (Raspberry Pi 3) | **99 %** accuracy | phone alerts easy to miss, Wi-Fi dependent |
| CV-ResNet (Shatnawi 2023) | 5 CNNs on **internet images** of drowning vs not | **100 %** validation and test | limited dataset, no rescue link |
| Improved YOLOv5 | 8,572 drone images of **simulated** drowning in pools; ICA + BiFPN modules | "high accuracy" | small data, drone constraints, not tested in the real world |
| AquaYOLO | YOLOv5 on drone video of **open water** | handles weather and lighting | **high false positives** from lighting and reflections |

Plus tables of sensor wearables (heart rate, pressure, goggles, strain yarn) and rescue
mechanisms (gantry robot throwing a ring buoy, a lifting pool floor, a robotic arm).

The review's own "open issues": no standardized large datasets, heavy class imbalance
(few drowning cases), and many false alarms. It proposes oversampling, cost-sensitive
learning and ensembles.

How it relates to Hydro-Knight:
- Every CV row is either **single-image classification** (some at 99 to 100 % on internet
  or staged images) or **underwater / drone** cameras. **None** works from fixed
  above-water pool video with per-swimmer tracking, measures time-to-alert, or reports
  false alarms per hour. That's the gap Hydro-Knight sits in, and it's worth saying in the
  README's "related work".
- The 100 % accuracy rows are the kind of result our 2026-09 audit taught us to distrust:
  no held-out-by-source split, no random baseline, no event-level scoring.
- Its prescription for imbalance (oversample, cost-sensitive loss) is generic. Our
  structure (the victim is one track among ~20 in the same clip, so every clip supplies
  matched negatives) is a better answer than resampling. It's a ranking problem within a
  scene, not a global class balance problem.
- Useful as a citable survey of the field's state in 2024.

### Manikin drills: a realistic way to create test events (fit: medium)
- **Vigilance Awareness Training (VAT) / DROP drills** (Ellis & Associates and others): a
  realistic manikin is placed on the bottom in a guard's zone without warning, and the guard
  must spot and retrieve it, often within 30 s; some wave pools use a 10 s recognition
  standard ([Redwoods guide](https://redwoodsgroup.com/resources/aquatic-drop-drills/);
  [Raging Waves](https://ragingwaves.com/safety/lifeguard-training-whats-it-all-about/)).
- A **2001 Ellis & Associates / Poseidon study of 500+ submerged manikins** reported an
  **average detection time of 1 min 14 s**, far beyond the 10/20 ideal. It's often cited as
  the reason DROP drills spread *(as summarized by Redwoods; primary report not found)*.
  Critics call the drill unscientific
  ([Water Safety Guy](https://www.watersafetyguy.org/tag/vat/)).
For us: parks already run these drills on their own cameras. **A partner park's drill
footage would be realistic, consented, labeled test events** (exact drop time and place)
for the late-stage "body under / head gone" signature, without waiting for real
emergencies. It doesn't cover the surface-struggle stage, but it would test the
disappearance / ghost-track logic.

### Heatmap detectors trained on little data (fit: medium)
[Jacquelin, Vuillemot & Duffner, "Detecting Swimmers in Unconstrained Videos with Few
Training Data" (MLSA 2021)](https://dtai.cs.kuleuven.be/events/MLSA21/papers/MLSA21_paper_jacquelin.pdf),
code and weights at [njacquelin/swimmers_detection](https://github.com/njacquelin/swimmers_detection)
(no license): a small **U-Net that outputs a heatmap of swimmer positions**, trained on a
small hand-labeled set with heavy augmentation (colour shifts, zoom-outs), robust across
camera distance, angle, light, reflections and resolution. Like P2PNet, it learns from
*positions*, not boxes, which matches our click labels. A heatmap model also degrades
gracefully on a half-visible head (a weaker peak instead of a missing box). That's good
for the "seen faintly" evidence that ghost tracks need.
Related: [Aalborg pool occupancy from low-quality video](https://vbn.aau.dk/en/publications/swimming-pool-occupancy-analysis-using-deep-learning-on-low-quali)
(YOLOv2 93.5 AUC, Tiny-YOLO 79.3 AUC running on a cheap embedded board).

### Camera hardware notes for a real install (fit: low now, high for deployment)
- **Polarizing filter for glare:** water-surface reflection is strongest near Brewster's
  angle (about 53° from vertical), where a rotatable linear polarizer can remove most of it;
  polarizer plus per-row fitting removed sun glint in maritime imaging
  ([Laser & Optoelectronics Progress 2021](https://m.researching.cn/articles/OJ67171597bb709b5f);
  [Sensors 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10490668)). Glare is one of
  Lynxight's named false-alarm sources and 4 % of water-park rescues were "limited by glare".
- **Thermal camera** (see "Sensing submersion directly"): only above-water body parts are
  visible.
- **Mounting:** above ~1.8 m (higher is better), viewing angle to the far water ≤ 70°
  (Lynxight patent); ISO 20380's worked example is 6 m high over a 20 × 12 m pool.
  **Two angles on every patch of water** is the commercial answer to crowd occlusion.

### Autoencoders for drowning: others tried our v1/v2 idea (fit: medium, cautionary)
- **He, Yuan & Liu, "A video system based on convolutional autoencoder for drowning
  detection"** ([Neural Computing and Applications 35(21), 2023](https://link.springer.com/article/10.1007/s00521-023-08526-9)):
  a lightweight convolutional autoencoder trained **unsupervised on normal swimming** to
  sidestep the lack of real drowning video, on an edge device, with their own underwater
  pool dataset. The abstract gives no numbers ("good comprehensive performance").
- The same group's **IET 2023 underwater device**
  ([IET Image Processing](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/ipr2.12765))
  first detects **near-vertical bodies with YOLOv5n** on a Jetson Nano, then applies an
  anomaly detector. Verticality as the gate, anomaly scoring after.
Both used **underwater** cameras, where a drowning body is large and clear. Our v2
autoencoder is the above-water, crowded, small-head version of the same idea, and it's
where the approach broke down. That supports the redesign: the autoencoder's problem was
the input (tiny, unreliable keypoints), and gating on a geometric cue (verticality /
head-only visible) before anomaly scoring is a pattern others found useful.
- **randhana/Drowning-Detection-** ([GitHub, MIT, 47 stars](https://github.com/randhana/Drowning-Detection-)):
  a person detector plus simple motion analysis on a tracked box. Another instance of
  "detect, then rules on the track".
- r/computervision thread, see the next subsection.

### Periodicity, done by a model (fit: medium)
**RepNet** ([Dwibedi et al., CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Dwibedi_Counting_Out_Time_Class_Agnostic_Video_Repetition_Counting_in_the_CVPR_2020_paper.pdf);
[Google blog](https://research.google/blog/repnet-counting-repetitions-in-videos/)):
class-agnostic repetition detection from raw frames via a **temporal self-similarity
matrix**, giving a per-frame "is this part of a repeating pattern?" score and a period
length. Trained on synthetic repetitions, no labels. For a head crop track, *regular*
bobbing (riding waves, playful bouncing) scores as periodic while distress bobbing should
not. The self-similarity matrix of a head crop over 10 s is cheap to compute even without
RepNet. A hand-built version: the autocorrelation of the head's vertical position and
visibility, compared with the pool's wave period.

### Evaluating with very few events (fit: high)
With 18 to 25 events per evaluation, intervals are wide and the usual shortcuts mislead:
- Report recall with an **exact binomial interval** (Clopper-Pearson or Wilson), not
  plus-or-minus a normal approximation; very small counts give unstable intervals
  ([ISPOR, rare-event CI robustness](https://www.ispor.org/heor-resources/presentations-database/presentation/ispor-17th-annual-international-meeting/robustness-of-confidence-intervals-for-rare-events)).
  For example, 0 of 25 caught has a 95 % upper bound of about 14 %; 5 of 25 is about 7 % to 41 %.
- **Resample whole clips** when bootstrapping, not windows or tracks (the clinical analogue:
  resample patients, not measurements).
- When comparing two models on the same clips, use **paired** tests: the standard few-shot
  confidence interval underestimates uncertainty, and paired comparisons partly fix that
  ([Lafargue et al., TMLR 2024](https://arxiv.org/pdf/2409.02850)).
- To find missed victims in unlabeled footage efficiently, **sample by model score**
  (stratified), not uniformly ([recall-estimation patent](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/12718121)).
- Thresholds with a false-alarm *guarantee*: conformal anomaly detection turns scores into
  p-values on held-out normal data, so a threshold picks a false-alarm rate. Temporal
  dependence breaks the standard assumption; adaptive weighted variants address it
  ([ICLR 2026, arXiv 2604.20122](https://arxiv.org/pdf/2604.20122); constant-false-alarm
  streaming detection: [Ozkan et al., IEEE TSP 2016](https://www.ee.bilkent.edu.tr/~kozat/papers/39_1.pdf)).
  This needs the hours of normal footage from section 6 to calibrate on.

## 12. Tooling worth knowing

- **EgoBlur** (Meta, **Apache 2.0**; [GitHub](https://github.com/facebookresearch/EgoBlur),
  [paper](https://ar5iv.labs.arxiv.org/html/2308.13093)): a face and licence-plate detector
  plus Gaussian blur for images and MP4, CPU or GPU. Could replace whole-frame pixelation in
  `make_pose_figure.py` for README figures and demo videos: blur only faces, keep the pool
  readable. Built for first-person footage, with no published small-face numbers, so check
  it on our distant faces (and keep pixelation as the fallback, since minors appear).
- **Local vision-language models on the Mac:** [MLX-VLM](https://www.mintlify.com/yocxy2/mlx-vlm/introduction)
  runs 160+ VLM architectures on Apple silicon (CLI, Python, local OpenAI-compatible
  server), including the Qwen-VL families. Qwen3-VL adds video understanding with event
  indexing; a 2B size exists ([model page](https://llmrun.dev/model/qwen-qwen3-vl-8b-instruct)).
  This is how a "second-opinion verifier" (section 4) could stay fully local and private.
  On the 8 GB M1, only the smallest sizes are realistic.
- **SAM 2 inside a labeling tool:** Label Studio's open-source ML backend runs SAM 2 video
  tracking (points or boxes as prompts, tracks forward and backward, survives leaving the
  frame), but the VideoVector tag that displays it is in paid tiers only
  ([Label Studio guide](https://labelstud.io/guide/ml_tutorials/segment_anything_2_video));
  CVAT's is enterprise-only. Both point the same way: **add a SAM 2 "propagate this victim"
  action to our own labeler**, where the rescue-timeline tab already has the clicks.

### RL attention ancestry (for the back-burner project)
- **Recurrent Attention Model** ([Mnih et al. 2014, arXiv 1406.6247](https://arxiv.org/abs/1406.6247)):
  a recurrent network sees one retina-like glimpse at a time (sharp centre, blurry
  surround) and learns *where to look next* with policy gradients, because there's no label
  for where to look. Extended to multiple objects ([Ba et al., ICLR 2015](https://liner.com/review/multiple-object-recognition-with-visual-attention)).
  A one-page conceptual match for "lifeguard scanning".
- **Dynamic Zoom-in Network** (section 9) is the detection-specific, practical descendant.
- Suggested learning path for Nick: RAM on a toy task, then a tile-picking policy replayed
  on our saved per-tile detections, rewarded by victim-track coverage minus tiles processed.

## 13. Who drowns, and where (context)
- US pool and spa fatal drownings: **about 358 per year** (2019 to 2021), **75 % under age 5**;
  380 child fatalities in 2021; **81 %** of child (under 15) fatalities with a known location
  were **residential** ([CPSC 2024 report](https://www.cpsc.gov/Newsroom/News-Releases/2024/CPSC-New-Drowning-Report-Shows-Increase-in-Child-Fatalities);
  [2020 submersion report](https://www.cpsc.gov/s3fs-public/2020-Submersion-Report-4-29-20_0.pdf)).
  Lifeguarded public water parks are a small share of deaths. Their value for this project
  is that they are where lots of near-drownings are *filmed and rescued*, giving labeled
  events.
- Water park rescues cluster in wave pools (26 to 38 %) and slides (section 3, Fielding).
- Implication for the story: the most deaths happen in unguarded home pools with one
  camera and no lifeguard. Hydro-Knight trained on guarded wave pools is an *assistant to
  a guard*. Saying that scope plainly in the README is honest and still meaningful.

### Offline tracklet linking: published versions of the "ghost track" (fit: high)
Our tracks are ~1 s pieces. Several tracking papers stitch short pieces back into long
trajectories after tracking, which is exactly the second stage the ghost-track idea needs:
- **AFLink** (appearance-free link, from [StrongSORT, IEEE TMM 2023](https://arxiv.org/pdf/2202.13514);
  [code](https://github.com/dyhBUPT/StrongSORT)): a tiny model that decides whether two
  tracklets are the same person **from positions and frame times only** (the last 30
  frames of each), at 1.7 ms per image on MOT17. Appearance-free is right for us: heads
  in water all look alike, and appearance features break under occlusion.
- **GSI** (Gaussian-smoothed interpolation, same paper): fills gaps in a linked trajectory
  with Gaussian-process regression instead of straight lines, with an adaptive smoothness
  (7.1 ms per image). It gives the "where was the head while hidden" estimate, and the gap
  length *is* the time under.
- **GIAOTracker** (VisDrone 2021, [arXiv 2202.11983](https://arxiv.org/pdf/2202.11983)):
  global Hungarian linking of short tracklets into long ones using appearance plus
  space-time distance.
- **Classic:** suspend a track during occlusion and match suspended tracklets to new ones
  with a motion model ([Kaucic et al., CVPR 2005](https://mlanthology.org/cvpr/2005/kaucic2005cvpr-unified)),
  and reject links when the gap is too long.
Design implication: build the ghost stage as **offline tracklet linking** with a
drowning-specific twist: a candidate link "same spot, reappeared after N seconds" is
allowed (and the gap recorded) where a generic linker would demand motion continuity.
AFLink's position-only input format is a ready template, trained on our own long tracks
(cut artificial gaps into them).

### Distressed swimmer, active victim, passive victim: three targets, three detectors (fit: high)
The lifeguard-training taxonomy (Red Cross / Pia; summarized by
[Mining Journal (Red Cross)](https://www.miningjournal.net/news/front-page-news/2018/09/red-cross-has-these-recommendations/),
[durfdenken.be](https://www.durfdenken.be/en/research-and-society/how-lifesguards-spot-someone-whos-drowning),
[Water Safety Guy](https://www.watersafetyguy.org/water-safety/water-assists/)):

| Stage | What it looks like | Camera signal (our terms) |
|---|---|---|
| **Distressed swimmer** | still swimming but **little or no forward progress**; aware, may signal or call | forward movement in head-widths drops while effort (splash, arm motion) stays |
| **Active drowning victim** | **vertical, slightly leaning back, no locomotion**, arms pressing down at the sides, **head low and tilted back, mouth at the waterline**, can't call out; 20 to 60 s | upright box, no displacement, irregular bobbing with the head going under, lateral motion beside the head |
| **Passive drowning victim** | unconscious: **motionless, face down at or below the surface**; causes include seizures, heart problems, breath-holding games, head strikes | a horizontal, still shape, or a head track that stops and disappears: the ghost-track / disappearance rule; ISO 20380 / Poseidon territory |

This lines up with our five distress signatures, and it suggests the alert should say
*which* stage it thinks it sees: "no progress", "active struggle" or "went under, not back".
Each stage maps to a different rule or feature, and a lifeguard would respond to each
differently. Passive victims explain a gap in our data: wave-pool rescue videos are mostly
active struggles, so the disappearance logic will need the manikin-drill footage or the
Fielding-style "found on the bottom" cases to be tested at all.

### Practitioner advice: r/computervision "Drowning recognition on video" (fit: medium; shared by Nick)
[Thread, about 3 years old](https://www.reddit.com/r/computervision/comments/17p1mjc/drowning_recognition_on_video/)
(Nick pasted the text; Reddit is blocked for my tools). The poster used YOLOv5 plus a
CNN-LSTM on short, low-quality clips, mostly *staged* instructional drownings, aiming for
a Jetson Orin. Useful points from the replies, paraphrased:
- **"When does drowning start?" is the annotation problem.** Exactly our onset label, and why
  our labels carry onset *and* guard contact, set by a lifeguard.
- **Zone "danger coefficient" by crowding:** the poster weighted beach zones as more dangerous
  the *fewer* people were nearby (nobody close to help). An **isolation feature** (distance
  to the nearest other head) is cheap to compute from head tracks. Nick should judge
  whether isolation raises risk in a guarded wave pool, where crowds also hide victims.
- **A former lifeguard in ML security:** many real drownings "just sink" with no drama, so
  data sparsity bites and a detector for the general case ("person in the water, then
  gone") is the product-safe default. That's the passive-victim / disappearance signature.
- **Someone who built one:** the naive "detect drowning class repeatedly in a region" is
  actually solid; the blocker was data. Their working rule: **track everyone, time above vs
  below water; past ~10 s under, focus on that person; alarm after another ~10 s.** They
  also tried **kids vs adults, alarming when a child is far from any adult too long**. It
  worked poorly as a drowning detector but well as an "unattended child" flag.
- **Track disappearance** as the core signal (needs a tracker that doesn't jump between
  people), possibly plus a motion model.
- **Streaming cost:** a CNN-LSTM is cheap at inference because only the hidden state is kept
  (linear in frames), while transformers are quadratic over the window.

Takeaways for us: the 10 s + 10 s submersion rule is a concrete, practitioner-tested
starting threshold for the ghost-track alarm (between Poseidon's 10 s and ISO 20380's
≤ 15 s). "Kid far from adults" and "isolated swimmer" are cheap risk-prior features, and
match Fielding's finding that guards profile at-risk swimmers in advance (11 to 12 % of
rescues). The streaming point favours **causal, stateful models** (causal TCN with cached
activations, or a GRU) for live use, over attention across long windows.

### Small-object detector tricks (fit: medium)
- **A stride-4 "P2" detection head:** YOLO normally predicts at strides 8/16/32, so an 8 px
  head covers about one cell at stride 8 but about 2×2 at stride 4. Adding P2 helps tiny
  objects in drone imagery at a cost in FLOPs and memory; dropping the stride-32 head to pay
  for it doesn't always help ([LAF-YOLOv10, arXiv 2609.14560](https://arxiv.org/pdf/2609.14560);
  [UAV P2 study, arXiv 2606.09081](https://arxiv.org/pdf/2606.09081);
  [Ultralytics thread](https://community.ultralytics.com/t/how-to-add-p2-head-on-yolov8/444)).
  A P2 head alone didn't fix recall for microcalcifications; tiling did more
  ([OpenCV forum](https://forum.opencv.org/t/detecting-breast-microcalcifications-with-yolo-small-object-detection-challenges/23780)).
  For the head fine-tune: try a P2 variant (Ultralytics' `-p2` model configs) as a cheap
  ablation against plain tiling.
- **Context decides tiny-object recognition:** "Finding Tiny Faces"
  ([Hu & Ramanan, CVPR 2017](https://openaccess.thecvf.com/content_cvpr_2017/papers/Hu_Finding_Tiny_Faces_CVPR_2017_paper.pdf))
  used scale-specific detectors whose templates are 99 % *context*, extending far beyond the
  face, and halved error on WIDER FACE. For heads in water, the shoulders, ripples and tube
  around a head carry most of the evidence. Don't crop head training patches too tightly,
  and let the detector see the waterline.
- **Crowds break NMS:** duplicate removal can delete a real second person standing close;
  **Adaptive NMS** raises the suppression threshold where density is high
  ([Liu et al., CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_Adaptive_NMS_Refining_Pedestrian_Detection_in_a_Crowd_CVPR_2019_paper.pdf));
  [R²NMS](https://arxiv.org/pdf/2003.12729) pairs full and visible boxes; density-guided and
  region-adaptive thresholds follow ([CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Wu_Boosting_Detection_in_Crowd_Analysis_via_Underutilized_Output_Features_CVPR_2023_paper.pdf)).
  Relevant to our new same-tile merge: it should merge near-copies of *one* person
  (contained, faint), not two touching heads, so an IoS test plus a confidence gap is safer
  than IoU alone. Heads overlap far less than bodies, a point in favour of head boxes in
  crowds.

### Dailymotion, Internet Archive (fit: low to medium)
Dailymotion's API search in English, French and Italian. The harvest pipeline already
rejected the Dailymotion re-uploads of our channel as duplicates (x2v1rjq, x127stf,
x2un5o7: frame-matched to our clips), which shows its duplicate check works. Unscreened
and plausibly in scope (titles only):
- "Bimbo cade tra le onde di una piscina" (child falls among the waves of a pool, 26 s):
  https://www.dailymotion.com/video/x7zv47x
- "Il sauve un enfant de la noyade dans une piscine blindée !" (saves a child in a packed
  pool): https://www.dailymotion.com/video/x2um56r
- "Son réflexe sauve un enfant de la noyade": https://www.dailymotion.com/video/x2udina
- "Les réflexes inouïs d'un maître nageur qui sauve une fillette…": https://www.dailymotion.com/video/x4613mo
- "Cette fille sauve sa sœur de la noyade dans la piscine": https://www.dailymotion.com/video/x7a28a6
- Context, not data: French news on Poseidon-style installs ("Dispositif anti-noyade à la
  piscine d'Andenne", x7yh1ov; "Noyades dans les piscines : l'IA à la rescousse", x7up8tq)
  and a staged drowning drill ("Altkirch : noyade fictive à la piscine", x7b8zzk).
Searching in other languages (French *noyade piscine*, Italian *piscina a onde*, Chinese
*泳池溺水监控*) finds footage that English searches don't; the harvest queries are
English-only today.

Internet Archive: nothing usable (wave-pool marketing and news; no CCTV rescues).

### "Intelligent eyes on water: YOLOv11-based real-time drowning detection" (shared by Nick) (fit: low, cautionary)
[Amer et al., Journal of Supercomputing 81:1242 (2025), open access](https://link.springer.com/article/10.1007/s11227-025-07732-7)
(11 citations, 6,500+ accesses). YOLOv11n on a Raspberry Pi 5, phone alerts, **mAP50 0.98**.
What's inside, from the full text:
- **Camera: underwater**, on the pool walls. A different problem from our above-water view.
- **Data: 633 original images** (from a cited dataset), **all drowning scenes simulated** by
  lifeguards and volunteers; "drowning" labeled as full submersion for more than 10 s.
- **Augmented to 1,519 images first, then split 80/20** (1,215 / 304). Crops, blur,
  brightness and greyscale variants of the same 633 photos therefore land in both train
  and test, so the 0.98 measures recognizing near-copies, not generalizing. The same
  leakage PoolWatch found in the Roboflow set.
- Single images: no tracking, no time-to-alarm, no false alarms per hour. The Pi runs at
  **3 to 4 FPS** (about 300 ms per frame at 384×640); the authors argue a "14 s temporal
  window" makes that enough.
- A training-size curve: 10 % of the data gave 71 % accuracy, rising with more data.
Useful bits: their "drowning = submerged more than 10 s" label is the same threshold the
Reddit practitioner and Poseidon use; and **3 to 4 FPS on a Raspberry Pi 5** is a real edge
data point (whole frame, nano model; our 17-pass tiling would be far slower there). As an
example for the README's "related work", it's the clearest case of why we split by whole
clip *before* any augmentation, and report a random baseline.

### What real drowning videos actually show: Carballo-Fazanes & Bierens 2020 (fit: very high)
["The Visible Behaviour of Drowning Persons: A Pilot Observational Study Using Analytic
Software and a Nominal Group Technique", IJERPH 17(18):6930](https://pmc.ncbi.nlm.nih.gov/articles/PMC7559254/).
20 international water-safety experts coded **real** drowning videos (mostly YouTube, **71 %
CCTV**) frame by frame. Quantitative results for 24 drowning persons in 23 videos;
qualitative coding for 52 persons in 42 videos. **This is the closest thing to an
empirical behaviour catalogue for exactly our kind of footage.**

Behaviours (52 persons):
| Behaviour | n | What it looks like |
|---|---|---|
| **Backward water milling** | **19** | arms out to the sides, **forcefully rotating backwards** to keep the front of the head out; **clear splashes** |
| Instinctive drowning response | 6 (+3 "climbing ladder") | arms out laterally pressing down on the surface |
| Distress | 6 | voluntary arm and leg movements keeping the mouth up |
| **Immediate disappearance** | **5** | went under on entry with **no visible behaviour** |
| **Nobody waved or shouted** | — | — |

Timing (24 persons; median, IQR; rescued n=19 vs not rescued before unconsciousness n=5):
| Measure | Rescued | Unconscious first |
|---|---|---|
| Length of each disappearance | **13 s** (7–20.5) | 90 s |
| Number of disappearances | 5 (2–6.5) | 10 |
| Number of resurfacings | 5 (3–6.5) | 13 |
| Incident start to rescue | **15 s** (10–45) | 229 s |
- **About six resurfacings per 30 s** (one every ~5 s); struggle length vs resurfacings
  R² = 0.53.
- **Lifeguards rescued everyone within 30 s** (median 17 s), against 48 to 63 s for
  bystanders.
- Recognition window: **at most about 2 minutes**.

Context (52 persons): **children apparently playing alone: 19**; children who jumped in
confidently and couldn't cope: 9; **bystanders watching without acting, sometimes within
arm's reach: 13**; a person in trouble pulling others under: 5; **toddlers in floating
rings who turned upside down: 3**.

What this changes or confirms for Hydro-Knight:
1. **"Head goes under repeatedly" is quantified:** real victims disappear and resurface
   about 5 times, each disappearance ~7 to 20 s, ~one resurfacing every 5 s. That's a
   concrete target for the ghost-track feature: **count of disappear-and-reappear-at-the-same-spot
   events and their lengths over the last 30 s**.
2. **Splash is real evidence, not noise:** the most common pattern (backward water milling,
   19 of 52) comes with clear splashes from arms rotating backwards at the sides. That
   backs Nick's "arms flapping at the sides" and the side-patch motion feature, and
   corrects the impression that drowning is always silent and still.
3. **"Immediate disappearance" (5 of 52)** has no visible struggle at all: only the
   disappearance rule can catch it.
4. **Two more cheap features:** *child alone* (small, isolated head) and *tube flip*
   (a person separating from an inverted ring). Both appear in the counts.
5. **Wave-pool caution:** one resurfacing every ~5 s is in the same range as wave
   periods (2 to 5 s), so periodicity alone won't separate distress from riding waves.
   Irregularity, staying in place and arm-splash have to come with it.
6. **The latency target is confirmed:** rescued victims were reached ~15 s after the
   incident started; an alert has to land within ~10 to 15 s of onset to be faster than a
   good guard, and well inside 2 minutes to matter at all.
7. The authors explicitly hope the findings "may assist automatic drowning detection
   developers", and call for coding more videos "until no new drowning behaviours are
   observed". **Our 66 labeled rescues, with a lifeguard's labels, are exactly that kind of
   data.** Coding behaviour types per clip (milling / IDR / distress / disappearance /
   passive) would be a small, citable research contribution in its own right.

### Recognizing actions of tiny people (fit: medium)
**TinyVIRAT** ([Demir, Rawat & Shah, ICPR 2020](https://arxiv.org/pdf/2007.07355);
[code + data](https://github.com/UgurDemir/Tiny-VIRAT)): natural low-resolution activities
cut from surveillance video, multi-label, with a progressive generative "enhance the tiny
clip" step plus weakly trained attention. It's the benchmark for "what is this 20-pixel
person doing?", the question behind a head-crop clip classifier. A model that classifies
short head-centred crops (milling / bobbing / swimming / riding a wave) could pretrain on
it. Low priority until head tracks exist.

### Alarm fatigue: why the false-alarm budget is the product (fit: high for framing)
Hospital monitoring is the best-studied case:
- On general wards **99 %** of monitor alarms were false or non-actionable (86.7 % in an
  ICU); nurses exposed to more false alarms responded **more slowly** (CHOP video study,
  4,962 alarms; [ECRI award summary](https://www.ecri.org/components/HDJournal/Pages/2014-HD-Award-CHOP.aspx)).
- Response time rose with **the number of non-actionable alarms in the preceding 2 hours**
  (36 nurses, 210 h, 5,070 alarms; [J. Hospital Medicine](https://cdn.mdedge.com/files/s3fs-public/pdfs/journals/jhm2331.pdf)).
- Across studies only **1 to 26 %** of alarms were actionable; lab psychology shows people
  quickly learn to ignore frequent false alarms
  ([review](https://reference-global.com/download/article/10.2478/fon-2022-0023.pdf)).
For a lifeguard, every alert costs a look away from their scan. Nick's 10 to 20 per hour
is a start; the evidence says the *rate in the last couple of hours* drives whether
alerts get ignored, so false alarms should be reported per hour *and* checked for bursts
(e.g. many alerts during wave-on periods).

### Data ethics and copyright practice (fit: medium, matches what we do)
Standard practice for research video datasets (Kinetics, AudioSet) is to **publish video
IDs, timestamps and labels, never the media**; YouTube's terms restrict redistribution,
and fair-use arguments for redistributing clips are untested
([HF forum summary](https://discuss.huggingface.co/t/share-and-make-a-dataset-of-youtube-videos-publicly-available-with-a-link-in-research-paper/173769);
[NeurIPS 2026 poster on legal data infrastructure](https://neurips.cc/virtual/2026/poster/139996);
[VICTOR, dataset copyright auditing](https://arxiv.org/pdf/2512.14439)). Our manifest
(URLs, times, labels; no video in git) already follows this, which is worth saying
explicitly in `DATA_SOURCING.md` and the README with those references.

### Risk-based attention, already patented: Lynxight "swimmer analytics" (fit: high, design validation)
[US 11159769, "Drowning detection enhanced by swimmer analytics"](https://patents.google.com/patent/US11159769),
assignee **Lynxight Ltd.** Each swimmer gets a *descriptor*:
- age group from height (**under 1.20 m = child**), estimated skill from pace, style and dive rate
- static vs moving, speed, continuous swim time and distance, predicted path
- **interaction:** watched by, playing with, or surrounded by others; a child **without an
  adult within ~2, 3 or 5 m**
- location vs markers (deep water, diving board); for example, higher risk for under-12s
  in deep water
- whether a spotter or guard can see them (a guard fixated on one swimmer for minutes
  raises everyone else's risk)
Risk is re-estimated every 5 to 10 minutes and **changes two things**:
1. **Compute:** higher-risk swimmers get more features, higher resolution and frame rate,
   smaller patches, more frequent inspection. That is the "scan harder where it matters"
   idea Nick described, and what the RL tile-scheduler would learn.
2. **Thresholds:** lower alert thresholds for higher-risk swimmers (a detection between two
   confidence levels alerts for a high-risk swimmer only), or routing to a slower,
   lower-miss algorithm.
Also: a laser or projector above the pool **points at the swimmer of concern**.

Takeaways: (a) the risk-prior features suggested earlier (child-sized, alone, deep zone,
tube) are the ones a commercial system actually uses; (b) "risk-weighted thresholds" is a
principled way to spend the false-alarm budget where misses are likeliest; (c) for a
non-commercial research project, building on these ideas is fine, but they're patented,
which matters if it ever became a product.

### DEWS (Eng et al., the field-tested academic system)
[IEEE TCSVT 18(2):196–210, 2008](https://data.epo.org/gpi/EP3408841A4) (citation via EPO);
[ICCV 2003 version](https://mlanthology.org/iccv/2003/eng2003iccv-automatic/): block-based
background model for outdoor pools, a Markov Random Field tracker for partial occlusion, and
**lifeguard-defined "visual indicators of water crises" turned into swimmer descriptors and
fused by a small neural network (a functional link network)**. Deployed live at an outdoor
pool; the WIRH summary reports 97 % over six months. Full texts are paywalled.
**This is the closest published precedent for the redesign: expert-defined measurements
plus a small learned combiner, on overhead pool video.**

### More "spot the drowning" quiz videos (fit: medium)
Recognition-quiz videos are exactly our kind of data (an uncut view, then the reveal), and
they exist beyond the main channel. Unscreened (titles only):
- "Spot The Drowning Swimmer #1" and "#2" (Jack Yeo, 2:17 and 3:50):
  https://www.youtube.com/watch?v=BCdnZ5LKJ0s, https://www.youtube.com/watch?v=nc51fmaUQz0
- "Can You Spot a Drowning Person? Swim Safety Test" (WebMD): https://www.youtube.com/watch?v=beNheoRRdKk
- "Can YOU spot the drowning child?" (The Sun, also in Z5cc's set): https://www.youtube.com/watch?v=gY0ASobuVok
- "Summer camp video shows one lifeguard watching as child drowned" (FOX 2, CCTV, a
  fatal case; sensitive): https://www.youtube.com/watch?v=8sLJ11pW22o
- "Terrifying video shows hotel worker rescuing drowning toddler" (TODAY): https://www.youtube.com/watch?v=C72ndG4lM6c
("How no swimmers noticed toddler drowning at crowded water park wave pool",
f8O-gVzpEAg, is already on our blocklist.)
Search terms that find these: *"spot the drowning"*, *"can you spot"*, *"drowning test"*,
*"caught on camera pool"*. Lifeguard-training organizations (StarGuard, Ellis, Red Cross,
YMCA) publish similar recognition videos.

### Modelling the water itself (fit: medium)
- **Water as a "dynamic texture":** the standard mixture-of-Gaussians background model fails
  on rippling water; dynamic-texture models (Mahadevan & Vasconcelos, "Background subtraction
  in highly dynamic scenes", CVPR 2008; "Spatiotemporal saliency in dynamic scenes", PAMI
  2010) and generalized Stauffer-Grimson models handle it
  ([MIT CSAIL background subtraction](https://groups.csail.mit.edu/vision/app/research/backgroundsubtraction.html);
  [VisionBib listing](https://visionbib.com/bibliography/motion-i763dbs1.html)); fuzzy colour
  histograms damp colour flicker from moving backgrounds while keeping moving objects
  ([KAIST](https://koasas.kaist.ac.kr/handle/10203/103121)). Use for us: a per-camera water
  background model gives "how unusual is the motion around this head *compared with the
  water's normal motion here*", the relative splash measure, without any learning.
- **Cheap optical flow for the side patches:** NeuFlow runs about 30 FPS on a Jetson Orin
  Nano with accuracy similar to RAFT, 10 to 80× faster; NeuFlow v2 over 20 FPS at 512×384
  ([arXiv 2403.10425](https://arxiv.org/html/2403.10425v1); [v2 review](https://www.themoonlight.io/en/review/neuflow-v2-high-efficiency-optical-flow-estimation-on-edge-devices)).
  Only small patches around in-range heads need flow, so cost is modest even offline on
  the Mac. OpenCV's Farnebäck flow is the zero-dependency first try.

### Where in the pool: zones, depth transitions, blind spots (fit: medium, for zone drawing)
- **Depth transitions are the hot spot:** one water-park supervision designer argues the
  **transition from one depth to another** is the most dangerous zone, and that more
  incidents happen in shallow "risk zones" than in deep "danger zones"
  ([Amusement Logic](https://amusementlogic.com/company-news/position-of-lifeguards-and-pool-design/)).
  Risk-management checklists ask for the drop-off to the deep end to be marked, the steps
  watched, and a zone for zero-depth entry where non-swimmers gather
  ([Silver Lining zones checklist](https://www.thesilverlining.com/businessblog/blog/aquatics-identifying-zones-of-protection-in-your-pool)).
- **Elevated stands have a blind spot directly in front of and below them**; zones should
  overlap and fit a 180° sweep
  ([Silver Lining bulletin](https://www.thesilverlining.com/hubfs/Safety%20Summaries%20and%20Sample%20Safety%20Policies/Lifeguard%20in%20the%20Zone%20Technical%20Bulletin%20WB-3091%203-22.pdf)).
  Our clips are often filmed from beside a stand, so the near-field blind spot of the
  *guard* is well covered by the *camera*, an argument for the camera as a complement.
- **Wave-pool incident reports** mention children carried toward the deep end by waves
  after losing the grip on a rope, a child found face down in ~4 ft, and a 4-year-old
  pulled unconscious from a 2 ft section where tubes were handed out without size limits
  (news reports: [Oneindia](https://www.oneindia.com/2012/02/28/bangalore-12-yr-old-girl-dies-in-wonderla-wave-pool.html),
  [Las Vegas Sun](https://m.lasvegassun.com/news/1997/aug/19/near-drowning-victim-in-very-critical-condition/),
  [Fox News](https://www.foxnews.com/story/4-year-old-boy-drowns-in-wave-pool-at-california-water-park.amp)).
Idea for the Zones tab: alongside the coverage polygon, let Nick mark a **depth-transition
line** (where the wave pool goes from standing depth to over-head). "Drifted past the
transition and stopped making progress" is a strong, explainable feature, and it's the
"followed bigger kids past the lifeline" story in Fielding's reports.

### Our labels already match published weak-supervision schemes (fit: high)
- **Pointly supervised action localization** ("Spot On", [Mettes, van Gemert & Snoek, ECCV 2016](https://ivi.fnwi.uva.nl/isis/publications/2016/MettesECCV2016/MettesECCV2016.pdf);
  journal version [arXiv 1805.11333](https://ar5iv.labs.arxiv.org/html/1805.11333)):
  annotators click **points on a sparse set of frames** instead of drawing boxes; an
  overlap term between candidate spatio-temporal "tubes" and the points goes into a
  multiple-instance learning objective. **Points matched box supervision** on UCF Sports and
  UCF101 at a fraction of the labeling time. That's exactly our victim clicks (onset,
  extras, last seen): a candidate head track is scored by how well it passes through the
  clicks, the principled version of our victim matcher, and it supports *training* a
  localizer, not just evaluating.
- **Single-frame (timestamp) supervision** (SF-Net, [Ma et al., ECCV 2020](https://mlanthology.org/eccv/2020/ma2020eccv-sfnet/),
  [code](https://github.com/Flowerfan/SF-Net)): one marked frame per action; the method
  grows pseudo-labels around each mark and mines background from unmarked frames, and comes
  close to full supervision. Our onset and guard-contact times are timestamp labels of
  this kind, so frame-level "distress" pseudo-labels between onset and contact are a
  standard, defensible way to train a per-head classifier.
- Taken together: **we don't need dense boxes or per-frame labels to train a supervised
  head-track model.** The existing labels plus multiple-instance learning are enough.

### Weakly supervised video anomaly detection, the detection-free family (fit: medium)
If the head pipeline stalls, the established fallback learns from clip-level labels only:
- **Sultani et al. 2018 / UCF-Crime** ([CRCV project](https://www.crcv.ucf.edu/projects/real-world/)):
  each video split into fixed segments forming a "bag"; a MIL ranking loss pushes the
  top-scoring segment of anomalous videos above the top segment of normal videos.
- **RTFM** ([Tian et al. 2021, arXiv 2101.10030](https://ar5iv.arxiv.org/html/2101.10030)):
  selects positives by feature *magnitude* and adds dilated convolutions plus self-attention
  for long and short temporal context; **MGFN** (AAAI 2022) adds magnitude-contrastive loss.
  The plain ranking loss is known to be fragile to noisy segment labels
  ([WACV 2024 workshop, MBZUAI](https://dclibrary.mbzuai.ac.ae/cvfp/396)).
- **Spatio-temporal weak supervision** ([Gianchandani, Tirupattur & Shah, arXiv 2605.13746, May 2026](https://arxiv.org/abs/2605.13746)):
  localizes anomalies **in space as well as time** from video-level labels with MIL over
  spatiotemporal regions (UCF Crime2Local). The closest to "find *where* in the pool, from
  clip-level labels".
For us these are second-best: we have much stronger labels than "this clip contains an
anomaly" (onset, contact, victim position), so we can supervise at the track level. But
MIL over zone tiles × time is a sound baseline to beat, and it doesn't depend on the
detector seeing the victim.

### The "small supervised model" on per-head measurements: which model (fit: high)
Once each head track has a handful of measurements, the classifier is a small **tabular**
problem: dozens of positives (victims) and thousands of negatives (everyone else).
- **Gradient-boosted trees vs neural nets:** a large benchmark finds the gap usually
  negligible, light tuning of boosted trees often matters more, and **boosted trees handle
  skewed, heavy-tailed features much better** ([McElfresh et al., "When do neural nets
  outperform boosted trees on tabular data?"](https://arxiv.org/html/2305.02997v4)). Our
  features (time under, displacement, splash energy) will be heavy-tailed, the same
  problem that hit v2's box-speed features.
- **TabPFN** ([Hollmann et al., Nature 637:319, 2025](https://pubmed.ncbi.nlm.nih.gov/39780007/);
  [Hugging Face](https://huggingface.co/Prior-Labs/TabPFN-v2)): a pretrained transformer
  that classifies small tables *without training*. It beat tuned baselines on datasets up
  to 10,000 rows, in seconds rather than hours of tuning. v2.5 (Nov 2025) reportedly handles
  50,000 rows; CPU is fine for about 1,000 rows. Ideal for a few hundred labeled windows.
- Practice: split **by whole clip** (leave-one-clip-out or grouped k-fold), tune only
  inside the training folds, compute any normalization from training clips only, and
  report per-clip variance, not just a pooled score.
- Order of attack: rules first (explainable baseline), then boosted trees / TabPFN on the
  same measurements, then any neural model only if it clearly beats them on the same
  splits and false-alarm budget.

### Calibration and new cameras (fit: medium)
- **Turning scores into probabilities with little data:** with fewer than ~2,000 calibration
  examples, **Platt scaling** (fit a sigmoid) beats isotonic regression, which overfits
  ([Niculescu-Mizil & Caruana, arXiv 1207.1403](https://arxiv.org/pdf/1207.1403));
  calibrate on held-out clips, never on training data. Platt has known pitfalls after
  undersampling negatives ([arXiv 2410.18144](https://arxiv.org/pdf/2410.18144)). Calibrated
  scores make risk-weighted thresholds (Lynxight-style) and per-camera thresholds meaningful.
- **Every camera is a new domain:** mounting angle, height and lighting are a
  non-negligible domain gap; one EPFL thesis narrows it by learning homographies per camera
  ([EPFL Infoscience](https://infoscience.epfl.ch/record/306176?ln=en)) (our perspective
  warp idea again). Test-time adaptation on a little unlabeled footage helps but can
  collapse in fast-changing scenes ([arXiv 2503.02170](https://arxiv.org/pdf/2503.02170);
  [domain adaptation survey, arXiv 2105.13502](https://arxiv.org/pdf/2105.13502)).
  Practical version: at setup, record ~1 hour of normal footage per camera and use it to
  (a) fit the size/perspective map, (b) set that camera's alarm threshold for the
  false-alarm budget, and (c) fit the water background model. That's a "commissioning"
  step, matching ISO 20380's requirement to test each installed system on site before
  go-live.

### Water polo: heads in splashing water (fit: medium)
- **Berkeley "Cal Water Polo"** ([data-x project](https://datax.berkeley.edu/projects/cal-water-polo/)):
  single-camera player and ball tracking with Fast R-CNN plus a hierarchical
  track-by-detection; ~0.73 mAP for players. Their key difficulty: **half the players wear
  light caps that look like splashes**, the same confusion we'd face between white water
  and heads. No public dataset mentioned.
- **"Player tracking in water polo"** ([Noordover, LIACS MSc thesis 2023](https://theses.liacs.nl/pdf/2022-2023-NoordoverKevin.pdf)):
  colour features plus detection; notes water polo is barely covered in public research.
- Swim-competition swimmer datasets ([Woinoski et al. 2020](https://ar5iv.arxiv.org/html/2001.04433))
  describe collection methods but no public release was found.
Takeaway: no ready head-in-water dataset exists to pretrain on. Our own labeled pool heads
(plus CroHD for generic heads) is the realistic route, and the water polo final in Z5cc's
list (79 min of 1080p) is a ready source of hard negatives *and* of unlabeled heads in
churning water for self-training.

### Frozen foundation-model features for head-crop clips (fit: medium, later)
For classifying a short head-centred clip (milling / bobbing / riding waves / swimming)
without training a video network from scratch:
- **V-JEPA 2** (Meta, [arXiv 2506.09985](https://arxiv.org/pdf/2506.09985);
  [code](https://github.com/aod321/vjepa2)): a self-supervised video encoder; with the
  encoder frozen and a small attentive probe trained on top, it reaches 77.3 % on
  Something-Something v2 (motion-heavy actions) and 90.2 % on Diving-48 (fine-grained
  body motion) for the largest model *(authors' numbers)*. Motion-sensitive features from
  frozen weights are what a few dozen labeled victims can afford. Smaller checkpoints
  should be the starting point on our hardware.
- **DINOv3** (Meta; [model card](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m);
  [guide](https://lightly.ai/guides/the-ml-engineers-guide-to-dino-dinov2-dinov3)):
  per-frame dense features that hold up for video segmentation and tracking with frozen
  weights, with distilled small ViT/ConvNeXt sizes. Per-head appearance embeddings could
  help re-link a head after a gap (as a check on a position-only link, not the main cue).
Neither has published numbers on tiny, low-resolution crops (TinyVIRAT's territory), so
it's an experiment: frozen features plus TabPFN or logistic regression on clips around
victim clicks vs random other heads, evaluated by clip.

### Trackers built for similar-looking, erratic, occluded people (fit: medium)
DanceTrack (look-alike people, non-linear motion, constant occlusion) is the public
stand-in for a crowded wave pool:
- **GHOST, "Simple Cues Lead to a Strong Multi-Object Tracker"**
  ([Seidenschwarz et al., CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Seidenschwarz_Simple_Cues_Lead_to_a_Strong_Multi-Object_Tracker_CVPR_2023_paper.pdf)):
  a simple motion model plus keeping **inactive tracks** around and re-matching them with
  appearance, adapted per sequence, beats far more complex trackers. The name is a
  coincidence with our "ghost track" idea, but the mechanism (hold lost tracks, re-match
  later) is the same family.
- **Learned motion predictors:** a Mamba-based predictor of past trajectories reached 63.2
  HOTA on DanceTrack ([U. Waterloo thesis 2025](https://uwspace.uwaterloo.ca/items/0123e2e6-8212-4663-b034-b7fde8202da9));
  a self-supervised depth-and-mask association cue improved SportsMOT / DanceTrack
  ([Khanchi et al., BMVC 2025](https://bmvc2025.bmva.org/proceedings/333/)).
For heads in water, a learned motion model has a natural target: **a submerged head doesn't
move far**, so "where will it reappear" is a tight prior that a generic pedestrian motion
model gets wrong. That's a cheap thing to learn from our own long tracks.

### Other sensors, for completeness (fit: low)
- **Thermal:** a thermal head-shape detector reached 92.4 % inside the pool area (off-hours
  intruder detection, [MMU 2013](https://shdl.mmu.edu.my/4442/)); a FLIR patent uses a
  long-wave infrared camera to decide whether a person is submerged and alert past a time
  limit ([patent 7839291](https://patents.justia.com/patent/7839291)); an Aalborg student
  project built drowning detection on thermal cameras
  ([AAU](https://projekter.aau.dk/detecting-and-preventing-drowning-accidents-using-thermal-cameras-d9484db7.html));
  multimodal (RGB + thermal) swimmer detection in reservoirs from Gdańsk
  ([PG repository](https://pub.pg.edu.pl/en/publication/161539/detection-of-people-swimming-in-water-reservoirs-with-the-use-of-multimodal-imaging-and-machine-lear)).
  Low resolution limits it to presence, not behaviour
  ([allPCB note](https://www.allpcb.com/allelectrohub/infrared-thermal-imaging-in-anti-drowning-systems)).
- **Event cameras:** ARQUIMEA tested a Prophesee event camera with a BrainChip Akida
  neuromorphic processor on a drone to spot distressed swimmers, claiming lower latency and
  power (vendor announcement, [Business Wire 2025](https://www.businesswire.com/news/home/20250416985472/en)).
  Event sensors respond only to change, a natural fit for splash, but there's no public
  pool data.

## 14. State of the field, measured (fit: high for positioning)
- **arXiv:** only **14 papers** have "drowning" in the title, and just one is pool drowning
  detection (NEPTUNE, 2018, which claims no false positives from 1 to 5 s of video with
  hand-made equations, [arXiv 1805.02530](https://www.arxiv.org/pdf/1805.02530)). Two 2026
  entries are rescue robotics or drones ([QuadBoat](https://arxiv.org/abs/2607.13633),
  [UAS search and rescue](https://arxiv.org/abs/2604.18088)). The academic work lives in
  MDPI / Springer / IEEE journals and is dominated by single-image YOLO classifiers on staged
  or augmented data.
- **GitHub:** the most-starred "drowning detection" repository has **47 stars**; most are
  student YOLO demos (Roboflow data, single frames) or wearables. No repo has real rescue
  video with per-victim labels and an honest evaluation (see Z5cc and PoolWatch for the
  closest).
- **Datasets:** no public dataset combines real incidents, fixed above-water cameras,
  per-person tracks and event timing. SwimID (on request, indoor CCTV) and Z5cc (YouTube
  boxes) come closest.
What this means for the portfolio story: a write-up of Hydro-Knight with **real lifeguard-labeled
rescues, victim-scoped scoring, a random baseline, alerts per hour and time-to-alert**
would be one of very few rigorous public pieces of work on this problem. An arXiv preprint
or a workshop paper (sports analytics, e.g. MLSA, or CVPR's computer vision in sports) is
realistic once the head pipeline produces a real result, *including* a negative one
explained well.
- **Where to publish:** MLSA (Machine Learning and Data Mining for Sports Analytics) runs
  each September with ECML/PKDD (2026 in Naples; past calls took 9-page long papers and
  2-page extended abstracts; 2025 accepted 11 of 24) ([MLSA 2023 call](https://dtai.cs.kuleuven.be/events/MLSA23/submission.php);
  [proceedings](https://link.springer.com/book/9783032151643)). Jacquelin et al.'s swimmer
  detector appeared there. The CVPR computer-vision-in-sports workshop is the other natural
  venue; the 2027 call wasn't found yet. A 2-page extended abstract ("an honest evaluation
  of drowning detection on real rescue video") is a realistic near-term target.
- Kaggle: no drowning *video* dataset found. The 5,613-image underwater set and its
  69,512-frame big sibling (Alzaabi et al.) are image sets, not clearly public.

### "Head tilted back": is it even measurable at our resolution? (fit: low to medium)
Low-resolution head-pose work suggests coarse orientation is recoverable from very small
heads: **randomised ferns classified pose from heads ~10 px square**
([Benfold & Reid, BMVC 2008](https://www.bmva-archive.org.uk/bmvc/2008/papers/272.pdf));
CNNs trained on mixed resolution then fine-tuned for continuous pose ran in 1.2 ms per head
([Mukherjee et al. 2016](https://www.scitepress.org/Papers/2016/57869/57869.pdf)); transfer
learning handled larger tilt ranges across camera views
([Rajagopal et al., IJCV 2014](https://cdn.iiit.ac.in/cdn/cvit.iiit.ac.in/images/JournalPublications/2014/raman_surveillanc.pdf));
perspective distortion needs region-specific classifiers
([FEGA-MTL](https://cris.fbk.eu/handle/11582/300802)). None target "tilted back with the
mouth at the waterline", and from an elevated camera a tilted-back face looks like a face
seen from above. Verdict: the instinctive-drowning "head back" cue is a long shot at
40 to 60 px swimmer height. Visibility, bobbing and splash are the realistic signals, with
head pose at most a near-field bonus.

### The human side: an assistant can make the guard worse (fit: high for framing and design)
- **Automation bias causes misses:** in a monitoring study, accuracy on the events the aid
  *didn't* flag fell from **97 % without the aid to 65 % with it**: people stop watching
  for what the system is supposed to catch ([Wikipedia: automation bias](https://en.wikipedia.org/wiki/Automation_bias);
  [Parasuraman & Manzey review, PubMed 25886768](https://pubmed.ncbi.nlm.nih.gov/25886768/)).
  Complacency is strongest when the aid is highly and constantly reliable and the person is
  multitasking, which describes a lifeguard on a stand.
- **False alarms damage trust more than misses** in alerting systems; aids should signal
  when they're uncertain; letting operators experience automation misses in training
  reduced omission errors toward the alarm function
  ([NASA NTRS](https://ntrs.nasa.gov/api/citations/19980048379/downloads/19980048379.pdf);
  [Alberdi et al., SAFECOMP 2009](https://openaccess.city.ac.uk/id/eprint/384/2/Alberdi-et-alSAFECOMP09.pdf)).
- **Likelihood alarms:** alarms graded by confidence and urgency keep trust better than
  binary alarms with many false positives (MIT line of work, via
  [ASU modelling paper](https://asu.elsevierpure.com/en/publications/modeling-the-interplay-between-human-trust-and-monitoring/)).
- **Industry framing:** Royal Life Saving Australia insists the AI is "an assistant, not an
  authority" and is building complacency training; Lynxight answers complacency with
  training, zone discipline and two-angle coverage
  ([iTWire](https://itwire.com/business-it-news/data/ai-becomes-the-quiet-lifeguard-watching-over-australia%E2%80%99s-pools);
  [Lynxight](https://www.lynxight.com/content/will-ai-make-lifeguards-stop-watching-the-water)).
Design implications for us: (1) Nick chose a single "alert" level for now, which is right
for evaluation. Note though that the research favours alerts that **carry a confidence**
(even if shown simply) over a bare buzz. (2) The system should **show what it can't see**
(out-of-range water, lost tracks, glare), so the guard doesn't assume coverage it doesn't
have; ISO 20380's "report degraded performance" requirement says the same. (3) The README
should state the risk plainly: an assistant that guards rely on can increase misses on
what it doesn't flag.
- **Australia is the live deployment to watch:** Royal Life Saving Australia partners with
  Lynxight, ~100 facilities nationwide, framed strictly as a supplement ("cannot have a
  conversation with a patron or pull someone from the water")
  ([RLSA Dec 2024](https://www.royallifesaving.com.au/about/news-and-updates/news/2024/dec/royal-life-saving-australia-champions-ai-aquatic-safety-technology-with-lynxight);
  [The Nightly](https://thenightly.com.au/australia/inside-the-lifesaving-ai-technology-that-has-already-saved-two-australians-as-it-rolls-out-nationwide-c-22710898)).
  **Brisbane City Council's two-pool trial** is explicitly meant to assess detection
  accuracy and response times
  ([Australasian Leisure Management](https://www.ausleisure.com.au/news/brisbane-city-council-to-trial-lynxight-anti-drowning-technology-at-two-public-pools)).
  If it publishes results, they'd be the first independent field numbers for an overhead
  system. Worth checking back. (Australia: 357 drowning deaths in 2024–25, 35 in pools.)

## 5. Convolution padding, edges and missing frames

Nick asked specifically how to handle the edges of convolutional padding. The short answer
from the literature: **don't pad with made-up values when real data exists; pad only where
it doesn't, and tell the model which values are real.**

1. **Use real context instead of padding (what v2 already does).** v2's encoder uses
   unpadded ("valid") convolutions fed the scored window plus 44 real steps on each side,
   so no output ever depends on invented values. That's the same fix the streaming
   literature arrives at: chunked inference with zero padding causes "dramatic accuracy
   reduction" because the network never saw padding at chunk edges during training; the
   cure is a **history cache** of real past inputs
   ([Real-time streaming Wave-U-Net, arXiv 2104.01923](https://arxiv.org/pdf/2104.01923);
   [pytorch-tcn streaming mode](https://pypi.org/project/pytorch-tcn/1.1.0/)).
2. **Where real data runs out (track start or end, gaps), mask instead of pad.** Partial
   convolutions treat padded or missing positions as holes and re-weight each output by
   the fraction of real inputs, beating zero, reflection and replication padding on
   ImageNet and segmentation ([Liu et al., NVIDIA 2018](https://arxiv.org/pdf/1811.11718);
   [project page](https://research.nvidia.com/labs/adlr/publication/partialconv-padding)).
   The same operator over time handles a track's undetected steps. v2's decoder still uses
   *reflect* padding inside the reconstruction blocks; partial-convolution padding is the
   drop-in upgrade there.
3. **Make missingness an input.** GRU-D's lesson (section 4): the mask and the time since
   the last real observation are informative. For drowning, the gap *is* the signal.
4. **Causal vs centred windows is a latency choice.** v2's encoder looks **44 steps (1.47 s)
   into the future** of each scored window, so every alarm waits for that right context
   (part of v2's 4.5 s alarm delay). A **causal** network (all context in the past,
   padding only on the left, never the right) alarms ~1.5 s sooner at some accuracy cost,
   and streams naturally: cache each layer's last activations (Fast WaveNet, MoViNets
   stream buffers), so each new frame costs one step, not a whole window
   ([MoViNets, arXiv 2103.11511](https://arxiv.org/pdf/2103.11511);
   [keyword spotting with dilated causal convs, arXiv 1811.07684](https://arxiv.org/pdf/1811.07684);
   [TinyDéjàVu, arXiv 2512.09786](https://arxiv.org/pdf/2512.09786)).
   The Reddit thread's point about LSTMs (constant cost per frame) is the same idea.
5. **Rule of thumb for any future temporal model here:** train and evaluate with exactly
   the padding regime it will see live. If live inference is causal and streaming, train
   causally, and include track starts (little history) in training rather than dropping
   short tracks.

### Two more evidence points on timing and supervision (fit: medium)
- **A fatal drowning on CCTV, analysed second by second** ([Marrone et al., Forensic Science
  International: Reports 2021, Bari](https://ricerca.uniba.it/handle/11586/374100)):
  surveillance footage of a 26-year-old man's death in a private pool; the submersion
  sequence broadly matched experimental physiology, with timing discrepancies the video made
  measurable. The full text was not accessible here; its phase timings would calibrate the
  "under too long" threshold for the passive-victim case.
- **Supervision gaps in child submersions:** in 1,537 paediatric submersion incidents, fatal
  cases had far longer *inactive supervision* (mean 16.1 min vs 3.2 min for non-fatal)
  ([PMC8392632](https://pmc.ncbi.nlm.nih.gov/articles/PMC8392632)). A system's value is
  largest exactly when nobody is watching that spot, a point for "show coverage" and for the
  attention/scanning design.
- Research ethics practice from the video-coding studies: **watch footage with audio muted**
  and code events, not identities (Carballo-Fazanes & Bierens). A cheap rule to adopt for
  our labeling and any shared figures.

## 15. License check for everything recommended (verified 2026-10-17 via GitHub / Hugging Face)
Hydro-Knight is AGPL-3.0. "Use" below means it can be a dependency or adapted code in this
repo; "eval only" means local experiments without vendoring its code or weights into the repo.

| Thing | License | For us |
|---|---|---|
| SAM 2 ([facebookresearch/sam2](https://github.com/facebookresearch/sam2)) | Apache-2.0 | use |
| SAMURAI | Apache-2.0 | use |
| `trackers` (BoT-SORT, OC-SORT, ByteTrack) | Apache-2.0 | already a dependency |
| RF-DETR (core sizes) | Apache-2.0 (XL/2XL separate) | use |
| YOLOV (multi-frame detection) | Apache-2.0 | use |
| TabPFN | code Apache-2.0; v2 weights under "priorlabs-1-1" | use (check the weights' terms) |
| EgoBlur | Apache-2.0 | use |
| NeuFlow v2 | Apache-2.0 | use |
| HeadHunter (head detector/tracker) | MIT | use |
| V-JEPA 2 | MIT | use |
| MLX-VLM | MIT | use |
| wesjos YOLO head models | AGPL-3.0 | use |
| SwimID repo | MIT (dataset under its own research-only agreement) | request data |
| StrongSORT (AFLink, GSI) | **GPL-3.0** | compatible with AGPL; reimplement the ideas or vendor with notice |
| P2PNet | **academic research only** (Tencent custom) | eval only |
| STG-NF | **CC BY-NC 4.0** | reimplement from paper (already our plan) |
| CoTracker | **CC BY-NC 4.0** | eval only |
| DINOv3 | **custom DINOv3 License** | read terms first |
| Tiny-VIRAT, SF-Net, Z5cc, zseng0912, PoolWatch, Jacquelin swimmers | **no license** (all rights reserved) | eval only / ask the authors |
| CroHD / HT21 | MOTChallenge terms | check before training |
| Roboflow drowning sets | CC BY 4.0 (most) | use with attribution, but see the leakage audit |

The pattern: Meta's and Roboflow's recent tools are permissive; academic research code is
often non-commercial or unlicensed. Reimplementing ideas from papers is always fine, which is
how STG-NF, AFLink-style linking and point supervision should come in.

### How a commercial vendor specifies cameras (fit: medium, supports the size cutoff)
Lynxight's camera guidance ([which cameras work](https://www.lynxight.com/content/which-cameras-work-with-ai-drowning-detection-systems))
gives no numbers but states the requirement in exactly our terms: **"enough detail that a
swimmer's head and shoulders remain distinct at the far end of the tank"**; higher
resolution beyond that costs bandwidth without helping. Also: steady frame rate (heavy
compression degrades in busy sessions), lenses matched to the tank (wide lenses distort the
edges, "where distress often occurs"), every patch of water from two angles, glare
control, PoE / ONVIF / RTSP, chlorine-proof housings. Our 60 px size cutoff is a measured
version of "head and shoulders distinct", which is a good line for the README: the cutoff
is how commercial installs are specified, not an arbitrary trim of hard cases.

### SAM 3: text-prompted detection plus tracking of every instance (fit: high as a labeler)
[SAM 3, "Segment Anything with Concepts" (Meta, Nov 2025; arXiv 2511.16719)](https://arxiv.org/pdf/2511.16719);
[Ultralytics docs](https://docs.ultralytics.com/models/sam-3);
[Roboflow video concept tracking](https://inference-models.roboflow.com/models/sam3-video-concept-tracking/).
Give it a short noun phrase (or an example crop) and it returns masks **and stable IDs for
every matching instance** in a video, picking up new ones as they enter. Architecture: an
image-level detector with a "presence head" plus a memory-based video tracker sharing one
backbone. Open-vocabulary; code, checkpoints and fine-tuning released under Meta's **SAM
License** (custom, read before vendoring).

Why it matters here:
1. **Auto-labeling the head fine-tune:** prompt "head" (or "swimmer's head") on sampled pool
   frames, keep SAM 3's masks as pre-labels, and Nick only corrects. That could replace
   most of the 2 to 3 hour pilot labeling estimate with review time.
2. **A tracking teacher:** SAM 3's IDs survive brief occlusions through its memory, so on the
   labeled rescues it can show what a strong tracker *could* keep for the victim (an upper
   bound for our ghost-track stage), and supply long tracks to train the relinking model.
3. **Not the live system:** far too heavy for a Jetson or for 30 fps over a whole pool; its
   role is offline teacher and labeler, like SAM 2 / SAMURAI above.
Caveats from users: **memory grows with object count more than resolution** (out-of-memory
on aerial scenes with hundreds of small objects, [HF discussion](https://huggingface.co/facebook/sam3/discussions/301));
the first release tracked objects one pass each, and **SAM 3.1 batches them (~7× faster at 128
objects on an H100)** and gained on the crowded-scene MOSEv2 benchmark
([Labellerr](https://www.labellerr.com/blog/sam-3-vs-sam-3-1-performance-comparison/)). Our
scenes have 30 to 200 heads, so use 3.1, run on Colab, and tile.
Unknown: how it does on 30 to 60 px heads in glare. Run it on the 10 hand-counted frames
first (we have exact head points to score against), with the same one-box-per-head rule.
- **Real-time open-vocabulary YOLOs:** YOLO-World (CVPR 2024; 35.4 AP on LVIS at 52 FPS;
  a 1280×1280 variant for small objects, [paper](https://openaccess.thecvf.com/content/CVPR2024/html/Cheng_YOLO-World_Real-Time_Open-Vocabulary_Object_Detection_CVPR_2024_paper.html))
  and **YOLOE** (2025; text, visual or prompt-free modes, faster than YOLO-World v2 at higher
  AP, runs on a T4 and an iPhone 12, [arXiv 2503.07465](https://arxiv.org/pdf/2503.07465)),
  both in Ultralytics (AGPL, compatible). Prompting "head" costs no labeling, so they belong in
  the head bake-off next to the fine-tune. They're the cheap middle ground between SAM 3
  (accurate, heavy) and a closed-set head detector (fast, needs labels).

### Wave-pool rules that show up on camera (fit: medium)
- **California SB 107** (after a 4-year-old drowned in a Great America wave pool): free
  Coast Guard vests for under-48-inch children and non-swimmers, **mandatory vests for under-48s
  in wave pools**, a **15-second alarm before waves start**, and guards positioned to reach
  anyone in **30 s** ([Aquatics International](https://www.aquaticsintl.com/?p=6291)). Cedar
  Fair applied an under-4-ft vest rule across 18 parks
  ([CoasterBuzz](https://coasterbuzz.com/News/cedar-fair-sets-new-pool-rule-after-death-of-4-year-old)).
  The Model Aquatic Health Code calls for free life jackets at wave pools; some parks use
  coloured wristbands so under-48s stay inside the 3-ft line
  ([Aquatics International, "Jacket Required"](https://www.aquaticsintl.com/?p=6495)).
- For us: (1) **life vests are visible** (bright, bulky): a vest-wearing swimmer is much
  lower risk of going under, a candidate feature or simply an exclusion; (2) the **wave-start
  horn and wave on/off cycle** are part of the operating rhythm, so a "waves on" scene state
  is legitimate context; (3) the **3-ft line / depth transition** is a real, rule-based
  boundary worth drawing in the Zones tab.

### "Spot the Drowning Child": an independent, MIT-licensed label set on our own clips (fit: very high)
[spotthedrowningchild.com](http://spotthedrowningchild.com/) (2015, Francisco Saldaña), an
educational game on the Charleston County training clips: watch a crowded wave pool, click
the drowning child before the lifeguard does. **Its code and data are MIT-licensed**
([FrankSalad/spot-the-drowning-child](https://github.com/FrankSalad/spot-the-drowning-child),
89 stars). `videos.js` gives, for each of **15 videos**: `drowningStartSecs`,
**`whistleSecs` (when the guard blew the whistle, i.e. human recognition)**,
`swimmerSavedSecs`, and a box around the victim (in the game's player pixels).

Compared with our manifest (computed 2026-10-17):
- **14 of the 15 are clips we have labeled** (the 15th, T5mDQeDkca0 / Rescue 19, is on our
  blocklist).
- **Onset agreement:** the game's onset is a median **2.4 s earlier** than ours, mean absolute
  difference **2.7 s**. Two independent labelers agree within a few seconds, a useful
  sanity check on our labels. One clip disagrees by **13.5 s** (`92b58607c295`, one of the 9
  un-held clips, whose onset may still be Nick's June coarse mark): worth a re-look.
- **Human recognition latency:** the guard's whistle came a **median 5.6 s after drowning
  onset** (IQR 2.6 to 7.8 s, range 2.0 to 24.3 s), and a median 6.5 s before our guard-contact
  time.

Why this matters:
1. **A concrete, human benchmark for the system:** "alerted before the guard's whistle" on
   these 14 clips. The whistle time is the best available measure of when a trained guard
   *recognized* the drowning, a finer reference than guard contact (which includes the
   swim to the victim). Beating 5.6 s median is a meaningful bar; matching it at a low
   false-alarm rate would already be a strong result for an assistant.
2. **A second annotator for free:** an inter-annotator check on onset (and, with care, the
   victim box, rescaled from the player size) that can be cited.
3. **Our onset is a little late** relative to the game's, so catches measured from our onset
   are slightly generous on timing. Report time-to-alert against both.
(The game page also has a comparison article and a creator tool for adding videos.)
Notes for using it: the game's player is **854 px wide (480p embed)**; `findBoxStyle` gives
the victim box as a CSS offset relative to that player, so it rescales to our frames by
the video's width ratio (1920/854 for 1080p). A systematic **GitHub code search for our
clips' YouTube IDs** found only two label sets: this game, and Z5cc's dataset (plus an
unlicensed copy at VuongQuocHung/Drowing-Dection-System). Nothing else public labels this
channel.
- More Dailymotion hits (titles only, unscreened): "Une maître-nageuse sauve un enfant de la
  noyade dans une piscine à vagues bondée" (a guard saves a child in a packed wave pool,
  https://www.dailymotion.com/video/x87kvut, possibly a re-upload of a channel clip; the
  harvest duplicate check would tell), and "How Hard It Could Be To Spot A Drowning Child"
  (https://www.dailymotion.com/video/x2um52q).

### More water-park sources (fit: low to medium)
- **Morey's Piers (NJ):** a 2009 study analysed **79 real rescues at its two water parks**:
  77 % of victims under 20, many aged 7 to 12, boys under 7 and women over 20 at higher risk
  ([Abraldes, Manzanares & Fernández, *Cultura, Ciencia y Deporte* 4(12), 2009](https://portalinvestigacion.um.es/documentos/63c0b2fc3df4c204fbaffc55)).
  Whether rescues were filmed isn't stated (the abstract says "registered"). Morey's Piers
  also posts lifeguard training clips ("Scanning", https://www.youtube.com/watch?v=tbDx03FIKtw),
  so it's a second operator with a visible training culture and a research link.
- YouTube searches naming other parks (Hurricane Harbor, Typhoon Texas, Great Wolf, Raging
  Waters, Wet 'n' Wild) returned mostly marketing and news. Possible normal-footage or
  scanning clips: "Lifeguard at Hurricane Harbor routine, watch head" (guard POV,
  https://www.youtube.com/watch?v=NznMuaJxd0U); a news clip of a wave-pool rescue
  ("Lifeguard rescues 17-year-old from wave pool", KJRH, https://www.youtube.com/watch?v=jVADtS4QZIE);
  a long pool-rescue upload ("Lifeguard saves a drowning lady at Cardinals Swimming Center,
  live on camera", 9 min, https://www.youtube.com/watch?v=Y9vIvu8Ewrs). Vimeo has no public
  search and returned nothing via web search.

### Cross-check: two independent label sets vs our onsets (fit: very high; computed 2026-10-17)
Z5cc's CVAT labels (per-frame boxes, 24,506 "drown" and 75,096 "swim" boxes over 55 videos)
overlap **49** of our labeled clips. Taking each video's first "drown" box as their onset:

| | Game (15 clips, 14 overlap) | Z5cc (55 videos, 49 overlap) |
|---|---|---|
| Their onset minus ours, median | **−2.4 s** | **−2.4 s** (IQR −5.3 to +0.1) |
| Mean absolute difference | 2.7 s | 4.0 s |
| Within 3 s / 5 s | most | 26 / 49, 34 / 49 |
| Their end vs our guard contact | whistle 6.5 s before contact | last drown box 1.7 s before contact |

What it says:
1. **Our onsets are consistent with outside labelers, but systematically ~2.4 s later.** Both
   independent sources start "drowning" earlier than we do. We mark the first clear sign of
   trouble; they include a little lead-in. That's a definitional difference, not an error,
   but it means time-to-alert measured from our onset flatters the system by ~2 s. Report
   latency from both, or adopt an "earliest plausible onset".
2. **Clips to re-check:**
   - `92b58607c295`: earlier by **11.4 s** (Z5cc) and **13.5 s** (game). Two independent
     sources agree, so our onset is very likely late. It's one of the 9 un-held clips, which
     may still carry a coarse June mark.
   - `27fd21052cf8`: −11.3 s (another un-held clip, same probable cause).
   - `41d0d2580dd1` −8.0, `cc9bd9e965e3` −7.2, `f5d86fafbf0e` −6.8, `8c83f0f8a0da` −6.7,
     `b91bc5f720f5` −6.0: worth a look in the labeler.
   - `7096dfde4791` is +35.8 s: Z5cc labeled a *later* drowning at ~52 s, after our trim end
     (32.75 s). Either a second event in the video, or something we trimmed away; worth
     checking whether that clip holds two rescues.
3. **A ready inter-annotator statistic** for the README or a paper: "independent labelers
   agreed with our onset within 5 s on 34 of 49 clips; median offset 2.4 s".
(The per-clip comparison is saved at `scratch/research/onset_compare.json` (gitignored), with the game data in `scratch/research/stdc_videos.js`;
re-running needs only the public label XMLs, no video.)

## 11. Ranked recommendations (written 13:40, after ~70 minutes of searching)

**Data (biggest gaps first)**
1. **Normal footage for false alarms per hour:** Divertiland's ~12 h wave-pool livestream
   archives (varQEogbBLU and siblings). Nick's scope call.
2. **Request SwimID** (CCTV, tracks, incident labels) as an independent test set.
3. **Hard negatives** (water polo, dunking games) from Z5cc's list, for the test set.
4. Queue the unscreened candidates in section 6 through `harvest.py add`; add
   non-English search terms to harvest.

**Labels and evaluation**
5. Re-check onsets flagged by the cross-check (`92b58607c295` first); report latency from
   both our onset and the independent ones (~2.4 s earlier).
6. Use the game's **guard-whistle times** as the human benchmark: "alert before the whistle".
7. Lead every report with catches at a fixed alerts/hour plus an AMOC-style delay curve;
   exact binomial intervals; clip-level resampling.

**Detection and tracking**
8. Auto-label heads with **SAM 3.1 / YOLOE prompts**, then fine-tune; add an
   "at / under the surface" class; try a P2 head and 3-frame input.
9. Try **BoT-SORT / OC-SORT** (already installed) on the pilot detections, then build the
   ghost stage as **AFLink-style offline relinking** with "same spot after a gap" allowed.

**Features and scoring**
10. Per-head measurements calibrated by Carballo-Fazanes & Bierens: disappear/reappear count
    and lengths over 30 s (~5 per event, 7 to 20 s each), side-splash relative to the water
    model, displacement in head-widths, verticality, plus risk priors (small and alone, past
    the depth transition, tube flip, vest).
11. Scorers in order: rules, then boosted trees / TabPFN, then k-NN anomaly on the same
    vectors, then any neural model. Practitioner starting rule: under ~10 s, focus; alarm
    after another ~10 s.

**Design and story**
12. Alerts carry a snapshot, a location and a confidence; show what the system can't see.
13. README: related-work positioning (section 14), the size cutoff as how commercial installs
    are specified, scope (an assistant in guarded pools), and the automation-bias risk.
