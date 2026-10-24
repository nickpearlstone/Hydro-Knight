# The model from first principles, 2026-10-24

The question: setting evaluation and pipeline aside, what should the model that decides "this
person is drowning" actually be, given what our cameras can see, how few rescues we have, and a
Jetson-class target? This builds on the 10-17 sweep (`2026-10-17-research-sweep.md`), which
covered behaviour science, detection, tracking and tabular scorers. Here the focus is on the
shape of the model. Every claim has a source link; numbers are the sources' own. Anything
argued rather than sourced is marked *(reasoning)*. "Fit" says how directly a finding applies:
high, medium, low.

## Bottom line

- **Anchor the model to places in the water.** Our tracks break exactly when the signal
  happens: the head goes under. The model's unit should be a remembered head position that
  persists through disappearances. Track ids can't be that unit.
- **Use the few rescues we have as labels.** Don't rely on pure anomaly detection. With
  about ten labelled anomalies, deviation-style learning beats unsupervised baselines by 8
  to 43 AUC points in the one controlled study found. Our natural form is a ranking: within
  a clip, the victim should outrank everyone else in the water.
- **Turn per-moment evidence into alarms with a sequential test (CUSUM).** It is the
  textbook answer to the latency vs false-alarm trade-off. A duration-aware state model
  matches what drowning looks like: under for 7 to 20 s, about 5 times.
- **Add a learned head-crop motion feature only after the place-anchored measurements.**
  Pretrained on our own unlabelled pool video, it can see arm-milling splash, which boxes
  can't. Watch for shortcuts in context (lifeguards, tubes, camera edits).

## Contents
1. What the model has to see
2. What kind of learning problem this is
3. Representation options, ranked
4. From scores to alarms: sequential decisions
5. What the model can learn from
6. Where the field is
7. Proposed model, and the experiments that would falsify it
8. Disagreements and uncertainty
9. Sources

---

## 1. What the model has to see

### The signal, from real drowning video (fit: high)
From the 10-17 sweep's reading of
[Carballo-Fazanes & Bierens 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7559254/) (real
CCTV-style drowning videos coded by 20 experts):
- rescued victims disappeared a median of **5 times**, each disappearance a median of **13 s**
  (IQR 7 to 20.5 s); start of incident to rescue a median of **15 s**;
- the most common pattern, **backward water milling** (19 of 52), comes with clear splashes
  from the arms at the sides; **immediate disappearance** (5 of 52) shows no struggle at all;
- nobody waved or shouted.

So the evidence lives in **whether a head is visible, where, for how long, and what the water
next to it is doing, over 10 to 30 s**. Fine body pose matters much less.

### What our cameras deliver (fit: high, our own measurements)
- Heads are about 5 to 40 px across (`synthetic/train/README.md`).
- The victim is in the detector's input for only **50 to 51%** of the distress window. Victim
  tracks have a median length of **1.3 s** (157 tracks in 46 clips), while the v2 model needs
  about 4.5 s of track before its first score (`docs/EVALUATION.md`).
- The pose model can't see a submerged victim (2026-09-23 audit: 0.02 box confidence on every
  tile).

### Tiny actors are a known hard case (fit: medium)
[TinyVIRAT](https://arxiv.org/pdf/2007.07355) exists because standard action benchmarks have
large, close actors. Its actors are 10 to 128 px, which is our regime, and its authors add a
generative enhancement stage and learned attention over activity regions to improve on
standard baselines. Off-the-shelf action recognisers, trained on close-ups, are not built
for footage like ours.

**First principle 1** *(reasoning)*: when the defining event is "the object disappears",
anything keyed on a tracker's identity loses the signal at the moment it appears. The model's
memory should be keyed on **place** (a head slot in pool coordinates that survives gaps), and
"is there a head here now" should be an input, not a precondition. This is the ghost-track
idea from 10-16, put as the core of the model, not a patch on top of the tracker.

## 2. What kind of learning problem this is

### We have labelled anomalies, and they are worth a lot (fit: high)
- [Pang et al., "Explainable Deep Few-shot Anomaly Detection with Deviation Networks"](https://ar5iv.labs.arxiv.org/html/2108.00462)
  train with only **10 labelled anomalies** per dataset. Their method learns a score that
  pushes the labelled anomalies into the tail of a prior fitted to normal data. AUC-ROC
  against unsupervised or semi-supervised baselines (KDAD): MVTec AD 0.945 vs 0.861, AITEX
  0.887 vs 0.576, ELPV 0.846 vs 0.744, Hyper-Kvasir 0.829 vs 0.401. The paper's argument is
  ours: unsupervised methods "have no knowledge" of what the anomalies look like.
- Video surveys report the same ordering: weak supervision beats unsupervised training,
  including with small labelled subsets
  ([cross-domain limited-supervision paper](https://arxiv.org/pdf/2408.05191)).
- **Caveat:** a benchmark review argues the gap is overstated because AUC-ROC is unreliable
  on very unbalanced data
  ([INESC TEC review](https://repositorio.inesctec.pt/handle/123456789/15342)). This is the
  same reason our protocol scores within-clip rankings and false alarms per hour.

### Our labels have a natural ranking form (fit: high)
[Sultani et al. 2018](https://openaccess.thecvf.com/content_cvpr_2018/papers/Sultani_Real-World_Anomaly_Detection_CVPR_2018_paper.pdf)
started weakly supervised video anomaly detection with a **multiple-instance ranking loss**:
the most anomalous segment of an anomalous video must outscore the most anomalous segment of
a normal video. We have something stronger *(reasoning)*:
- inside one rescue clip, the victim's head slot is the positive and the roughly 20 other
  people in the water are the negatives, seen under the same camera, light and waves;
- so the loss is "victim above same-clip others during onset to contact". That is exactly
  the evaluation protocol's same-clip AUC, so training and evaluation measure the same thing;
- comparing within a clip cancels the camera and scene, so the model can't win by telling
  pools apart.

**First principle 2** *(reasoning)*: the autoencoder's premise was "learn normal, flag the
rest". With about 57 labelled victims and thousands of same-clip negatives, the better use of
the data is a normal-dynamics model for the bulk of the signal (section 3b), then a small
supervised ranking layer on top. This fits the open revisit of the "anomaly detection, not
classification" decision in CLAUDE.md.

## 3. Representation options, ranked

### 3a. Hand-built per-slot measurements, then trees or TabPFN (fit: high, first)
Already recommended on 10-17: disappear/reappear counts and lengths over 30 s, displacement
in head-widths, verticality, splash energy beside the head, risk priors. With dozens of
positives these are the most data-efficient and the easiest to explain. Nothing new found
that beats this as a starting point.

### 3b. A learned model of normal head dynamics, scored by surprise (fit: high)
[Kanu-Asiegbu et al., "Leveraging Trajectory Prediction for Pedestrian Video Anomaly Detection"](https://arxiv.org/pdf/2207.02279)
and [BiPOCO](https://arxiv.org/pdf/2207.02281) train a forecaster on normal trajectories or
poses only, and score anomalies by forecast error. BiPOCO weights each joint's error by the
pose detector's confidence and reports detection rates up to 87.0%.

For us *(reasoning)*:
- train a small forecaster on every non-victim head slot (we have far more of those than
  victims);
- predict next-step position, visibility and box size; the error, and especially "it
  stayed under longer than predicted", becomes a per-step evidence stream;
- weighting by detection confidence (as BiPOCO does) handles faint heads honestly.

This keeps what was good about the autoencoder (it learns normal from plentiful data), but on
the right unit (head slots) and with a forward-in-time question, the one a live system faces.

### 3c. Head-crop clips with self-supervised video features (fit: medium, second)
- Boxes can't see what the arms and water do beside the head. A small clip around each slot
  (say 64 x 64 px for 1 to 2 s) can.
- [VideoMAE](https://ar5iv.labs.arxiv.org/html/2203.12602) is "data-efficient": it pretrains
  well on **3k to 4k videos** with no extra data, tolerates 90 to 95% masking, and the
  authors stress that data quality matters more than quantity and that domain shift between
  pretraining and target hurts.
- So pretrain on **our own unlabelled pool crops** (57 clips x thousands of head slots is far
  more than 3k clips), then train a light head on top. Fine with no new footage.
- [V-JEPA](https://arxiv.org/html/2404.08471v1) is strongest on motion-heavy benchmarks
  (+6% on Something-Something-v2 with a frozen backbone), but no source tested either model
  on objects this small. *(unverified for our scale)*

### 3d. Track-free dense occupancy over time (fit: medium, a cross-check)
*(reasoning)* A grid over the pool records "head present here" and local motion energy per
cell over time. A disappearance is then a local drop in occupancy with no matching appearance
nearby. This needs no tracker at all, so it can't fail when tracks break, and on a Jetson it
is cheap: one pass of the head detector plus counters. It is a good independent second
opinion for the "immediate disappearance" case (5 of 52), where there is no struggle to see.

### Shortcut risk in any learned representation (fit: high)
[Li et al., ICCV 2023](https://arxiv.org/abs/2211.12883) show that every action model they
examined relies on **static** cues in the background and also in the foreground (clothing,
for example). Debiasing methods that fix the background still fail on foreground bias.
Our likely shortcuts *(reasoning)*:
- a lifeguard approaching (cut by scoring only before contact, already done);
- inner tubes and small children (real risk priors, but they would fire on every child);
- YouTube edits around the rescue (zoom, slow motion, captions) that real cameras never have.

Test for them directly: score a clip with its frames shuffled in time. A motion model should
collapse; one that keeps its score is reading static appearance.

## 4. From scores to alarms: sequential decisions

### Quickest change detection is the matching theory (fit: high)
- The problem "alarm as soon as possible after the change, with a capped false-alarm rate"
  is quickest change detection. **CUSUM** is optimal for worst-case delay at a fixed
  false-alarm rate (Lorden's criterion), given the before and after distributions
  ([score-based multi-stream QCD](https://arxiv.org/pdf/2511.03967)).
- Video work in this framing: [Lao & Sundaramoorthi, "Quickest Moving Object Detection"](https://arxiv.org/abs/1605.07369)
  minimise average delay under a false-alarm cap and beat simple thresholding at every
  false-alarm level.
- A close analogue for per-object scores: CUSUM on trajectory-prediction errors for
  self-driving cars, where a mixture CUSUM gave the lowest delay at the same mean time to
  false alarm ([Building Real-time Awareness of Out-of-distribution in Trajectory Prediction](https://arxiv.org/pdf/2409.17277)).

For us *(reasoning)*: run one CUSUM per head slot on the evidence stream from 3a/3b. Its
threshold maps directly onto "mean time between false alarms", which is the 10 to 20 per hour
budget (or ISO 20380's ~0.5 per hour). Evidence accumulates across disappearances, so three
10 s submersions in 30 s can alarm even if none alone would. A plain per-window threshold
can't do that.

### Duration-aware state models fit the behaviour (fit: medium)
[Duong et al.'s switching hidden semi-Markov model](https://espace.curtin.edu.au/handle/20.500.11937/10349)
models activities with explicit state durations and detects abnormality **without training
on abnormal data**. The Coxian duration version is compact and cheap
([AI journal 2009](https://www.datalearner.com/academic/journal-papers/0004-3702/volumes-and-issues/339/paper-detail/68681)).

The fit *(reasoning)*: states such as swimming, bobbing (wave rhythm) and under, with
learned "how long normal people stay under, in this spot, in these waves". A victim's
pattern (repeated 7 to 20 s submersions with little displacement) gets low likelihood under
normal durations. This is also the survival-analysis view: model the probability that a
submerged head reappears by time t, learned from normal swimmers, and treat a falling
reappearance probability as rising evidence.

## 5. What the model can learn from

### Rendered behaviour sequences (fit: high, matches the 10-19 decision)
[Varol et al., "Synthetic Humans for Action Recognition from Unseen Viewpoints" (IJCV 2021)](https://www.di.ens.fr/willow/research/surreact)
**raised unseen-viewpoint accuracy from 53.6% to 69.0% on NTU and from 49.4% to 66.4% on
UESTC** by adding rendered people to limited real data. Diversity of motions and viewpoints
was what mattered. For us *(reasoning)*:
- our Blender pipeline already renders matched cameras with exact labels;
- animated milling, instinctive drowning response and submersion, mixed with normal
  bobbing, would give the normal-dynamics model and the ranking layer positives with exact
  onsets;
- vary the motions widely, not just the scenes.

### Unlabelled pool video for pretraining (fit: medium)
The VideoMAE result (section 3c) is the case for pretraining on our own footage rather than
on large external datasets. Kinetics-style pretraining sees close, large actors, a domain
shift the authors flag.

## 6. Where the field is (fit: medium, for positioning)
- A systematic review of 30 drowning-detection studies found **only 5 (16.7%) with
  real-world validation and none with cross-dataset evaluation**; all used private data,
  evaluation was mostly accuracy, and fewer than 10% addressed occlusion, glare or crowding
  ([JMIR preprint listing; treat with some caution](https://preprints.jmir.org/preprint/100939)).
- Another review notes real drowning footage is "virtually nonexistent", so most work uses
  staged footage ([PMC10820385](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10820385/)).
- Recent published detectors are per-frame object detectors, for example
  [MS-YOLO](https://pmc.ncbi.nlm.nih.gov/articles/PMC11548417), 86.4% accuracy on a self-built
  set. Commercial and patent systems track heads across cameras
  ([Hangzhou patent](https://patents.justia.com/patent/20230222804)), but no published learned
  temporal model over head histories was found.

So a place-anchored temporal model, evaluated on real rescues with a lifeguard's labels and
false alarms per hour, would be new work, not a re-implementation.

## 7. Proposed model, and the experiments that would falsify it

*(reasoning, assembled from sections 1 to 5)*

```
head detector (synthetic-pretrained RT-DETR, real fine-tune later)
  -> head-slot memory in pool coordinates: survives disappearances (ghost slots)
  -> per-slot evidence each step:
       visible? time under; displacement in head-widths; box shape; motion energy beside head
       [later] head-crop clip features (VideoMAE pretrained on our crops)
  -> normal-dynamics model (trained on every non-victim slot): surprise per step
     + duration likelihood (how long normal heads stay under, here, in these waves)
  -> small supervised ranking layer: victim above same-clip others, onset to contact
  -> CUSUM per slot -> alarm, with the stage it saw ("went under, not back" / "struggling")
```

Experiments, cheapest first. Each states what would kill the idea:
1. **Slot memory alone:** on the 57 rescues, how often is the victim's slot alive
   (visible or a remembered ghost) from onset to contact, compared with track coverage
   (50 to 51%)? *If it doesn't rise well above 51%, place-anchoring isn't enough and
   detection is still the bottleneck.*
2. **Measurements plus trees with a ranking objective** (3a), grouped by clip: within-clip AUC
   and top-3 share against the shuffled baseline. *If it can't beat "1 minus mean box
   confidence", the measurements carry nothing beyond detection quality.*
3. **CUSUM on (2)'s scores:** catches at 20 false alarms per hour (proxy) and the delay curve,
   compared with a per-window threshold. *If the delay at matched false alarms doesn't fall,
   accumulating evidence isn't helping.*
4. **Normal-dynamics surprise** (3b) as one extra feature in (2). *If it adds nothing over
   hand-built time-under, skip the neural forecaster.*
5. **Head-crop features** (3c), with the frame-shuffle shortcut test. *If the frame-shuffled
   score stays high, it is reading appearance, not motion.*
6. **Rendered behaviour sequences** added to (2) to (5): does recall on real rescues rise?

## 8. Disagreements and uncertainty
- **How large the supervised advantage is:** the 8 to 43 point gains come from image and
  defect datasets with 10 labelled anomalies. No study found measured it for per-person video
  with about 57 positives. The video-level evidence (weak beats unsupervised) is consistent
  but disputed on metric grounds.
- **Tiny-object video features are untested:** VideoMAE and V-JEPA claims come from
  benchmarks with large actors. Whether they help at 5 to 40 px heads is open, so they are
  ranked behind measurements.
- **CUSUM needs a model of normal evidence:** its false-alarm guarantee holds only if the
  normal score distribution is right. Without hours of incident-free footage, the threshold
  can only be set on pre-onset rescue footage (the `--pseudo-normal` stopgap), which may be
  calmer or busier than a normal hour.
- **Waves vs bobbing:** resurfacing every ~5 s overlaps wave periods (10-17 sweep), so the
  duration model needs the wave phase or it will learn that bobbing in waves is normal
  everywhere.
- **No published drowning model to compare against on equal terms**; reviews say the field
  lacks cross-dataset tests, so claims from other papers can't be taken at face value.

## 9. Sources
- Carballo-Fazanes & Bierens 2020, visible behaviour of drowning persons: https://pmc.ncbi.nlm.nih.gov/articles/PMC7559254/
- Pang et al., deviation networks, few-shot anomaly detection: https://ar5iv.labs.arxiv.org/html/2108.00462
- Cross-domain video anomaly detection with limited supervision: https://arxiv.org/pdf/2408.05191
- Benchmark-based review of video anomaly detection metrics: https://repositorio.inesctec.pt/handle/123456789/15342
- Sultani, Chen & Shah 2018, MIL ranking for surveillance anomalies: https://openaccess.thecvf.com/content_cvpr_2018/papers/Sultani_Real-World_Anomaly_Detection_CVPR_2018_paper.pdf
- Kanu-Asiegbu et al., trajectory prediction for pedestrian anomaly detection: https://arxiv.org/pdf/2207.02279
- BiPOCO, pose-constrained bidirectional prediction: https://arxiv.org/pdf/2207.02281
- VideoMAE: https://ar5iv.labs.arxiv.org/html/2203.12602
- V-JEPA: https://arxiv.org/html/2404.08471v1
- TinyVIRAT: https://arxiv.org/pdf/2007.07355
- Li et al., static bias in action representations (ICCV 2023): https://arxiv.org/abs/2211.12883
- Score-based quickest change detection, multi-stream: https://arxiv.org/pdf/2511.03967
- Lao & Sundaramoorthi, quickest moving object detection: https://arxiv.org/abs/1605.07369
- CUSUM on trajectory-prediction errors (autonomous vehicles): https://arxiv.org/pdf/2409.17277
- Duong et al., switching hidden semi-Markov model: https://espace.curtin.edu.au/handle/20.500.11937/10349
- Duong et al., Coxian duration HSMM (AI 2009): https://www.datalearner.com/academic/journal-papers/0004-3702/volumes-and-issues/339/paper-detail/68681
- Varol et al., SURREACT: https://www.di.ens.fr/willow/research/surreact
- Systematic review of drowning-detection studies (JMIR preprint): https://preprints.jmir.org/preprint/100939
- Review of drowning-detection technology: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10820385/
- MS-YOLO: https://pmc.ncbi.nlm.nih.gov/articles/PMC11548417
- Multi-camera pool alarm patent: https://patents.justia.com/patent/20230222804
