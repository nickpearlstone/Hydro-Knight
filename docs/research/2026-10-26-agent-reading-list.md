# Reading list for the research agents (2026-10-26)

The corpus for the sandboxed research experiment: agents read these, propose improvements aimed
at a specific Hydro-Knight bottleneck, then prove or disprove them on our own data against the
Tier 1 protocol (`docs/EVALUATION.md`).

**How papers were picked.** Peer-reviewed at a strong venue or journal, or a widely cited
preprint from an established lab. Preference for papers with honest ablations and results other
groups have reproduced. Each entry says which Hydro-Knight problem it touches and what not to
over-read in it.

**Markers.** *in use*: already part of the pipeline. *sweep*: also cited in the 10-17 research
sweeps. Citation counts are Semantic Scholar on 2026-10-26; counts marked (CR) are from Crossref,
which runs lower. Venue, year and authors were checked for every paper; the notes on content
come from abstracts and my own reading, so anything an agent builds on should be checked
against the paper itself.

**Not committed:** PDFs go in a gitignored folder (copyright), never into git.

33 papers in six groups:

| Group | Papers |
|---|---|
| A. Detection and tracking of small people | 7 |
| B. Temporal and anomaly models | 5 |
| C. Synthetic data | 5 |
| D. Efficient models | 5 |
| E. Similar tasks and drowning science | 6 |
| F. Research rigor | 5 |

---

## A. Detection and tracking of small people

**A1. Slicing Aided Hyper Inference and Fine-Tuning for Small Object Detection (SAHI)**
Akyon, Altinuc, Temizel. ICIP 2022. 571 cites. [arXiv 2202.06934](https://arxiv.org/abs/2202.06934). *in use*
- **For Hydro-Knight:** the tiling behind our extraction. Read for the fine-tuning half, which
  we have never tried: training on slices, so the detector sees distant swimmers at a larger size.
- **Don't over-read:** gains are on aerial benchmarks (VisDrone, xView). We already found that
  tiling only helps when the inference size is larger than the tile.

**A2. Scale Match for Tiny Person Detection (TinyPerson)**
Yu, Gong, Jiang, Ye, Han. WACV 2020. 391 cites. [arXiv 1912.10664](https://arxiv.org/abs/1912.10664). *sweep*
- **For Hydro-Knight:** the closest benchmark to our crowd: people under 20 px at seaside scenes.
  Scale Match rescales the pretraining data so its object sizes match the target's, which
  speaks to "detection collapses below 60 to 80 px".
- **Don't over-read:** absolute accuracy on tiny people stays low even with the method; the
  gains are a few points.

**A3. A Normalized Gaussian Wasserstein Distance for Tiny Object Detection (NWD)**
Wang, Xu, Yang, Yu. arXiv 2021; extended version in ISPRS J. Photogrammetry and Remote Sensing 2022. 662 cites. [arXiv 2110.13389](https://arxiv.org/abs/2110.13389)
- **For Hydro-Knight:** shows that IoU swings wildly for tiny boxes when they shift a few
  pixels, and swaps it for a distance between boxes modeled as Gaussians. ByteTrack matches
  boxes by IoU, and our tracks last about 1 s. That match step can be retested offline from
  `data/detections/` without rerunning YOLO.
- **Don't over-read:** tested on aerial tiny-object detection, not tracking. The tracking link
  is a hypothesis.

**A4. ByteTrack: Multi-Object Tracking by Associating Every Detection Box**
Zhang et al. ECCV 2022. 2,910 cites. [arXiv 2110.06864](https://arxiv.org/abs/2110.06864). *in use*
- **For Hydro-Knight:** its core idea is a second matching pass that uses low-confidence boxes
  to keep tracks alive without starting new ones. Our 0.25 extraction floor throws those boxes
  away before the tracker sees them, while pose sees 90% of victims at 0.05.
- **Don't over-read:** pedestrian benchmarks (MOT17/20) with large, well-separated people.

**A5. Observation-Centric SORT (OC-SORT)**
Cao, Pang, Weng, Khirodkar, Kitani. CVPR 2023. 1,077 cites. [arXiv 2203.14360](https://arxiv.org/abs/2203.14360)
- **For Hydro-Knight:** fixes the Kalman filter's drift while an object is hidden, and
  re-associates it when it reappears. It's the published cousin of the "ghost track" idea
  (freeze the position, don't extrapolate velocity) for relinking resurfacing heads.
- **Don't over-read:** occlusions in its benchmarks (DanceTrack, MOT) are short and the
  objects are large; going fully underwater is a harder gap.

**A6. Tracking Pedestrian Heads in Dense Crowd (CroHD, HeadHunter-T)**
Sundararaman, de Almeida Braga, Marchand, Pettré. CVPR 2021. 109 cites. [arXiv 2103.13516](https://arxiv.org/abs/2103.13516). *sweep*
- **For Hydro-Knight:** head detection and tracking from elevated cameras in crowds, the same
  shape as the head-anchored redesign. Also proposes a metric that scores tracks by how far
  they follow the true head.
- **Don't over-read:** only 9 sequences; the dataset is non-commercial, so use it for comparison,
  not training.

**A7. Rethinking Counting and Localization in Crowds: A Purely Point-Based Framework (P2PNet)**
Song et al. ICCV 2021. 443 cites. [arXiv 2107.12746](https://arxiv.org/abs/2107.12746)
- **For Hydro-Knight:** trains from point labels alone and predicts points. Our victim clicks
  and the 736 counted swimmers are points, not boxes.
- **Don't over-read:** crowd-counting benchmarks; it gives locations, not boxes or identities.

## B. Temporal and anomaly models

**B1. Normalizing Flows for Human Pose Anomaly Detection (STG-NF)**
Hirschorn, Avidan. ICCV 2023. 121 cites. [arXiv 2211.10946](https://arxiv.org/abs/2211.10946). *sweep*
- **For Hydro-Knight:** the stated model priority: about 1K parameters, scores by likelihood.
  Code is CC BY-NC, so reimplement from the paper.
- **Don't over-read:** it normalizes pose by default, which hides displacement. Later
  benchmarks rank it differently from dataset to dataset, so run several seeds.

**B2. An Attribute-Based Method for Video Anomaly Detection**
Reiss, Hoshen. TMLR 2025 (arXiv 2022). 32 cites. [arXiv 2212.00789](https://arxiv.org/abs/2212.00789). *sweep*
- **For Hydro-Knight:** each person becomes a few readable attributes (velocity, pose) scored by
  nearest-neighbour density. A simple, interpretable baseline, close to the per-head
  measurements in the redesign.
- **Don't over-read:** benchmarks use staged anomalies; its Avenue number changed between
  versions (93.3 vs 93.7 AUROC).

**B3. A Survey of Single-Scene Video Anomaly Detection**
Ramachandra, Jones, Vatsavai. TPAMI 2022. 267 cites. [arXiv 2004.05993](https://arxiv.org/abs/2004.05993)
- **For Hydro-Knight:** v1 is one wave pool, so this is exactly our setting. It is critical of
  frame-level scoring that never checks *where* the anomaly was, which is the bug our 0.539
  score had. Read with the same authors' [Street Scene paper (WACV 2020)](https://openaccess.thecvf.com/content_WACV_2020/html/Ramachandra_Street_Scene_A_new_dataset_and_evaluation_protocol_for_video_WACV_2020_paper.html),
  which proposes stricter evaluation criteria.
- **Don't over-read:** a survey; method comparisons in it come from each paper's own numbers.

**B4. Deep Anomaly Detection with Outlier Exposure**
Hendrycks, Mazeika, Dietterich. ICLR 2019. 1,849 cites. [arXiv 1812.04606](https://arxiv.org/abs/1812.04606)
- **For Hydro-Knight:** trains an anomaly detector with a set of known outliers. Speaks to the
  open "anomaly detection vs small supervised model" question once victim tracks are labeled.
- **Don't over-read:** its outlier sets are large and varied; ours is a few dozen victims, so
  overfitting to those victims is the risk to measure.

**B5. Recurrent Neural Networks for Multivariate Time Series with Missing Values (GRU-D)**
Che, Purushotham, Cho, Sontag, Liu. Scientific Reports 2018. 2,678 cites. [arXiv 1606.01865](https://arxiv.org/abs/1606.01865)
- **For Hydro-Knight:** treats *when* data goes missing as information in its own right. For
  us, a head that stops being detected is the submersion signal; our series fill gaps with
  confidence 0 and held coordinates.
- **Don't over-read:** clinical data; the margin over well-imputed baselines is modest.

## C. Synthetic data

**C1. Training Deep Networks with Synthetic Data: Bridging the Reality Gap by Domain Randomization**
Tremblay et al. (NVIDIA). CVPR Workshops 2018. 1,050 cites. [arXiv 1804.06516](https://arxiv.org/abs/1804.06516)
- **For Hydro-Knight:** non-realistic but heavily varied renders, then fine-tuning on real
  images, beat real-only training for detection. Directly relevant to the Blender head renders.
- **Don't over-read:** cars on KITTI; the benefit shrinks as real data grows.

**C2. Cut, Paste and Learn: Surprisingly Easy Synthesis for Instance Detection**
Dwibedi, Misra, Hebert. ICCV 2017. 727 cites. [arXiv 1708.01642](https://arxiv.org/abs/1708.01642)
- **For Hydro-Knight:** shows that how a crop is blended in matters, because detectors learn
  to spot paste boundaries. That's the main risk when pasting heads into water.
- **Don't over-read:** household objects on tabletops, not people in moving water.

**C3. Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation**
Ghiasi et al. (Google). CVPR 2021. 1,427 cites. [arXiv 2012.07177](https://arxiv.org/abs/2012.07177). *sweep*
- **For Hydro-Knight:** the 10-17 sweep's top pick for the head detector: random pasting with
  scale jitter, strongest when real labels are scarce.
- **Don't over-read:** uses segmentation masks; we have boxes and points, so the pasted crop
  will carry some background.

**C4. BEDLAM: A Synthetic Dataset of Bodies Exhibiting Detailed Lifelike Animated Motion**
Black, Patel, Tesch, Yang. CVPR 2023. 332 cites. [arXiv 2306.16940](https://arxiv.org/abs/2306.16940)
- **For Hydro-Knight:** models trained on synthetic humans alone reached state of the art on
  real benchmarks. The strongest evidence that rendered people can work, and a record of what
  it took (hair, clothing, lighting).
- **Don't over-read:** it took a large team's engineering, on land, with no water.

**C5. What Makes Good Synthetic Training Data for Learning Disparity and Optical Flow Estimation?**
Mayer et al. IJCV 2018. 231 cites. [arXiv 1801.06397](https://arxiv.org/abs/1801.06397)
- **For Hydro-Knight:** one of the few careful studies of *which properties* of synthetic data
  matter (variety, realism, training order), which is the question behind the synthetic
  realism loop.
- **Don't over-read:** optical flow and depth, not detection.

## D. Efficient models

**D1. DETRs Beat YOLOs on Real-time Object Detection (RT-DETR)**
Zhao et al. (Baidu). CVPR 2024. 4,817 cites. [arXiv 2304.08069](https://arxiv.org/abs/2304.08069). *in use (licence path)*
- **For Hydro-Knight:** the Apache-licensed detector family that beat YOLO11n-pose at matched
  false boxes in our licence-path benchmark.
- **Don't over-read:** speed numbers are on a T4 with TensorRT, not a Jetson.

**D2. Distilling the Knowledge in a Neural Network**
Hinton, Vinyals, Dean. NeurIPS Deep Learning Workshop 2014. 27,076 cites. [arXiv 1503.02531](https://arxiv.org/abs/1503.02531)
- **For Hydro-Knight:** train a small Jetson-sized student from a large teacher. The teacher can
  be a big model run offline that never ships, which may help the licence path too.
- **Don't over-read:** gains depend on the teacher-student gap; check licence terms of any teacher.

**D3. A White Paper on Neural Network Quantization**
Nagel et al. (Qualcomm AI Research). arXiv 2021. 993 cites. [arXiv 2106.08295](https://arxiv.org/abs/2106.08295)
- **For Hydro-Knight:** practical recipes and debugging steps for INT8, which the Nano needs
  through TensorRT.
- **Don't over-read:** its near-float INT8 results are measured on overall benchmark accuracy.
  Tiny-object recall can drop more, so measure on our own metric.

**D4. What is the State of Neural Network Pruning?**
Blalock, Gonzalez Ortiz, Frankle, Guttag. MLSys 2020. 1,326 cites. [arXiv 2003.03033](https://arxiv.org/abs/2003.03033)
- **For Hydro-Knight:** a review of 81 pruning papers found inconsistent comparisons and
  claims that didn't hold up. Doubles as a rigor paper for any efficiency claim we make.
- **Don't over-read:** about pruning specifically; the lesson is the comparison method.

**D5. Efficient Object Detection in Large Images Using Deep Reinforcement Learning**
Uzkent, Yeh, Ermon. WACV 2020. 81 cites (CR). [arXiv 1912.03966](https://arxiv.org/abs/1912.03966)
- **For Hydro-Knight:** an RL agent decides where high resolution is needed. It matched a
  high-resolution-only detector while using high resolution about 30% of the time. That's
  the Jetson problem (16 tiles per frame) and the RL back-burner project.
- **Don't over-read:** satellite images with static objects; our objects move and drown.

## E. Similar tasks and drowning science

**E1. The Visible Behaviour of Drowning Persons: A Pilot Observational Study**
Carballo-Fazanes, Bierens et al. Int. J. Environ. Res. Public Health 2020. 33 cites (CR). [open access](https://www.ilsf.org/wp-content/uploads/2020/10/The-visible-behaviour-of-drowning-persons.pdf)
- **For Hydro-Knight:** 23 real drowning videos (most in pools) scored by 20 experts. Everyone
  drowned within 2 minutes. Patterns: backward water milling (19 of 24), instinctive drowning
  response (6), distress (6), immediate disappearance (5), climbing-ladder motion (3), plus
  frequent short resurfacing. The best evidence for which tells to measure.
- **Don't over-read:** a pilot with 24 people; the patterns are expert judgements, not
  measurements.

**E2. The Effect of Lifeguard Experience upon the Detection of Drowning Victims in a Realistic Dynamic Visual Search Task**
Laxton, Crundall. Applied Cognitive Psychology 2018. 21 cites (CR). [post-print](https://irep.ntu.ac.uk/id/eprint/32044/)
- **For Hydro-Knight:** the human baseline. Lifeguards beat non-lifeguards on pool videos, and
  passive drownings were missed more often than active ones.
- **Don't over-read:** staged drownings by confederates in an indoor pool, small sample.

**E3. DEWS: A Live Visual Surveillance System for Early Drowning Detection at Pool**
Eng, Toh, Yau, Wang. IEEE TCSVT 2008. 62 cites (CR). [DOI](https://doi.org/10.1109/TCSVT.2007.913960) (paywalled)
- **For Hydro-Knight:** the most-cited peer-reviewed camera system for drowning at an outdoor
  pool, and a reference for the alarm logic and the water-surface problems (glare, ripples)
  any system faces.
- **Don't over-read:** check how its drowning episodes were obtained; the same group's earlier
  system was tested on simulated crises. Hand-built features from before deep learning.

**E4. The Detection, Tracking, and Temporal Action Localisation of Swimmers for Automated Analysis**
Hall, Victor, He, Langer, Elipot, Nibali. Neural Computing and Applications 2021. 12 cites (CR). [DOI](https://doi.org/10.1007/s00521-020-05485-3)
- **For Hydro-Knight:** swimmer *head* detection plus a SORT-based tracker that keeps
  identities longer. Reports 20.8% higher head-detection F1 than Faster R-CNN at 6x the speed.
  The closest published work to our head tracks.
- **Don't over-read:** competition swimming in lanes, a far cleaner scene than a wave pool.

**E5. SwimXYZ: A Large-Scale Dataset of Synthetic Swimming Motions and Videos**
Fiche et al. ACM SIGGRAPH Motion, Interaction and Games 2023. 11 cites. [arXiv 2310.04360](https://arxiv.org/abs/2310.04360). *sweep*
- **For Hydro-Knight:** the only synthetic swimmer dataset with rendered water, useful as a
  record of how others rendered swimmers.
- **Don't over-read:** competitive strokes, not distress; a smaller venue with few citations.

**E6. Evaluation of Accelerometer-Based Fall Detection Algorithms on Real-World Falls**
Bagalà et al. PLoS ONE 2012. 402 cites (CR). [open access](https://pmc.ncbi.nlm.nih.gov/articles/PMC3353905/)
- **For Hydro-Knight:** the cautionary tale for rare-event alarms. 13 published fall detectors
  were tested on 29 *real* falls: mean sensitivity 57% (best 83%), far below their simulated-fall
  numbers, and 3 to 85 false alarms per day. Same risk for staged drownings and synthetic data.
- **Don't over-read:** wearable sensors, not video.

## F. Research rigor

**F1. Leakage and the Reproducibility Crisis in Machine-Learning-Based Science**
Kapoor, Narayanan. Patterns 2023. 1,338 cites (CR). [arXiv 2207.07048](https://arxiv.org/abs/2207.07048)
- **For Hydro-Knight:** a taxonomy of leakage found in 294 papers across 17 fields, plus a
  checklist. We have hit one already: held-out windows overlapping training windows in
  `--train-fresh`.
- **Don't over-read:** about scientific ML, but the leakage types transfer directly.

**F2. Shortcut Learning in Deep Neural Networks**
Geirhos et al. Nature Machine Intelligence 2020. 3,813 cites. [arXiv 2004.07780](https://arxiv.org/abs/2004.07780)
- **For Hydro-Knight:** why Tier 1 demands trivial baselines: victims are small and near inner
  tubes, so a model can win on box size alone.
- **Don't over-read:** a perspective paper; it frames the problem rather than measuring it.

**F3. Accounting for Variance in Machine Learning Benchmarks**
Bouthillier et al. MLSys 2021. 344 cites. [arXiv 2103.03098](https://arxiv.org/abs/2103.03098)
- **For Hydro-Knight:** how much of a "win" is seed and data-split noise, and how many runs it
  takes to tell. Matters for K-fold over 57 clips and for STG-NF's seed sensitivity.
- **Don't over-read:** computed on larger benchmarks, where the noise is smaller than ours.

**F4. The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets**
Saito, Rehmsmeier. PLoS ONE 2015. 5,028 cites (CR). [open access](https://pmc.ncbi.nlm.nih.gov/articles/PMC4349800/)
- **For Hydro-Knight:** ROC curves look reassuring when positives are rare. Victim windows are
  a tiny fraction of all windows.
- **Don't over-read:** bioinformatics examples; our operating metric is still alarms per hour.

**F5. Machine Learning for Medical Imaging: Methodological Failures and Recommendations for the Future**
Varoquaux, Cheplygina. npj Digital Medicine 2022. 191 cites. [arXiv 2103.10292](https://arxiv.org/abs/2103.10292)
- **For Hydro-Knight:** small test sets give error bars wider than the gaps between methods,
  and benchmarks get overfit. Our test set is a few dozen clips from one pool.
- **Don't over-read:** medical imaging, but the statistics are the same.

---

## Gaps worth knowing

- **No trustworthy drowning-detection deep learning paper evaluated on real rescues turned up.**
  Most recent work reports accuracy on small staged or self-collected sets, with no false-alarm
  rate. That gap is also where a novel result from our 57 real rescues could land.
- **Underwater re-identification after submersion** has no direct literature; OC-SORT (A5) and
  the drowning-behaviour study (E1) are the nearest pieces.
- **Lifeguard rescue outcomes** at scale: no peer-reviewed source found.
