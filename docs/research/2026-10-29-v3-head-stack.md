# v3 head-anchored stack (2026-10-27 to 2026-10-29)

**Result:** On the same 46 rescue clips, folds and seeds as the v2 benchmark, the best v3 version
(place-anchored head features + appearance + motion appearance) ranks the victim in the top 3 of
everyone in the pool for 46-47% of the catch window, against 7% for random scores. Its
same-clip AUC is 0.92-0.93 (v2: 0.73), and the victim reaches the top spot at some point in 100% of
clips. False-alarm rates are unknown: there is no normal pool footage to set alarm thresholds on.

v2 (the TCN autoencoder on pose tracks) stays the benchmark. v3 is a stack of small pieces: a head
detector, head tracks, per-head measurements, and a within-moment ranker that orders everyone in the
pool every 0.5 s and asks how often the victim is near the top.

## Results

Full protocol, whole-frame head detection, 46 clips, 5 folds, seeds 0-2. Ranges span the seeds.

| Version | Victim scored | Same-clip AUC | Time in top 3 | Time at top | Reaches top spot |
|---|---|---|---|---|---|
| Ranker (per-track features) | 48% | 0.79-0.80 | 26% | - | - |
| + appearance | 48% | 0.89 | 36-37% | 24% | 87% of clips |
| + stitching | 49% | 0.89 | 37-38% | - | - |
| Place-anchored ranker | 58% | 0.81 | 28-29% | - | - |
| Place + appearance | 58% | 0.91-0.92 | 44% | 31% | 96% of clips |
| Place + motion only | 58% | 0.89 | 38-40% | - | - |
| **Place + appearance + motion** | 58% | **0.92-0.93** | **46-47%** | **32%** | **100% of clips** |
| Random baseline | | 0.50 | 7% | 2% | 25% |

- **v2 benchmark:** AUC 0.73, time in top 3 7%, reaches top 3 in 48-50% of clips. v2 is scored on pose
  tracks, where random scores reach the top 3 for 3-4% of the window (more rivals per moment).
- **Same-clip AUC** is the chance that a victim score in the catch window beats another swimmer's
  score in the same clip.
- **Victim detected** (has a head box in the input tracks): 67% of the window in every whole-frame row.
  "Victim scored" is the share of the window where the victim has a row with a score.
- **Motion appearance, paired per clip** against the same stack without it: AUC +0.009 to +0.013 and
  time in top 3 +2.2 to +2.7 points, with the 95% interval above 0 on every seed. 29-34 clips improve,
  12-17 get worse.
- **Ranker weights** with everything in: appearance ~1.3-1.7, motion ~1.0, size flicker ~0.7,
  churn ~0.5 (~0.8 before appearance was added).
- **Alarm budgets:** 0 caught at every budget for every version. 21 minutes of pseudo-normal footage
  cannot support an alarm threshold.

## Pipeline

All commands are `uv run hydroknight stack ...`. Code is in `src/hydro_knight/stack/` and
`src/hydro_knight/cli/stack.py`.

1. **Head detection** (`heads-detect`). Fine-tuned RT-DETRv2-R18 head model (seed 1, last checkpoint;
   see the 2026-10-27 note), run once per frame on the whole frame resized to 640 px wide, at about
   10 fps. Transformers is pinned to 5.21.0 through the `heads` optional extra.
2. **Tracks** (`heads-track`). Cross-tile merge and ByteTrack with a "weak-kept" rule: every box on a
   confirmed track is kept and flagged weak under confidence 0.4. Boxes on no track are kept only at
   0.4 or above.
3. **Rows** (`heads-rows --place`). One row per head in the pool zone every 0.5 s. Victim truth is set
   per row: the victim is the nearest head within 2 head heights of the path interpolated between
   Nick's victim clicks. This fixes clip 644cb555b4d9, where a whole victim track followed the wrong
   person.
4. **Features** per row, causal, over the last 2 s, in head heights where relevant:
   - churn: smoothed path length minus net displacement ("busy but going nowhere"), plus churn over 4 s;
   - size flicker: standard deviation of log head-box height (bobbing);
   - confidence flicker;
   - aspect: median log(width / height) over 1 s. Nick's head-tilt cue: the box widens as the face
     tips back;
   - splash: motion beside the head divided by motion in a ring 3-5 head widths out;
   - head height, as a control.
5. **Place-anchored paths** (`--place`). Each row's history walks backward to the nearest box of any
   track, confirmed or not, within 1.5 head heights and of similar size. A row's past no longer depends
   on one track id surviving.
6. **Appearance** (`heads-embed`, `heads-fit --appear`). A head-and-shoulder crop (4 head heights,
   centred 0.75 head heights below the head) goes through frozen DINOv2-small (Apache-2.0, never
   trained here) to 768 numbers. A per-fold probe (standardise, PCA to 64, logistic regression with
   balanced classes) turns them into a score. The probe is stacked: rows are scored only by probes
   that never saw their clip, including inner folds for the ranker's own training. Embeddings are
   cached once per track every 2 s.
7. **Motion appearance** (`heads-embed --kind motion`, `heads-fit --motion`). The same frozen DINOv2
   on a motion image: frame-to-frame change in the same crop over the last 1.5 s, with the three colour
   channels holding the newest, middle and oldest half second. Same probe and caching.
8. **Ranker** (`heads-fit`). Conditional logit within each moment, trained from onset to guard
   contact minus 3 s. Clips are weighted equally, L2 penalty 0.01, and weights on suspicion features
   are kept at 0 or above. Scores are relative to the moment's median.
9. **Evaluation.** `uv run hydroknight eval protocol <run> --size <name> --tracks-dir <run>/tracks`,
   always reported with random and shuffled baselines.

## Head detector and tiling

Five checks on head tracks (Kaggle tiled detection, 51 labelled clips):

| Check | Head tracks | Pose tracks |
|---|---|---|
| Track ends per swimmer-minute | 8.1 | 28.2 |
| Median track length | 2.3 s | 1.3 s |
| Victim has a box within 1.5 head heights (first click to contact - 3 s) | 84% | 34-47% |

- Nick's clicks are on heads: 87% fall within one head height of a head box.
- Victims stay visible until near contact in 48 of 51 clips.

**Whole frame vs 16 tiles.** One clip (0168422d9be4) on the M1 GPU:

| Input | Time per frame |
|---|---|
| Whole frame, 640 px wide | 88 ms |
| Whole frame, 1280 px | 231 ms |
| Whole frame, native 1920 px | 480 ms |
| 16 tiles | ~1.4 s (estimate) |

The 640 px pass found 76% of the in-zone heads the tiled pass found and boxed the victim 86% of the
time (tiled: 93%). Most misses are heads under 15 px at the far edge, where it finds 35%.

On the full protocol the ranking barely moves:

| Detection | Ranker AUC | Time in top 3 | Reaches top 3 | Detected | Scored | Random reaches top 3 |
|---|---|---|---|---|---|---|
| Tiled | 0.79-0.80 | 22-23% | 78-80% | 69% | 52% | 39% |
| Whole frame | 0.79-0.80 | 26% | 83-85% | 67% | 48% | 46% |

With appearance added: tiled AUC 0.89-0.91 and top 3 36-38%; whole frame AUC 0.89 and top 3 36-37%.
Random reaches the top 3 more often on whole frames because fewer rivals are detected.

- Dropping tiles costs almost nothing on ranking at about 16x less detector compute.
- Direction for the Jetson: whole frame at 640 px, possibly with tiles for the far band only.

## What did not work

- **Stitching track fragments** (`heads-rows --stitch`). Settings were picked without victim labels by
  cutting long tracks and checking the rejoins: a 2 s gap and 0.5 head heights rejoined 20.5% of cuts
  with 4.8% wrong. Victim scored moved from 48% to 49%, with no ranking gain. The victim's breaks do
  not fit the rule: 32% overlap in time (two boxes on the victim at once), and the rest reappear a
  median 1.4 head heights away. Only 5% fall inside the gates, and gates wide enough to catch them join
  the wrong people about 1 time in 5. Kept as an option, off by default.
- **Ghosts and time unseen.** A frozen ghost position beat a velocity-predicted one (median error 1.57
  vs 2.71 head heights). Victims stay visible until near contact, so time-unseen features lowered the
  ranking. Dropped.
- **Staying in place** (net displacement) had no signal: AUC 0.45-0.49.
- **Learned vs equal weights.** At first the learned ranker matched an unfitted equal-weight sum of
  churn and size flicker: the features were the limit.
- **Top-k cascade.** Running DINOv2 only on the motion ranker's top k loses the victims motion misses:
  k = 5 gave AUC 0.82 vs 0.92 fused (labelled moments, exploratory script). Per-track caching every
  2 s keeps the full gain at about a quarter of the crops and was chosen instead.

## Exploratory checks on labelled moments

These used uncommitted scratch scripts on 506 labelled moments in 44 clips (onset to contact - 3 s,
seeds 0-2). They are checks, not headline results.

- **Splash over time.** Nick: victims splash rhythmically while bobbing and thrash in intense cases.
  Splash was measured at every detection frame along place paths. AUC of the victim against v3's top-3
  look-alikes: snapshot splash 0.49, upper splash level over 4 s 0.64, persistence 0.62, in step with
  the bob 0.59, rhythm 0.48. Adding them to v3 gave no gain (AUC 0.940 either way), so the information
  is already there. This check led to the motion-image feature.
- **Look-alikes.** Two of the strongest look-alikes were women lowering their heads to tie or adjust
  their hair, arms up and out of the water (Nick's read). Hard-negative candidate.
- **Greyscale.** Colour vs grey crops: appearance alone 0.90-0.91 vs 0.89-0.90, inside v3 0.935 vs
  0.930. The differences are within noise on every seed, so appearance does not rely on hue.
- **Brightness** (a fairness proxy; there are no skin-tone labels):
  - Among non-victims at the same moment, the final v3 score barely tracks head brightness relative
    to water (correlation -0.01 to +0.02).
  - Top-3 flag rate: darkest third 9.3% vs 7.7-8.3% for the rest. Per-clip difference +1.1 points
    [-1.3, +3.4], 23 clips higher vs 19 lower. No clear bias.
  - Victims' heads are no darker or lighter than others (-51.6 vs -52.1 relative to water).
  - Appearance rises slightly with a lighter area below the head (+0.07 to +0.09), most likely water
    and foam around a submerged body.
  - Brightness mixes skin, hair, wetness and shade. A real fairness check needs varied footage.
- **Where the victim is unscored** (best version, seed 0, pooled). The victim is scored for 63% of the
  window and is in the top 3 for 86% of that. The rest: head vanished under 1 s ago 12%, 1-3 s ago 12%,
  over 3 s ago 4%; not yet seen since onset 9%.
- **Cost.** Place-anchored scoring takes about 3.5 ms per 0.5 s step on one M1 CPU core on the busiest
  clip (67 heads in the zone).

## Limits

- **False alarms are unknown.** All 46 clips are rescues. Without normal pool footage there is no
  honest alarm threshold, so every alarm budget catches 0 and false alarms per hour cannot be measured.
- **Detection caps the result.** The victim has a head box for 67% of the window and is scored for 58%.
  Every ranking number above is limited by that.
- **One venue.** The 46 clips come from one wave pool. Results may not carry over to other pools,
  cameras or lighting.
- **Fairness is untested.** The brightness check is a weak proxy on one pool's footage.

## Next steps

1. **Hold-score test.** When a head vanishes, its place keeps competing with its last score at the
   last-seen spot for 1, 2 or 3 s, with gentle decay. This covers up to ~24 points of the rescue window.
2. **Hard negatives for the head detector** (Nick to label): hair buns, round high-contrast patterns,
   tube rims and handles, and people adjusting their hair. Then retrain.
3. **Normal pool footage**, so alarm thresholds and false alarms per hour can be measured.
4. **Jetson budget.** The whole-frame 640 px detector (~90 ms per frame on the M1) is the main cost.
   Two DINOv2 passes per head every 2 s and ~3.5 ms per step of place features are small next to it.
