# Licence-path benchmark: permissive detectors vs YOLO11n-pose (2026-10-18)

**Result: every Apache-2.0 detector tested finds more swimmers and far more victims than
YOLO11n-pose (AGPL) at the same number of false boxes, at similar speed on the M1.** Leaving
Ultralytics looks like an upgrade, not a sacrifice. Code and boxes: `scratch/licbench/`
(gitignored); throwaway environment via `uv run --with transformers --with rfdetr`, project
lockfile untouched.

## Setup

- Same production tiling for every model: 480 px tiles plus the whole frame, IoS merge.
  Person boxes kept down to confidence 0.05. All models are COCO-pretrained, nothing
  fine-tuned.
- **Swimmers:** the 10 hand-counted frames (736 people in the water). A swimmer counts as
  found if a box holds their click (box padded 10%, one box per click), the rule from
  FINDINGS §4.
- **Victims:** the 78 labelled victim clicks from the 6 head-bake-off clips. A click counts as
  seen if a box holds it or lies within half a local swimmer height.
- **Matched operating points:** each model's threshold is set so its false boxes on the count
  frames stay at or under 20 / 30 / 50, and its victim rate is read at that same threshold.
  A model that fires everywhere can't win.
- The YOLO11n-pose baseline reuses the boxes saved by the earlier benchmarks. It reproduces
  FINDINGS §4: 157 of 736 found at 20 false boxes, versus 146 at about 20 there.

## Results

Share of counted swimmers found / share of victim clicks seen, at each false-box budget:

| Model | Licence | 20 false boxes | 30 false boxes | 50 false boxes | s / frame (M1, tiled) |
|---|---|---|---|---|---|
| YOLO11n-pose (baseline) | AGPL-3.0 | 21% / 32% | 30% / 41% | 39% / 60% | 1.1 (FINDINGS §4) |
| D-FINE nano | Apache-2.0 | 32% / 69% | 40% / 82% | 44% / 82% | 1.16 |
| D-FINE small | Apache-2.0 | 35% / 80% | 44% / 85% | 53% / 92% | 1.65 |
| RF-DETR Nano | Apache-2.0 | 31% / 85% | 38% / 88% | 50% / 96% | 1.02 |
| RF-DETR Small | Apache-2.0 | 33% / 87% | 45% / 96% | 55% / 99% | 1.84 |
| RT-DETRv2 R18 | Apache-2.0 | **42%** / 86% | 46% / 87% | 52% / 92% | 1.82 |

At the 20-false-box point, the permissive models stack 7 to 24 duplicate boxes over the 10
frames (YOLO: 3) and draw 26 to 36 boxes per frame (YOLO: 18). The extra boxes are mostly
extra real people (227 to 312 found vs 157), so duplicates don't explain the victim gain.

## Caveats

- **The baseline is a pose model.** YOLO11n-pose's boxes come from a keypoint model; the plain
  YOLO11n detector wasn't tested (its weights weren't on the approved download list). Part of
  the gap may be "detector vs pose model" rather than "DETR vs YOLO". Worth one more run.
- **Victim clicks come from 6 clips and are correlated.** 78 clicks are not 78 independent
  victims, and the half-height tolerance is lenient. Treat the victim column as direction, the
  swimmer column as the firmer number.
- **Speed is M1 + PyTorch (MPS with CPU fallback for some DETR ops).** Jetson TensorRT
  numbers will differ. DETR-family models export to TensorRT, but the head-to-head on the Nano
  is still to be measured.
- **These are person detectors.** The plan is a head detector; this tells us which
  *architecture and licence* to fine-tune from, not the final model.

## Recommendation

1. **Fine-tune the head detector from an Apache model, not Ultralytics.** RT-DETRv2-R18 (best
   swimmer recall at the strict budget) and RF-DETR Nano (fastest, near-best victims) are the
   two to try first; D-FINE-N is the smallest if the Nano's speed decides.
2. **Re-run with plain YOLO11n detect** (5 MB download) to settle the pose-vs-detector caveat
   before writing this into FINDINGS.
3. **Benchmark the two front-runners on the Jetson with TensorRT FP16** once it's set up.
