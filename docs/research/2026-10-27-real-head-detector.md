# Head detector trained on real footage (2026-10-27)

**Result:** RT-DETRv2-R18 and YOLO11n trained on real pool stills labelled by SAM 3, plus the synthetic
tiles, both match SAM 3 on Nick's 8 hand-boxed test frames. Every RT-DETR number before this one was
produced by a broken training setup (see "The transformers bug").

## Data

- **Teacher labels.** SAM 3 (`sam3_pcs`, text prompt "head") ran on 1,461 training stills from 46 clips.
  None of those clips are in the test frames (8) or the validation frames (12). Boxes at confidence
  0.40 or higher became labels. 0.40 is the best-F1 cut on Nick's validation dots: SAM 3 finds 93% of
  his heads there, and 87% of its boxes land on a head.
- **Tiles.** `synthetic/train/make_real_tiles.py` cuts the stills into 640 px tiles (same grid and
  60%-inside rule as `make_tiles.py`), with 4 whole clips held out for the trainer's validation loss:
  7,158 training tiles. The 1,620 synthetic tiles (batches c + d) are mixed in. Stills and tiles show
  real people, so they live only in a private Kaggle dataset, never in git.
- **Picking.** Seed and checkpoint are chosen on the validation frames (`score_heads.py --val`), and only
  the pick is scored on the test frames, once.

## Training

`synthetic/kaggle/mix_train_kernel.py`, 2x T4:
- RT-DETR: 4 seeds, 20-epoch schedule from COCO weights, frozen backbone, weights saved every 2 epochs,
  ~7.5 min per epoch.
- YOLO11n: 2 seeds, 30 epochs.

## Results

Share of heads found at a matched number of wrong boxes over all test frames (20 / 50 / 100).
"In range" counts only swimmers inside the clip's zone whose body is at least 60 px tall.

| Model | All 641 heads | In range (123) |
|---|---|---|
| RT-DETR real+synth (seed 1, last epoch) | 61 / 83 / 93% | 98 / 99 / 99% |
| YOLO11n real+synth (seed 1, epoch 20) | 62 / 81 / 93% | 98 / 98 / 99% |
| SAM 3, the teacher (too slow for live use) | 63 / 81 / 95% | 98 / 99 / 99% |
| YOLO11n synthetic only | 47 / 67 / 77% | 90 / 94 / 94% |

Validation frames, at up to 100 wrong boxes, across all seeds and later checkpoints:
- RT-DETR: 83 to 88%.
- YOLO: 82 to 85%.
- Synthetic-only baselines: RT-DETR 30% (broken; see below), YOLO 67%.

The two detectors are tied. A gap of a few points on 8 frames is noise, and both sit at their teacher's
level, which is the most this way of labelling allows. To go beyond SAM 3, the labels need to improve
(hand corrections, or more teachers voting).

## The transformers bug

Under transformers **5.16.1** (what Kaggle installed for every earlier run), RT-DETRv2 fine-tuning leaves
the last decoder layer's classifier untrained. The first two decoder layers reach confident scores
(about 0.99), but the last layer, which produces the output, stays near 0.06 to 0.35. As a result:
- Nothing ever passes a 0.25 confidence cut.
- Recall at matched wrong boxes stays low.

The same 300-step memorisation test on 8 tiles reaches 0.996 on all layers under **5.21.0**.

Ruled out along the way: the EMA weights, mixed precision, augmentation, batch-norm statistics, and
saving and reloading.

The kernel now pins `transformers==5.21.0` and stops if the pin doesn't take. This invalidates every
RT-DETR result from before 2026-10-27, including the synthetic-only 75.5% and the 53.6 ± 15.4% seed
spread.
