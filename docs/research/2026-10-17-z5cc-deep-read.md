# Z5cc/drowning-detection: deep read

Read on 2026-10-17 from [github.com/Z5cc/drowning-detection](https://github.com/Z5cc/drowning-detection)
(David Nicklaser, September 2024, last touched December 2025). Every number below comes from
reading his code and label files, or from our own comparison script
(`scratch/z5cc/mine_labels.py`, gitignored). His label files were read from a local clone in
`scratch/`; none are copied into this repo.

**License.** The MIT notices in his README cover the YOWOv2 and ConvLSTM code he built on.
His own code and his labels state no license, and the README asks anyone who uses the
project "in any form" to cite him. If any of his labels or results end up in committed work,
cite him in the README:

```bibtex
@misc{nicklaser2024drowning,
    title={Drowning Detection via Real-Time Spatio-Temporal Action Localization},
    author={Nicklaser, David},
    year={2024},
    url={https://github.com/Z5cc/drowning-detection}
}
```

## 1. The approach in one paragraph

He treats drowning as an **action to localize**: for every frame, draw a box around each
person and label the box *swim* or *drown*. The model is **YOWOv2** ("You Only Watch Once"),
a YOLO-style detector with a second branch that looks at the last 32 frames through 3D
convolutions, so the box label can depend on motion and not just on one frame. His own
contribution is **2Dseq**: he drops the 3D branch and adds a **convolutional LSTM** (a
memory cell that keeps a running summary of each spot in the feature map from frame to
frame). That model carries its memory forward through the video, so each new frame costs one
2D pass. It's 3x faster than YOWOv2 at the same size and almost as accurate.

## 2. Data

| | |
|---|---|
| Source | 82 YouTube videos, mostly the Lifeguard Rescue channel (55 are in our manifest) |
| Snippets | 111 listed, 91 with label files; median 14.5 s (10 to 89 s) |
| Crop | a **hand-placed square crop around the incident**, scaled to 224×224; median crop side = 30% of frame height (7% to 97%) |
| Labels | CVAT boxes on every person in the crop, *swim* or *drown*, drawn every 6th frame and interpolated; about 3 people per crop |
| With a drown box | 55 of 91 snippets; the rest are crops with only swimming (some from the same videos) |

The crop is the most important design choice to understand. **He never asks the model to
find the victim in a full crowded frame.** A person picked the 224-pixel square where the
rescue happens, so the model sees about three people at a large size. In our wave-pool
frames, about 21 swimmers are in view and the victim is often under 60 px tall at 1080p. His
problem is "given this small patch, which of these three people is drowning". Ours is
"which of the 21, and also find them".

Reconstruction: `dataset_utils/LifeguardUtils.py` downloads each video with yt-dlp
(`format: bv`), then ffmpeg cuts `[start, end]`, crops, scales to 224 and resamples to
30 fps. The XML frame ids index those 30 fps frames, starting at 0 at the snippet start.

## 3. The models

All four variants share YOWOv2's parts:
- **2D backbone** ("YOLO-free", an anchor-free YOLO) on the newest frame, giving features at
  three scales (strides 8, 16, 32).
- **3D backbone** (ShuffleNetV2 3D) over the last 32 frames, every 6th frame here
  (`T_len 32`, `T_distance 6`, so about 6.4 s of history).
- **Fusion and decoupled heads**: separate branches predict *is there a person* (confidence),
  *swim or drown* (class) and the box.

| Variant | Where the ConvLSTM goes | Result |
|---|---|---|
| YOWOv2 (reference) | none | baseline |
| 3Dseq | on the 3D features | lost to 2Dseq |
| 2D3Dseq | on 3D features, 2D kept | lost to 2Dseq |
| **2Dseq** | on each of the three 2D feature scales, separately for the class and box branches; 3D branch removed | final pick |

2Dseq details (`models/yowo/yowo_2Dseq.py`, `config/2Dseq_tiny.py`):
- Training sequences: 48 frames, every 3rd frame (`S_len 48`, `S_distance 3`), so 4.8 s per
  sequence; a new sequence starts every 6 frames. Two ConvLSTM layers, 64 channels (tiny).
- Inference streams: one frame at a time, every 3rd frame (10 fps), with the memory carried
  forward and reset at the start of each snippet. That is causal and fits live use.
- The ConvLSTM memory is **per location in the feature map**: it remembers "what happened at
  this spot". That works in a fixed small crop. It weakens when the camera moves or the
  person swims across the patch, because their history stays behind at the old spot.

Training: AdamW, lr 1e-4, 10 epochs with the learning rate halved at epochs 2, 3 and 4,
batch 8, colour and position jitter, ImageNet-pretrained backbones. He also tried a
self-attention ConvLSTM (`sa-convlstm` branch), and an earlier SlowFast attempt (`slowfast`
branch, June 2024) that he dropped for YOWOv2. About 50 configs in `config/old/` sweep
sequence length, spacing, layer count, ConvLSTM width and frozen backbones.

## 4. Results, and what they measure

From his `readme/model_selection.png` (one split, epoch 6):

| Model | AP | FPS (i5 laptop CPU) | GFLOPs | M params |
|---|---|---|---|---|
| YOWOv2 tiny | 49.8 | 8 | 4.5 | 10.9 |
| YOWOv2 medium | 54.7 | 5 | 13.7 | 52.0 |
| **2Dseq tiny** | 48.4 | **25** | 2.2 | 10.1 |
| 2Dseq medium | 43.4 | 8 | 15.6 | 63.5 |
| 2Dseq SA tiny | 47.6 | 12 | 2.5 | 11.0 |

How to read it:
- **AP is frame mAP at IoU 0.5** (`evaluator/cal_frame_mAP.py`, PASCAL VOC style): a box
  counts if it overlaps a labeled box by half and has the right class, averaged over the
  *swim* and *drown* classes (the plot doesn't say whether "AP" is drown alone or the mean).
  It measures box quality frame by frame. It doesn't measure what a pool needs: *was each
  drowning caught, how fast, and how many false alarms per hour*.
- **The ConvLSTM didn't win on accuracy.** 2Dseq tiny is 1.4 points below YOWOv2 tiny, and
  medium 2Dseq is worse than tiny. The README's claim that 2Dseq "addresses" bobbing and
  crawling rests on example GIFs, not on a number. Its real win is **3x speed**.
- **One split, one seed, no error bars**, and test sets of 10 snippets, about 6 with
  drowning. A 1 to 5 point gap is within noise at that size.

### His train/test splits leak

He splits **by snippet**, and 19 videos give more than one snippet, often overlapping in
time (17 overlapping pairs; sometimes the same moment cropped twice). Counting test snippets
whose footage also appears, at the same seconds of the same video, in that fold's training
set:

| Fold | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | main |
|---|---|---|---|---|---|---|---|---|---|---|
| Test snippets overlapping training | 3/10 | 2/10 | 4/10 | 3/10 | 4/10 | 3/10 | 2/10 | 4/10 | 4/11 | 2/9 |

So 20 to 40% of each test set was seen in training (sometimes cropped differently). The AP
numbers are optimistic by an unknown amount. It's the same lesson as our own audit: split
by whole video.

## 5. His labels against ours (55 shared videos, 41 with our victim clicks)

**Timing.** On 48 snippets where both of us mark a drowning:
- His first *drown* box is a median **2.4 s earlier** than our onset. 20 are within 2 s, 34
  within 5 s, 2 differ by more than 10 s (`92b58607c295` -11.4 s, `27fd21052cf8` -11.3 s).
  His onset can't be earlier than his snippet start, so a few gaps are understated.
- His last *drown* box ends a median 1.7 s before our guard contact. We agree on where
  the event ends.

The research sweep's "34 of 49 within 5 s" figure is this same comparison.

**Who the victim is.** We mapped his crop boxes back to full-frame coordinates and checked
each of our victim clicks that falls inside one of his labeled frames (388 clicks):
- 329 fall inside his crop. Of those, 317 sit within half a box height of one of his boxes:
  **295 on a *drown* box, 22 on a *swim* box.** 12 have no box nearby.
- So on most clips we picked the same person. The disagreements cluster in a few clips,
  worth re-checking:

| Clip | What disagrees |
|---|---|
| `53edb0ec3c31` (sBlxMGGOpVk) | all 10 of our clicks sit on people he labeled *swim*; his drown box is the person just above, in a tight cluster |
| `8c83f0f8a0da` (u8gJy6Jeuzg) | 7 of 10 clicks on a *swim* box; handheld camera |
| `cfe1a153af39` (lvS0S6X3_xc) | all 11 clicks are outside his drown crop: **we may have marked a different person, or a different rescue** |
| `0168422d9be4` (yXHrMeNdyaE) | our onset click is on a person he labels *swim*; his drown starts 5.8 s after ours |
| `92b58607c295`, `27fd21052cf8` | onset 11 s apart (already flagged in the sweep) |

## 6. What to take, and what not to

**Take**
1. **Use his drown boxes as a second labeler**: re-check the six clips above first. Then,
   with his permission or a citation, use his boxes as extra victim boxes for the head
   detector fine-tune and as *swim* negatives in the same frames.
2. **His hard cases are our hard cases.** "Descending and ascending" (bobbing for fun) and
   "crawling" (front crawl) were what fooled his models, while splashing caused fewer errors
   than expected. That matches your tells, and it says the features to get right are bobbing
   regularity and forward progress.
3. **Streaming memory is the right shape for live use.** One cheap pass per frame plus a
   carried state, reset per camera. Our per-head tracks with a running state (ghost tracks, a
   GRU-style "time since last seen") get the same benefit without the per-location weakness.
4. **The video list**, including the 25 videos we don't have (water polo, dunking games and
   other hard negatives), already queued as harvest candidates in the sweep.

**Don't take**
1. **The hand-placed crop.** It solves the part of the problem that is hardest for us
   (finding the small victim among 21 swimmers). Any comparison with his numbers has to
   account for it.
2. **Frame mAP as the headline metric.** Keep ours: catches at a fixed alarms per hour,
   delay, random baseline.
3. **His splits.** Use whole-video splits.
4. **An end-to-end box classifier as the main detector, for now.** It needs many boxed
   drowning frames, and its memory is tied to positions in the image. Our head-anchored plan
   keeps identity in the tracker and the features explainable. YOWOv2 or 2Dseq stays a
   reasonable *second-stage* idea: run it on a 224 crop around each in-range head that the
   rules flag, which is close to the setting his model was trained for.

## 7. Open questions for Nick

- Re-check the six disputed clips (an hour or less in the labeler)?
- Message him? Not needed for a citation, but he offers the videos and may have the trained
  weights, which would let us run 2Dseq on our crops as a second-stage test without
  retraining.
