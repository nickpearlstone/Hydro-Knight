# synthetic/train: head detector from synthetic frames

Goal: a detector that finds swimmers' heads, including small, far and half-submerged ones. It is
trained on rendered frames whose labels are exact, then checked on real frames that Nick
hand-counted. Real fine-tuning comes later, once real head boxes exist.

## The pipeline, file by file

```
render (Kaggle, Blender)            synthetic/blender/scene.py + synthetic/configs/ref_*.json
   │  rgb + clip frames, seg + headseg masks, labels_*.json
   ▼
post (Kaggle)                       synthetic/tools/post.py
   │  camera look, fish-eye, x264 twice, grain  ->  rgbv_*.png (the training image)
   ▼
boxes (Kaggle)                      synthetic/tools/export_yolo.py
   │  person + head boxes from the VISIBLE mask pixels  ->  yolo/rgbv_*.txt
   ▼
tiles (Mac)                         synthetic/train/make_tiles.py
   │  1920x1080 -> six 640 px tiles each, head boxes only, train / val split by frame
   ▼
upload tiles (private Kaggle dataset hk-synth-tiles; synthetic images only)
   ▼
train (Kaggle T4 x2)                synthetic/kaggle/train_kernel.py
   │  GPU 0: synthetic/train/train_rtdetr.py   (RT-DETRv2-R18, main model)
   │  GPU 1: synthetic/train/train_yolo.py     (YOLO11n, baseline)
   ▼
score (Mac, real frames never leave it)  synthetic/train/score_heads.py
      recall of Nick's 736 swimmer clicks at matched false boxes, vs the off-the-shelf 11n-head
```

### 1. `make_tiles.py`: why tiles
- A head in these frames is 5–40 px across. Detectors take a 640 px input, so feeding the whole
  1920 px frame would shrink every head three times over.
- Cutting 640 px tiles keeps heads at their real size, and testing runs on the same kind of tiles.
- A box goes into a tile only if 60% of it is inside; it's clipped at the edge.
- Validation tiles come from whole frames the model never trains on, so the validation score
  can't be inflated by near-duplicate tiles.

### 2a. `train_rtdetr.py`: fine-tuning, not training from scratch
1. **Download** `PekingU/rtdetr_v2_r18vd` from Hugging Face. Its weights already know what objects
   and people look like, from COCO.
2. **Replace the output layer** (80 COCO classes) with one class, "head". The rest of the network
   keeps its pretrained weights.
3. **Freeze the backbone** (ResNet-18, the general-purpose feature extractor) and train the encoder,
   decoder and new head layer. With 150 frames this overfits less and trains faster.
   `--backbone-lr-scale 0.1` instead lets the backbone learn slowly too, which is the RT-DETR
   authors' fine-tuning recipe. Kaggle runs both versions.
4. **Loop:** AdamW optimiser, cosine learning-rate decay, gradient clipping, mixed precision on the
   GPU, flips and brightness jitter. After every epoch it measures validation loss and keeps the
   best checkpoint (`best/`, loadable with `from_pretrained`).

### 2b. `train_yolo.py`: the baseline
- Same idea through Ultralytics: COCO `yolo11n.pt`, one class, first 10 layers (the backbone)
  frozen, its own augmentation (mosaic, flips, colour) and early stopping on validation mAP.
- AGPL licence, so it's a comparison, not the commercial path.

### 3. `score_heads.py`: the honest test
- Runs a model on the 10 real hand-counted frames (736 swimmers) with the same 640 px tiling,
  merging boxes across tile overlaps.
- A swimmer counts as found when a box covers their click, allowing half a box height of slack,
  since a head box can sit just above a click placed on the body. One box per click.
- Models are compared at the same number of false boxes (20 / 50 / 100 over the 10 frames), so a
  detector that fires everywhere can't win.
- Reference: the off-the-shelf `11n-head` (trained on real crowd photos).

## What this test can and can't say
- **It can say:** whether synthetic-only training already finds real swimmers' heads better than an
  off-the-shelf head detector.
- **It can't say:** how much synthetic pretraining helps a model that is also fine-tuned on real
  head boxes. That needs real head boxes (labelling still to do), and is the next experiment.
