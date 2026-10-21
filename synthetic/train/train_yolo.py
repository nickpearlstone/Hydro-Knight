"""Step 2a (baseline): fine-tune YOLO11n (COCO-pretrained detector) into a 1-class head detector.

Starts from Ultralytics' COCO weights (yolo11n.pt, downloaded on first use), replaces the 80-class
output with one class ("head"), freezes the first --freeze layers (the backbone: general image
features learned on COCO) and trains the rest on the synthetic tiles at their native 640 px.
Ultralytics handles augmentation (flips, scale, colour), the loss and validation; the best weights by
validation mAP land in <project>/<name>/weights/best.pt. Licence note: Ultralytics is AGPL-3.0, so this
model is a comparison baseline, not the commercial path.

    uv run python synthetic/train/train_yolo.py synthetic/renders/tiles_c/data.yaml --epochs 60
"""

import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", help="data.yaml from make_tiles.py")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument(
        "--freeze",
        type=int,
        default=10,
        help="first N layers frozen (10 = YOLO11 backbone)",
    )
    ap.add_argument("--device", default=None)
    ap.add_argument("--project", default="synthetic/renders/train")
    ap.add_argument("--name", default="yolo11n_head")
    a = ap.parse_args()
    from ultralytics import YOLO

    model = YOLO("yolo11n.pt")  # COCO-pretrained detection weights (not the pose model)
    model.train(
        data=a.data,
        epochs=a.epochs,
        imgsz=640,
        batch=a.batch,
        freeze=a.freeze,
        single_cls=True,
        device=a.device,
        project=str(
            Path(a.project).resolve()
        ),  # relative paths get a runs/detect/ prefix
        name=a.name,
        exist_ok=True,
        patience=15,  # stop early if validation mAP stops improving
        cos_lr=True,
        mosaic=1.0,  # stitches 4 tiles per sample: more crowd context per batch
        degrees=0.0,  # cameras are never rolled much; keep heads upright
        fliplr=0.5,
        plots=True,
    )


if __name__ == "__main__":
    main()
