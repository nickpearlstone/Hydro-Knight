"""Fine-tune YOLO11n into a 1-class head detector. Ultralytics is AGPL-3.0: comparison baseline only.

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
    ap.add_argument(
        "--seed", type=int, default=0, help="Ultralytics' seed (its default is 0)"
    )
    ap.add_argument(
        "--save-period",
        type=int,
        default=-1,
        help="also keep weights/epochN.pt every N epochs (-1 = off)",
    )
    a = ap.parse_args()
    from ultralytics import YOLO

    model = YOLO("yolo11n.pt")
    model.train(
        data=a.data,
        epochs=a.epochs,
        imgsz=640,
        batch=a.batch,
        freeze=a.freeze,
        single_cls=True,
        device=a.device,
        project=str(Path(a.project).resolve()),
        name=a.name,
        exist_ok=True,
        seed=a.seed,
        save_period=a.save_period,
        patience=15,
        cos_lr=True,
        mosaic=1.0,
        degrees=0.0,
        fliplr=0.5,
        plots=True,
    )


if __name__ == "__main__":
    main()
