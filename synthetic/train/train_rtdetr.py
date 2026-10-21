"""Step 2b (main model): fine-tune RT-DETRv2-R18 (COCO-pretrained, Apache-2.0) into a head detector.

1. Load PekingU/rtdetr_v2_r18vd from the Hugging Face Hub with its COCO weights, but with a fresh
   1-class output ("head"); ignore_mismatched_sizes lets the 80-class classifier be replaced.
2. Freeze the ResNet-18 backbone (--freeze-backbone, default on) so only the hybrid encoder, the
   transformer decoder and the new class/box heads learn. With --backbone-lr-scale > 0 instead, the
   backbone trains too at a lower learning rate (the RT-DETR authors' fine-tuning recipe).
3. Train on the 640 px synthetic tiles (YOLO txt labels read directly), with horizontal flips and
   small brightness / contrast jitter, AdamW, cosine schedule, mixed precision on CUDA.
4. After each epoch, compute validation loss and keep the best weights (save_pretrained) in
   <out>/best/, loadable with AutoModelForObjectDetection.from_pretrained.

    uv run --with transformers python synthetic/train/train_rtdetr.py synthetic/renders/tiles_c --epochs 40
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch

NAME = "PekingU/rtdetr_v2_r18vd"


class Tiles(torch.utils.data.Dataset):
    """640 px tiles + YOLO head boxes -> (RGB image, COCO-style target) for the HF image processor."""

    def __init__(self, root, split, augment):
        self.imgs = sorted((Path(root) / "images" / split).glob("*.jpg"))
        self.labels = Path(root) / "labels" / split
        self.augment = augment

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        img = cv2.cvtColor(cv2.imread(str(self.imgs[i])), cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]
        boxes = []
        txt = self.labels / f"{self.imgs[i].stem}.txt"
        for line in txt.read_text().split("\n") if txt.exists() else []:
            if line.strip():
                _, cx, cy, w, h = map(float, line.split())
                boxes.append(
                    [(cx - w / 2) * W, (cy - h / 2) * H, w * W, h * H]
                )  # COCO xywh, pixels
        if self.augment:
            if random.random() < 0.5:
                img = img[:, ::-1].copy()
                boxes = [[W - x - w, y, w, h] for x, y, w, h in boxes]
            a, b = random.uniform(0.8, 1.2), random.uniform(-20, 20)
            img = np.clip(img.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
        anns = [
            {"bbox": b, "category_id": 0, "area": b[2] * b[3], "iscrowd": 0}
            for b in boxes
        ]
        return img, {"image_id": i, "annotations": anns}


class Collate:
    """Batch tiles through the HF image processor (resize, normalise, boxes -> its label format).

    A class rather than a closure so DataLoader workers can pickle it (macOS spawns them).
    """

    def __init__(self, proc):
        self.proc = proc

    def __call__(self, batch):
        imgs, targets = zip(*batch, strict=True)
        return self.proc(
            images=list(imgs), annotations=list(targets), return_tensors="pt"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tiles", help="folder from make_tiles.py")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument(
        "--backbone-lr-scale", type=float, default=0.0, help="0 = backbone frozen"
    )
    ap.add_argument("--max-steps", type=int, default=0, help="stop early (smoke tests)")
    ap.add_argument("--out", default="synthetic/renders/train/rtdetrv2_head")
    a = ap.parse_args()
    from transformers import AutoImageProcessor, AutoModelForObjectDetection

    dev = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    proc = AutoImageProcessor.from_pretrained(NAME, size={"height": 640, "width": 640})
    model = AutoModelForObjectDetection.from_pretrained(
        NAME,
        num_labels=1,
        id2label={0: "head"},
        label2id={"head": 0},
        ignore_mismatched_sizes=True,
    ).to(dev)
    backbone = [p for n, p in model.named_parameters() if ".backbone." in n]
    rest = [p for n, p in model.named_parameters() if ".backbone." not in n]
    if a.backbone_lr_scale <= 0:
        for p in backbone:
            p.requires_grad = False
        groups = [{"params": rest, "lr": a.lr}]
    else:
        groups = [
            {"params": rest, "lr": a.lr},
            {"params": backbone, "lr": a.lr * a.backbone_lr_scale},
        ]
    n_train = sum(p.numel() for g in groups for p in g["params"])
    print(
        f"device {dev}; training {n_train / 1e6:.1f} M of {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M params"
    )
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    dl = torch.utils.data.DataLoader(
        Tiles(a.tiles, "train", True),
        batch_size=a.batch,
        shuffle=True,
        num_workers=2,
        collate_fn=Collate(proc),
    )
    dv = torch.utils.data.DataLoader(
        Tiles(a.tiles, "val", False), batch_size=a.batch, collate_fn=Collate(proc)
    )
    total = a.epochs * len(dl)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, total) / total))
    )
    use_amp = dev == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    def to_dev(b):
        return {
            "pixel_values": b["pixel_values"].to(dev),
            "labels": [{k: v.to(dev) for k, v in t.items()} for t in b["labels"]],
        }

    best, history, step = float("inf"), [], 0
    out = Path(a.out)
    for ep in range(a.epochs):
        model.train()
        t0, tr = time.time(), []
        for b in dl:
            with torch.autocast("cuda", enabled=use_amp):
                loss = model(**to_dev(b)).loss
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tr.append(loss.item())
            step += 1
            if a.max_steps and step >= a.max_steps:
                break
        model.eval()
        with torch.no_grad():
            va = [model(**to_dev(b)).loss.item() for b in dv]
        row = {
            "epoch": ep,
            "train_loss": float(np.mean(tr)),
            "val_loss": float(np.mean(va)),
            "s": round(time.time() - t0),
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if row["val_loss"] < best:
            best = row["val_loss"]
            model.save_pretrained(out / "best")
            proc.save_pretrained(out / "best")
        if a.max_steps and step >= a.max_steps:
            break
    out.mkdir(parents=True, exist_ok=True)
    (out / "history.json").write_text(json.dumps(history, indent=1))


if __name__ == "__main__":
    main()
