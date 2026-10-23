"""Step 2b (main model): fine-tune RT-DETRv2-R18 (COCO-pretrained, Apache-2.0) into a head detector.

1. Load PekingU/rtdetr_v2_r18vd from the Hugging Face Hub with its COCO weights, but with a fresh
   1-class output ("head"); ignore_mismatched_sizes lets the 80-class classifier be replaced.
2. Freeze the ResNet-18 backbone (--freeze-backbone, default on) so only the hybrid encoder, the
   transformer decoder and the new class/box heads learn. With --backbone-lr-scale > 0 instead, the
   backbone trains too at a lower learning rate (the RT-DETR authors' fine-tuning recipe).
3. Train on the 640 px synthetic tiles (YOLO txt labels read directly), with horizontal flips and
   small brightness / contrast jitter, AdamW, cosine schedule, mixed precision on CUDA.
4. Scale/crop jitter (0.5-1.5x), warm-up, an EMA of the weights. After each epoch, compute
   validation loss (EMA weights) and keep the best weights (save_pretrained) in
   <out>/best/, loadable with AutoModelForObjectDetection.from_pretrained.
5. Seed runs: --seed fixes the random start (new class head, shuffling, augmentation);
   --stop-after N trains only the first N epochs of the --epochs learning-rate schedule;
   --save-every K also keeps the EMA weights at epochs K, 2K, ... in <out>/epoch_NNN/.

    uv run --with transformers python synthetic/train/train_rtdetr.py synthetic/renders/tiles_c --epochs 40
"""

import argparse
import copy
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
            if random.random() < 0.8:  # zoom 0.5-1.5x: the same head seen at many sizes
                img, boxes = rescale(img, boxes, random.uniform(0.5, 1.5))
            a, b = random.uniform(0.8, 1.2), random.uniform(-20, 20)
            img = np.clip(img.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
        anns = [
            {"bbox": b, "category_id": 0, "area": b[2] * b[3], "iscrowd": 0}
            for b in boxes
        ]
        return img, {"image_id": i, "annotations": anns}


def rescale(img, boxes, s):
    """Zoom a tile by s, then crop (s > 1) or pad with the tile's mean colour (s < 1) back to its size.

    Boxes (COCO xywh) move with the image; a box keeps going only if 40% of it survives the crop.
    """
    H, W = img.shape[:2]
    big = cv2.resize(img, (round(W * s), round(H * s)), interpolation=cv2.INTER_LINEAR)
    bh, bw = big.shape[:2]
    if s >= 1:
        ox, oy = random.randint(0, bw - W), random.randint(0, bh - H)
        out = big[oy : oy + H, ox : ox + W]
        dx, dy = -ox, -oy
    else:
        out = np.empty_like(img)
        out[:] = img.reshape(-1, 3).mean(0).astype(np.uint8)
        dx, dy = random.randint(0, W - bw), random.randint(0, H - bh)
        out[dy : dy + bh, dx : dx + bw] = big
    kept = []
    for x, y, w, h in boxes:
        x0, y0, x1, y1 = x * s + dx, y * s + dy, (x + w) * s + dx, (y + h) * s + dy
        cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
        if (
            cx1 - cx0 >= 2
            and cy1 - cy0 >= 2
            and (cx1 - cx0) * (cy1 - cy0) >= 0.4 * (x1 - x0) * (y1 - y0)
        ):
            kept.append([cx0, cy0, cx1 - cx0, cy1 - cy0])
    return out, kept


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
    ap.add_argument(
        "--warmup-epochs", type=float, default=2.0, help="linear learning-rate warm-up"
    )
    ap.add_argument(
        "--ema", type=float, default=0.999, help="EMA decay of the weights (0 = off)"
    )
    ap.add_argument("--max-steps", type=int, default=0, help="stop early (smoke tests)")
    ap.add_argument(
        "--stop-after",
        type=int,
        default=0,
        help="train only the first N epochs of the --epochs schedule (0 = all)",
    )
    ap.add_argument(
        "--save-every",
        type=int,
        default=0,
        help="also keep the EMA weights every N epochs in <out>/epoch_NNN/ (0 = off)",
    )
    ap.add_argument(
        "--seed", type=int, default=None, help="seed Python, NumPy and torch"
    )
    ap.add_argument("--out", default="synthetic/renders/train/rtdetrv2_head")
    a = ap.parse_args()
    if a.seed is not None:  # before the model loads: the new class head is random too
        random.seed(a.seed)
        np.random.seed(a.seed)
        torch.manual_seed(a.seed)  # all devices, and the DataLoader workers' seeds
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
    total, warm = a.epochs * len(dl), max(1, int(a.warmup_epochs * len(dl)))

    def lr_factor(s):  # linear warm-up, then cosine decay to zero
        if s < warm:
            return (s + 1) / warm
        return 0.5 * (
            1 + math.cos(math.pi * min(s - warm, total - warm) / max(total - warm, 1))
        )

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    # EMA: a running average of the weights, smoother than the last step; used for val + saving
    ema = copy.deepcopy(model).eval() if a.ema > 0 else model
    for p_ in ema.parameters():
        p_.requires_grad_(False)

    def ema_update():
        if a.ema <= 0:
            return
        with torch.no_grad():
            for e, m in zip(
                ema.state_dict().values(), model.state_dict().values(), strict=True
            ):
                if e.dtype.is_floating_point:
                    e.mul_(a.ema).add_(m.detach(), alpha=1 - a.ema)
                else:
                    e.copy_(m)

    use_amp = dev == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    def to_dev(b):
        return {
            "pixel_values": b["pixel_values"].to(dev),
            "labels": [{k: v.to(dev) for k, v in t.items()} for t in b["labels"]],
        }

    best, history, step = float("inf"), [], 0
    out = Path(a.out)
    for ep in range(a.stop_after or a.epochs):
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
            ema_update()
            tr.append(loss.item())
            step += 1
            if a.max_steps and step >= a.max_steps:
                break
        ema.eval()
        with torch.no_grad():
            va = [ema(**to_dev(b)).loss.item() for b in dv]
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
            ema.save_pretrained(out / "best")
            proc.save_pretrained(out / "best")
        if a.save_every and ep and ep % a.save_every == 0:
            ema.save_pretrained(out / f"epoch_{ep:03d}")
            proc.save_pretrained(out / f"epoch_{ep:03d}")
        if a.max_steps and step >= a.max_steps:
            break
    ema.save_pretrained(out / "last")
    proc.save_pretrained(out / "last")
    out.mkdir(parents=True, exist_ok=True)
    (out / "history.json").write_text(json.dumps(history, indent=1))


if __name__ == "__main__":
    main()
