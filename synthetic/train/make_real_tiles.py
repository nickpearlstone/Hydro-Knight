"""Cut real pool stills into 640 px training tiles, with SAM 3's head boxes as the labels.

SAM 3 (text prompt "head") is the teacher: its boxes at or above --cut become the labels the student
detector (RT-DETR) learns from. The cut is chosen on Nick's 12 validation frames, never on the test
frames (0.40 = best F1 there: 93% of his heads found, 87% of boxes on a head). Tiles use make_tiles.py's
grid and 60%-inside rule, so real and synthetic tiles look the same to the trainer and can share a folder.
Whole clips are held out for the trainer's validation loss (never frames of one clip on both sides).

    uv run python synthetic/train/make_real_tiles.py <stills dir> heads_gpu0.json heads_gpu1.json \
        --out synthetic/renders/tiles_real --cut 0.40

Tile names: real_<clip_id>_<frame>_<x>_<y>. Output layout matches make_tiles.py (images/, labels/,
tiles.json); stills and tiles show real people, so they stay out of git and go only to private datasets.
"""

import argparse
import json
import random
from pathlib import Path

import cv2
from make_tiles import MIN_INSIDE, TILE, tile_origins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stills", help="folder of <clip_id>_<frame>.jpg")
    ap.add_argument(
        "heads", nargs="+", help="SAM 3 output: {name: {boxes, scores, size}}"
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--cut", type=float, default=0.40)
    ap.add_argument("--val-clips", type=int, default=4, help="whole clips held out")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    heads = {}
    for p in a.heads:
        heads.update(json.loads(Path(p).read_text()))
    clips = sorted({name.rsplit("_", 1)[0] for name in heads})
    val = set(random.Random(a.seed).sample(clips, a.val_clips))
    out = Path(a.out)
    meta, counts = [], {"train": [0, 0], "val": [0, 0]}  # tiles, head boxes
    for name, h in sorted(heads.items()):
        img = cv2.imread(str(Path(a.stills) / f"{name}.jpg"))
        H, W = img.shape[:2]
        assert list(h["size"]) == [W, H], name
        boxes = [b for b, s in zip(h["boxes"], h["scores"], strict=True) if s >= a.cut]
        split = "val" if name.rsplit("_", 1)[0] in val else "train"
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for ox, oy in tile_origins(W, H):
            lines = []
            for x0, y0, x1, y1 in boxes:
                cx0, cy0 = max(x0, ox), max(y0, oy)
                cx1, cy1 = min(x1, ox + TILE), min(y1, oy + TILE)
                if cx1 <= cx0 or cy1 <= cy0:
                    continue
                if (cx1 - cx0) * (cy1 - cy0) < MIN_INSIDE * (x1 - x0) * (y1 - y0):
                    continue
                lines.append(
                    f"0 {((cx0 + cx1) / 2 - ox) / TILE:.6f} {((cy0 + cy1) / 2 - oy) / TILE:.6f} "
                    f"{(cx1 - cx0) / TILE:.6f} {(cy1 - cy0) / TILE:.6f}"
                )
            tile = f"real_{name}_{ox}_{oy}"
            cv2.imwrite(
                str(out / "images" / split / f"{tile}.jpg"),
                img[oy : oy + TILE, ox : ox + TILE],
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            )
            (out / "labels" / split / f"{tile}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else "")
            )
            meta.append(
                {
                    "tile": tile,
                    "split": split,
                    "frame": name,
                    "x": ox,
                    "y": oy,
                    "heads": len(lines),
                }
            )
            counts[split][0] += 1
            counts[split][1] += len(lines)
    (out / "tiles.json").write_text(
        json.dumps({"cut": a.cut, "val_clips": sorted(val), "tiles": meta})
    )
    print(
        f"{len(heads)} stills, cut {a.cut} -> train {counts['train'][0]} tiles / {counts['train'][1]} heads, "
        f"val {counts['val'][0]} tiles / {counts['val'][1]} heads (clips {sorted(val)})"
    )


if __name__ == "__main__":
    main()
