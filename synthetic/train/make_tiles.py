"""Step 1: cut rendered frames into 640 px training tiles with head boxes (YOLO format).

Heads are 5-40 px across in a 1920x1080 frame. Shrinking the whole frame to a detector's 640 px input
would make them 3x smaller still, so we cut tiles at native resolution instead (the detector at test
time also runs on tiles). A head box goes into a tile if at least 60% of it falls inside; it is clipped
to the tile edge. Whole frames are split into train / val (never tiles of one frame on both sides).

Output: <out>/images/{train,val}/*.jpg, <out>/labels/{train,val}/*.txt (class 0 = head), data.yaml,
and tiles.json (provenance: source frame, camera config, tile offset).

    uv run python synthetic/train/make_tiles.py synthetic/renders/batch_out_c synthetic/renders/tiles_c
"""

import argparse
import json
import random
from pathlib import Path

import cv2

TILE = 640
MIN_INSIDE = 0.6


def tile_origins(w, h, t=TILE):
    """Grid of tile corners covering the frame, last row / column flush with the edge."""
    xs = list(range(0, max(w - t, 0) + 1, t))
    ys = list(range(0, max(h - t, 0) + 1, t))
    if xs[-1] + t < w:
        xs.append(w - t)
    if ys[-1] + t < h:
        ys.append(h - t)
    return [(x, y) for y in ys for x in xs]


def frames(root: Path):
    """Every finished frame: (image, yolo txt, camera config name)."""
    for txt in sorted(root.glob("**/yolo/rgbv_*.txt")):
        img = txt.parent.parent / f"{txt.stem}.png"
        lab = txt.parent.parent / f"labels_{txt.stem.split('_')[1]}.json"
        if img.exists() and lab.exists():
            yield img, txt, json.loads(lab.read_text())["config"]["name"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--val", type=float, default=0.1, help="share of frames held out")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    items = list(frames(Path(a.src)))
    random.Random(a.seed).shuffle(items)
    n_val = max(1, int(len(items) * a.val))
    meta = []
    counts = {"train": [0, 0], "val": [0, 0]}  # tiles, head boxes
    for k, (img_path, txt, cam) in enumerate(items):
        split = "val" if k < n_val else "train"
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        img = cv2.imread(str(img_path))
        H, W = img.shape[:2]
        heads = []
        for line in txt.read_text().split("\n"):
            if line.startswith("1 "):  # class 1 = head in the export
                _, cx, cy, w, h = map(float, line.split())
                heads.append(
                    (
                        (cx - w / 2) * W,
                        (cy - h / 2) * H,
                        (cx + w / 2) * W,
                        (cy + h / 2) * H,
                    )
                )
        for ox, oy in tile_origins(W, H):
            lines = []
            for x0, y0, x1, y1 in heads:
                cx0, cy0, cx1, cy1 = (
                    max(x0, ox),
                    max(y0, oy),
                    min(x1, ox + TILE),
                    min(y1, oy + TILE),
                )
                if cx1 <= cx0 or cy1 <= cy0:
                    continue
                if (cx1 - cx0) * (cy1 - cy0) < MIN_INSIDE * (x1 - x0) * (y1 - y0):
                    continue
                lines.append(
                    f"0 {((cx0 + cx1) / 2 - ox) / TILE:.6f} {((cy0 + cy1) / 2 - oy) / TILE:.6f} "
                    f"{(cx1 - cx0) / TILE:.6f} {(cy1 - cy0) / TILE:.6f}"
                )
            name = f"{cam}_{img_path.parent.name}_{img_path.stem}_{ox}_{oy}"
            cv2.imwrite(
                str(out / "images" / split / f"{name}.jpg"),
                img[oy : oy + TILE, ox : ox + TILE],
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            )
            (out / "labels" / split / f"{name}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else "")
            )
            meta.append(
                {
                    "tile": name,
                    "split": split,
                    "frame": str(img_path),
                    "camera": cam,
                    "x": ox,
                    "y": oy,
                    "heads": len(lines),
                }
            )
            counts[split][0] += 1
            counts[split][1] += len(lines)
    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: head\n"
    )
    (out / "tiles.json").write_text(json.dumps(meta))
    print(
        f"{len(items)} frames -> train {counts['train'][0]} tiles / {counts['train'][1]} heads, "
        f"val {counts['val'][0]} tiles / {counts['val'][1]} heads"
    )


if __name__ == "__main__":
    main()
