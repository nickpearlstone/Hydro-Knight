"""Export person and head boxes from the render masks in YOLO format (run after post.py).

Boxes come from the visible pixels in seg_<n>.npy (whole person) and headseg_<n>.npy (head only), so a
half-submerged head gets a box around the part above the water. Both masks are already distorted to
match the fish-eye image when post.py applied a lens. Per frame:
  yolo/rgbv_<n>.txt   "class cx cy w h", normalised; class 0 = person, 1 = head
  yolo/rgbv_<n>.json  the same boxes with person id, role, pose, share of head under water, pixels

    uv run python synthetic/tools/export_yolo.py synthetic/renders/b20
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

MIN_PX = {0: 20, 1: 6}  # fewer visible pixels than this = nobody could label it


def boxes(mask, min_px):
    """{id: (x0, y0, x1, y1, pixels)} for every non-zero id with at least min_px pixels."""
    ids = np.unique(mask)
    ids = ids[ids > 0]
    if len(ids) == 0:
        return {}
    counts = ndimage.sum_labels(np.ones_like(mask), mask, ids)
    objs = ndimage.find_objects(mask.astype(np.int32))
    out = {}
    for i, c in zip(ids, counts, strict=True):
        sl = objs[i - 1]
        if c >= min_px and sl is not None:
            out[int(i)] = (sl[1].start, sl[0].start, sl[1].stop, sl[0].stop, int(c))
    return out


def add_deck_heads(head, seg, people):
    """Procedural deck people and lifeguards carry no is_head mark, so the head pass misses them.

    Their head = their own visible pixels inside the projected head circle (centre u, v; radius
    head_r_px, padded 20%). Swimmers are left alone: no head pixels there means the head is under
    water or hidden, which is the correct label.
    """
    head = head.copy()
    H, W = head.shape
    yy, xx = np.ogrid[:H, :W]
    for p in people.values():
        if p.get("role", "swimmer") == "swimmer" or not p.get("in_front", True):
            continue
        r = 1.2 * p["head_r_px"]
        if r < 1 or (head == p["id"]).any():
            continue
        x0, x1 = int(max(p["u"] - r, 0)), int(min(p["u"] + r + 1, W))
        y0, y1 = int(max(p["v"] - r, 0)), int(min(p["v"] + r + 1, H))
        if x0 >= x1 or y0 >= y1:
            continue
        circle = (xx[:, x0:x1] - p["u"]) ** 2 + (yy[y0:y1] - p["v"]) ** 2 <= r * r
        mine = (seg[y0:y1, x0:x1] == p["id"]) & circle
        head[y0:y1, x0:x1][mine] = p["id"]
    return head


def export(d: Path):
    (d / "yolo").mkdir(exist_ok=True)
    (d / "yolo" / "classes.txt").write_text("person\nhead\n")
    for lf in sorted(d.glob("labels_*.json")):
        n = lf.stem.split("_")[1]
        lab = json.loads(lf.read_text())
        people = {p["id"]: p for p in lab["people"]}
        lines, rows = [], []
        seg_mask = None  # person mask, needed by the head pass for deck people
        for cls, name in ((0, "seg"), (1, "headseg")):
            f = d / f"{name}_{n}.npy"
            if not f.exists() and f.with_suffix(".npz").exists():
                f = f.with_suffix(".npz")  # Kaggle batches store masks compressed
            if not f.exists():
                continue
            m = np.load(f)["m"] if f.suffix == ".npz" else np.load(f)
            if cls == 1:
                m = add_deck_heads(m, seg_mask, people)
            else:
                seg_mask = m
            H, W = m.shape
            for pid, (x0, y0, x1, y1, px) in boxes(m, MIN_PX[cls]).items():
                p = people.get(pid, {})
                cx, cy, w, h = (
                    (x0 + x1) / 2 / W,
                    (y0 + y1) / 2 / H,
                    (x1 - x0) / W,
                    (y1 - y0) / H,
                )
                lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                rows.append(
                    {
                        "cls": cls,
                        "id": pid,
                        "box": [x0, y0, x1, y1],
                        "px": px,
                        "role": p.get("role"),
                        "pose": p.get("pose"),
                        "under": p.get("under"),
                    }
                )
        (d / "yolo" / f"rgbv_{n}.txt").write_text("\n".join(lines) + "\n")
        (d / "yolo" / f"rgbv_{n}.json").write_text(json.dumps(rows))
        print(
            f"{n}: {sum(r['cls'] == 0 for r in rows)} person, {sum(r['cls'] == 1 for r in rows)} head boxes"
        )


if __name__ == "__main__":
    export(Path(sys.argv[1]))
