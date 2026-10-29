"""Pre-label head boxes on the head validation frames with an ensemble of trained detectors.

Each model runs with score_heads.py's tiling. A box is kept when at least --votes models put
a box on the same head (IoU >= --iou with the most confident box of the group) at confidence
>= --conf; the kept box is the mean of the group's boxes. Pre-labels go into the frame's
head boxes (in the water) with "heads_reviewed": false and "heads_source" naming the models,
so the labeler draws them dashed until a person has checked every box. Frames that already
have head boxes are skipped unless --overwrite, so a person's work is never replaced.

Only the validation frames get pre-labels: the benchmark (test) frames are labelled from
scratch, so the test labels don't lean toward any model's boxes.

    uv run --with transformers python synthetic/train/prelabel_heads.py \\
        synthetic/renders/train_out3/train/rtdetr_v3_seed*/best --votes 4 --conf 0.05
"""

import argparse
from pathlib import Path

import numpy as np
from score_heads import HF, Yolo, detect

from hydro_knight.annotate import count


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU of every box in a (N, 4) against every box in b (M, 4), xyxy."""
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def vote(
    dets: list[tuple[np.ndarray, np.ndarray]], conf: float, votes: int, iou: float
):
    """Boxes that at least `votes` of the models agree on, as their mean (K, 4).

    Greedy: the most confident unused box starts a group; each other model adds its
    best-overlapping unused box if the IoU reaches `iou`.
    """
    boxes, scores, model = [], [], []
    for m, (b, c) in enumerate(dets):
        keep = c >= conf
        boxes.append(b[keep])
        scores.append(c[keep])
        model.append(np.full(keep.sum(), m))
    if not sum(len(s) for s in scores):
        return np.zeros((0, 4), np.float32)
    boxes, scores, model = map(np.concatenate, (boxes, scores, model))
    ious = iou_matrix(boxes, boxes)
    used = np.zeros(len(boxes), bool)
    out = []
    for i in np.argsort(-scores):
        if used[i]:
            continue
        group = [i]
        for m in set(model.tolist()) - {model[i]}:
            cand = np.flatnonzero((model == m) & ~used)
            if len(cand):
                j = cand[np.argmax(ious[i, cand])]
                if ious[i, j] >= iou:
                    group.append(j)
        if len(group) >= votes:
            used[group] = True
            out.append(boxes[group].mean(0))
        else:
            used[i] = True
    return np.asarray(out, np.float32).reshape(-1, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+", help="HF checkpoint folders or YOLO .pt files")
    ap.add_argument("--votes", type=int, default=4)
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--iou", type=float, default=0.4)
    ap.add_argument("--labels", default=str(count.VAL_PATH))
    ap.add_argument(
        "--limit", type=int, default=0, help="first N frames only (0 = all)"
    )
    ap.add_argument(
        "--source", default=None, help="heads_source text (default: from args)"
    )
    ap.add_argument(
        "--overwrite", action="store_true", help="replace existing head boxes"
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="print counts, write nothing"
    )
    a = ap.parse_args()
    path = Path(a.labels)
    data = count.load(path)
    todo = [
        i
        for i, f in enumerate(data["frames"])
        if a.overwrite or not f.get("heads", {}).get("water")
    ]
    if a.limit:
        todo = todo[: a.limit]
    images = {i: count.read_frame(data["frames"][i]) for i in todo}
    dets = {i: [] for i in todo}
    for p in a.models:  # one model in memory at a time
        model = HF(p) if Path(p).is_dir() else Yolo(p)
        for i in todo:
            dets[i].append(detect(model, images[i]))
        del model
        print("ran", p, flush=True)
    source = a.source or f"{len(a.models)} models, {a.votes}+ votes, conf {a.conf}"
    for i in todo:
        kept = vote(dets[i], a.conf, a.votes, a.iou)
        f = data["frames"][i]
        print(f"{f['clip_id']} frame {f['frame']}: {len(kept)} pre-labelled heads")
        if a.dry_run:
            continue
        boxes = [[round(float(v), 1) for v in b] for b in kept]
        f["heads"] = {"water": boxes, "deck": [], "no_head": []}
        f["heads_reviewed"] = False
        f["heads_source"] = source
    if not a.dry_run:
        count.save(data, path)
        print(f"wrote pre-labels for {len(todo)} frames -> {path}")


if __name__ == "__main__":
    main()
