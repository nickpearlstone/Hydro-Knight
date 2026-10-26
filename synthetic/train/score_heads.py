"""Step 3: score head detectors on the real hand-counted frames (runs on the Mac; real frames stay local).

Each model runs on the 10 swimmer-count frames with the same tiling it was trained for: 640 px tiles
at native resolution with 25% overlap, boxes merged across tiles (IoU-or-IoS, as in extraction).
Ground truth (count.head_truth): on frames whose head boxes Nick reviewed, the centre of each
in-water head box (out-of-water heads and "no head visible" marks are allowed but not counted);
elsewhere Nick's clicks (one per swimmer; deck and hidden clicks allowed but not counted). The
Jupiter Reef frames were removed on 2026-10-26 (too far, wrong angle), leaving 8 frames. A click is found when a box covers it with the "fair" head rule from the 10-16 bake-off:
padding of half the box height (min 6 px), since a head box can sit just above a click on the body.
One box per click (greedy, most confident first); boxes covering no click are false boxes.

--in-range (the 10-16 range rule): only swimmers inside the clip's hand-drawn zone whose expected
body height there is >= 60 px at 1080p count. The expected height comes from a perspective fit
(log height vs image row) of YOLO11n-pose boxes >= 0.25 on that frame. Clicks out of range, and
boxes centred outside the zone, are ignored: neither found nor false.

Models are compared at MATCHED false boxes over all 10 frames (20 / 50 / 100), so a detector that
fires everywhere cannot win, plus the raw numbers at conf 0.25.

    uv run --with transformers python synthetic/train/score_heads.py \\
        rtdetr=synthetic/renders/train/rtdetrv2_head/best yolo=runs/yolo11n_head/weights/best.pt \\
        ref11n=scratch/heads/11n-head.pt
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

from hydro_knight.annotate import count, zones
from hydro_knight.preprocess.tiled_pose import merge_detections, tile_grid

TILE, OVERLAP, FLOOR = 640, 0.25, 0.01
BUDGETS = (20, 50, 100)


class Yolo:
    def __init__(self, path, device="mps"):
        from ultralytics import YOLO

        self.m, self.device = YOLO(path), device

    def __call__(self, crops):
        out = []
        for r in self.m(
            crops, imgsz=TILE, conf=FLOOR, verbose=False, device=self.device
        ):
            out.append((r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()))
        return out


class HF:
    def __init__(self, path, device="mps"):
        import torch
        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        self.t, self.device = torch, device
        self.proc = AutoImageProcessor.from_pretrained(path)
        self.m = AutoModelForObjectDetection.from_pretrained(path).to(device).eval()

    def __call__(self, crops):
        out = []
        for k in range(0, len(crops), 4):
            chunk = [cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in crops[k : k + 4]]
            inp = self.proc(images=chunk, return_tensors="pt").to(self.device)
            with self.t.no_grad():
                o = self.m(**inp)
            for r in self.proc.post_process_object_detection(
                o, threshold=FLOOR, target_sizes=[c.shape[:2] for c in chunk]
            ):
                out.append((r["boxes"].cpu().numpy(), r["scores"].cpu().numpy()))
        return out


def detect(model, frame):
    h, w = frame.shape[:2]
    origins = tile_grid(w, h, TILE, OVERLAP)
    res = model([frame[oy : oy + TILE, ox : ox + TILE] for ox, oy in origins])
    boxes, confs, crops = [], [], []
    for k, ((b, c), (ox, oy)) in enumerate(zip(res, origins, strict=True)):
        b = np.asarray(b, np.float32).reshape(-1, 4).copy()
        b[:, [0, 2]] += ox
        b[:, [1, 3]] += oy
        boxes.append(b)
        confs.append(np.asarray(c, np.float32))
        crops.append(np.full(len(c), k, np.int8))
    boxes, confs, crops = (
        np.concatenate(boxes),
        np.concatenate(confs),
        np.concatenate(crops),
    )
    keep = merge_detections(boxes, confs, crops)
    return boxes[keep], confs[keep]


def match(boxes, confs, water, deck):
    """(found water clicks, false boxes) with the fair head rule, one box per click."""
    pts = np.asarray(list(water) + list(deck), np.float32).reshape(-1, 2)
    nw = len(water)
    if len(boxes) == 0:
        return 0, 0
    h = boxes[:, 3] - boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    px, py = (
        np.maximum(0.1 * w, np.maximum(0.5 * h, 6)),
        np.maximum(0.1 * h, np.maximum(0.5 * h, 6)),
    )
    x, y = pts[None, :, 0], pts[None, :, 1]
    inside = (
        (x >= (boxes[:, 0] - px)[:, None])
        & (x <= (boxes[:, 2] + px)[:, None])
        & (y >= (boxes[:, 1] - py)[:, None])
        & (y <= (boxes[:, 3] + py)[:, None])
    )
    taken = np.zeros(len(pts), bool)
    found = fp = 0
    for i in np.argsort(-confs, kind="stable"):
        cand = np.flatnonzero(inside[i])
        if len(cand) == 0:
            fp += 1
            continue
        free = cand[~taken[cand]]
        if len(free) == 0:
            continue
        cx, cy = (boxes[i, 0] + boxes[i, 2]) / 2, (boxes[i, 1] + boxes[i, 3]) / 2
        j = free[np.argmin(np.hypot(pts[free, 0] - cx, pts[free, 1] - cy))]
        taken[j] = True
        found += j < nw
    return found, fp


MIN_PX = 60.0


def in_range_split(frame, img, zone):
    """Split water clicks into in-range and out-of-range (zone + expected height >= MIN_PX)."""
    import importlib.util

    try:
        from hydro_knight.cli import benchmark_pose as bp
    except ImportError:
        spec = importlib.util.spec_from_file_location("bp", "scripts/benchmark_pose.py")
        bp = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bp)
    bp.IMGSZ = 1280
    if not hasattr(in_range_split, "pose"):
        in_range_split.pose = bp.YoloBackend("yolo11n-pose", "mps", 4, conf=0.25)
    b, _ = bp.detect(in_range_split.pose, img, tiled=True)
    b = np.asarray(b, float).reshape(-1, 4)
    h, yc = b[:, 3] - b[:, 1], (b[:, 1] + b[:, 3]) / 2
    keep = h <= np.quantile(h, 0.9)
    a, c = np.polyfit(yc[keep], np.log(h[keep]), 1)
    pts = np.asarray(frame["water"], float).reshape(-1, 2)
    ok = zones.in_zone(zone, pts[:, 0], pts[:, 1]) & (
        np.exp(a * pts[:, 1] + c) * 1080.0 / img.shape[0] >= MIN_PX
    )
    return [tuple(p) for p in pts[ok]], [tuple(p) for p in pts[~ok]]


def main():
    in_range = "--in-range" in sys.argv
    sys.argv = [x for x in sys.argv if x != "--in-range"]
    gt = json.loads(count.COUNT_PATH.read_text())["frames"]
    frames = [f for f in gt if f["water"]]
    for f in frames:  # reviewed head boxes replace the dots (count.head_truth)
        f["water"], f["deck"] = count.head_truth(f)
        f["hidden"] = []
    imgs = {
        f"{f['clip_id']}_{f['frame']}": cv2.imread(
            f"raw_local/pose_bench/{f['clip_id']}_{f['frame']}.png"
        )
        for f in frames
    }
    if in_range:  # out-of-range clicks become "allowed" points like deck clicks
        zs = zones.load()
        for f in frames:
            inr, out = in_range_split(
                f, imgs[f"{f['clip_id']}_{f['frame']}"], zs.get(f["clip_id"])
            )
            f["water"], f["deck"], f["hidden"] = inr, count.ignored(f) + out, []
            f["zone"] = zs.get(f["clip_id"])
    total = sum(len(f["water"]) for f in frames)
    results = {}
    for arg in sys.argv[1:]:
        name, path = arg.split("=", 1)
        model = HF(path) if Path(path).is_dir() else Yolo(path)
        dets = {k: detect(model, im) for k, im in imgs.items()}
        rows = []
        allc = np.concatenate([c for _, c in dets.values()])
        # 200 thresholds spread over the confidence distribution, plus 0.25
        cuts = set(np.quantile(allc, np.linspace(0, 1, 200)).tolist()) | {0.25}
        for cut in sorted(cuts, reverse=True):
            f_tot = fp_tot = 0
            for fr in frames:
                b, c = dets[f"{fr['clip_id']}_{fr['frame']}"]
                k = c >= cut
                if in_range and fr.get(
                    "zone"
                ):  # boxes centred outside the zone are ignored
                    k &= zones.in_zone(
                        fr["zone"], (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2
                    )
                f, p = match(b[k], c[k], fr["water"], count.ignored(fr))
                f_tot, fp_tot = f_tot + f, fp_tot + p
            rows.append((cut, f_tot / total, fp_tot))
        res = {"total_clicks": total}
        for budget in BUDGETS:
            ok = [r for r in rows if r[2] <= budget]
            res[f"recall_at_{budget}_false"] = (
                round(max(r[1] for r in ok), 3) if ok else 0.0
            )
        at25 = min(rows, key=lambda r: abs(r[0] - 0.25))
        res["at_conf_0.25"] = {"recall": round(at25[1], 3), "false_boxes": at25[2]}
        results[name] = res
        print(name, json.dumps(res), flush=True)
    Path("synthetic/renders/train").mkdir(parents=True, exist_ok=True)
    Path(
        f"synthetic/renders/train/head_scores{'_in_range' if in_range else ''}.json"
    ).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
