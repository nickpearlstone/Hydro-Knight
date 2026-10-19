"""Detector-level realism check: score a rendered frame and a real frame with identical rules.

Both detectors run with the production tiling (480 px tiles + whole frame, IoS merge):
YOLO11n-pose (current backend) and RT-DETRv2-R18 (likely head-detector base). For each
frame we report, at conf 0.05 and 0.25: recall by size band, false boxes, and the median
confidence of matched people. Size = body height at that image row from a perspective fit
of the detector's own >= 0.25 boxes (YOLO), scaled to 1080p, so real and synthetic frames
are measured the same way. Matching = benchmark_pose's greedy one-box-per-point rule.

    uv run --with transformers python synthetic/tools/score.py real 6f5c67826a76_720
    uv run --with transformers python synthetic/tools/score.py synth synthetic/renders/b01 [--prefix rgbv]
    uv run python synthetic/tools/score.py compare synthetic/renders/b01

Results are cached as <name>.dets.npz / scores.json beside the frame.
"""

import argparse
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np

try:  # main: the benchmark lives in the package (hydroknight command)
    from hydro_knight.cli import benchmark_pose as bp
except ImportError:  # older branches: scripts/benchmark_pose.py
    _spec = importlib.util.spec_from_file_location("bp", "scripts/benchmark_pose.py")
    bp = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(bp)

BINS = [0, 30, 45, 60, 80, 120, 1e9]
LABELS = ["<30", "30-45", "45-60", "60-80", "80-120", "120+"]
CUTS = (0.05, 0.25)
REAL_DIR = Path("synthetic/renders/real")
MIN_VISIBLE_PX = 6


class HFBackend:
    """A Hugging Face transformers detector (RT-DETRv2, Apache-2.0), person class only, conf >= 0.05."""

    def __init__(self, name, device="mps", batch=4):
        import torch
        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        self.torch, self.device, self.batch = torch, device, batch
        self.proc = AutoImageProcessor.from_pretrained(name)
        self.model = AutoModelForObjectDetection.from_pretrained(name).to(device).eval()
        self.person = [
            int(i) for i, lab in self.model.config.id2label.items() if lab == "person"
        ][0]

    def __call__(self, images):
        out = []
        for k in range(0, len(images), self.batch):
            chunk = [
                cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in images[k : k + self.batch]
            ]
            inputs = self.proc(images=chunk, return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                o = self.model(**inputs)
            res = self.proc.post_process_object_detection(
                o, threshold=0.05, target_sizes=[im.shape[:2] for im in chunk]
            )
            for r in res:
                keep = (r["labels"] == self.person).cpu().numpy()
                out.append(
                    (
                        r["boxes"].cpu().numpy()[keep].astype(np.float32),
                        r["scores"].cpu().numpy()[keep],
                    )
                )
        return out


def detectors(device):
    bp.IMGSZ = 1280
    yolo = bp.YoloBackend("yolo11n-pose", device, 4, conf=0.05)
    rtdetr = HFBackend("PekingU/rtdetr_v2_r18vd", device)
    return {"yolo11n": yolo, "rtdetrv2": rtdetr}


def run_dets(img, device, cache: Path):
    if cache.exists():
        z = np.load(cache)
        return {k: (z[f"{k}_b"], z[f"{k}_c"]) for k in ("yolo11n", "rtdetrv2")}
    out = {}
    for name, be in detectors(device).items():
        b, c = bp.detect(be, img, tiled=True)
        out[name] = (np.asarray(b, float).reshape(-1, 4), np.asarray(c, float))
    np.savez(
        cache,
        **{f"{k}_b": v[0] for k, v in out.items()},
        **{f"{k}_c": v[1] for k, v in out.items()},
    )
    return out


def size_fit(boxes, H):
    h = boxes[:, 3] - boxes[:, 1]
    y = (boxes[:, 1] + boxes[:, 3]) / 2
    keep = h <= np.quantile(h, 0.9)
    a, b = np.polyfit(y[keep], np.log(h[keep]), 1)
    return lambda yy: np.exp(a * np.asarray(yy, float) + b) * 1080.0 / H


def match(boxes, confs, pts):
    """Greedy one-box-per-point (benchmark rule). Returns (taken bool per point, conf per point, n false)."""
    inside = bp._inside(boxes, pts)
    taken = np.zeros(len(pts), bool)
    pconf = np.zeros(len(pts))
    fp = 0
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
        pconf[j] = confs[i]
    return taken, pconf, fp


def score(img, water, deck, dets):
    H = img.shape[0]
    pts = np.asarray(list(water) + list(deck), float).reshape(-1, 2)
    nw = len(water)
    yb, yc = dets["yolo11n"]
    fit = size_fit(yb[yc >= 0.25], H)
    size = fit(pts[:nw, 1])
    band = np.digitize(size, BINS[1:-1])
    res = {
        "n_water": nw,
        "band_n": [int((band == k).sum()) for k in range(len(LABELS))],
    }
    for name, (b, c) in dets.items():
        for cut in CUTS:
            k = c >= cut
            taken, pconf, fp = match(b[k], c[k], pts)
            tw = taken[:nw]
            res[f"{name}@{cut}"] = {
                "recall": round(float(tw.mean()), 3),
                "false_boxes": int(fp),
                "recall_by_band": [
                    round(float(tw[band == j].mean()), 2) if (band == j).any() else None
                    for j in range(len(LABELS))
                ],
                "median_conf_found": round(float(np.median(pconf[:nw][tw])), 3)
                if tw.any()
                else None,
            }
    return res


def cmd_real(a):
    gt = json.loads(bp.GT_PATH.read_text())["frames"]
    cid, fr = a.name.rsplit("_", 1)
    rec = next(f for f in gt if f["clip_id"] == cid and f["frame"] == int(fr))
    img = cv2.imread(f"raw_local/pose_bench/{a.name}.png")
    REAL_DIR.mkdir(parents=True, exist_ok=True)
    dets = run_dets(img, a.device, REAL_DIR / f"{a.name}.dets.npz")
    res = score(img, rec["water"], rec["deck"], dets)
    (REAL_DIR / f"{a.name}.scores.json").write_text(json.dumps(res, indent=1))
    show(a.name, res)


def cmd_synth(a):
    d = Path(a.dir)
    allres = {}
    for lf in sorted(d.glob("labels_*.json")):
        n = lf.stem.split("_")[1]
        lab = json.loads(lf.read_text())
        img = cv2.imread(str(d / f"{a.prefix}_{n}.png"))
        H, W = img.shape[:2]
        ok = [
            p
            for p in lab["people"]
            if p["visible_px"] >= MIN_VISIBLE_PX
            and p.get("in_front", True)
            and 0 <= p["u"] < W
            and 0 <= p["v"] < H
        ]
        water = [(p["u"], p["v"]) for p in ok if p.get("role", "swimmer") == "swimmer"]
        deck = [(p["u"], p["v"]) for p in ok if p.get("role", "swimmer") != "swimmer"]
        dets = run_dets(img, a.device, d / f"{a.prefix}_{n}.dets.npz")
        res = score(img, water, deck, dets)
        allres[f"{a.prefix}_{n}"] = res
        show(f"{a.prefix}_{n}", res)
    (d / f"scores_{a.prefix}.json").write_text(json.dumps(allres, indent=1))


def show(name, res):
    print(
        f"\n== {name}: {res['n_water']} in water; per band n = {dict(zip(LABELS, res['band_n'], strict=False))}"
    )
    for k, v in res.items():
        if "@" in k:
            bands = " ".join(
                f"{lab}:{'-' if r is None else f'{r:.2f}'}"
                for lab, r in zip(LABELS, v["recall_by_band"], strict=False)
            )
            print(
                f"  {k:16s} recall {v['recall']:.2f}  false {v['false_boxes']:3d}  medconf {v['median_conf_found']}  | {bands}"
            )


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("real")
    r.add_argument("name")
    s = sub.add_parser("synth")
    s.add_argument("dir")
    s.add_argument("--prefix", default="rgbv")
    for p in (r, s):
        p.add_argument("--device", default="mps")
    a = ap.parse_args()
    {"real": cmd_real, "synth": cmd_synth}[a.cmd](a)


if __name__ == "__main__":
    main()
