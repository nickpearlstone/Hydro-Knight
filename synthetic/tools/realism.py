"""Feature-level realism: do person crops from a render look like person crops from the real frame?

Crops are fixed 64x64 windows at native resolution around each person point (real: the
hand-counted clicks; render: projected head centres of visible swimmers), never per-person
rescaled. Features: DINOv2-small CLS token (Apache-2.0). Two measures, each next to a
real-vs-real baseline (real crops split into left/right halves of the frame):
  - KID: unbiased squared MMD, cubic polynomial kernel, with a bootstrap interval
  - C2ST: logistic regression on PCA-32 features, GroupKFold over 256 px spatial tiles,
    AUC (0.5 = indistinguishable; ~0.6 or less = close)
Also a water-only C2ST (crops away from people) to see whether the water alone gives it away.

    uv run --with transformers --with scikit-learn python synthetic/tools/realism.py synthetic/renders/b01 [--prefix rgbv]
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

REAL = "6f5c67826a76_720"
GT = Path("data/annotations/pose_bench_gt.json")
C = 32  # half crop


def crops(img, pts):
    H, W = img.shape[:2]
    out, keep = [], []
    for x, y in pts:
        x, y = int(round(x)), int(round(y))
        if C <= x < W - C and C <= y < H - C:
            out.append(img[y - C : y + C, x - C : x + C])
            keep.append((x, y))
    return out, np.array(keep)


def water_pts(img, people, rng, n=200):
    """Random points at least 40 px from any person (background-only crops)."""
    H, W = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    pts = []
    for _ in range(n * 50):
        x, y = rng.integers(C, W - C), rng.integers(C, H - C)
        if 80 < hsv[y, x, 0] < 105 and (
            len(people) == 0 or np.hypot(*(people - (x, y)).T).min() > 40
        ):
            pts.append((x, y))
            if len(pts) >= n:
                break
    return pts


class Dino:
    def __init__(self, device="mps"):
        import torch
        from transformers import AutoImageProcessor, AutoModel

        self.torch, self.device = torch, device
        self.proc = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
        self.model = (
            AutoModel.from_pretrained("facebook/dinov2-small").to(device).eval()
        )

    def __call__(self, ims):
        feats = []
        for k in range(0, len(ims), 64):
            batch = [
                cv2.cvtColor(
                    cv2.resize(im, (224, 224), interpolation=cv2.INTER_CUBIC),
                    cv2.COLOR_BGR2RGB,
                )
                for im in ims[k : k + 64]
            ]
            inp = self.proc(
                images=batch, return_tensors="pt", do_resize=False, do_center_crop=False
            ).to(self.device)
            with self.torch.no_grad():
                feats.append(self.model(**inp).last_hidden_state[:, 0].cpu().numpy())
        return np.concatenate(feats)


def kid(x, y):
    d = x.shape[1]
    k = lambda a, b: (a @ b.T / d + 1) ** 3  # noqa: E731
    kxx, kyy, kxy = k(x, x), k(y, y), k(x, y)
    m, n = len(x), len(y)
    return float(
        (kxx.sum() - np.trace(kxx)) / (m * (m - 1))
        + (kyy.sum() - np.trace(kyy)) / (n * (n - 1))
        - 2 * kxy.mean()
    )


def kid_ci(x, y, rng, reps=200):
    v = [
        kid(x[rng.integers(0, len(x), len(x))], y[rng.integers(0, len(y), len(y))])
        for _ in range(reps)
    ]
    return kid(x, y), float(np.percentile(v, 5)), float(np.percentile(v, 95))


def c2st(xa, xb, ga, gb):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X = np.vstack([xa, xb])
    y = np.r_[np.zeros(len(xa)), np.ones(len(xb))]
    g = np.r_[ga, gb + 10_000]  # different images never share a group
    aucs = []
    for rep in range(5):
        p = np.zeros(len(y))
        perm = np.random.default_rng(rep).permutation(np.unique(g))
        gmap = {k: i for i, k in enumerate(perm)}
        gg = np.array([gmap[k] for k in g])
        for tr, te in GroupKFold(5).split(X, y, gg):
            clf = make_pipeline(
                StandardScaler(),
                PCA(min(32, len(tr) - 1)),
                LogisticRegression(C=1.0, max_iter=2000),
            )
            clf.fit(X[tr], y[tr])
            p[te] = clf.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y, p))
    return round(float(np.mean(aucs)), 3)


def tile_groups(pts):
    return (pts[:, 1] // 256) * 100 + pts[:, 0] // 256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--prefix", default="rgbv")
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    d = Path(a.dir)
    rec = next(
        f
        for f in json.loads(GT.read_text())["frames"]
        if f"{f['clip_id']}_{f['frame']}" == REAL
    )
    rimg = cv2.imread(f"raw_local/pose_bench/{REAL}.png")
    rc, rp = crops(rimg, rec["water"])
    rw, rwp = crops(
        rimg, water_pts(rimg, np.array(rec["water"] + rec["deck"], float), rng)
    )
    dino = Dino(a.device)
    fr, frw = dino(rc), dino(rw)
    left = rp[:, 0] < np.median(rp[:, 0])
    base_kid = kid_ci(fr[left], fr[~left], rng)
    base_c2st = c2st(fr[left], fr[~left], tile_groups(rp[left]), tile_groups(rp[~left]))
    out = {"real_vs_real": {"kid": base_kid, "c2st_auc": base_c2st, "n": len(rc)}}
    for lf in sorted(d.glob("labels_*.json")):
        n = lf.stem.split("_")[1]
        img = cv2.imread(str(d / f"{a.prefix}_{n}.png"))
        H, W = img.shape[:2]
        lab = json.loads(lf.read_text())
        sp = [
            (p["u"], p["v"])
            for p in lab["people"]
            if p.get("role", "swimmer") == "swimmer"
            and p["visible_px"] >= 6
            and p.get("in_front", True)
            and 0 <= p["u"] < W
            and 0 <= p["v"] < H
        ]
        sc, spp = crops(img, sp)
        sw, swp = crops(img, water_pts(img, np.array(sp, float), rng))
        fs, fsw = dino(sc), dino(sw)
        out[f"{a.prefix}_{n}"] = {
            "n": len(sc),
            "kid": kid_ci(fr, fs, rng),
            "c2st_auc": c2st(fr, fs, tile_groups(rp), tile_groups(spp)),
            "water_c2st_auc": c2st(frw, fsw, tile_groups(rwp), tile_groups(swp)),
        }
    for k, v in out.items():
        print(k, json.dumps(v))
    (d / f"realism_{a.prefix}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
