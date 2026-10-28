"""Appearance: how a head-and-shoulders crop looks, scored by a frozen image model + a linear probe.

Nick's idea (2026-10-28): fuse a single-frame appearance signal into v3. Each row's head box gives
a square crop CROP_SIDE head heights wide, centred CROP_DROP head heights below the head (shoulders
and the waterline). A frozen DINOv2-small turns the crop into a 768-number embedding (CLS token +
mean of the patch tokens); nothing in it is trained.

The probe (standardise -> PCA 64 -> logistic, balanced classes) learns victim vs every other row on
the training folds' labelled moments only. For the ranker it is stacked: training rows get scores
from probes that never saw their fold (inner K-fold), test rows from a probe trained on all
training folds. Without that, the ranker would learn from scores the probe had memorised.

Caching (`every`): the Jetson can't afford a crop per head every 0.5 s, and appearance changes
slowly, so each tracker track is embedded once per `every` seconds and the score is carried forward
within that bucket (causal). On the labelled moments, a 2 s cache kept the full gain at ~1/4 of the
crops; scoring only the motion ranker's top k lost the victims motion misses.

Feasibility (448 labelled moments, 41 clips, scratch run): appearance alone same-moment AUC 0.90,
v3 ranker 0.79, ranker + appearance 0.92. The probe mostly separates "head low, body under, no tube"
from people sitting on tubes.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

CROP_SIDE = 4.0  # head heights
CROP_DROP = 0.75  # head heights below the head centre
CROP_PX = 224
MODEL = "facebook/dinov2-small"  # Apache-2.0
PCA_DIM = 64
C = 0.1


def crop(img: np.ndarray, x: float, y: float, hh: float) -> np.ndarray:
    """CROP_PX square around (x, y + CROP_DROP * hh), CROP_SIDE * hh wide; edges replicated."""
    s = max(int(round(CROP_SIDE * hh)), 2)
    x0 = int(round(x - s / 2))
    y0 = int(round(y + CROP_DROP * hh - s / 2))
    big = cv2.copyMakeBorder(img, s, s, s, s, cv2.BORDER_REPLICATE)
    c = big[y0 + s : y0 + 2 * s, x0 + s : x0 + 2 * s]
    return cv2.resize(c, (CROP_PX, CROP_PX), interpolation=cv2.INTER_CUBIC)


def cache_rows(rows: pd.DataFrame, every: float) -> np.ndarray:
    """Mask of the rows that get a fresh crop: the first row of each (clip, track, `every` s) bucket."""
    if every <= 0:
        return np.ones(len(rows), bool)
    key = [rows.clip_id, rows.src_track, (rows.t // every).astype(int)]
    first = rows.t.groupby(key).transform("min")
    return (rows.t == first).to_numpy()


def carry_forward(rows: pd.DataFrame, score: np.ndarray, every: float) -> np.ndarray:
    """Fill each bucket's rows with the score of its first (embedded) row."""
    if every <= 0:
        return score
    key = [rows.clip_id, rows.src_track, (rows.t // every).astype(int)]
    s = pd.Series(score, index=rows.index)
    return s.groupby(key).transform("first").to_numpy()


class Embedder:
    """Frozen DINOv2 (needs the `heads` extra: transformers)."""

    def __init__(self, model: str = MODEL, device: str | None = None):
        import torch
        from transformers import AutoImageProcessor, AutoModel

        self.torch = torch
        self.dev = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        proc = AutoImageProcessor.from_pretrained(model)
        self.mean = np.array(proc.image_mean, np.float32)
        self.std = np.array(proc.image_std, np.float32)
        self.m = AutoModel.from_pretrained(model).to(self.dev).eval()

    def __call__(self, crops_bgr: list[np.ndarray], batch: int = 64) -> np.ndarray:
        out = []
        with self.torch.no_grad():
            for k in range(0, len(crops_bgr), batch):
                a = np.stack(crops_bgr[k : k + batch])[..., ::-1].astype(np.float32)
                a = (a / 255 - self.mean) / self.std
                x = self.torch.from_numpy(a.transpose(0, 3, 1, 2).copy()).to(self.dev)
                h = self.m(pixel_values=x).last_hidden_state
                out.append(
                    self.torch.cat([h[:, 0], h[:, 1:].mean(1)], 1).float().cpu().numpy()
                )
        return np.concatenate(out) if out else np.zeros((0, 0), np.float32)


def embed_rows(
    rows: pd.DataFrame, fps: dict[str, float], videos: Path, embedder: Embedder
) -> np.ndarray:
    """One embedding per row (rows.t = file frame / fps); decodes each clip once, in order."""
    out = np.full(len(rows), None, object)
    t0 = time.time()
    for cid, rc in rows.groupby("clip_id"):
        f_of = np.round(rc.t.to_numpy() * fps[cid]).astype(int)
        want: dict[int, list[int]] = {}
        for pos, f in zip(range(len(rc)), f_of, strict=True):
            want.setdefault(int(f), []).append(pos)
        cap = cv2.VideoCapture(str(videos / f"{cid}.mp4"))
        crops, where = [], []
        f, last = 0, max(want)
        while f <= last and cap.grab():
            if f in want:
                _, img = cap.retrieve()
                for pos in want[f]:
                    r = rc.iloc[pos]
                    crops.append(crop(img, r.x, r.y, r.hh))
                    where.append(pos)
            f += 1
        cap.release()
        if len(where) != len(rc):
            raise RuntimeError(f"{cid}: decoded {len(where)} of {len(rc)} rows' frames")
        e = embedder(crops)
        idx = np.flatnonzero(rows.index.isin(rc.index))  # rc keeps rows' order
        for pos, v in zip(where, e, strict=True):
            out[idx[pos]] = v
        print(f"{cid}: {len(rc)} crops, {time.time() - t0:.0f} s", flush=True)
    return np.stack(out).astype(np.float32)


def _probe(X: np.ndarray, y: np.ndarray):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(
        StandardScaler(),
        PCA(PCA_DIM, random_state=0),
        LogisticRegression(C=C, class_weight="balanced", max_iter=2000),
    ).fit(X, y)


def stacked_feature(
    X: np.ndarray,
    victim: np.ndarray,
    labelled: np.ndarray,
    fold: np.ndarray,
    test_fold,
) -> np.ndarray:
    """Appearance score for every row with an embedding (NaN elsewhere), for one outer fold.

    X: (N, D) embeddings, NaN rows = not embedded. labelled: rows the probe may learn from
    (training-window moments). Test rows: probe on all other folds. Training rows: probe on the
    training folds minus their own (inner K-fold), so the ranker sees honest scores.
    """
    have = ~np.isnan(X).any(1)
    out = np.full(len(X), np.nan)
    learn = labelled & have
    test = fold == test_fold
    out[test & have] = _probe(
        X[learn & ~test], victim[learn & ~test]
    ).decision_function(X[test & have])
    for f in np.unique(fold[~test]):
        inner = fold == f
        tr = learn & ~test & ~inner
        out[inner & have] = _probe(X[tr], victim[tr]).decision_function(X[inner & have])
    return out
