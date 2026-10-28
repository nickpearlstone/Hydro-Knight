"""Top level of the stack: an equal-weight score and a small within-moment ranker.

Features (from slots + splash): log1p(seconds unseen), log1p(dunks in 30 s), log1p(splash),
log(box height). Each is standardised with the TRAINING folds' median and IQR only.

- Equal weight: z_unseen + z_dunks + z_splash (Nick's signs: more of each = more suspicious).
  With ~51 victims, learned weights often fail to beat this, so it is always reported.
- Ranker: a conditional logit. Each moment of each training rescue (onset to contact - 3 s,
  the last seconds hold the lifeguard's approach) is one group: the victim's rows against
  every other swimmer row at that moment. Loss = -log(sum of softmax over the victim's
  rows); groups are weighted so every clip counts equally; L2 on the weights. Weights on
  unseen, dunks and splash are kept >= 0 (sign constraints); box height is free (a control).

Scores for alarms are relative: each row's score minus the median score of its moment, since
a within-moment softmax only learns order, never the absolute level.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize

FEATURES = ["f_unseen", "f_dunks", "f_splash", "f_height"]
SIGNED = ["f_unseen", "f_dunks", "f_splash"]  # weight >= 0
L2 = 0.01
GUARD_CUT_S = 3.0


def features(rows: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "f_unseen": np.log1p(rows["unseen_s"].to_numpy()),
            "f_dunks": np.log1p(rows["dunks_30s"].to_numpy()),
            "f_splash": np.log1p(rows["splash"].to_numpy()),
            "f_height": np.log(np.maximum(rows["h"].to_numpy(), 1.0)),
        },
        index=rows.index,
    )


@dataclass
class Scaler:
    center: dict = field(default_factory=dict)
    scale: dict = field(default_factory=dict)

    @classmethod
    def fit(cls, F: pd.DataFrame) -> Scaler:
        s = cls()
        for c in F.columns:
            v = F[c].dropna().to_numpy()
            q1, med, q3 = np.percentile(v, [25, 50, 75]) if len(v) else (0, 0, 1)
            s.center[c], s.scale[c] = float(med), float(max(q3 - q1, 1e-6) / 1.349)
        return s

    def __call__(self, F: pd.DataFrame) -> pd.DataFrame:
        Z = (F - pd.Series(self.center)) / pd.Series(self.scale)
        return Z.fillna(0.0)  # a missing measurement counts as typical


def training_groups(rows: pd.DataFrame, clips: dict) -> pd.DataFrame:
    """Rows in the training window (onset to contact - 3 s) of moments that hold the victim."""
    on = rows["clip_id"].map(lambda c: clips[c]["onset"])
    cut = rows["clip_id"].map(lambda c: clips[c]["contact"]) - GUARD_CUT_S
    w = rows[(rows["t"] >= on) & (rows["t"] <= cut)]
    has = w.groupby(["clip_id", "t"])["victim"].transform("any")
    mixed = w.groupby(["clip_id", "t"])["victim"].transform(lambda v: (~v).any())
    return w[has & mixed]


def fit_ranker(
    Z: pd.DataFrame,
    groups: pd.DataFrame,
    features: list[str] = FEATURES,
    signed: list[str] = SIGNED,
) -> np.ndarray:
    """Weights for `features` (see module docstring); groups = training_groups rows, aligned with Z."""
    X = Z.loc[groups.index, features].to_numpy(np.float64)
    pos = groups["victim"].to_numpy()
    gid = groups.groupby(["clip_id", "t"]).ngroup().to_numpy()
    n_g = gid.max() + 1
    clip_of = groups.groupby(["clip_id", "t"])["clip_id"].first().to_numpy()
    per_clip = pd.Series(clip_of).value_counts()
    gw = 1.0 / pd.Series(clip_of).map(per_clip).to_numpy()
    gw /= gw.sum()

    def loss(w):
        s = X @ w
        m = np.full(n_g, -np.inf)
        np.maximum.at(m, gid, s)
        e = np.exp(s - m[gid])
        all_ = np.bincount(gid, e, n_g)
        vic = np.bincount(gid, e * pos, n_g)
        lg = -np.log(vic / all_)
        # gradient: E_all[x] - E_victim[x], per group
        pa = e / all_[gid]
        pv = np.where(pos, e / np.maximum(vic[gid], 1e-300), 0.0)
        gx = np.zeros((n_g, X.shape[1]))
        np.add.at(gx, gid, (pa - pv)[:, None] * X)
        return float(gw @ lg + L2 * w @ w), gw @ gx + 2 * L2 * w

    bounds = [(0, None) if f in signed else (None, None) for f in features]
    res = minimize(
        loss, np.zeros(len(features)), jac=True, bounds=bounds, method="L-BFGS-B"
    )
    return res.x


def relative(rows: pd.DataFrame, score: np.ndarray) -> np.ndarray:
    """Score minus the median score of the same moment in the same clip."""
    s = pd.Series(score, index=rows.index)
    return (s - s.groupby([rows["clip_id"], rows["t"]]).transform("median")).to_numpy()
