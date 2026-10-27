"""Exploratory grid table (not preregistered): every (G, R) against today and both random baselines.

uv run python research/pilot_tracking/grid_table.py OUT_DIR [victims_v2.csv]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

out = Path(sys.argv[1])
df = pd.read_csv(out / (sys.argv[2] if len(sys.argv) > 2 else "victims_v2.csv"))
gaps = pd.read_csv(out / "gaps_summary.csv").set_index(["max_gap_s", "max_dist_h"])
today = df[df.method == "today"].set_index("clip")
rng = np.random.default_rng(0)


def ci(d):
    m = d[rng.integers(0, len(d), (2000, len(d)))].mean(1)
    return (
        f"{d.mean():+.3f} [{np.percentile(m, 2.5):+.3f}, {np.percentile(m, 97.5):+.3f}]"
    )


lines = [
    "| G s | R | wrong joins (gap check) | joins/clip | continuity gain | clips better | "
    "minus gated random | minus any-distance random | moved-path gain | detected gain |",
    "|---|---|---|---|---|---|---|---|---|---|",
]
for (g, r), s in df[df.method != "today"].groupby(["max_gap_s", "max_dist_h"]):
    p = {
        m: s[s.method == m].groupby("clip").mean(numeric_only=True)
        for m in s.method.unique()
    }

    def gain(m, col="continuity"):
        return (p[m][col] - today[col]).to_numpy(float)

    d = gain("relink")
    rel = p["relink"].continuity
    gated = p.get("gated_random_relink")
    lines.append(
        f"| {g} | {r} | {gaps.loc[(g, r), 'wrong']:.1%} | {p['relink'].n_links.mean():.0f} | {ci(d)} | "
        f"{(d > 1e-9).sum()} of {len(d)} | "
        f"{ci((rel - gated.continuity).to_numpy(float)) if gated is not None else '-'} | "
        f"{ci((rel - p['random_relink'].continuity).to_numpy(float))} | "
        f"{gain('relink', 'moved_continuity').mean():+.3f} | {gain('relink', 'detected').mean():+.3f} |"
    )
print("\n".join(lines))
