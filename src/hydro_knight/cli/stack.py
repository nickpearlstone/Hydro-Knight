"""Two-level stack MVP (spike): build per-swimmer rows, then score them with K-fold.

    uv run hydroknight stack rows                     # rows + splash for every labelled clip
    uv run hydroknight stack fit --seeds 0 1 2        # equal-weight + ranker + baselines, 5-fold
    uv run hydroknight eval protocol runs/stack_mvp --size ranker
    uv run hydroknight stack video CLIP_ID            # ghost timers drawn on a clip, to time by hand

`fit` writes a run folder in the format `hydroknight eval protocol` reads (summary.json +
alarms_<name>_s<seed>.parquet), one <name> per scorer, so every scorer gets the same
protocol report, baselines and bootstrap.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import pandas as pd

RUN = Path("runs/stack_mvp")


def cmd_rows(args) -> None:
    from hydro_knight.stack.data import VIDEOS, iter_clips
    from hydro_knight.stack.slots import SlotParams, build_rows
    from hydro_knight.stack.splash import add_splash

    out = Path(args.run)
    out.mkdir(parents=True, exist_ok=True)
    parts, meta = [], {}
    t0 = time.time()
    for c in iter_clips():
        cap = cv2.VideoCapture(str(VIDEOS / f"{c.clip_id}.mp4"))
        size = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        cap.release()
        rows = add_splash(build_rows(c, size), c)
        parts.append(rows)
        meta[c.clip_id] = {
            "start": c.start,
            "onset": c.onset,
            "contact": c.contact,
            "fps": c.fps,
            "victim_tracks": sorted(c.victim_tracks),
            "path_fragments": sorted(c.victim_path_tracks),
        }
        print(f"{c.clip_id}: {len(rows)} rows, {time.time() - t0:.0f} s", flush=True)
    pd.concat(parts, ignore_index=True).to_parquet(out / "rows.parquet")
    (out / "clips.json").write_text(
        json.dumps({"params": SlotParams().__dict__, "clips": meta}, indent=1)
    )
    print(f"wrote {out}/rows.parquet ({len(meta)} clips)")


def cmd_fit(args) -> None:
    import numpy as np

    from hydro_knight.eval.split import assign_folds
    from hydro_knight.stack import rank

    run = Path(args.run)
    rows = pd.read_parquet(run / "rows.parquet")
    meta = json.loads((run / "clips.json").read_text())["clips"]
    F = rank.features(rows)
    clip_ids = sorted(meta)
    raw = {  # trivial baselines and single features: no training
        "unseen": F["f_unseen"].to_numpy(),
        "splash": F["f_splash"].fillna(F["f_splash"].median()).to_numpy(),
        "neg_height": -rows["h"].to_numpy(),
        "lowconf": 1.0 - rows["box_conf"].fillna(0.0).to_numpy(),
    }
    reports = {}
    for seed in args.seeds:
        folds = assign_folds(clip_ids, args.folds, seed)
        scores = {
            name: np.full(len(rows), np.nan) for name in ["equal", "ranker", *raw]
        }
        weights = {}
        for f in range(args.folds):
            test = rows["clip_id"].map(folds).to_numpy() == f
            scaler = rank.Scaler.fit(F[~test])
            Z = scaler(F)
            w = rank.fit_ranker(Z[~test], rank.training_groups(rows[~test], meta))
            weights[f] = dict(zip(rank.FEATURES, np.round(w, 4).tolist(), strict=True))
            scores["ranker"][test] = Z[test][rank.FEATURES].to_numpy() @ w
            scores["equal"][test] = Z[test][rank.SIGNED].sum(1).to_numpy()
        for name, s in raw.items():
            scores[name] = s
        for name, s in scores.items():
            table = rows[["clip_id", "track_id", "t"]].assign(
                score=rank.relative(rows, s)
            )
            table.to_parquet(run / f"alarms_{name}_s{seed}.parquet")
        reports[str(seed)] = {
            "used_test": clip_ids,
            "victim_tracks": {c: meta[c]["victim_tracks"] for c in clip_ids},
            "path_fragments": {c: meta[c]["path_fragments"] for c in clip_ids},
            "folds": folds,
            "ranker_weights": weights,
        }
        print(f"seed {seed}: ranker weights per fold {weights}", flush=True)
    (run / "summary.json").write_text(
        json.dumps({"reports": reports, "folds": args.folds}, indent=1)
    )
    print(
        f"wrote {run}: score with  uv run hydroknight eval protocol {run} --size ranker"
    )


def cmd_video(args) -> None:
    from hydro_knight.stack.video import render

    run = Path(args.run)
    rows = pd.read_parquet(run / "rows.parquet")
    meta = json.loads((run / "clips.json").read_text())["clips"]
    for cid in args.clips:
        out = render(rows, meta[cid], cid, run / "videos" / f"{cid}_ghosts.mp4")
        print(f"{cid}: {out or 'not written (ffmpeg or video missing)'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rows", help="per-swimmer rows + splash for every labelled clip")
    r.add_argument("--run", default=str(RUN))
    f = sub.add_parser("fit", help="equal-weight + ranker + baselines, grouped K-fold")
    f.add_argument("--run", default=str(RUN))
    f.add_argument("--folds", type=int, default=5)
    f.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    v = sub.add_parser("video", help="ghost timers drawn on clips, to time by hand")
    v.add_argument("clips", nargs="+")
    v.add_argument("--run", default=str(RUN))
    args = ap.parse_args()
    {"rows": cmd_rows, "fit": cmd_fit, "video": cmd_video}[args.cmd](args)


if __name__ == "__main__":
    main()
