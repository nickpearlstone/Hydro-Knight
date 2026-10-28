"""Two-level stack MVP (spike): build per-swimmer rows, then score them with K-fold.

    uv run hydroknight stack rows                     # rows + splash for every labelled clip
    uv run hydroknight stack fit --seeds 0 1 2        # equal-weight + ranker + baselines, 5-fold
    uv run hydroknight eval protocol runs/stack_mvp --size ranker
    uv run hydroknight stack video CLIP_ID            # ghost timers drawn on a clip, to time by hand

Head tracks (RT-DETR head boxes in data/detections_heads/raw/, from Kaggle hk-heads-detect):

    uv run hydroknight stack heads-track              # merge + track + weak flag -> tracks_kept/
    uv run hydroknight stack heads-run                # churn + size flicker -> runs/stack_heads_{rank,raw}
    uv run hydroknight stack heads-rows               # v3 rows: head features + splash -> runs/v3_ranker
    uv run hydroknight stack heads-embed              # DINOv2 crop embeddings, cached per track (heads extra)
    uv run hydroknight stack heads-fit --appear       # v3 ranker + appearance feature (size ranker_appear)
    uv run hydroknight stack heads-detect             # whole-frame head boxes on this machine (no tiles)
    uv run hydroknight stack heads-fit                # v3: within-moment ranker, v2's folds and seeds

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


def cmd_heads_track(args) -> None:
    from hydro_knight.stack import heads

    kept, ev = heads.HEADS / "tracks_kept", heads.HEADS / "tracks_eval"
    kept.mkdir(exist_ok=True)
    ev.mkdir(exist_ok=True)
    for p in sorted((heads.HEADS / "raw").glob("*.npz")):
        if (kept / f"{p.stem}.parquet").exists() and not args.force:
            continue
        t0 = time.time()
        df = heads.track_clip(p.stem)
        df.to_parquet(kept / f"{p.stem}.parquet", index=False)
        heads.write_eval_tracks(df, p.stem, ev)
        print(
            f"{p.stem}: {df.track_id.nunique()} tracks, {time.time() - t0:.0f} s",
            flush=True,
        )


def cmd_heads_run(args) -> None:
    from hydro_knight.annotate import zones
    from hydro_knight.ingest.manifest import Manifest
    from hydro_knight.stack import heads

    base = json.loads((Path(args.base) / "summary.json").read_text())["reports"]
    clips = base[str(args.seeds[0])]["used_test"]
    recs = {
        r.clip_id: r for r in Manifest(Path("data/manifests/pool_footage.jsonl")).load()
    }
    zs = zones.load()
    parts, vt, pf = [], {}, {}
    for cid in clips:
        rows, tracks = heads.clip_rows(recs[cid], zs.get(cid), zones.in_zone)
        heads.write_eval_tracks(tracks, cid, Path(f"{args.run}_tracks"))
        parts.append(rows)
        vt[cid], pf[cid] = [heads.VICTIM_ID], []
        print(
            f"{cid}: {len(rows)} rows, victim rows {int(rows.victim.sum())}", flush=True
        )
    rows = pd.concat(parts, ignore_index=True)
    for name in ("rank", "raw"):
        out = Path(f"{args.run}_{name}")
        out.mkdir(parents=True, exist_ok=True)
        reports = {}
        for seed in args.seeds:
            rep = dict(base[str(seed)])
            score = (
                heads.rank_score(rows)
                if name == "rank"
                else heads.raw_score(rows, rep["folds"])
            )
            rows[["clip_id", "track_id", "t"]].assign(
                score=score, fold=rows.clip_id.map(rep["folds"])
            ).to_parquet(out / f"alarms_small_s{seed}.parquet", index=False)
            reports[str(seed)] = rep | {
                "victim_tracks": vt,
                "path_fragments": pf,
                "unmatched_clicks": {},
            }
        (out / "summary.json").write_text(
            json.dumps({"reports": reports, "base": args.base, "score": name}, indent=1)
        )
        print(
            f"wrote {out}: uv run hydroknight eval protocol {out} "
            f"--tracks-dir {args.run}_tracks"
        )


def cmd_heads_rows(args) -> None:
    from types import SimpleNamespace

    from hydro_knight.annotate import zones
    from hydro_knight.ingest.manifest import Manifest
    from hydro_knight.stack import heads
    from hydro_knight.stack.splash import add_splash

    base = json.loads((Path(args.base) / "summary.json").read_text())["reports"]
    clips = base[next(iter(base))]["used_test"]
    recs = {
        r.clip_id: r for r in Manifest(Path("data/manifests/pool_footage.jsonl")).load()
    }
    zs = zones.load()
    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    parts, meta = [], {}
    old = pd.read_parquet(args.splash_from) if args.splash_from else None
    t0 = time.time()
    for cid in clips:
        r = recs[cid]
        rows, tracks = heads.clip_rows(r, zs.get(cid), zones.in_zone)
        heads.write_eval_tracks(tracks, cid, run / "tracks")
        fps = heads.read_meta(cid)["fps"]
        if old is not None:  # splash depends only on the tracker's box: reuse it
            prev = old.loc[
                old.clip_id == cid,
                ["src_track" if "src_track" in old else "track_id", "t", "splash"],
            ]
            rows = rows.merge(
                prev.set_axis(["src_track", "t", "splash"], axis=1),
                on=["src_track", "t"],
                how="left",
            )
        else:
            # add_splash sizes its patches as 0.3 x w, so w = head width / 0.3 gives head-sized patches
            rows = add_splash(
                rows.assign(w=rows.hw / 0.3),
                SimpleNamespace(clip_id=cid, fps=fps, zone=zs.get(cid)),
            ).drop(columns="w")
        parts.append(rows)
        ev = r.events[0]
        meta[cid] = {
            "fps": fps,
            "onset": r.to_file_time(ev["start"]),
            "contact": r.to_file_time(ev["contact"]),
            "victim_tracks": [heads.VICTIM_ID],
            "path_fragments": [],
        }
        print(
            f"{cid}: {len(rows)} rows, victim rows {int(rows.victim.sum())}, {time.time() - t0:.0f} s",
            flush=True,
        )
    pd.concat(parts, ignore_index=True).to_parquet(run / "rows.parquet")
    (run / "clips.json").write_text(
        json.dumps({"base": args.base, "clips": meta}, indent=1)
    )
    print(f"wrote {run}/rows.parquet ({len(meta)} clips)")


def cmd_heads_fit(args) -> None:
    import numpy as np

    from hydro_knight.stack import heads, rank

    run = Path(args.run)
    rows = pd.read_parquet(run / "rows.parquet")
    meta = json.loads((run / "clips.json").read_text())
    base = json.loads((Path(meta["base"]) / "summary.json").read_text())["reports"]
    clips = meta["clips"]
    F = heads.v3_features(rows)
    names = ["ranker", "equal"]
    if args.appear:
        from hydro_knight.stack import appearance

        a = np.load(run / "appear.npz")
        X = np.full((len(rows), a["emb"].shape[1]), np.nan, np.float32)
        X[a["idx"]] = a["emb"]
        every = float(a["every"])
        g = rank.training_groups(rows, clips)
        labelled = rows.index.isin(g.index)
        names.append("ranker_appear")
        print(f"appearance: {len(a['idx'])} embedded rows, cache every {every} s")
    reports = {}
    for seed in args.seeds:
        rep = dict(base[str(seed)])
        folds = rep["folds"]
        fold = rows.clip_id.map(folds).to_numpy()
        scores = {k: np.full(len(rows), np.nan) for k in names}
        weights = {}
        for f in sorted(set(folds.values())):
            test = fold == f
            Z = rank.Scaler.fit(F[~test])(F)
            w = rank.fit_ranker(
                Z[~test],
                rank.training_groups(rows[~test], clips),
                heads.V3_FEATURES,
                heads.V3_SIGNED,
            )
            weights[f] = dict(
                zip(heads.V3_FEATURES, np.round(w, 3).tolist(), strict=True)
            )
            scores["ranker"][test] = Z[test][heads.V3_FEATURES].to_numpy() @ w
            scores["equal"][test] = Z[test][heads.V3_SIGNED].sum(1).to_numpy()
            if args.appear:
                ap = appearance.stacked_feature(
                    X, rows.victim.to_numpy(), labelled, fold, f
                )
                Fa = F.assign(f_appear=appearance.carry_forward(rows, ap, every))
                Za = rank.Scaler.fit(Fa[~test])(Fa)
                feats = [*heads.V3_FEATURES, "f_appear"]
                wa = rank.fit_ranker(
                    Za[~test],
                    rank.training_groups(rows[~test], clips),
                    feats,
                    [*heads.V3_SIGNED, "f_appear"],
                )
                weights[f]["with_appear"] = dict(
                    zip(feats, np.round(wa, 3).tolist(), strict=True)
                )
                scores["ranker_appear"][test] = Za[test][feats].to_numpy() @ wa
        for name, s in scores.items():
            rows[["clip_id", "track_id", "t"]].assign(
                score=rank.relative(rows, s), fold=fold
            ).to_parquet(run / f"alarms_{name}_s{seed}.parquet", index=False)
        reports[str(seed)] = rep | {
            "victim_tracks": {c: clips[c]["victim_tracks"] for c in clips},
            "path_fragments": {c: clips[c]["path_fragments"] for c in clips},
            "unmatched_clicks": {},
            "ranker_weights": weights,
        }
        print(f"seed {seed}: weights per fold {weights}", flush=True)
    (run / "summary.json").write_text(
        json.dumps({"reports": reports, "base": meta["base"]}, indent=1)
    )
    print(
        f"wrote {run}: uv run hydroknight eval protocol {run} --size ranker "
        f"--tracks-dir {run / 'tracks'}"
    )


def cmd_heads_embed(args) -> None:
    import numpy as np

    from hydro_knight.stack import appearance

    run = Path(args.run)
    rows = pd.read_parquet(run / "rows.parquet")
    clips = json.loads((run / "clips.json").read_text())["clips"]
    pick = appearance.cache_rows(rows, args.every)
    print(f"{pick.sum()} of {len(rows)} rows get a crop (cache every {args.every} s)")
    emb = appearance.embed_rows(
        rows[pick],
        {c: m["fps"] for c, m in clips.items()},
        Path(args.videos),
        appearance.Embedder(),
    )
    np.savez(
        run / "appear.npz",
        idx=np.flatnonzero(pick),
        emb=emb,
        every=args.every,
        model=appearance.MODEL,
    )
    print(
        f"wrote {run}/appear.npz; next: hydroknight stack heads-fit --run {run} --appear"
    )


def cmd_heads_detect(args) -> None:
    from hydro_knight.stack import whole_frame

    whole_frame.detect_clips(
        args.clips or None,
        Path(args.model),
        Path(args.out),
        args.width,
        args.base,
    )


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
    ht = sub.add_parser(
        "heads-track", help="RT-DETR head boxes -> weak-kept head tracks"
    )
    ht.add_argument("--force", action="store_true")
    hr = sub.add_parser(
        "heads-run", help="churn + size flicker -> protocol run folders"
    )
    hr.add_argument("--run", default="runs/stack_heads")
    hr.add_argument(
        "--base", default="runs/v2_kfold_small", help="clips, folds, seeds from here"
    )
    hr.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    hw = sub.add_parser("heads-rows", help="v3 rows: head-track features + splash")
    hw.add_argument("--run", default="runs/v3_ranker")
    hw.add_argument("--base", default="runs/v2_kfold_small", help="clips from here")
    hw.add_argument(
        "--splash-from", help="rows.parquet of an earlier run: reuse its splash"
    )
    hf = sub.add_parser(
        "heads-fit", help="v3 within-moment ranker, base run's folds and seeds"
    )
    hf.add_argument("--run", default="runs/v3_ranker")
    hf.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    hf.add_argument(
        "--appear",
        action="store_true",
        help="add the appearance feature (run heads-embed first) as size ranker_appear",
    )
    he = sub.add_parser(
        "heads-embed", help="frozen DINOv2 embeddings of head-and-shoulder crops"
    )
    he.add_argument("--run", default="runs/v3_ranker")
    he.add_argument(
        "--every",
        type=float,
        default=2.0,
        help="one crop per track per this many seconds, carried forward (0 = every row)",
    )
    he.add_argument("--videos", default="raw_local")
    hd = sub.add_parser(
        "heads-detect",
        help="head detector on the whole frame (no tiles), on this machine",
    )
    hd.add_argument("clips", nargs="*", help="default: the base run's clips")
    hd.add_argument("--model", default="data/detections_heads/model_rtdetr_v7_s1")
    hd.add_argument("--out", default="data/detections_heads_whole")
    hd.add_argument("--width", type=int, default=640)
    hd.add_argument("--base", default="runs/v2_kfold_small", help="clips from here")
    for p in (ht, hw):
        p.add_argument(
            "--heads-dir",
            default="data/detections_heads",
            help="raw/ in, tracks_kept/ out (data/detections_heads_whole for no tiling)",
        )
    args = ap.parse_args()
    if getattr(args, "heads_dir", None):
        from hydro_knight.stack import heads

        heads.HEADS = Path(args.heads_dir)
    {
        "rows": cmd_rows,
        "fit": cmd_fit,
        "video": cmd_video,
        "heads-track": cmd_heads_track,
        "heads-run": cmd_heads_run,
        "heads-rows": cmd_heads_rows,
        "heads-fit": cmd_heads_fit,
        "heads-embed": cmd_heads_embed,
        "heads-detect": cmd_heads_detect,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
