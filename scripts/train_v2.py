"""
Train the v2 TCN autoencoder sizes on non-victim swimmers and compare them on
held-out clips.

    uv run python scripts/train_v2.py --sizes small medium large --seeds 0 1 2 --out runs/v2_sizes
    uv run python scripts/train_v2.py --sizes medium --rates capped --out runs/v2_rates_capped
    uv run python scripts/train_v2.py --smoke        # code-path check before labels exist

Each seed picks its own whole-clip train/test split and initialization. For every
(size, seed) it reports two checks on the held-out clips:
  1. reconstruction error on normal swimmers (median, p90): can the size describe
     normal swimming it never saw?
  2. victim separation (eval/separation.py): do the victim's windows between onset and
     guard contact score as more unusual than everyone else's? Always next to the same
     number for random scores.
Clips whose labels a review marked "not_good" (data/annotations/label_reviews.json) are
left out until fixed, and Claude-labeled clips with no review verdict yet are held out
until reviewed. Writes runs/<out>/summary.json and one checkpoint per (size, seed),
runs/<out>/model_<size>_s<seed>.pt (load with tcn_ae_v2.load_checkpoint).
--load-dir skips training and re-scores the checkpoints of an earlier run, so an
evaluation change can be measured on the exact same models.
"""

from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from hydro_knight.annotate import review
from hydro_knight.eval import run_meta
from hydro_knight.eval.detection_v2 import alarm_table, clip_evals, operating_points
from hydro_knight.eval.figures import loss_curves
from hydro_knight.eval.separation import victim_separation
from hydro_knight.features.dataset_v2 import build_datasets, victim_windows
from hydro_knight.features.track_series import RATE_MODES
from hydro_knight.ingest.manifest import Manifest
from hydro_knight.models.tcn_ae_v2 import (
    SIZES,
    load_checkpoint,
    n_params,
    save_checkpoint,
    score_v2,
    train_v2,
)


def _clips_of(ws) -> np.ndarray:
    """Clip id of every window in a window set."""
    by_series = np.array([ts.clip_id for ts in ws.series], dtype=object)
    return by_series[ws.index[:, 0]] if len(ws) else np.empty(0, dtype=object)


def _mean_sd(rows: list[dict], key: str) -> tuple[float | None, float | None]:
    vals = [r[key] for r in rows if r.get(key) is not None]
    if not vals:
        return None, None
    return float(np.mean(vals)), float(np.std(vals))


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--manifest", default="data/manifests/pool_footage.jsonl")
    p.add_argument("--keypoints", default="data/keypoints_tiled")
    p.add_argument("--sizes", nargs="+", default=list(SIZES), choices=list(SIZES))
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--coord-weight", type=float, default=0.0)
    p.add_argument("--joint-velocity", action="store_true")
    p.add_argument("--rates", default="raw", choices=RATE_MODES)
    p.add_argument(
        "--no-path-fragments",
        action="store_true",
        help="remove only click-matched victim tracks (the old behavior)",
    )
    p.add_argument(
        "--min-track-s",
        type=float,
        default=None,
        help="also score tracks this long or longer that are shorter than a window",
    )
    p.add_argument("--reviews", default=str(review.REVIEW_PATH))
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--out", default=None, help="default runs/v2_sizes (runs/v2_smoke with --smoke)"
    )
    p.add_argument(
        "--detection",
        action="store_true",
        help="also report recall / time to alarm / false alarms per hour (eval/detection_v2.py)",
    )
    p.add_argument("--fa-targets", nargs="+", type=float, default=[30, 60, 120, 240])
    p.add_argument(
        "--load-dir",
        default=None,
        help="re-score model_<size>_s<seed>.pt from this earlier run instead of training",
    )
    p.add_argument(
        "--mlflow",
        action="store_true",
        help="log a parent run + one child per (size, seed) (needs --extra tracking)",
    )
    p.add_argument("--mlflow-experiment", default="train-v2")
    p.add_argument(
        "--video",
        action="store_true",
        help="with --detection: render a best/worst held-out demo video per (size, seed)"
        " (unblurred footage, local only, never commit)",
    )
    p.add_argument("--videos-dir", default="raw_local")
    p.add_argument(
        "--smoke", action="store_true", help="3 unlabeled clips, 30 steps; not a result"
    )
    args = p.parse_args()
    if args.smoke and args.mlflow:
        raise SystemExit("--smoke runs are never logged to MLflow")
    if (
        args.load_dir
        and Path(args.load_dir).resolve() == Path(args.out or "").resolve()
    ):
        raise SystemExit(
            "--load-dir and --out must differ (re-scoring never overwrites)"
        )
    if args.smoke:
        print("SMOKE TEST: unlabeled clips, victims NOT removed. Not a result.")
        args.steps = 30
    # A smoke run must never overwrite a real result.
    args.out = args.out or ("runs/v2_smoke" if args.smoke else "runs/v2_sizes")

    records = Manifest(Path(args.manifest)).load()
    verdicts = review.load(Path(args.reviews))
    rejected = {cid for cid, v in verdicts.items() if v.get("verdict") == "not_good"}
    # Claude's labels count only after a review marks them good (held until reviewed).
    unreviewed = {
        cid
        for cid in review.load(review.CLAUDE_PATH)
        if verdicts.get(cid, {}).get("verdict") is None
    }
    if rejected:
        print(f"leaving out {len(rejected)} clips whose labels a review marked wrong")
    if unreviewed:
        print(f"leaving out {len(unreviewed)} Claude-labeled clips not reviewed yet")
    rejected |= unreviewed

    out = Path(args.out)
    tracking = None
    with contextlib.ExitStack() as stack:
        if args.mlflow:
            from hydro_knight.eval import tracking  # imports mlflow (optional extra)

            stack.enter_context(
                tracking.run(
                    out.name,
                    experiment=args.mlflow_experiment,
                    tags=run_meta.standard_tags(
                        "train_v2",
                        Path(args.manifest),
                        model_family="tcn_ae_v2",
                        run_folder=out.name,
                        load_dir=args.load_dir,
                    ),
                    note=_parent_note(args, rejected),
                )
            )
            tracking.log_config(_flat_args(args))
        results, reports, histories = _train_all(args, records, rejected, tracking)
        summary_rows = _summarize(args, results)

        out.mkdir(parents=True, exist_ok=True)
        summary = {
            "smoke": args.smoke,
            "args": vars(args),
            "rejected_clips": sorted(rejected),
            "reports": reports,
            "results": results,
            "summary": summary_rows,
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
        print(f"wrote {out / 'summary.json'}")

        if tracking is not None:
            for agg in summary_rows:
                tracking.log_metrics(run_meta.numeric_items(agg, f"{agg['size']}."))
            tracking.log_figure(loss_curves(histories), "figures/loss_curves.png")
            tracking.log_artifact(out / "summary.json")


def _flat_args(args) -> dict:
    """Command-line settings as MLflow params (lists joined, so sizes/seeds read as one value)."""
    return {
        k: " ".join(map(str, v)) if isinstance(v, list) else v
        for k, v in vars(args).items()
    }


def _parent_note(args, rejected) -> str:
    return (
        f"**train_v2** `{Path(args.out).name}`: sizes {' '.join(args.sizes)}, "
        f"seeds {' '.join(map(str, args.seeds))}, {args.steps} steps, rates `{args.rates}`"
        + (f", re-scoring `{args.load_dir}`" if args.load_dir else "")
        + (", with detection tables" if args.detection else "")
        + f". {len(rejected)} clips left out (rejected or unreviewed labels).\n\n"
        "One child run per (size, seed): per-step `train_loss`, train/test clip lists "
        "under Datasets, the checkpoint under Artifacts. Mean and SD across seeds here."
    )


def _child(tracking, args, size: str, seed: int):
    """Nested MLflow run for one (size, seed), or a no-op without --mlflow."""
    if tracking is None:
        return contextlib.nullcontext()
    return tracking.run(
        f"{size}-s{seed}",
        nested=True,
        tags=run_meta.standard_tags(
            "train_v2",
            Path(args.manifest),
            model_family="tcn_ae_v2",
            size=size,
            seed=seed,
            run_folder=Path(args.out).name,
        ),
    )


def _log_child(tracking, args, row, model, rep, train, test, ckpt: Path) -> None:
    """Everything one (size, seed) child run carries: params, loss per step, metrics, datasets, files."""
    out = Path(args.out)
    size, seed = row["size"], row["seed"]
    tracking.set_tags({"checkpoint_sha256": run_meta.file_sha256(ckpt)})
    tracking.log_config(_flat_args(args) | {"sizes": size, "seeds": seed})
    tracking.log_series("train_loss", model.loss_history_)
    metrics = run_meta.numeric_items(row)
    for key in ("detection", "detection_random"):
        for pt in row.get(key, []):
            metrics |= run_meta.numeric_items(pt, f"{key}.fa{pt['fa_target']:g}.")
    tracking.log_metrics(metrics)
    tracking.log_dataset(
        run_meta.split_table(rep.used_train, _clips_of(train)),
        f"train-s{seed}",
        "training",
        args.manifest,
    )
    tracking.log_dataset(
        run_meta.split_table(rep.used_test, _clips_of(test)),
        f"test-s{seed}",
        "evaluation",
        args.manifest,
    )
    tracking.log_artifact(ckpt, "checkpoint")
    for name in (
        f"sweep_{size}_s{seed}.csv",
        f"sweep_random_{size}_s{seed}.csv",
        f"alarms_{size}_s{seed}.parquet",
    ):
        if (out / name).exists():
            tracking.log_artifact(out / name, "detection")
    demo = out / f"demo_{size}_s{seed}.mp4"
    if args.video and demo.exists():
        tracking.log_artifact(demo, "demo")
        tracking.set_tags({"private_footage": "true"})


def _demo_video(args, records, rep, alarms, det, held, size, seed) -> None:
    """Best and worst held-out clip with boxes coloured by score -> runs/<out>/demo_<size>_s<seed>.mp4 (local only)."""
    from hydro_knight.eval import demo_video
    from hydro_knight.eval import protocol as P

    truths = P.truths_from_records(records, rep.used_test, rep.victim_tracks)
    groups = dict(tuple(alarms.groupby("clip_id")))
    auc = {
        c: P.within_clip_auc(groups[c], t, P.ProtocolConfig().onset_tol_s)
        for c, t in truths.items()
        if c in groups
    }
    lo, hi = demo_video.score_scale(
        {"test_median": float(np.median(held)) if len(held) else None}
        | {"detection": det["points"]},
        alarms["score"].to_numpy(),
    )
    path = demo_video.render_demo(
        Path(args.out) / f"demo_{size}_s{seed}.mp4",
        alarms,
        truths,
        demo_video.pick_best_worst(auc),
        Path(args.keypoints),
        Path(args.videos_dir),
        lo,
        hi,
        subtitle=f"{Path(args.out).name}: {size}, seed {seed}, never trained on",
    )
    print(f"  demo video: {path or 'skipped (no ffmpeg, scored clips or video files)'}")


def _summarize(args, results: list[dict]) -> list[dict]:
    summary_rows = []
    for size in args.sizes:
        rows = [r for r in results if r["size"] == size]
        agg = {"size": size, "params": rows[0]["params"], "seeds": len(rows)}
        for key in (
            "test_median",
            "test_p90",
            "auc",
            "auc_random",
            "clip_rank_median",
            "clip_rank_median_random",
        ):
            agg[key], agg[key + "_sd"] = _mean_sd(rows, key)
        summary_rows.append(agg)
        print("SUMMARY", json.dumps(agg))
    return summary_rows


def _train_all(args, records, rejected, tracking):
    """Train (or re-load) and score every (size, seed); returns (result rows, dataset reports, loss histories)."""
    results, reports = [], {}
    histories: dict[str, dict[int, list[float]]] = {s: {} for s in args.sizes}
    for seed in args.seeds:
        train, test, rep = build_datasets(
            args.keypoints,
            records,
            test_frac=args.test_frac,
            seed=seed,
            joint_velocity=args.joint_velocity,
            require_labels=not args.smoke,
            max_clips=3 if args.smoke else None,
            rates=args.rates,
            exclude_clips=frozenset(rejected),
            path_fragments=not args.no_path_fragments,
            min_track_s=args.min_track_s,
        )
        reports[seed] = rep.__dict__
        n_frag = sum(len(v) for v in rep.path_fragments.values())
        print(
            f"seed {seed}: train {len(rep.used_train)} clips, {len(train)} windows | "
            f"test {len(rep.used_test)} clips, {len(test)} windows | "
            f"{n_frag} path fragments removed"
        )
        vic, in_event = (None, None)
        if not args.smoke:
            vic, in_event = victim_windows(
                args.keypoints,
                records,
                rep,
                joint_velocity=args.joint_velocity,
                rates=args.rates,
                min_track_s=args.min_track_s,
            )
        for size in args.sizes:
            with _child(tracking, args, size, seed):
                if args.load_dir:
                    ckpt = Path(args.load_dir) / f"model_{size}_s{seed}.pt"
                    model, scaler, _ = load_checkpoint(ckpt, args.device)
                    model.loss_history_ = [float("nan")]
                else:
                    model, scaler = train_v2(
                        train,
                        size,
                        args.steps,
                        args.batch,
                        args.lr,
                        seed,
                        args.coord_weight,
                        args.device,
                    )
                    Path(args.out).mkdir(parents=True, exist_ok=True)
                    ckpt = Path(args.out) / f"model_{size}_s{seed}.pt"
                    save_checkpoint(
                        ckpt,
                        model,
                        scaler,
                        {
                            "size": size,
                            "seed": seed,
                            "rates": args.rates,
                            "joint_velocity": args.joint_velocity,
                            "coord_weight": args.coord_weight,
                            "min_track_s": args.min_track_s,
                            "train_clips": rep.used_train,
                            "test_clips": rep.used_test,
                        },
                    )
                held = score_v2(
                    model, test, scaler, args.coord_weight, device=args.device
                )
                row = {
                    "size": size,
                    "seed": seed,
                    "params": n_params(model),
                    "train_loss_final": float(np.mean(model.loss_history_[-20:])),
                    "test_windows": int(len(held)),
                    "test_median": float(np.median(held)) if len(held) else None,
                    "test_p90": float(np.percentile(held, 90)) if len(held) else None,
                }
                if vic is not None and len(vic):
                    vs = score_v2(
                        model, vic, scaler, args.coord_weight, device=args.device
                    )
                    sep = victim_separation(
                        held,
                        _clips_of(test),
                        vs[in_event],
                        _clips_of(vic)[in_event],
                        seed=seed,
                    )
                    row.update({k: v for k, v in sep.items() if k != "per_clip_rank"})
                    row["per_clip_rank"] = sep["per_clip_rank"]
                if args.detection and vic is not None:
                    vs_all = (
                        score_v2(
                            model, vic, scaler, args.coord_weight, device=args.device
                        )
                        if len(vic)
                        else np.empty(0)
                    )
                    alarms = pd.concat(
                        [alarm_table(test, held), alarm_table(vic, vs_all)]
                    )
                    det = operating_points(
                        clip_evals(alarms, records, rep), args.fa_targets
                    )
                    rng = np.random.default_rng(seed)
                    alarms_rand = alarms.assign(score=rng.random(len(alarms)))
                    det_rand = operating_points(
                        clip_evals(alarms_rand, records, rep), args.fa_targets
                    )
                    out_dir = Path(args.out)
                    out_dir.mkdir(parents=True, exist_ok=True)
                    det["sweep"].to_csv(
                        out_dir / f"sweep_{size}_s{seed}.csv", index=False
                    )
                    det_rand["sweep"].to_csv(
                        out_dir / f"sweep_random_{size}_s{seed}.csv", index=False
                    )
                    alarms.to_parquet(out_dir / f"alarms_{size}_s{seed}.parquet")
                    if args.video:
                        _demo_video(args, records, rep, alarms, det, held, size, seed)
                    row["detection"] = det["points"]
                    row["detection_random"] = det_rand["points"]
                    row["detection_events"] = det["events"]
                    row["detection_hours"] = det["hours"]
                    for pt, pr in zip(det["points"], det_rand["points"], strict=True):
                        print(
                            f"  <= {pt['fa_target']:g} FA/h: recall {pt['recall']:.2f}"
                            f" (random {pr['recall']:.2f}), median time to alarm"
                            f" {pt.get('median_latency_s')} s"
                        )
                histories[size][seed] = list(model.loss_history_)
                if tracking is not None:
                    _log_child(tracking, args, row, model, rep, train, test, ckpt)
                results.append(row)
                print(
                    json.dumps(
                        {
                            k: v
                            for k, v in row.items()
                            if k
                            not in ("per_clip_rank", "detection", "detection_random")
                        }
                    )
                )
    return results, reports, histories


if __name__ == "__main__":
    main()
