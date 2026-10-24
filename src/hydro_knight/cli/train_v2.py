"""
Train the v2 TCN autoencoder sizes on non-victim swimmers and compare them on
clips they never trained on.

    uv run hydroknight train tcn-v2 --sizes small medium large --seeds 0 1 2 --out runs/v2_sizes
    uv run hydroknight train tcn-v2 --sizes medium --rates capped --out runs/v2_rates_capped
    uv run hydroknight train tcn-v2 --holdout --seeds 0      # old single 20% hold-out, one model
    uv run hydroknight train tcn-v2 --smoke        # code-path check before labels exist

Grouped K-fold (default, --folds 5): each seed gives every usable clip one fold
(eval/split.assign_folds) and its own initialization. Each fold's model trains on the other
folds and scores only its own, so every clip is tested exactly once per seed, by a model
that never saw it; the K held-out score tables are then pooled (eval/kfold.py), each model's
scores divided by the median score of its own training windows so one alarm threshold
means the same for every fold. --holdout instead trains one model per seed on a random
~80% of clips (split_clips) and tests the rest, as before.

For every (size, seed) it reports two checks on the pooled held-out clips, and the same
per fold:
  1. reconstruction error on normal swimmers (median, p90): can the size describe
     normal swimming it never saw?
  2. victim separation (eval/separation.py): do the victim's windows between onset and
     guard contact score as more unusual than everyone else's? Always next to the same
     number for random scores.
--detection adds recall at false-alarm targets with the threshold picked on the scored
clips themselves (an upper bound) and, for K-fold, picked on the other folds (honest).
Clips whose labels a review marked "not_good" (data/annotations/label_reviews.json) are
left out until fixed, and Claude-labeled clips with no review verdict yet are held out
until reviewed. Writes runs/<out>/summary.json and one checkpoint per model,
runs/<out>/model_<size>_s<seed>_f<fold>.pt (model_<size>_s<seed>.pt with --holdout; load
with tcn_ae_v2.load_checkpoint). --load-dir skips training and re-scores the checkpoints
of an earlier run, so an evaluation change can be measured on the exact same models.
"""

from __future__ import annotations

import argparse
import contextlib
import json
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from hydro_knight.annotate import review
from hydro_knight.eval import kfold, run_meta
from hydro_knight.eval.detection_v2 import (
    alarm_table,
    clip_evals,
    crossfit_points,
    operating_points,
)
from hydro_knight.eval.figures import loss_curves
from hydro_knight.eval.separation import victim_separation
from hydro_knight.eval.split import assign_folds, split_clips
from hydro_knight.features.dataset_v2 import (
    DatasetReport,
    load_clips,
    split_windows,
    victim_windows,
)
from hydro_knight.features.track_series import RATE_MODES
from hydro_knight.features.windows_v2 import WindowSet
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
    p.add_argument(
        "--folds", type=int, default=5, help="grouped K-fold: number of folds"
    )
    p.add_argument(
        "--holdout",
        action="store_true",
        help="old single random hold-out per seed (about --test-frac of clips) instead of K-fold",
    )
    p.add_argument("--test-frac", type=float, default=0.2, help="with --holdout")
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
        args.folds = min(args.folds, 3)  # only 3 clips
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
        f"seeds {' '.join(map(str, args.seeds))}, "
        + ("one random hold-out per seed" if args.holdout else f"{args.folds}-fold")
        + f", {args.steps} steps, rates `{args.rates}`"
        + (f", re-scoring `{args.load_dir}`" if args.load_dir else "")
        + (", with detection tables" if args.detection else "")
        + f". {len(rejected)} clips left out (rejected or unreviewed labels).\n\n"
        "One child run per (size, seed): per-step `train_loss` (one series per fold), "
        "train/test clip lists under Datasets, the checkpoints under Artifacts, pooled "
        "held-out metrics plus `fold<f>.` per-fold ones. Mean and SD across seeds here."
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


def _log_child(tracking, args, row, fits, parts, ckpts) -> None:
    """Everything one (size, seed) child run carries: params, loss per step, metrics, datasets, files.

    K-fold runs log one loss series, dataset pair and checkpoint per fold, the pooled
    metrics unprefixed and each fold's metrics under `fold<f>.`.
    """
    out = Path(args.out)
    size, seed = row["size"], row["seed"]
    shas = {
        p.fold: run_meta.file_sha256(ck) for p, ck in zip(parts, ckpts, strict=True)
    }
    tags = {"checkpoint_sha256": next(iter(shas.values()))}  # fold 0 with K-fold
    if not args.holdout:
        tags |= {f"checkpoint_sha256.f{f}": sha for f, sha in shas.items()}
    tracking.set_tags(tags)
    tracking.log_config(_flat_args(args) | {"sizes": size, "seeds": seed})
    metrics = run_meta.numeric_items(row)
    for key in (
        "detection",
        "detection_random",
        "detection_crossfit",
        "detection_crossfit_random",
    ):
        for pt in row.get(key, []):
            metrics |= run_meta.numeric_items(pt, f"{key}.fa{pt['fa_target']:g}.")
    for part, fit in zip(parts, fits, strict=True):
        tag = "" if part.fold is None else f".f{part.fold}"
        tracking.log_series(f"train_loss{tag}", fit["model"].loss_history_)
        if part.fold is not None:
            metrics |= run_meta.numeric_items(fit["row"], f"fold{part.fold}.")
        tracking.log_dataset(
            run_meta.split_table(part.rep.used_train, _clips_of(part.train)),
            f"train-s{seed}{tag}",
            "training",
            args.manifest,
        )
        tracking.log_dataset(
            run_meta.split_table(part.rep.used_test, _clips_of(part.test)),
            f"test-s{seed}{tag}",
            "evaluation",
            args.manifest,
        )
    tracking.log_metrics(metrics)
    for ck in ckpts:
        tracking.log_artifact(ck, "checkpoint")
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
    """Mean and SD across seeds of each size's pooled rows (per-fold rows are skipped)."""
    summary_rows = []
    for size in args.sizes:
        rows = [r for r in results if r["size"] == size and "fold" not in r]
        agg = {"size": size, "params": rows[0]["params"], "seeds": len(rows)}
        if not args.holdout:
            agg["folds"] = args.folds
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


@dataclass
class _Part:
    """One model's data: its fold (None for --holdout), windows, report and victim windows."""

    fold: int | None
    train: WindowSet
    test: WindowSet
    rep: DatasetReport
    vic: WindowSet | None
    in_event: np.ndarray | None


def _splits(args, clip_ids: list[str], seed: int) -> list[tuple[int | None, list[str]]]:
    """(fold, test clips) for every model of one seed: the hold-out, or one per fold."""
    if args.holdout:
        return [(None, split_clips(clip_ids, args.test_frac, seed)[1])]
    folds = assign_folds(clip_ids, args.folds, seed)
    return [(f, [c for c in clip_ids if folds[c] == f]) for f in range(args.folds)]


def _ckpt_path(args, size: str, seed: int, fold: int | None, load: bool) -> Path:
    tag = "" if fold is None else f"_f{fold}"
    return Path(args.load_dir if load else args.out) / f"model_{size}_s{seed}{tag}.pt"


def _fold_label(fold: int | None) -> str:
    return "" if fold is None else f" fold {fold}"


def _fit(args, part: _Part, size: str, seed: int) -> dict:
    """Train (or re-load) one model, score its held-out clips; returns its scores and row.

    K-fold models also score a sample of their own training windows: its median is the
    model's common-scale unit (eval/kfold.py). The hold-out model keeps raw scores.
    """
    if args.load_dir:
        ckpt = _ckpt_path(args, size, seed, part.fold, load=True)
        model, scaler, _ = load_checkpoint(ckpt, args.device)
        model.loss_history_ = [float("nan")]
    else:
        model, scaler = train_v2(
            part.train,
            size,
            args.steps,
            args.batch,
            args.lr,
            seed,
            args.coord_weight,
            args.device,
        )
    score = partial(score_v2, model, scaler=scaler, coord_weight=args.coord_weight)
    ref = 1.0
    if part.fold is not None:
        ref = kfold.scale_reference(
            score(kfold.training_sample(part.train), device=args.device)
        )
    if not args.load_dir:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        ckpt = _ckpt_path(args, size, seed, part.fold, load=False)
        save_checkpoint(
            ckpt,
            model,
            scaler,
            {
                "size": size,
                "seed": seed,
                "fold": part.fold,
                "folds": None if part.fold is None else args.folds,
                "score_scale": ref,
                "rates": args.rates,
                "joint_velocity": args.joint_velocity,
                "coord_weight": args.coord_weight,
                "min_track_s": args.min_track_s,
                "train_clips": part.rep.used_train,
                "test_clips": part.rep.used_test,
            },
        )
    held = score(part.test, device=args.device)
    vs = (
        score(part.vic, device=args.device)
        if part.vic is not None and len(part.vic)
        else np.empty(0)
    )
    row = _score_row(size, seed, model, held, part.test, vs, part)
    if part.fold is not None:
        row = {"fold": part.fold, "score_scale": ref} | row
    return {
        "model": model,
        "ckpt": ckpt,
        "held": held,
        "vs": vs,
        "ref": ref,
        "row": row,
    }


def _score_row(size, seed, model, held, test, vs, part) -> dict:
    """Reconstruction error and victim separation of one set of held-out scores."""
    row = {
        "size": size,
        "seed": seed,
        "params": n_params(model),
        "train_loss_final": float(np.mean(model.loss_history_[-20:])),
        "test_windows": int(len(held)),
        "test_median": float(np.median(held)) if len(held) else None,
        "test_p90": float(np.percentile(held, 90)) if len(held) else None,
    }
    if part.vic is not None and len(part.vic):
        sep = victim_separation(
            held,
            _clips_of(test),
            vs[part.in_event],
            _clips_of(part.vic)[part.in_event],
            seed=seed,
        )
        row.update({k: v for k, v in sep.items() if k != "per_clip_rank"})
        row["per_clip_rank"] = sep["per_clip_rank"]
    return row


def _pooled(
    parts: list[_Part], fits: list[dict], rep
) -> tuple[_Part, np.ndarray, np.ndarray]:
    """All held-out windows of one seed as one part, with their scores on the common scale.

    The hold-out run has one part and scale 1, so this returns it unchanged.
    """
    if len(parts) == 1:
        return parts[0], fits[0]["held"], fits[0]["vs"]
    vics = [p.vic for p in parts if p.vic is not None]
    pooled = _Part(
        None,
        parts[0].train,  # unused: every clip trained K-1 of the models
        WindowSet.concat([p.test for p in parts]),
        rep,
        WindowSet.concat(vics) if vics else None,
        np.concatenate([p.in_event for p in parts]) if vics else None,
    )
    held = np.concatenate([f["held"] / f["ref"] for f in fits])
    vs = np.concatenate([f["vs"] / f["ref"] for f in fits])
    return pooled, held, vs


def _alarms(parts: list[_Part], fits: list[dict]) -> pd.DataFrame:
    """Pooled alarm table: one row per held-out window (normal and victim).

    K-fold tables also carry `raw_score` (before scaling) and `fold`; `score` is the
    common-scale score, which every evaluation reads.
    """
    tables = {}
    for part, fit in zip(parts, fits, strict=True):
        frames = [alarm_table(part.test, fit["held"] / fit["ref"])]
        if part.vic is not None:
            frames.append(alarm_table(part.vic, fit["vs"] / fit["ref"]))
        t = pd.concat(frames)
        if part.fold is None:
            return t
        tables[part.fold] = t.assign(raw_score=t["score"] * fit["ref"])
    return kfold.pool(tables)


def _detection(args, records, rep, parts, fits, size, seed, held) -> dict:
    """Recall at false-alarm targets on the pooled held-out clips, plus random scores; writes sweeps and alarms."""
    alarms = _alarms(parts, fits)
    clips = clip_evals(alarms, records, rep)
    det = operating_points(clips, args.fa_targets)
    rng = np.random.default_rng(seed)
    alarms_rand = alarms.assign(score=rng.random(len(alarms)))
    clips_rand = clip_evals(alarms_rand, records, rep)
    det_rand = operating_points(clips_rand, args.fa_targets)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    det["sweep"].to_csv(out_dir / f"sweep_{size}_s{seed}.csv", index=False)
    det_rand["sweep"].to_csv(out_dir / f"sweep_random_{size}_s{seed}.csv", index=False)
    alarms.to_parquet(out_dir / f"alarms_{size}_s{seed}.parquet")
    if args.video:
        _demo_video(args, records, rep, alarms, det, held, size, seed)
    row = {
        "detection": det["points"],
        "detection_random": det_rand["points"],
        "detection_events": det["events"],
        "detection_hours": det["hours"],
    }
    if rep.folds:
        row["detection_crossfit"] = crossfit_points(clips, rep.folds, args.fa_targets)
        row["detection_crossfit_random"] = crossfit_points(
            clips_rand, rep.folds, args.fa_targets
        )
    honest = row.get("detection_crossfit", [{}] * len(det["points"]))
    for pt, pr, ph in zip(det["points"], det_rand["points"], honest, strict=True):
        line = (
            f"  <= {pt['fa_target']:g} FA/h: recall {pt['recall']:.2f}"
            f" (random {pr['recall']:.2f}; threshold picked on these clips, upper bound)"
        )
        if ph:
            line += (
                f" | threshold from the other folds: recall {ph['recall']:.2f},"
                f" {ph['fa_per_hour']:.0f} FA/h"
            )
        print(line)
    return row


def _train_all(args, records, rejected, tracking):
    """Train (or re-load) and score every (size, seed, fold); returns (result rows, dataset reports, loss histories)."""
    results, reports = [], {}
    histories: dict[str, dict] = {s: {} for s in args.sizes}
    sets, base = load_clips(
        args.keypoints,
        records,
        joint_velocity=args.joint_velocity,
        require_labels=not args.smoke,
        max_clips=3 if args.smoke else None,
        rates=args.rates,
        exclude_clips=frozenset(rejected),
        path_fragments=not args.no_path_fragments,
        min_track_s=args.min_track_s,
    )
    n_frag = sum(len(v) for v in base.path_fragments.values())
    print(f"{len(sets)} usable clips, {n_frag} path fragments removed")
    for seed in args.seeds:
        parts = []
        for fold, test_ids in _splits(args, list(sets), seed):
            train, test, rep = split_windows(sets, base, test_ids)
            print(
                f"seed {seed}{_fold_label(fold)}: train {len(rep.used_train)} clips, "
                f"{len(train)} windows | test {len(rep.used_test)} clips, {len(test)} windows"
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
            parts.append(_Part(fold, train, test, rep, vic, in_event))
        if args.holdout:
            seed_rep = parts[0].rep
        else:
            fold_of = {c: p.fold for p in parts for c in p.rep.used_test}
            seed_rep = kfold.pooled_report(base, {c: fold_of[c] for c in sets})
        reports[seed] = seed_rep.__dict__
        for size in args.sizes:
            with _child(tracking, args, size, seed):
                fits = [_fit(args, part, size, seed) for part in parts]
                pooled, held, vs = _pooled(parts, fits, seed_rep)
                first = fits[0]["model"]
                row = _score_row(size, seed, first, held, pooled.test, vs, pooled)
                if not args.holdout:
                    row["folds"] = args.folds
                    row["train_loss_final"] = float(
                        np.mean([f["row"]["train_loss_final"] for f in fits])
                    )
                if args.detection and not args.smoke:
                    row |= _detection(
                        args, records, seed_rep, parts, fits, size, seed, held
                    )
                for part, fit in zip(parts, fits, strict=True):
                    key = seed if part.fold is None else f"{seed} fold {part.fold}"
                    histories[size][key] = list(fit["model"].loss_history_)
                if tracking is not None:
                    ckpts = [f["ckpt"] for f in fits]
                    _log_child(tracking, args, row, fits, parts, ckpts)
                if not args.holdout:
                    results += [f["row"] for f in fits]
                    for f in fits:
                        print(json.dumps(_printable(f["row"])))
                results.append(row)
                print(json.dumps(_printable(row)))
    return results, reports, histories


def _printable(row: dict) -> dict:
    skip = (
        "per_clip_rank",
        "detection",
        "detection_random",
        "detection_crossfit",
        "detection_crossfit_random",
    )
    return {k: v for k, v in row.items() if k not in skip}


if __name__ == "__main__":
    main()
