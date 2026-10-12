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
left out until fixed. Writes runs/<out>/summary.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hydro_knight.annotate import review
from hydro_knight.eval.separation import victim_separation
from hydro_knight.features.dataset_v2 import build_datasets, victim_windows
from hydro_knight.features.track_series import RATE_MODES
from hydro_knight.ingest.manifest import Manifest
from hydro_knight.models.tcn_ae_v2 import SIZES, n_params, score_v2, train_v2


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
    p.add_argument("--reviews", default=str(review.REVIEW_PATH))
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--out", default=None, help="default runs/v2_sizes (runs/v2_smoke with --smoke)"
    )
    p.add_argument("--mlflow", action="store_true")
    p.add_argument(
        "--smoke", action="store_true", help="3 unlabeled clips, 30 steps; not a result"
    )
    args = p.parse_args()
    if args.smoke and args.mlflow:
        raise SystemExit("--smoke runs are never logged to MLflow")
    if args.smoke:
        print("SMOKE TEST: unlabeled clips, victims NOT removed. Not a result.")
        args.steps = 30
    # A smoke run must never overwrite a real result.
    args.out = args.out or ("runs/v2_smoke" if args.smoke else "runs/v2_sizes")

    records = Manifest(Path(args.manifest)).load()
    rejected = {
        cid
        for cid, v in review.load(Path(args.reviews)).items()
        if v.get("verdict") == "not_good"
    }
    if rejected:
        print(f"leaving out {len(rejected)} clips whose labels a review marked wrong")

    results, reports = [], {}
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
            )
        for size in args.sizes:
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
            held = score_v2(model, test, scaler, args.coord_weight, device=args.device)
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
                vs = score_v2(model, vic, scaler, args.coord_weight, device=args.device)
                sep = victim_separation(
                    held,
                    _clips_of(test),
                    vs[in_event],
                    _clips_of(vic)[in_event],
                    seed=seed,
                )
                row.update({k: v for k, v in sep.items() if k != "per_clip_rank"})
                row["per_clip_rank"] = sep["per_clip_rank"]
            results.append(row)
            print(json.dumps({k: v for k, v in row.items() if k != "per_clip_rank"}))

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

    out = Path(args.out)
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

    if args.mlflow:
        from hydro_knight.eval import tracking

        for row in results:
            with tracking.run(name=f"{out.name}-{row['size']}-s{row['seed']}"):
                tracking.log_config(
                    {k: v for k, v in vars(args).items() if k not in ("sizes", "seeds")}
                    | {"size": row["size"], "seed": row["seed"]}
                )
                tracking.log_metrics(
                    {
                        k: v
                        for k, v in row.items()
                        if isinstance(v, (int, float)) and not isinstance(v, bool)
                    }
                )


if __name__ == "__main__":
    main()
