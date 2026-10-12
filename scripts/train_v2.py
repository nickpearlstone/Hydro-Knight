"""
Train the v2 TCN autoencoder sizes on non-victim swimmers and compare them on
held-out clips.

    uv run python scripts/train_v2.py --sizes small medium large --steps 2000 --out runs/v2_sizes
    uv run python scripts/train_v2.py --smoke        # code-path check before labels exist

Writes runs/<out>/summary.json: per size, parameter count, final training loss, and
the held-out reconstruction error (median, p90) on normal swimmers. Lower held-out
error means a size can describe normal swimming it never saw; it is the first of the
two size checks (the second, separating victims, needs the rebuilt evaluation).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hydro_knight.features.dataset_v2 import build_datasets
from hydro_knight.ingest.manifest import Manifest
from hydro_knight.models.tcn_ae_v2 import SIZES, n_params, score_v2, train_v2


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--manifest", default="data/manifests/pool_footage.jsonl")
    p.add_argument("--keypoints", default="data/keypoints_tiled")
    p.add_argument("--sizes", nargs="+", default=list(SIZES), choices=list(SIZES))
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--coord-weight", type=float, default=0.0)
    p.add_argument("--joint-velocity", action="store_true")
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
    train, test, rep = build_datasets(
        args.keypoints,
        records,
        test_frac=args.test_frac,
        seed=args.seed,
        joint_velocity=args.joint_velocity,
        require_labels=not args.smoke,
        max_clips=3 if args.smoke else None,
    )
    print(
        f"train: {len(rep.used_train)} clips, {len(train)} windows | test: {len(rep.used_test)} clips, {len(test)} windows"
    )
    print(
        f"skipped: {len(rep.unlabeled)} unlabeled, {len(rep.held)} held, {len(rep.missing)} without keypoints"
    )
    unmatched = sum(rep.unmatched_clicks.values())
    if unmatched:
        print(
            f"note: {unmatched} victim clicks matched no track (victim not detected then)"
        )

    results = []
    for size in args.sizes:
        model, scaler = train_v2(
            train,
            size,
            args.steps,
            args.batch,
            args.lr,
            args.seed,
            args.coord_weight,
            args.device,
        )
        held = score_v2(model, test, scaler, args.coord_weight, device=args.device)
        row = {
            "size": size,
            "params": n_params(model),
            "train_loss_final": float(np.mean(model.loss_history_[-20:])),
            "test_windows": int(len(held)),
            "test_median": float(np.median(held)) if len(held) else None,
            "test_p90": float(np.percentile(held, 90)) if len(held) else None,
        }
        results.append(row)
        print(json.dumps(row))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "smoke": args.smoke,
        "args": vars(args),
        "report": rep.__dict__,
        "results": results,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(f"wrote {out / 'summary.json'}")

    if args.mlflow:
        from hydro_knight.eval import tracking

        for row in results:
            with tracking.run(name=f"{out.name}-{row['size']}"):
                tracking.log_config(
                    {k: v for k, v in vars(args).items() if k != "sizes"}
                    | {"size": row["size"]}
                )
                tracking.log_metrics(
                    {k: v for k, v in row.items() if isinstance(v, (int, float))}
                )


if __name__ == "__main__":
    main()
