"""
Score a saved run with the two-tier evaluation protocol (src/hydro_knight/eval/protocol.py).

Reads the per-window alarm tables and dataset reports that `hydroknight train tcn-v2 --detection`
saves, so nothing is retrained. Writes, per seed, a markdown report, the tier-1 per-clip
table and a JSON summary into --out. For a K-fold run each seed's table already pools every
fold's held-out clips (each clip scored once, by a model that never saw it), so it is scored
once per seed; the report adds the proxy curve with thresholds picked on the other folds and
the per-fold spread. Old hold-out runs score as before.

    uv run hydroknight eval protocol runs/v2_ckpt_after
    uv run hydroknight eval protocol runs/v2_ckpt_after --pseudo-normal      # tier 2 on pre-onset footage
    uv run hydroknight eval protocol runs/v2_ckpt_after --onset-alt scratch/z5cc/label_compare.json
    uv run hydroknight eval protocol runs/RUN_B --compare runs/RUN_A          # is B better than A?
    uv run --extra tracking hydroknight eval protocol runs/RUN --mlflow       # also log to MLflow

--onset-alt takes either {clip_id: onset in source seconds} or the Z5cc label comparison
rows (clip_id, his_onset, our_onset); only onsets within 30 s of ours are used.
--normal takes a JSON list of {clip_id, start, end} incident-free spans (file seconds)
whose scores are in the run's alarm tables; until such footage exists, tier 2 reports
that no budget can be verified, and --pseudo-normal gives a clearly labelled stopgap.
"""

from __future__ import annotations

import argparse
import contextlib
import json
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from hydro_knight.eval import protocol as P
from hydro_knight.ingest.manifest import Manifest
from hydro_knight.preprocess.build_tracks import read_provenance


def load_onset_alt(path: Path | None) -> dict[str, float]:
    if path is None:
        return {}
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        return {str(k): float(v) for k, v in data.items()}
    out: dict[str, float] = {}
    for row in data:
        cid, his, ours = row.get("clip_id"), row.get("his_onset"), row.get("our_onset")
        if cid is None or his is None or ours is None or abs(his - ours) > 30:
            continue
        out[cid] = min(out.get(cid, his), his)
    return out


def load_normal(path: Path | None) -> list[P.NormalSpan]:
    if path is None:
        return []
    return [
        P.NormalSpan(d["clip_id"], float(d["start"]), float(d["end"]))
        for d in json.loads(path.read_text())
    ]


def load_tracks(clip_ids, tracks_dir: Path) -> dict[str, pd.DataFrame]:
    """The track table each clip's model was fed, as (t in file seconds, track_id), confirmed tracks only.

    Any track file with columns frame and track_id and a video fps in its provenance works,
    so head tracks can replace the default pose keypoint tracks later.
    """
    out = {}
    for cid in clip_ids:
        pq = tracks_dir / f"{cid}.parquet"
        if not pq.exists():
            continue
        fps = read_provenance(pq)["video"]["fps"]
        df = pd.read_parquet(pq, columns=["frame", "track_id"])
        df = df[df["track_id"] >= 0]
        out[cid] = pd.DataFrame(
            {
                "t": df["frame"].to_numpy(float) / fps,
                "track_id": df["track_id"].to_numpy(),
            }
        )
    return out


def seeds_in(run: Path, size: str) -> list[int]:
    return sorted(
        int(p.stem.split("_s")[-1]) for p in run.glob(f"alarms_{size}_s*.parquet")
    )


def evaluate_seed(
    run: Path, size: str, seed: int, records, args, cfg: P.ProtocolConfig
) -> dict:
    summary = json.loads((run / "summary.json").read_text())
    rep = summary["reports"][str(seed)]
    table = pd.read_parquet(run / f"alarms_{size}_s{seed}.parquet")[P.SCORE_COLUMNS]
    test = [c for c in rep["used_test"] if c not in set(args.drop)]
    table = table[table["clip_id"].isin(test)]
    truths = P.truths_from_records(
        records, test, rep["victim_tracks"], onset_alt=load_onset_alt(args.onset_alt)
    )
    normal = load_normal(args.normal)
    label = "normal footage"
    if args.pseudo_normal and not normal:
        normal, label = (
            P.pre_onset_spans(truths, cfg.onset_tol_s),
            "PSEUDO-normal (pre-onset rescue footage)",
        )
    tracks = load_tracks(test, args.tracks_dir)
    folds = {c: rep["folds"][c] for c in test} if rep.get("folds") else None
    t1 = P.tier1(table, truths, cfg, tracks, folds)
    t2 = P.tier2(table, truths, normal, cfg)
    # Second victim definition (Nick, 2026-10-18): click-matched tracks only; the path
    # fragments' rows are dropped from scoring, so they are neither catches nor rivals.
    frags = rep.get("path_fragments", {})
    click_only = {
        c: sorted(set(v) - set(frags.get(c, [])))
        for c, v in rep["victim_tracks"].items()
    }
    truths_click = P.truths_from_records(
        records,
        test,
        click_only,
        onset_alt=load_onset_alt(args.onset_alt),
        ignore_tracks=frags,
    )
    t1_click = P.tier1(table, truths_click, cfg, tracks, folds)
    return {
        "table": table,
        "truths": truths,
        "t1": t1,
        "t2": t2,
        "t1_click": t1_click,
        "normal_label": label,
    }


def _clip_mean(pc: pd.DataFrame, col: str):
    """Clip-mean of one per-clip column, as a statistic over a (resampled) list of clip ids."""
    return lambda ids: float(np.nanmean(pc.loc[ids, col].to_numpy(float)))


def compare(run_a: Path, run_b: Path, size: str, records, args, cfg) -> str:
    """Paired, clip-resampled difference B - A on tier-1 headline measures, per shared seed."""
    lines = [
        f"# {run_b.name} vs {run_a.name}",
        "",
        "| Seed | Measure | B - A [95%] | P(B better) |",
        "|---|---|---|---|",
    ]
    for seed in sorted(set(seeds_in(run_a, size)) & set(seeds_in(run_b, size))):
        a = evaluate_seed(run_a, size, seed, records, args, cfg)["t1"][
            "per_clip"
        ].set_index("clip_id")
        b = evaluate_seed(run_b, size, seed, records, args, cfg)["t1"][
            "per_clip"
        ].set_index("clip_id")
        shared = sorted(set(a.index) & set(b.index))
        for col in ("within_clip_auc", "time_in_top3", "scored", "detected"):
            stat = partial(_clip_mean, col=col)
            d = P.paired_difference(
                shared, stat(a), stat(b), cfg.n_boot, cfg.seed, cfg.conf
            )
            lines.append(
                f"| {seed} | {col} | {d['diff']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}] | {d['p_b_better']:.2f} |"
            )
    return "\n".join(lines) + "\n"


def jsonable(t1: dict, t2: dict) -> dict:
    tier1 = {
        "summary": t1["summary"],
        "random": t1["random"],
        "shuffled": t1["shuffled"],
        "proxy": t1["proxy"],
    }
    for key in (
        "proxy_random_caught",
        "proxy_shuffled_caught",
        "crossfit_random_caught",
        "crossfit_shuffled_caught",
    ):
        if key in t1:
            tier1[key] = {str(k): v for k, v in t1[key].items()}
    if "fold_spread" in t1:
        tier1["fold_spread"] = t1["fold_spread"].to_dict(orient="records")
    return {"tier1": tier1, "tier2": {k: v for k, v in t2.items() if k != "amoc"}}


def headline_metrics(r: dict) -> dict[str, float]:
    """Flat metric dict for MLflow: tier-1 values with intervals, both baselines, both victim rows."""
    out: dict[str, float] = {}
    for prefix, t1 in (("headline", r["t1"]), ("click_only", r["t1_click"])):
        for k, v in t1["summary"].items():
            if isinstance(v, dict):
                out[f"{prefix}.{k}"] = v["value"]
                out[f"{prefix}.{k}.lo"] = v["lo"]
                out[f"{prefix}.{k}.hi"] = v["hi"]
        for base in ("random", "shuffled"):
            for k, v in t1[base].items():
                out[f"{prefix}.{base}.{k}"] = v
    for pt in r["t1"]["proxy"]["points"]:
        out[f"headline.proxy_caught_at_{pt['budget_per_hour']:g}_per_h"] = pt.get(
            "caught", 0
        )
    for pt in r["t1"]["proxy"].get("crossfit", []):
        b = f"{pt['budget_per_hour']:g}"
        out[f"headline.crossfit_caught_at_{b}_per_h"] = pt["caught"]
        out[f"headline.crossfit_alarms_per_h_at_{b}_per_h"] = pt["fa_per_hour"]
    spread = r["t1"].get("fold_spread")
    if spread is not None and len(spread):
        for col in spread.columns.drop(["fold", "clips"]):
            out[f"fold_spread.{col}.min"] = spread[col].min()
            out[f"fold_spread.{col}.max"] = spread[col].max()
    out["tier2.normal_hours"] = r["t2"]["normal_hours"]
    out["tier2.min_verifiable_per_hour"] = r["t2"]["min_verifiable_per_hour"]
    return {k: float(v) for k, v in out.items() if v is not None and np.isfinite(v)}


def training_run_id(tracking, ckpt: Path, experiment: str) -> str:
    """Id of the train_v2 run that wrote this checkpoint (matched by file hash), or "none found".

    Re-scoring runs (--load-dir) carry the same hash; the earliest run is the one that trained it.
    """
    from hydro_knight.eval import run_meta

    sha = run_meta.file_sha256(ckpt)
    if sha == "missing":
        return "none found"
    runs = tracking.find_runs(experiment, f"tags.checkpoint_sha256 = '{sha}'")
    return runs[-1].info.run_id if runs else "none found"


def log_seed(
    tracking, args, seed: int, cfg: P.ProtocolConfig, r: dict, files: list[Path]
) -> dict[str, float]:
    """One nested MLflow child run per seed: params, the 94 headline metrics, three figures, the report files."""
    from hydro_knight.eval import figures as F
    from hydro_knight.eval import run_meta

    ckpt = args.run / f"model_{args.size}_s{seed}.pt"
    if not ckpt.exists():  # K-fold: the training run is found by its fold-0 checkpoint
        ckpt = args.run / f"model_{args.size}_s{seed}_f0.pt"
    params = {
        "run_folder": str(args.run),
        "size": args.size,
        "seed": seed,
        "onset_alt": str(args.onset_alt),
        "tracks_dir": str(args.tracks_dir),
        "normal": r["normal_label"],
        "dropped_clips": ",".join(args.drop) or "none",
        **{f"cfg.{k}": str(v) for k, v in cfg.__dict__.items()},
    }
    tags = run_meta.standard_tags(
        "eval_protocol",
        model_family="tcn_ae_v2",
        size=args.size,
        seed=seed,
        run_folder=args.run.name,
        checkpoint_sha256=run_meta.file_sha256(ckpt),
        trained_by_run_id=training_run_id(tracking, ckpt, args.train_experiment),
        trained_by_experiment=args.train_experiment,
    )
    metrics = headline_metrics(r)
    t1 = r["t1"]
    with tracking.run(f"s{seed}", nested=True, tags=tags):
        tracking.log_config(params)
        tracking.log_metrics(metrics)
        tracking.log_figure(
            F.victim_timelines(r["table"], r["truths"], cfg, t1["per_clip"]),
            "figures/victim_timelines.png",
        )
        tracking.log_figure(
            F.catch_vs_false_alarms(
                t1["proxy"], t1["proxy_random_caught"], t1["proxy_shuffled_caught"]
            ),
            "figures/catch_vs_false_alarms.png",
        )
        tracking.log_figure(
            F.per_clip_results(t1["per_clip"]), "figures/per_clip_results.png"
        )
        for f in files:
            tracking.log_artifact(f, "report")
    return metrics


def log_parent_summary(tracking, args, seed_metrics: list[dict]) -> None:
    """Parent run: mean and SD over seeds, plus a comparison against every evaluated run so far."""
    from hydro_knight.eval import figures as F
    from hydro_knight.eval import run_meta

    tracking.log_metrics(run_meta.mean_sd_metrics(seed_metrics))
    runs = run_meta.flatten_search_runs(tracking.runs_frame(args.mlflow_experiment))
    tracking.log_figure(
        F.cross_run_comparison(runs), "figures/cross_run_within_clip_auc.png"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run", type=Path)
    ap.add_argument("--size", default="small")
    ap.add_argument("--out", type=Path, help="default: <run>/protocol")
    ap.add_argument("--onset-alt", type=Path)
    ap.add_argument("--normal", type=Path)
    ap.add_argument("--pseudo-normal", action="store_true")
    ap.add_argument("--drop", nargs="*", default=[], help="clip ids to leave out")
    ap.add_argument("--compare", type=Path, help="baseline run A; reports run - A")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-random", type=int, default=20)
    ap.add_argument(
        "--tracks-dir",
        type=Path,
        default=Path("data/keypoints_tiled"),
        help="the track tables the model was fed (for the model-free 'victim detected' row)",
    )
    ap.add_argument(
        "--mlflow",
        action="store_true",
        help="log a parent run + one child per seed (needs --extra tracking)",
    )
    ap.add_argument("--mlflow-experiment", default="eval-protocol")
    ap.add_argument(
        "--train-experiment",
        default="train-v2",
        help="where to look up the training run of each checkpoint",
    )
    args = ap.parse_args()
    cfg = P.ProtocolConfig(n_boot=args.n_boot, n_random=args.n_random)
    records = Manifest(Path("data/manifests/pool_footage.jsonl")).load()
    out = args.out or args.run / "protocol"
    out.mkdir(parents=True, exist_ok=True)
    if args.compare:
        md = compare(args.compare, args.run, args.size, records, args, cfg)
        (out / f"compare_vs_{args.compare.name}.md").write_text(md)
        print(md)
        return
    with contextlib.ExitStack() as stack:
        tracking = None
        if args.mlflow:
            from hydro_knight.eval import run_meta, tracking  # imports mlflow

            stack.enter_context(
                tracking.run(
                    f"{args.run.name}-{args.size}",
                    experiment=args.mlflow_experiment,
                    tags=run_meta.standard_tags(
                        "eval_protocol",
                        model_family="tcn_ae_v2",
                        size=args.size,
                        run_folder=args.run.name,
                    ),
                    note=(
                        f"**eval_protocol** on `{args.run}` ({args.size}): one child "
                        "run per seed with tier-1/tier-2 metrics, the victim rank "
                        "timelines, catch vs false alarms against the random and "
                        "track-shuffled baselines, and per-clip results. Each child's "
                        "`trained_by_run_id` tag points at the train_v2 run that made "
                        "its checkpoint."
                    ),
                )
            )
        seed_metrics = run_seeds(args, records, cfg, out, tracking)
        if tracking is not None and seed_metrics:
            log_parent_summary(tracking, args, seed_metrics)
    print(f"wrote {out}")


def run_seeds(args, records, cfg, out: Path, tracking) -> list[dict]:
    """Evaluate, write and (optionally) log every seed in the run folder; returns each seed's metrics."""
    seed_metrics = []
    for seed in seeds_in(args.run, args.size):
        r = evaluate_seed(args.run, args.size, seed, records, args, cfg)
        md = P.format_report(
            f"{args.run.name} {args.size} seed {seed}", r["t1"], r["t2"], cfg
        )
        md = md.replace(
            "(false alarms on normal footage)", f"(false alarms on {r['normal_label']})"
        )
        variants = {
            "Click-matched tracks + path fragments (headline)": r["t1"],
            "Click-matched tracks only, fragment rows dropped": r["t1_click"],
        }
        md += (
            "\n## How much the result depends on the victim definition\n\n"
            + P.format_variants(variants, cfg)
            + "\n"
        )
        (out / f"report_{args.size}_s{seed}.md").write_text(md)
        r["t1"]["per_clip"].to_csv(
            out / f"tier1_per_clip_{args.size}_s{seed}.csv", index=False
        )
        r["t1_click"]["per_clip"].to_csv(
            out / f"tier1_per_clip_click_only_{args.size}_s{seed}.csv", index=False
        )
        r["t2"]["amoc"].to_csv(out / f"tier2_amoc_{args.size}_s{seed}.csv", index=False)
        if "fold_spread" in r["t1"]:
            r["t1"]["fold_spread"].to_csv(
                out / f"tier1_fold_spread_{args.size}_s{seed}.csv", index=False
            )
        (out / f"summary_{args.size}_s{seed}.json").write_text(
            json.dumps(jsonable(r["t1"], r["t2"]), indent=1, default=str)
        )
        if tracking is not None:
            files = sorted(out.glob(f"*_{args.size}_s{seed}.*"))
            seed_metrics.append(log_seed(tracking, args, seed, cfg, r, files))
        print(md)
    return seed_metrics


if __name__ == "__main__":
    main()
