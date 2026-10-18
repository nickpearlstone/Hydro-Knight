"""
Score a saved run with the two-tier evaluation protocol (src/hydro_knight/eval/protocol.py).

Reads the per-window alarm tables and dataset reports that `scripts/train_v2.py --detection`
saves, so nothing is retrained. Writes, per seed, a markdown report, the tier-1 per-clip
table and a JSON summary into --out.

    uv run python scripts/eval_protocol.py runs/v2_ckpt_after
    uv run python scripts/eval_protocol.py runs/v2_ckpt_after --pseudo-normal      # tier 2 on pre-onset footage
    uv run python scripts/eval_protocol.py runs/v2_ckpt_after --onset-alt scratch/z5cc/label_compare.json
    uv run python scripts/eval_protocol.py runs/RUN_B --compare runs/RUN_A          # is B better than A?

--onset-alt takes either {clip_id: onset in source seconds} or the Z5cc label comparison
rows (clip_id, his_onset, our_onset); only onsets within 30 s of ours are used.
--normal takes a JSON list of {clip_id, start, end} incident-free spans (file seconds)
whose scores are in the run's alarm tables; until such footage exists, tier 2 reports
that no budget can be verified, and --pseudo-normal gives a clearly labelled stopgap.
"""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from hydro_knight.eval import protocol as P
from hydro_knight.ingest.manifest import Manifest


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
    t1 = P.tier1(table, truths, cfg)
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
    t1_click = P.tier1(table, truths_click, cfg)
    return {
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
        for col in ("within_clip_auc", "time_in_top3", "coverage"):
            stat = partial(_clip_mean, col=col)
            d = P.paired_difference(
                shared, stat(a), stat(b), cfg.n_boot, cfg.seed, cfg.conf
            )
            lines.append(
                f"| {seed} | {col} | {d['diff']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}] | {d['p_b_better']:.2f} |"
            )
    return "\n".join(lines) + "\n"


def jsonable(t1: dict, t2: dict) -> dict:
    return {
        "tier1": {
            "summary": t1["summary"],
            "random": t1["random"],
            "proxy": t1["proxy"],
            "proxy_random_caught": {
                str(k): v for k, v in t1["proxy_random_caught"].items()
            },
        },
        "tier2": {k: v for k, v in t2.items() if k != "amoc"},
    }


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
        (out / f"summary_{args.size}_s{seed}.json").write_text(
            json.dumps(jsonable(r["t1"], r["t2"]), indent=1, default=str)
        )
        print(md)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
