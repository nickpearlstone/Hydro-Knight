"""Tests for the two-tier evaluation protocol (eval/protocol.py)."""

import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from hydro_knight.eval import protocol as P

FAST = P.ProtocolConfig(n_boot=50, n_random=5, n_thresholds=40)


def make_table(
    victim_score=1.0,
    n_tracks=10,
    clip="c1",
    t_end=40.0,
    seed=0,
    victim_from=15.0,
    victim_to=30.0,
):
    """Ten swimmers scored every 0.5 s with noise in [0, 0.5); track 1 is the victim and
    scores `victim_score` between victim_from and victim_to."""
    rng = np.random.default_rng(seed)
    rows = []
    for tid in range(1, n_tracks + 1):
        for t in np.arange(0, t_end, 0.5):
            s = rng.random() * 0.5
            if tid == 1 and victim_from <= t <= victim_to:
                s = victim_score
            rows.append((clip, tid, float(t), float(s)))
    return pd.DataFrame(rows, columns=P.SCORE_COLUMNS)


def truth(clip="c1", onset=20.0, contact=30.0, alt=None, silent=False):
    return P.EventTruth(
        clip, 0.0, onset, contact, frozenset({1}), onset_alt=alt, silent=silent
    )


def test_garwood_zero_alarms_matches_chi_square():
    rate, lo, hi = P.garwood(0, 1.0)
    assert rate == 0 and lo == 0
    assert hi == pytest.approx(3.689, abs=1e-3)  # two-sided 95%
    assert P.zero_alarm_bound(1.0) == pytest.approx(2.996, abs=1e-3)  # one-sided ~3/h
    assert P.garwood(3, 0.0)[2] == math.inf


def test_garwood_interval_brackets_rate():
    rate, lo, hi = P.garwood(5, 2.0)
    assert rate == 2.5 and lo < rate < hi


def test_clopper_pearson_edges():
    p, lo, hi = P.clopper_pearson(0, 25)
    assert p == 0 and lo == 0 and hi == pytest.approx(0.137, abs=1e-3)
    assert P.clopper_pearson(25, 25)[2] == 1.0
    assert math.isnan(P.clopper_pearson(0, 0)[0])


def test_onset_tolerance_early_alarm_is_a_catch_but_too_early_is_false():
    tr = truth(onset=20.0)
    early = pd.DataFrame([("c1", 1, 17.0, 1.0)], columns=P.SCORE_COLUMNS)
    assert P.first_catch(early, tr, 0.5, onset_tol_s=5.0) == pytest.approx(-3.0)
    assert P.false_alarms(early, 0.5, truth=tr, onset_tol_s=5.0) == 0
    too_early = pd.DataFrame([("c1", 1, 14.0, 1.0)], columns=P.SCORE_COLUMNS)
    assert P.first_catch(too_early, tr, 0.5, onset_tol_s=5.0) is None
    assert P.false_alarms(too_early, 0.5, truth=tr, onset_tol_s=5.0) == 1


def test_independent_onset_widens_the_catch_window():
    rows = pd.DataFrame([("c1", 1, 13.0, 1.0)], columns=P.SCORE_COLUMNS)
    assert P.first_catch(rows, truth(onset=20.0), 0.5, 5.0) is None
    assert P.first_catch(rows, truth(onset=20.0, alt=17.0), 0.5, 5.0) == pytest.approx(
        -7.0
    )


def test_after_contact_is_excluded():
    tr = truth(contact=30.0)
    rows = pd.DataFrame(
        [("c1", 2, 31.0, 1.0), ("c1", 1, 31.0, 1.0)], columns=P.SCORE_COLUMNS
    )
    assert P.false_alarms(rows, 0.5, truth=tr) == 0
    assert P.first_catch(rows, tr, 0.5, 5.0) is None


def test_false_alarm_episodes_are_per_track_and_merge_short_gaps():
    span = P.NormalSpan("n", 0.0, 100.0)
    two_tracks = pd.DataFrame(
        [("n", 1, 5.0, 1.0), ("n", 2, 5.0, 1.0)], columns=P.SCORE_COLUMNS
    )
    assert P.false_alarms(two_tracks, 0.5, span=span) == 2
    short_gap = pd.DataFrame(
        [("n", 1, 5.0, 1.0), ("n", 1, 6.0, 1.0)], columns=P.SCORE_COLUMNS
    )
    assert P.false_alarms(short_gap, 0.5, span=span, merge_gap_s=2.0) == 1
    long_gap = pd.DataFrame(
        [("n", 1, 5.0, 1.0), ("n", 1, 8.0, 1.0)], columns=P.SCORE_COLUMNS
    )
    assert P.false_alarms(long_gap, 0.5, span=span, merge_gap_s=2.0) == 2


def test_rank_timeline_marks_missing_victim():
    rows = make_table(victim_from=25.0, victim_to=30.0)
    rows = rows[
        ~((rows.track_id == 1) & (rows.t < 25.0))
    ]  # victim not tracked before 25 s
    tl = P.victim_rank_timeline(rows, truth(), FAST)
    assert tl.loc[tl.t < 24.0, "rank"].isna().all()
    assert (tl.loc[tl.t >= 26.0, "rank"] == 1).all()


def test_tier1_perfect_model_beats_random():
    table = make_table(victim_score=1.0)
    out = P.tier1(table, {"c1": truth()}, FAST)
    s, r = out["summary"], out["random"]
    assert s["coverage"]["value"] == pytest.approx(1.0)
    assert s["within_clip_auc"]["value"] > 0.9
    assert s["time_in_top1"]["value"] > 0.9
    assert r["time_in_top1"] < 0.5
    assert s["reach_top1"]["value"] == 1.0
    assert out["proxy"]["hours"] == pytest.approx(30 / 3600)


def test_tier1_per_clip_has_one_row_per_truth_even_without_scores():
    table = make_table(clip="c1")
    out = P.tier1(table, {"c1": truth("c1"), "c2": truth("c2")}, FAST)
    pc = out["per_clip"].set_index("clip_id")
    assert pc.loc["c2", "coverage"] == 0.0
    assert math.isnan(pc.loc["c2", "best_rank"])


def test_tier2_verifies_only_budgets_the_footage_can_prove():
    rescue = make_table(victim_score=1.0)
    normal = make_table(victim_score=0.2, clip="n").assign(
        score=lambda d: d.score * 0.5
    )
    spans = [P.NormalSpan("n", 0.0, 40.0)]
    cfg = P.ProtocolConfig(n_thresholds=60, budgets_per_hour=(0.1, 1000.0))
    # 40 s of normal footage cannot prove anything below ~270 alarms/hour
    out = P.tier2(pd.concat([rescue, normal]), {"c1": truth()}, spans, cfg)
    assert out["min_verifiable_per_hour"] == pytest.approx(
        P.zero_alarm_bound(40 / 3600)
    )
    low, high = out["points"]
    assert low["verified"] is False
    assert high["caught"] == 1 and high["verified"] is True
    assert high["alerts_per_day"] == pytest.approx(
        high["fa_per_hour"] * cfg.open_hours_per_day
    )
    assert not out["amoc"].empty


def test_tier2_without_normal_footage_verifies_nothing():
    out = P.tier2(make_table(), {"c1": truth()}, [], FAST)
    assert out["normal_hours"] == 0
    assert all(not p["verified"] for p in out["points"])


def test_paired_difference_prefers_the_better_model():
    truths = {f"c{i}": truth(f"c{i}") for i in range(6)}
    good = pd.concat(
        [make_table(victim_score=1.0, clip=c, seed=i) for i, c in enumerate(truths)]
    )
    bad = pd.concat(
        [make_table(victim_score=0.0, clip=c, seed=i) for i, c in enumerate(truths)]
    )
    pc_good = P.tier1(good, truths, FAST)["per_clip"].set_index("clip_id")
    pc_bad = P.tier1(bad, truths, FAST)["per_clip"].set_index("clip_id")

    def stat(pc):
        return lambda ids: float(pc.loc[ids, "time_in_top3"].mean())

    d = P.paired_difference(list(truths), stat(pc_bad), stat(pc_good), n_boot=100)
    assert d["diff"] > 0.5 and d["p_b_better"] == 1.0


def test_truths_from_records_converts_source_time():
    rec = SimpleNamespace(
        clip_id="c1",
        trim_start=12.0,
        events=[{"start": 20.0, "contact": 30.0}],
        to_file_time=lambda t: t - 10.0,  # file starts 10 s into the source
    )
    no_contact = SimpleNamespace(
        clip_id="c2", trim_start=None, events=[{"start": 5.0}], to_file_time=lambda t: t
    )
    out = P.truths_from_records(
        [rec, no_contact],
        ["c1", "c2"],
        {"c1": [4, 5]},
        onset_alt={"c1": 18.0},
        silent=["c1"],
    )
    assert set(out) == {"c1"}
    t = out["c1"]
    assert (t.start, t.onset, t.contact, t.onset_alt) == (2.0, 10.0, 20.0, 8.0)
    assert t.victim_tracks == frozenset({4, 5}) and t.silent and t.onset_lo == 8.0


def test_pre_onset_spans_stop_before_the_catch_window():
    spans = P.pre_onset_spans({"c1": truth(onset=20.0, alt=18.0)}, onset_tol_s=5.0)
    assert spans == [P.NormalSpan("c1", 0.0, 13.0)]
    assert P.pre_onset_spans({"c1": truth(onset=3.0)}, onset_tol_s=5.0) == []


def test_format_report_renders_both_tiers():
    table = make_table()
    t1 = P.tier1(table, {"c1": truth()}, FAST)
    t2 = P.tier2(table, {"c1": truth()}, [], FAST)
    md = P.format_report("demo", t1, t2, FAST)
    assert "Tier 1" in md and "Tier 2" in md and "Lowest alarm rate" in md
