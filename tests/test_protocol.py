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
    assert s["scored"]["value"] == pytest.approx(1.0)
    assert s["within_clip_auc"]["value"] > 0.9
    assert s["time_in_top1"]["value"] > 0.9
    assert r["time_in_top1"] < 0.5
    assert s["reach_top1"]["value"] == 1.0
    assert out["proxy"]["hours"] == pytest.approx(30 / 3600)


def test_tier1_per_clip_has_one_row_per_truth_even_without_scores():
    table = make_table(clip="c1")
    out = P.tier1(table, {"c1": truth("c1"), "c2": truth("c2")}, FAST)
    pc = out["per_clip"].set_index("clip_id")
    assert pc.loc["c2", "scored"] == 0.0
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


def test_ignored_tracks_count_as_nothing():
    """A fragment track that fires hard is neither a false alarm, a rival in the rank, nor in the AUC."""
    table = make_table(victim_score=0.9)
    frag = make_table(n_tracks=1).assign(track_id=99, score=1.0)
    both = pd.concat([table, frag])
    keep = {"c1": truth()}
    drop = {
        "c1": P.EventTruth(
            "c1", 0.0, 20.0, 30.0, frozenset({1}), ignore_tracks=frozenset({99})
        )
    }
    plain, ignored = P.tier1(both, keep, FAST), P.tier1(both, drop, FAST)
    # the fragment outranks the victim everywhere unless it is ignored
    assert plain["summary"]["time_in_top1"]["value"] == 0.0
    assert ignored["summary"]["time_in_top1"]["value"] > 0.9
    assert (
        ignored["summary"]["within_clip_auc"]["value"]
        > plain["summary"]["within_clip_auc"]["value"]
    )
    span = [P.NormalSpan("c1", 0.0, 10.0)]
    g_keep = P._drop_ignored(P._by_clip(both), keep)["c1"]
    g_drop = P._drop_ignored(P._by_clip(both), drop)["c1"]
    assert P.false_alarms(g_keep, 0.95, span=span[0]) == 1
    assert P.false_alarms(g_drop, 0.95, span=span[0]) == 0


def test_truths_from_records_carries_ignored_tracks():
    rec = SimpleNamespace(
        clip_id="c1",
        trim_start=None,
        events=[{"start": 5.0, "contact": 9.0}],
        to_file_time=lambda t: t,
    )
    out = P.truths_from_records(
        [rec], ["c1"], {"c1": [1]}, ignore_tracks={"c1": [7, 8]}
    )
    assert out["c1"].ignore_tracks == frozenset({7, 8})


def test_format_variants_one_row_per_definition():
    t1 = P.tier1(make_table(), {"c1": truth()}, FAST)
    md = P.format_variants({"headline": t1, "click only": t1}, FAST)
    assert md.count("\n") == 3 and "click only" in md


def _tied_table(n_tracks, score=0.0, clip="c1"):
    rows = [
        (clip, tid, float(t), score)
        for tid in range(1, n_tracks + 1)
        for t in np.arange(0, 40, 0.5)
    ]
    return pd.DataFrame(rows, columns=P.SCORE_COLUMNS)


@pytest.mark.parametrize("n", [2, 20])
def test_all_tied_scores_give_chance_not_a_free_win(n):
    """A detector that scores everyone the same is chance on time-in-top-k and never reaches it outright."""
    out = P.tier1(_tied_table(n), {"c1": truth()}, FAST)
    s = out["summary"]
    assert s["time_in_top3"]["value"] == pytest.approx(min(3 / n, 1.0))
    assert s["time_in_top1"]["value"] == pytest.approx(1 / n)
    assert s["reach_top3"]["value"] == (
        1.0 if n <= 2 else 0.0
    )  # with 2 swimmers, rank 2 <= 3 outright
    assert out["per_clip"].loc[0, "best_rank"] == n


def test_partial_tie_worked_example():
    """2 swimmers higher, victim tied with 5 others: chance of top 3 = (3 - 2) / 6."""
    rows = [("c1", 1, 25.0, 0.5)]
    rows += [("c1", tid, 25.0, 0.9) for tid in (2, 3)]
    rows += [("c1", tid, 25.0, 0.5) for tid in (4, 5, 6, 7, 8)]
    rows += [("c1", tid, 25.0, 0.1) for tid in (9, 10)]
    tl = P.victim_rank_timeline(
        pd.DataFrame(rows, columns=P.SCORE_COLUMNS), truth(), FAST
    )
    at = tl[(tl.t >= 25.0) & (tl.t < 27.0)].iloc[0]
    assert at["p_top3"] == pytest.approx(1 / 6)
    assert at["p_top1"] == 0.0 and at["p_top5"] == pytest.approx(3 / 6)
    assert at["rank"] == 8  # outright: 2 higher + 5 tied ahead + the victim


def test_distinct_scores_unchanged_by_tie_handling():
    rows = [("c1", tid, 25.0, s) for tid, s in [(1, 0.7), (2, 0.9), (3, 0.5), (4, 0.3)]]
    tl = P.victim_rank_timeline(
        pd.DataFrame(rows, columns=P.SCORE_COLUMNS), truth(), FAST
    )
    at = tl[(tl.t >= 25.0) & (tl.t < 27.0)].iloc[0]
    assert at["rank"] == 2 and at["p_top1"] == 0.0 and at["p_top3"] == 1.0


def _binary_rule(fires_on_victim_only: bool, rate=0.05, n_tracks=10, seed=3, clip="c1"):
    """A 0/1 detector: fires only on the victim during distress, or on random swimmer-moments."""
    rng = np.random.default_rng(seed)
    rows = []
    for tid in range(1, n_tracks + 1):
        for t in np.arange(0, 40, 0.5):
            if fires_on_victim_only:
                s = 1.0 if (tid == 1 and 18.0 <= t <= 30.0) else 0.0
            else:
                s = float(rng.random() < rate)
            rows.append((clip, tid, float(t), s))
    return pd.DataFrame(rows, columns=P.SCORE_COLUMNS)


def test_shuffle_moves_whole_track_sequences_at_offset_zero():
    """Equal-length tracks: each track gets exactly one other track's whole sequence, from its start."""
    table = pd.concat([make_table(clip="c1"), make_table(clip="c2", seed=1)])
    groups = P._by_clip(table)
    sh = P.shuffle_within_clip(groups, np.random.default_rng(0))
    for c, g in groups.items():
        out = sh[c]
        assert np.array_equal(out["t"].to_numpy(), g["t"].to_numpy())  # rows untouched
        orig = {
            k: v.sort_values("t")["score"].to_numpy() for k, v in g.groupby("track_id")
        }
        used = []
        for _, v in out.groupby("track_id"):
            new = v.sort_values("t")["score"].to_numpy()
            match = [j for j, o in orig.items() if np.array_equal(new, o)]
            assert len(match) == 1, "not one source sequence, unshifted"
            used.append(match[0])
        assert sorted(used) == sorted(orig)  # a permutation: every sequence used once
        assert any(u != k for u, k in zip(used, sorted(orig), strict=True))


def test_shuffle_cuts_long_and_repeats_short_sequences():
    """Two tracks of different lengths swapped: the long one is cut, the short one repeated from its start."""
    rows = [("c1", 1, float(t), float(t)) for t in range(6)]  # track 1: 0..5
    rows += [("c1", 2, float(t), 100.0 + t) for t in range(3)]  # track 2: 100..102
    groups = P._by_clip(pd.DataFrame(rows, columns=P.SCORE_COLUMNS))
    for seed in range(20):  # find a draw that swaps the two tracks
        out = P.shuffle_within_clip(groups, np.random.default_rng(seed))["c1"]
        t1 = out[out.track_id == 1].sort_values("t")["score"].to_numpy()
        if t1[0] == 100.0:
            break
    t2 = out[out.track_id == 2].sort_values("t")["score"].to_numpy()
    assert t1.tolist() == [100, 101, 102, 100, 101, 102]  # short source repeated
    assert t2.tolist() == [0, 1, 2]  # long source cut, from its start


def test_rule_that_fires_on_the_victim_beats_its_shuffle():
    # with offset 0 the victim gets its own sequence back in about 1 draw in 10, so average many
    cfg = P.ProtocolConfig(n_boot=20, n_random=50, n_thresholds=20)
    out = P.tier1(_binary_rule(True), {"c1": truth()}, cfg)
    assert out["summary"]["time_in_top1"]["value"] > 0.8
    assert out["shuffled"]["time_in_top1"] < 0.3


def test_rule_that_fires_at_random_matches_its_shuffle_but_not_noise():
    """A sparse 0/1 rule with no skill: its shuffle agrees with it, noise does not.

    Noise never ties, so by luck it puts the victim in the top 3 outright at some moment in
    almost every clip; a 0/1 rule mostly ties everyone at 0 and only gets there when it fires.
    """
    clips = [f"c{i}" for i in range(20)]
    table = pd.concat(
        [_binary_rule(False, rate=0.02, seed=i, clip=c) for i, c in enumerate(clips)]
    )
    out = P.tier1(
        table,
        {c: truth(c) for c in clips},
        P.ProtocolConfig(n_boot=20, n_random=10, n_thresholds=20),
    )
    model = out["summary"]["reach_top3"]["value"]
    assert abs(model - out["shuffled"]["reach_top3"]) < 0.15
    assert out["random"]["reach_top3"] - model > 0.3


def test_victim_detected_but_too_short_to_be_scored():
    """The input holds a 1 s victim track, but the model never scored it (shorter than its warm-up)."""
    scores = make_table()
    scores = scores[scores.track_id != 1]  # the model emitted nothing for the victim
    tracks = pd.DataFrame({"t": np.arange(22.0, 23.0, 1 / 30), "track_id": 1})
    out = P.tier1(scores, {"c1": truth()}, FAST, tracks={"c1": tracks})
    s = out["summary"]
    assert s["detected"]["value"] > 0  # 1 s of track inside a 2 s trailing window
    assert s["scored"]["value"] == 0.0  # "scored by this model"
    # 1 s track at 22-23 s, trailing 2 s window: moments 22.0 .. 24.5 see it (6 of 31)
    assert s["detected"]["value"] == pytest.approx(6 / 31)


def test_detected_is_empty_without_a_track_table():
    out = P.tier1(make_table(), {"c1": truth()}, FAST)
    assert math.isnan(out["summary"]["detected"]["value"])
    assert "Victim detected (input tracks" in P.format_report("x", out, None, FAST)


def _burst_rule(seed, clip, n_tracks=10, burst_steps=6, on_share=0.02):
    """A no-skill 0/1 rule that alarms in 3 s bursts (6 steps) on random swimmers."""
    rng = np.random.default_rng(seed)
    times = np.arange(0, 40, 0.5)
    rows = []
    for tid in range(1, n_tracks + 1):
        on = np.zeros(len(times))
        for st in np.flatnonzero(rng.random(len(times)) < on_share / burst_steps):
            on[st : st + burst_steps] = 1.0
        rows += [
            (clip, tid, float(t), float(v)) for t, v in zip(times, on, strict=True)
        ]
    return pd.DataFrame(rows, columns=P.SCORE_COLUMNS)


def _smooth_scores(seed, clip, n_tracks=10, width=20):
    """A no-skill continuous detector whose scores drift slowly (moving average of noise)."""
    rng = np.random.default_rng(seed)
    times = np.arange(0, 40, 0.5)
    rows = []
    for tid in range(1, n_tracks + 1):
        x = np.convolve(
            rng.normal(size=len(times) + width), np.ones(width) / width, "valid"
        )[: len(times)]
        rows += [(clip, tid, float(t), float(v)) for t, v in zip(times, x, strict=True)]
    return pd.DataFrame(rows, columns=P.SCORE_COLUMNS)


@pytest.mark.parametrize("make", [_burst_rule, _smooth_scores])
def test_clumpy_no_skill_detectors_match_their_track_shuffle(make):
    """A row shuffle would scatter each alarm into many blips and look far luckier than these
    detectors; the track shuffle keeps their rhythm, so it agrees with them."""
    clips = [f"c{i}" for i in range(30)]
    table = pd.concat([make(i, c) for i, c in enumerate(clips)])
    out = P.tier1(
        table,
        {c: truth(c) for c in clips},
        P.ProtocolConfig(n_boot=20, n_random=10, n_thresholds=20),
    )
    model = out["summary"]["reach_top3"]["value"]
    assert abs(model - out["shuffled"]["reach_top3"]) <= 0.12
    if (
        make is _burst_rule
    ):  # the old row shuffle, averaged like the protocol does, for contrast
        rng = np.random.default_rng(0)
        groups = P._by_clip(table)
        truths = {c: truth(c) for c in clips}
        old = []
        for _ in range(10):
            rowshuf = {
                c: g.assign(score=rng.permutation(g["score"].to_numpy()))
                for c, g in groups.items()
            }
            pc = P._tier1_per_clip(rowshuf, truths, FAST)
            old.append(P._tier1_summary(pc, FAST)["reach_top3"]["value"])
        assert np.mean(old) - model > 0.15
