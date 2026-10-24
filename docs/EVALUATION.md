# Evaluation protocol

How every Hydro-Knight model is scored. Code: `src/hydro_knight/eval/protocol.py`;
command: `uv run hydroknight eval protocol runs/<run>`.

The protocol has two tiers because the data does. Tier 1 works today with the labelled rescue
clips alone and answers "is this change heading the right way?". Tier 2 is the standard a
deployable system has to meet, and it needs hours of incident-free footage we don't have yet.

## Tier 1 is the proof-of-concept bar

Version 1 is a single-venue proof of concept: all rescue clips come from one wave pool, and
the claim is only that drowning is learnable at this pool. Tier 1 is the bar for that claim,
and a result clears it only if it beats **trivial baselines** as well as the random and
shuffled scores. Within one pool the risk is shortcuts: victims are mostly near inner tubes
and often small, and box size, box confidence and a detected flag are model inputs. So each
window is also scored, on the same clips and the same rules, by:

- the share of missing steps in the window,
- the number of times the track flips between detected and missing,
- 1 minus the mean box confidence,
- negative box height,

and the model is retrained with all pose coordinates removed. These baselines are required
but not yet built into the harness.

## Shared rules

- **Splits: grouped K-fold.** `hydroknight train tcn-v2` gives every labelled clip one of 5
  folds (`--folds`) from a hash of its id and the seed, so a clip keeps its fold as more clips
  are labelled. Each fold's model trains on the other four folds and scores only its own, so
  every clip is tested exactly once per seed, by a model that never saw any of its footage.
  The five held-out score tables are pooled and scored once per seed. Each seed draws new
  folds and new starting weights, and the spread across seeds shows how repeatable a result
  is. The report also lists each fold's own numbers, to show whether one fold carries the
  result. `--holdout` keeps the old single random 20% hold-out.
- **One score scale.** Each fold's model has its own error level, so its scores are divided
  by the median score of its own training windows before pooling. This never reorders
  swimmers within a clip, so it only matters for measures that use one alarm threshold across
  clips.
- **Thresholds.** Any threshold picked on the clips it is then scored on is an upper bound.
  The proxy budget curve and `--detection` report two versions: picked on the same clips
  (upper bound), and picked for each fold on the other folds' clips only, then applied to that
  fold (honest). The feature scaler is fit on training windows only. Choosing a model size by
  its K-fold result and then quoting that same result is mildly optimistic.
- **Input.** Any model reduces to one table: clip, track, time, score. The time is when a live
  system could have raised that score, never earlier.
- **Catch.** A score at or above the alarm threshold on the victim's own track, between the
  onset and guard contact. Onsets are a range: the earlier of our label and an independent
  labeller's (median gap 2.4 s), minus a 5 s tolerance. Alarms after guard contact are ignored.
- **Who the victim is.** The headline counts the tracks matched to the victim clicks plus the
  short fragments along the victim's path between clicks. A second row scores only the
  click-matched tracks and drops the fragments' rows entirely (neither catches, false alarms,
  nor rivals), showing how much a result leans on fragments that could be bystanders.
- **False alarm.** Any other run of alarms, counted per swimmer, so twenty swimmers firing at
  once are twenty false alarms.
- **Baselines and intervals.** A number alone can't say whether it's good: by pure luck a
  guard pointing at 3 of 4 swimmers finds the victim 75% of the time, but only 15% with 20
  swimmers. So every tier-1 number sits next to two zero-skill versions computed on the same
  clips. *Random* replaces every score with noise. *Shuffled* hands each swimmer's whole score
  sequence to a random swimmer in the same clip, starting at the start of that swimmer's track
  (cut if too long, repeated if too short), so it keeps the model's ties, firing rate and rhythm
  (one long alarm stays one long alarm) and changes only who gets it; it is the fair rival for
  0/1 rules and state-machine alarms, which noise can't imitate. Only "reaches top k" and the
  budget curves depend on how this rival is built: the same-clip AUC (chance 0.5) and the
  share of time in the top k (chance k out of N, ties split) have exact chance levels. Each number also gets a
  95% interval from resampling whole clips. Comparing two models uses the same
  resampled clips for both ("B minus A" with an interval and the share of resamples where B wins).
- **Never pooled frame AUC.** Pooling across clips rewards telling clips apart instead of
  finding the victim.

## Tier 1: direction check (available now)

The negatives are the other swimmers in the same clip, about 20 at a time, which is what a
lifeguard actually scans.

| Measure | Question it answers |
|---|---|
| Victim detected (input tracks) | Share of the distress window in which the track table the model was given holds the victim. Model-free: it measures the detector and tracker (pose tracks today, head tracks later via `--tracks-dir`). |
| Victim scored by this model | Share of the distress window in which the model actually produced a score for the victim. Depends on the model too: a window model needs a few seconds of track before its first score, and skips tracks shorter than its minimum. |
| Victim vs other swimmers, same clip | Chance a victim score beats another swimmer's score in the same clip, averaged over clips. |
| Share of window victim is top 1 / 3 / 5 (ties split by chance) | If a guard checked the k most suspicious swimmers, how often would the victim be among them? Ties with other swimmers are broken at random, so a detector that gives everyone the same score gets exactly chance (3 of 20 swimmers = 15%). |
| Clips where victim reaches top k outright | In how many rescues the victim is ever among the k most suspicious before contact, and how soon after onset. Ties count against the victim, so this yes/no measure stays conservative. |
| Proxy budget curve | Catches against false alarms counted on the rescue footage itself, with exact intervals. Crowded and short, so a proxy only. Shown with thresholds picked on these clips (upper bound) and, for K-fold runs, picked on the other folds (honest). |

## Tier 2: full protocol (needs normal footage)

- False alarms are counted **only on incident-free footage**, with exact Poisson intervals,
  and reported per hour and per pool per day (assuming 10 open hours). Commercial systems
  report about 2 to 3 alerts per pool per day.
- Recall at each budget comes with an exact binomial interval; delay is reported from our
  onset and from the earliest onset.
- A budget counts as **verified** only when the upper end of the alarm-rate interval is within
  it. Zero alarms in h hours can only prove a rate of about 3 / h per hour, so the report
  states the lowest rate the available footage can prove.
- A delay curve (average time to alarm, misses counted as the whole window, against false
  alarms per hour) and, once labelled, a separate count for "silent" submersions with no
  visible struggle.

Until normal footage exists, `--pseudo-normal` uses the rescue clips' footage before the catch
window as a clearly labelled stopgap.

## First reading: v2 autoencoder (5-fold, 46 rescues, 3 seeds)

Small v2 model, every labelled clip tested once per seed by a model that never trained on it
(`runs/v2_kfold_small`). Values are seeds 0 / 1 / 2; brackets are the seed-0 95% clip-resampled
interval. Onset ranges include the independent Z5cc onsets where they exist (`--onset-alt`).

| Measure | v2 | Random | Shuffled |
|---|---|---|---|
| Victim detected (input tracks) | 49% / 49% / 49% [40%, 58%] | (not score-based) | |
| Victim scored by this model | 26% / 26% / 26% [19%, 34%] | (not score-based) | |
| Victim vs other swimmers, same clip | 0.73 / 0.73 / 0.73 [0.67, 0.80] | 0.49 to 0.50 | 0.50 to 0.51 |
| Share of window victim is top 3 (ties split by chance) | 7% / 7% / 7% [4%, 10%] | 3 to 4% | 3 to 5% |
| Clips where victim reaches top 3 outright | 50% / 48% / 50% [35%, 63%] | 26% | 23 to 25% |
| Catches at 20 false alarms per hour (proxy, upper bound and honest) | 0 / 0 / 0 | 0 | 0 |

The old single hold-outs (6 to 11 clips each) put the same-clip AUC anywhere from 0.71 to 0.89.
Pooled over all 46 clips it is 0.73 for every seed, so the earlier spread came from which
clips were tested. Single folds still range from 0.56 to 0.89.

The model ranks victims above other swimmers better than chance, but it never catches one at
a usable alarm rate, and most of the distress window is lost before ranking starts. The loss
has two parts of similar size:

- **About half of the window, the input never holds the victim** (detected 49%). That is the
  detector and tracker: the head-detector and low-confidence work targets it.
- **About another quarter, the victim is detected but the model has no score yet** (scored
  26%). The v2 model's first score comes about 4.5 s into a track, and 85% of victim tracks
  are shorter than that (median 1.3 s, 157 tracks in 46 clips). Shorter-window models and stitching track
  fragments back together target this part.

So better detection alone is not enough: the scoring model also has to work on short tracks.
With only click-matched victim tracks (fragments dropped), detected falls to 39% and scored
to 18% (seed 0), and the same-clip AUC stays at 0.72.
