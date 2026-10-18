# Evaluation protocol

How every Hydro-Knight model is scored. Code: `src/hydro_knight/eval/protocol.py`;
command: `uv run python scripts/eval_protocol.py runs/<run>`.

The protocol has two tiers because the data does. Tier 1 works today with the labelled rescue
clips alone and answers "is this change heading the right way?". Tier 2 is the standard a
deployable system has to meet, and it needs hours of incident-free footage we don't have yet.

## Shared rules

- **Input.** Any model reduces to one table: clip, track, time, score. The time is when a live
  system could have raised that score, never earlier.
- **Catch.** A score at or above the alarm threshold on the victim's own track, between the
  onset and guard contact. Onsets are a range: the earlier of our label and an independent
  labeller's (median gap 2.4 s), minus a 5 s tolerance. Alarms after guard contact are ignored.
- **False alarm.** Any other run of alarms, counted per swimmer, so twenty swimmers firing at
  once are twenty false alarms.
- **Baselines and intervals.** Every number is shown next to the same number for random
  scores, with a 95% interval from resampling whole clips. Comparing two models uses the same
  resampled clips for both ("B minus A" with an interval and the share of resamples where B wins).
- **Never pooled frame AUC.** Pooling across clips rewards telling clips apart instead of
  finding the victim.

## Tier 1: direction check (available now)

The negatives are the other swimmers in the same clip, about 20 at a time, which is what a
lifeguard actually scans.

| Measure | Question it answers |
|---|---|
| Victim visible to the model | Share of the distress window in which the model scores the victim at all (detection and tracking health). |
| Victim vs other swimmers, same clip | Chance a victim score beats another swimmer's score in the same clip, averaged over clips. |
| Share of window victim is top 1 / 3 / 5 | If a guard checked the k most suspicious swimmers, how often would the victim be among them? |
| Clips where victim reaches top k | In how many rescues the victim is ever among the k most suspicious before contact, and how soon after onset. |
| Proxy budget curve | Catches against false alarms counted on the rescue footage itself, with exact intervals. Crowded and short, so a proxy only. |

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

## First reading: v2 autoencoder (3 seeds, 6 to 11 held-out rescues each)

Onset ranges include the independent Z5cc onsets where they exist (`--onset-alt`).

| Measure | v2 | Random |
|---|---|---|
| Victim visible to the model | 21 to 27% | (not score-based) |
| Victim vs other swimmers, same clip | 0.71 to 0.89 | about 0.5 |
| Share of window victim is top 3 | 7 to 8% | 1 to 3% |
| Clips where victim reaches top 3 | 50 to 67% | 13 to 31% |
| Catches at 20 false alarms per hour (proxy) | 0 | 0 |

The model ranks victims above other swimmers better than chance, but it sees the victim for
only about a quarter of the distress window and never catches one at a usable alarm rate.
Detection and tracking come first, which is where the project is heading.
