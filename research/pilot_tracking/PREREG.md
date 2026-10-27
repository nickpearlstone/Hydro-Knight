# Preregistration: offline relinking of track fragments (pilot, idea 2)

Written 2026-10-26 22:35, before any relinking code was run. Nick's pick: idea 2 only,
click-based scoring, detector-agnostic code. Track birth settings stay as today
(`TrackSettings()`), so any change comes from relinking alone.

## What is tested

An offline pass over a finished track table. When a track ends and another starts shortly
after, close to the first one's **last observed box** (OC-SORT's observation-centric
recovery, not the Kalman prediction) and at a similar size, the two get the same id. The
candidates are joined best first (smallest distance in box heights), one to one, and the
joins are chained. Settings live in `RelinkSettings`: maximum gap G (s), maximum distance R
(box heights of the earlier track's last box), maximum height ratio S (default 1.5).

Input is any track table with `frame, track_id, bx1, by1, bx2, by2` (confidence optional, never
used); `track_id < 0` is treated as unassigned. Nothing assumes pose keypoints or YOLO's
confidence scale, so head tracks can be run through the same code later.

## Data

The 46 labelled, active wave-pool clips (51 labelled minus the 5 reviewed "not good"),
today's tracks in `data/keypoints_tiled/`. Victim clicks are converted with
`ClipRecord.to_file_time()`. Only clicks at or after the trim start and at or before guard
contact are scored; clicks on the same frame count once.

## Metrics

A click is **held** by a track if one of that track's boxes, within 2 frames of the click,
contains the click or has its center within 0.75 box heights (the matcher's rule in
`eval/victims.py`).

- **Primary: continuity.** Per clip, the largest share of the clip's scored clicks held by
  one single track id. Averaged over clips (each clip weighs the same), with a 95% interval
  from resampling whole clips (2,000 resamples), and counts of clips that got better, stayed
  the same or got worse. This is the objective's "keep following them" from the moment
  trouble starts to guard contact.
- **Secondary: coverage with gap filling.** Relinking alone never changes which clicks hold
  a box, so coverage changes only if the gap between two joined pieces is filled with a
  straight-line box (OC-SORT's virtual trajectory). Reported both ways: observed boxes only
  (unchanged by construction, a sanity check) and with filled gaps.
- **For comparison:** the existing "victim detected (input tracks)" share from
  `eval/protocol.py` (victim tracks re-matched on the new ids with `match_victim_tracks` plus
  `near_path_tracks`), before and after relinking.

## Baselines, scored the same way

1. **Today's tracks** (no relinking).
2. **Random relinking:** the same number of joins per clip as the method makes, but each
   ended track joined to a random track that starts within the same gap window G, at any
   distance. If this raises victim continuity about as much, the gain comes from merging more,
   not from merging right.
3. **Random paths:** the victim's click path moved by a random offset that keeps every point
   inside the clip's pool zone (`zones.json`), 20 draws per clip; and **time-shifted paths**:
   the same click positions moved to a random other moment of the clip's scored footage, 20
   draws. These say what continuity a point that is not the victim gets, before and after
   relinking.

## Gaming check: artificial gaps (decides the configuration)

Merging everything raises continuity. To measure wrong joins without labels: in each clip,
take every confirmed track lasting at least 3 s, delete a stretch of g seconds from its
middle (g drawn from 0.3 to 3 s, uniform), and give the second part a new id. Run relinking
on the whole cut table. For each cut track: **correct** if the two parts are rejoined with
each other, **wrong** if either part is joined to any other track, **missed** otherwise.
Wrong-join rate = wrong / cut tracks. Two seeds.

**Choosing the configuration before looking at victims:** over the grid G in {0.5, 1, 2, 4}
s and R in {0.5, 1, 2}, the primary configuration is the one with the most correct rejoins
among those with a wrong-join rate at or below 5% in the artificial-gap check. This uses no
victim labels, so the reported victim result is not tuned on the victims. The full grid is
also reported on the victims, marked as exploratory.

## What counts as success or failure

- **Success:** at the preregistered configuration, mean victim continuity rises, the 95%
  interval of the per-clip difference (relinked minus today) excludes 0, more clips improve
  than get worse, and the gain is larger than random relinking's gain on the same clips.
- **Failure:** any of those does not hold, or no configuration meets the 5% wrong-join bar.
- Also reported whatever the outcome: the wrong-join rate at the chosen configuration, and the
  random-path and time-shifted continuity before and after.

## Known limits, stated now

- Clicks are about 1 to 3 s apart, so continuity scores the moments a human looked at.
- The artificial gaps come from long confident tracks, which are easier than a victim
  going under. The real wrong-join rate on victims may be higher.
- Pre-contact only, but relinking may use any frame of the clip (offline). An online
  version could only look backward; G bounds how far forward this one looks.
