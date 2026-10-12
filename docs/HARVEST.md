# Harvest procedure (for Claude, unattended)

You are running without Nick. Work through candidates one at a time until you have
accepted the target number of clips (default 5, given in the prompt) or the queue is
empty. Every label you write is held out of training until Nick reviews it, so be
accurate, not fast, and **skip anything you are unsure about**: a rejection costs
nothing, a wrong victim costs a review cycle.

Footage shows minors. Images stay in scratch/lbl/ (gitignored). Never commit video,
images, or detections. Do not run git; the wrapper script commits.

## Tools (the only commands you need)

```
uv run python scripts/harvest.py search            # once at the start if `next` says NONE
uv run python scripts/harvest.py next              # next pending candidate (JSON) or NONE
uv run python scripts/harvest.py fetch ID          # download it
uv run python scripts/label_sheets.py CLIP overview                 # 1 fps sheets, 36 s each
uv run python scripts/label_sheets.py CLIP zoom T0 T1 X0 Y0 X1 Y1 [STEP]
uv run python scripts/harvest.py accept ID --kind rescue|normal [--section A B]
uv run python scripts/harvest.py reject ID --reason "..."
uv run python scripts/write_label.py scratch/specs/CLIP.json
uv run python scripts/harvest.py status
```

CLIP is the clip id `fetch` prints. Check the frame size before reading positions: some
videos are 1280x720, not 1920x1080 (zoom coordinates are in the video's own pixels).

## 1. Judge each candidate

`next`, then `fetch`, then `overview`, and Read **every** sheet (overlays and cuts can be
anywhere). Accept only if ALL hold:

**Rescue clip** (`--kind rescue`):
- Wave pool (waves, sloped beach entry or a wave-pool deck; not a lap pool, lazy river,
  slide, hotel pool, ocean).
- Fixed camera: no panning, zooming, or handheld movement during the event. A camera
  that only cuts between fixed views is OK only if the whole event is in one view.
- The rescue is fully on camera: you can see the victim before the guard reaches them,
  the guard's arrival, and the save, with no edit cut in between.
- You can identify the victim with confidence (trace them back from the guard's
  contact to before the first sign of trouble). If you cannot, REJECT with that reason.
- Not a duplicate of a clip already in the manifest (same footage re-uploaded).

**Normal footage** (`--kind normal`): wave pool, fixed camera, people swimming, no rescue
or guard entry anywhere in the kept part, at least 60 s. Videos over 3 minutes need
`--section A B` (seconds, at most 180 s, a stretch with no overlays or cuts).

Otherwise `reject` with a short specific reason (e.g. "lap pool", "handheld camera",
"rescue cut mid-event", "victim not identifiable", "duplicate of <clip>").

## 2. Label accepted rescue clips

Follow exactly the procedure Nick calibrated (HANDOFF.md "Claude labels the remaining
clips"):

1. Overview sheets: find the guard's reaction (tube, entering water), whom they go to,
   and where live footage ends (overlay, replay, slow motion, highlight circle, end
   card, camera cut).
2. Zoom on the victim at 0.5 s steps from a few seconds before the first sign to just
   after the save; use 0.25 s or 0.1 s steps around onset and contact if unclear.
   Positions: full = X0 + k*tile (k = 2 for crops >= 600 px wide, else 1).
3. Conventions:
   - Onset: earliest visible sign of trouble (head low or back, arm flailing up,
     vertical with no progress, slipping under/off a tube), judged with hindsight.
   - Contact: the guard lands in or reaches the water right at the victim.
   - Saved: victim held on the tube or by the guard, head above water (null +
     off camera if the clip ends first).
   - Last visible: last frame before the victim is hidden (guard's body starting to
     block counts as hidden); if never hidden, the frame before contact.
   - Clicks: onset click on the head (visible body centre if under; never a guess);
     last_seen click; an `extra` click about every 1 s along the path in between.
   - Trim: first live frame to the last frame of continuous live footage before any
     overlay, replay, or cut; null when there is none.
4. Write scratch/specs/CLIP.json and run write_label.py. Notes must say who the victim
   is (appearance, position), what onset was judged on (and the alternative), any
   decoys, and "SPOT CHECK" / "LOW CONFIDENCE" when warranted. If low confidence on
   the victim's identity, reject the clip instead.

**Normal clips**: write `{"clip": CLIP, "kind": "normal", "trim_start": ..., "trim_end":
..., "notes": "..."}` and run write_label.py.

## 3. Finish

Run `status`, then end with a short report: accepted (clip, kind, one-line note),
rejected (reason), and anything Nick should look at first. Extraction and the commit
happen after you finish, in the wrapper script.
