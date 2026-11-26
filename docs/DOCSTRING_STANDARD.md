# Comment and docstring standard

Keep comments and docstrings to the minimum the code needs.

## Docstrings

- Public functions, classes and methods (no leading `_`) get a one- or two-line
  summary. No `Args:` / `Returns:` / `Raises:` sections; put units, shapes or a
  sentinel return in the summary only when the signature can't say it.
- Private helpers, nested functions and dunder methods get no docstring.
- Modules get a one- or two-line summary. CLI modules that pass `__doc__` to
  `argparse` keep their full docstring as the `--help` text.
- Tests get no docstrings; name the test for what it checks.

## Comments

A comment stays only if removing it would likely lead to a bug: a non-obvious
invariant or ordering, units or time base, a boundary, a library version
constraint, a known bias in the output, or a licence requirement. One short
line each. Tool pragmas (`# noqa`, `# type: ignore`, `# fmt:`) are always kept.

Kept comments and docstrings state the constraint, never the reason: "must run
inside the manifest lock", "source time, not file time". No history and no
"because".

Constraints that would break data, evaluation or the pipeline silently are
tagged `# CAUTION[<id>]: <constraint>` (`// CAUTION[<id>]:` in JS), so
`git grep -n 'CAUTION\['` lists them all. Each id's reason and pre-change check
are kept outside the code, keyed by id, and read only when that line or the
code under it is about to change. A new constraint gets a tag and an entry in
the same change.

## Evidence (2026-11-25)

A/B test of this policy against the fully commented code it replaced.

**Setup.** Two arms, each a fresh git worktree:

- **A:** `main` before the strip (full comments and docstrings, no tags).
- **B:** this policy (minimal comments, `CAUTION[id]` tags, reasons list, and an edit hook that
  shows an id's entry when an edit touches its line).

Each run was a new headless Claude Code session on Claude Sonnet 5.5 with the same
project instructions (B adds a short note about the tags), unattended, at most 40 turns.
There were 4 tasks, each run 3 times per arm (24 runs).

| Task | A: cost (median) | B: cost (median) | A: outcome | B: outcome |
|---|---|---|---|---|
| Find: where victim clicks become track ids | $0.218 | $0.117 | correct 3/3 | correct 3/3 |
| Reuse: event times to file time, with a test | $0.213 | $0.251 | reused `to_file_time` 3/3, tests pass 3/3 | same |
| Trap: "prefer AV1 in `download.py`" (breaks OpenCV decoding) | $0.208 | $0.117 | made the change 3/3 | declined and explained 3/3 |
| Plain: add `--limit` to `zip-videos` | $0.182 | $0.117 | correct 3/3, tests and ruff pass | same |
| **Total, 12 runs per arm** | **$2.60** | **$1.81** | | |

- **Trap.** In A the reason was written next to the line, and every run changed it anyway.
  All three also rewrote the comment to match. In B every run looked up the tagged entry
  before editing and declined. The edit hook never had to fire.
- **Reasons list use.** In B the list was opened only in the trap runs (4 reads). It was
  opened 0 times across the other 9 runs.
- **Reuse.** Both arms found and reused the existing helper every time. B cost more here: it
  read more code to work out what functions do (750k vs 482k cached input tokens, median).

**How cost was computed.** Each session reports `modelUsage` per model: input, cache-write,
cache-read and output tokens, plus `costUSD` priced at list (`costBasis: "list"`). All runs
used only `claude-sonnet-5-5`. A least-squares fit of `costUSD` on the four token counts
across the 24 runs recovers the per-million-token prices exactly (max error < $0.000001):

| Token type | $ / M tokens | A total tokens | B total tokens | Share of A cost | Share of B cost |
|---|---|---|---|---|---|
| Input (uncached) | 2.00 | 250 | 147 | 0.0% | 0.0% |
| Cache write (1 h) | 4.00 | 420,357 | 313,685 | 64.7% | 69.4% |
| Cache read | 0.10 | 5,540,601 | 3,217,137 | 21.3% | 17.8% |
| Output | 10.00 | 36,230 | 23,172 | 13.9% | 12.8% |

Cost = Σ tokens × price, summed over the 12 runs of an arm. Cache writes dominate: each
session pays once to cache its context (system prompt, instructions, files read), then pays
1/40 of that to re-read it on every later turn. B wrote less (shorter files to read) and
took fewer turns, so it re-read less. These are list-price estimates, not billed amounts.

**Limits.** 3 runs per cell. Wall-clock times are not reported because runs shared one
machine 4 at a time. The trap result (0/3 vs 3/3) is the clearest effect. The cost
difference held on 3 of 4 tasks.
