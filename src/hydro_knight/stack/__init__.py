"""
Two-level detector stack, MVP spike (branch two-level-stack; not for a PR yet).

Bottom level: per-swimmer measurements every 0.5 s, for visible swimmers and for "ghosts"
(a spot where a tracked head disappeared): time unseen at the spot, recent disappearances
near it, ring-normalised splash beside the head, and box height as a control. Top level:
an equal-weight sum and a small logistic ranker trained on the victims, within each
moment. Design and review: docs/superpowers/specs/2026-10-24-two-level-stack-design.md
(local only) and the stack-review report.
"""
