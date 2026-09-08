# Docstring standard

Concise **Google-style** docstrings on **public and non-obvious** functions.
Trivial one-line helpers need only a one-line summary — no schema.

## Template

```python
"""One-line summary of what it does. (+ one more sentence only if genuinely needed.)

Preconditions: <one line — omit if none>
Args:
    name: terse description (shape / units / dtype where it matters).
Returns:
    terse description (shape / units; what a sentinel like None means).
Raises:
    ErrorType: when/why (omit if it doesn't raise).
"""
```

## Rules

- The summary line is required and stays on one line. Add at most **one** more
  sentence, and only for a genuinely non-obvious "why".
- **Omit any section that doesn't apply** — a no-arg function has no `Args:`; a
  function that can't fail a precondition has no `Preconditions:`.
- Include **shapes, units, and dtypes** wherever they matter — this is
  array-heavy code and those are the facts a caller actually needs.
- Always state what a **sentinel return** means (e.g. `None` when the pose is unusable).
- **Module docstrings stay free prose** (purpose + context). This schema is for
  functions, not modules.
- **No repeated reasoning (DRY for docs).** Each piece of "why" appears once, at
  the right altitude: shared/overarching context lives in the *module* docstring;
  a *function* docstring carries only what's specific to it. If the same rationale
  would appear in two functions, it belongs in the module docstring instead.
- **Schema is a floor, not a ceiling.** Always include the structured fields;
  keep substantive function-specific "why" where it genuinely helps, but trim
  padding. Thin docstrings get tightened; already-rich ones get *structured*, not gutted.

## Scope of application

Public API + any function whose behavior isn't obvious from its name and
signature. Skip the full schema on dead-simple private helpers; a one-line
summary is enough there.
