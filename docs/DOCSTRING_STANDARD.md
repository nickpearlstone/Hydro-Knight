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
