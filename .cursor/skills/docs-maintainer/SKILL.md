---
name: docs-maintainer
description: >-
  Improve Wenuray's documentation: README, Sphinx pages (MyST Markdown, Read the Docs), examples and
  docstrings, keeping docs and code in step. Docs-only PRs. Use for documentation issues or when
  asked to audit or improve the docs.
---

# Docs maintainer

## Scope

| In | Paths |
| --- | --- |
| Front page and guides | `README.md` (it is the docs' front page), `docs/*.md`, `docs/migration.md` |
| Examples | `examples/<name>/recipe.toml` and `selfcal.py`, `examples/README.md` |
| API reference | Public docstrings (numpy style), rendered by sphinx-autoapi |
| Agent and contributor docs | `AGENTS.md`, `CONTRIBUTING.md` (only when that is the task) |

Out of scope: behaviour changes. If the docs can only be right after a code change, open an issue
or use `feature-dev`.

## Goals, in order

1. **Correctness:** every snippet runs against the current API; names, defaults and file paths
   match the code; instrument values in examples are marked as the example's, not Wenuray's.
2. **Parity:** every public class and option is documented; recipes and scripts in `examples/`
   build equal runs (a test checks it).
3. **Structure:** pages in the right toctree, no duplication (the README is included, not copied).
4. **Clarity:** short sentences, the theory where a choice depends on it (e.g. why a cell must be
   Nyquist-sampled), links to CASA guides for CASA parameters.

## Checks

```bash
pre-commit run --files <changed files>            # markdownlint, TOML formatting
pip install -r docs/requirements.txt
sphinx-build -W --keep-going -b html docs docs/_build/html   # no warnings allowed
pytest tests/examples                              # README and examples run against the API
```

Docstrings render as reStructuredText: quote `|u|` and `**kwargs` as literals (` ``|u|`` `), and
end a sentence with `::` before an indented code block.

## Pull request

Branch `chg/docs-<slug>` (or `chg/<issue>-<slug>`), PR into `development`: what was wrong, what
changed, how to verify (the Sphinx build, the example tests). Escalate when a fix needs an API
change, a structural decision, or domain review (self-calibration theory, CASA behaviour).
