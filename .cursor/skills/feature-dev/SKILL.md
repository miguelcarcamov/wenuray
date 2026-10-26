---
name: feature-dev
description: >-
  Implement a Wenuray GitHub issue end to end: plan, branch from development, code per the repo's
  rules, test (CASA tests in Docker when needed), pre-commit, push and open a PR. Use when asked to
  implement or develop an issue, typically labelled enhancement.
---

# Feature development

Implement one GitHub issue of Wenuray (`github.com/miguelcarcamov/wenuray`).

## Before coding

1. Read the issue and its comments (`gh issue view N --comments`): problem, proposal, acceptance
   criteria, open questions, linked issues and PRs.
2. Read `AGENTS.md` and the rules in `.cursor/rules/` (`general.mdc` and `selfcal-domain.mdc`
   first). Check the code the issue names: it may have moved since the issue was written.
3. **Decisions that are the maintainer's** (public API, defaults, behaviour, scope) get asked with
   a recommendation before coding; record the answer in the issue. Questions listed in the issue
   are such decisions.
4. A change too big for one PR or one session: use `staged-feature` and split it into PRs that
   each leave `development` green (e.g. #36 went recipes → CLI → run record).

## Branch and commits

- `git fetch origin && git checkout -b chg/<issue>-<slug> origin/development`.
- Commits: `CHG: <what and why> (#<issue>)`; the body explains the reasoning.
- Never base the branch on another unmerged PR.

## Implementation

Follow `.cursor/rules/`. In particular:

- Self-calibration invariants (`selfcal-domain.mdc`): gaincal solves against `DATA`, imagers image
  `CORRECTED_DATA` and write `MODEL_DATA`, one selection everywhere, `CalStack` builds every table
  list.
- Parameter budget and config objects (`design.mdc`); defaults from theory or the data, never
  from one instrument; extension through hooks and registries, not `if` chains.
- No I/O in constructors, no module-level CASA tools, MS reads in samples (`casa-io.mdc`,
  `implementation.mdc`).
- Stay in the issue's scope; open an issue for anything else you find.

## Tests

| Change | Tests | Where |
| --- | --- | --- |
| Logic in one module | Unit tests, CASA-free (mock the CASA tasks the module imports) | `tests/unit/<area>/` mirroring `src/wenuray/` |
| Anything touching CASA tasks or a measurement set | Integration tests on `wenuray.testing.simulate_point_source_ms` | `tests/integration/` |
| The self-cal loop, solving, applying, acceptance | Keep `test_selfcal_recovers_gains_casa.py` passing; extend it if behaviour changes | `tests/integration/` |
| Recipes, CLI, examples | Recipe and script build equal objects | `tests/unit/test_recipe.py`, `tests/examples/` |

Run the narrowest paths first, then the full suite:

```bash
pre-commit run --files <changed files>     # git add new files first
pytest tests/unit/<area> tests/examples
# CASA tests where casatools can't import natively (see AGENTS.md):
docker run --rm -v $PWD:/src:ro -e HOME=/tmp -e SETUPTOOLS_SCM_PRETEND_VERSION_FOR_WENURAY=0.0.dev0 \
  ghcr.io/miguelcarcamov/wenuray:base bash -c \
  "cp -r /src /tmp/wenuray && cd /tmp/wenuray && pip install -q -e '.[test]' && pytest tests -q"
```

Docs or docstrings changed: `sphinx-build -W -b html docs docs/_build/html`.
Rules changed: `python tools/sync_conventions.py`.

Before the PR, run `convention-review` on the diff. If results, numbers or speed of a run may
change, run `verify-ab` and put its tables in the PR.

## Pull request

1. Push the branch and open a PR into `development` (`gh pr create --base development`), filling
   `.github/pull_request_template.md`: `Closes #<issue>` (or "Part of #<issue>"), what and why,
   the theory for science changes, the tests and the commands run with their results.
2. Watch CI (`gh pr checks N --watch`) and fix failures caused by the PR.
3. Report the PR link, what changed, and anything left for the maintainer to decide.

## Review comments

Address each comment, rerun the targeted tests and pre-commit, push, and reply on the thread with
what changed. For CI failures and threads on an existing PR, use `pr-review`.
