---
name: staged-feature
description: >-
  Run a change too big for one session or one PR (a redesign, a new backend, a new calibration
  type) as plan -> test -> implement -> review -> verify -> remember -> improve, with a living plan
  and one verified, mergeable stage at a time. Use when starting or resuming such a change, or
  when asked to plan one.
---

# Staged feature

Big changes go wrong by drifting: the plan says one thing, the code another, and the next session
re-derives decisions. Keep one written source of truth and one green, merged stage at a time.

## 1. Plan

1. Keep the plan in the issue (a "Plan" comment updated as stages land) or, for a long design, in
   `docs/<topic>_plan.md`. Record each decision with who made it and why, the open questions, and
   per-stage progress.
2. **Check the plan against the code, not memory:** grep every class, function and file it names.
   Write down drift instead of silently reconciling it.
3. Cut stages that each leave `development` green and merge on their own, **one PR per stage,
   each based on `development`** (no stacked PRs). Don't build ahead of a real consumer: note it
   as deferred.
4. Questions that are the maintainer's (API, defaults, behaviour, scope): ask with a
   recommendation, record the answer.

## 2. Test first

- **Characterise before refactoring:** record a run's outputs on simulated data (`selfcal.toml`
  rounds, the tables' gains) and reproduce them after the change; name every intended difference
  before re-recording.
- Regression tests fail on the old code: check by reverting only the fix.
- Scientific claims get a scientific test: e.g. injected gains recovered within the noise the
  simulation implies.

## 3. Implement

One stage at a time, following `.cursor/rules/`. Note every place reality differed from the plan.

## 4. Review

`convention-review` on the stage's diff, with fresh eyes; fix or answer each finding.

## 5. Verify

Targeted tests, then the full suite (CASA in Docker if needed). Anything that changes a run's
results or speed: `verify-ab` against `development`. `pre-commit run --files <changed files>`.

## 6. Remember

Close the stage in the plan: the PR, what differed from the design, test counts, what comes next.
Commit subjects name the issue (`(#36)`).

## 7. Improve

Promote lessons that generalise into the rules: a bug class found twice becomes a line in the
right `.mdc` (then `python tools/sync_conventions.py`). One-off facts stay in the plan.
