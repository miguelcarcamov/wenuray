---
name: verify-ab
description: >-
  Check that a Wenuray branch gives the same self-calibration results as development, and is not
  slower or heavier: run the same simulated self-calibration with each tree's code and compare
  every round's metrics, decisions and gain solutions, plus time and peak memory. Use before
  merging anything that can change a run's results or speed.
---

# Verify A/B against development

"Tests pass" doesn't show that a change kept the results, the speed or the memory of a run.
Compare the branch with `development` directly.

## Setup

- Reference tree: `git worktree add --detach <scratch>/wt_dev origin/development`. Never
  `git stash` to switch trees.
- Each run uses its own tree's code: `PYTHONPATH=<tree>/src`. CASA runs in Docker where it can't
  import natively (mount both trees).
- One run at a time; remove the worktree afterwards.

## 1. Results

`ab_run.py` simulates the seeded test MS (`wenuray.testing.simulate_point_source_ms`), runs a
self-calibration and reports the time and the peak memory:

```bash
PYTHONPATH=<branch>/src python .cursor/skills/verify-ab/ab_run.py <out>/branch
PYTHONPATH=<scratch>/wt_dev/src python .cursor/skills/verify-ab/ab_run.py <out>/dev
python .cursor/skills/verify-ab/ab_compare.py <out>/branch <out>/dev --rtol 1e-6
```

`ab_run.py --recipe <file>` runs a recipe of your own instead (its `ms` is replaced by the
simulated one); cover what the change can affect: each imager (`tclean`, `pyralysis`), phase and
amplitude steps, `combine`, residual flagging, a rejected round, a resumed run.

`ab_compare.py` compares, round by round, the step, the decision and its reasons, PSNR, peak, rms,
model flux, flagged fractions and predicted S/N, then the gain solutions of the accepted tables.
Differences at floating-point rounding are expected when the order of operations changed; say so.

## 2. Speed and memory

`ab_run.py` writes `seconds` and `peak_rss_mb` (this process; the Pyralysis worker is a child
process and is reported as `children_peak_rss_mb`). Run branch and development alternately, best
of three; repeat a borderline row before calling it a regression.

## Report

A table per axis (results, speed, memory): branch vs development, the largest differences and
what causes them.
