---
name: qa-explorer
description: >-
  Find gaps in Wenuray's tests and add high-value ones, above all self-calibration invariants on
  simulated data; file a bug issue when a test finds a defect. Use when asked to improve tests or
  coverage, or when idle.
---

# QA explorer

## Goals, in order

1. **Self-calibration invariants** on simulated data (`wenuray.testing`), which hold for any array:
   - recovered gains match injected ones up to the reference phase and a source shift;
   - phase-only calibration doesn't change amplitudes, and keeps the model flux (or raises it);
   - a rejected round leaves `CORRECTED_DATA`, `MODEL_DATA`, the flags and the table stack as they
     were before it;
   - every applied table appears once, with its own `spwmap` and `interp`, in solve and apply;
   - the imager contract: `MODEL_DATA` written for exactly the selected rows;
   - a recipe and its script build equal objects; a recorded recipe reproduces the run.
2. **Regression safety** for untested paths in the loop, `CalStack`, acceptance, preflight,
   recipes and the CLI.
3. **Coverage gaps** in `src/wenuray` (`pytest --cov=wenuray tests/unit`).

## Workflow

1. Read `tests/` and `.cursor/rules/testing.mdc`.
2. Pick one focused improvement: one invariant or one module.
3. Test-only changes on a branch `chg/tests-<slug>`, PR titled `CHG: test <what>`.
4. Make fixtures non-trivial and assertions about values (`testing.mdc`); derive tolerances from
   the simulation's noise.

## If a test finds a bug

Don't fix it silently in a test-only PR. Search for duplicates, then open an issue with the bug
template (`.github/ISSUE_TEMPLATE/bug.md`): summary, actual and expected behaviour, a pytest
reproduction, environment, suspected code, labels `bug` and an `effort:N` estimate. Then fix it
with `bug-fix` on its own branch if asked.
