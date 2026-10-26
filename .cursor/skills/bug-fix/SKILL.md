---
name: bug-fix
description: >-
  Fix a Wenuray bug: reproduce it with a failing test, fix the root cause, keep the regression test,
  run pre-commit and pytest, push and open a PR. Use for issues labelled bug or when asked to fix a
  defect.
---

# Bug fix

## Before coding

1. Read the issue: steps to reproduce, expected vs actual behaviour, environment, logs.
2. Read `AGENTS.md` and `.cursor/rules/`. Search for duplicates and for the tests that cover the
   area.
3. Decide whether the bug is Wenuray's or upstream's (CASA, Pyralysis). An upstream bug gets an
   upstream issue with a minimal reproduction and, if needed, a workaround in Wenuray that names the
   upstream issue.

## Reproduce first

1. Write the smallest test that fails because of the bug: a unit test with mocks if the logic is
   pure, an integration test on simulated data if it needs CASA.
2. Run it and see it fail for the reason in the issue, not another one.
3. Check the test's data can show the bug: e.g. a cell must be Nyquist-sampled for an imaging
   check to mean anything (`testing.mdc`, "Assert values, not just form").

## Fix

- Branch `fix/<issue>-<slug>` from `origin/development`; commits `FIX: <what> (#<issue>)`.
- Fix the root cause, not the symptom. A silent fallback that changes the science becomes an
  error or an explicit warning (`error-handling.mdc`).
- Keep the regression test. Confirm it fails with only the fix reverted (`git checkout
  origin/development -- <file>`, run, restore), not by stashing unrelated work.

## Quality gates

```bash
pre-commit run --files <changed files>
pytest <targeted tests>            # CASA tests in Docker if needed, see AGENTS.md
```

Run `convention-review` on the diff; run `verify-ab` if the fix changes results of an existing
path.

## Pull request

Open a PR into `development` with `Closes #<issue>`, the reproduction, the root cause, the
regression test and the commands run. If the issue has no `effort:N` label, suggest one.
