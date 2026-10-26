---
name: convention-review
description: >-
  Deep review of a Wenuray diff (branch or PR) against the repository's rules: self-calibration
  invariants, genericity across arrays, CASA I/O, design budget, tests and docs. Reports findings
  ranked by severity, each with a concrete failure scenario. Use before opening or merging a PR, or
  when asked to review code. For CI failures and review threads use pr-review.
---

# Convention review

Review with fresh eyes: read the diff (`git diff origin/development...HEAD`) and the rules it
touches, not the author's explanation. Every finding needs a concrete scenario (data, selection,
configuration) that produces wrong calibration, a crash, a regression or a broken rule. No
scenario, no finding. Cite the rule file.

## Checklist

**Self-calibration theory** (`selfcal-domain.mdc`)

- gaincal solves against `DATA` with the accepted stack pre-applied; the imager images
  `CORRECTED_DATA` and writes `MODEL_DATA` for the selected data.
- One `Selection` for solve, apply and imaging; every table's `spwmap` and `interp` travel with it
  through `CalStack`.
- A rejected round restores data, flags, model and stack; acceptance is never PSNR alone.
- Amplitude steps come after a phase solution; schedules are validated before any I/O.

**Generic across arrays** (`general.mdc`, `design.mdc`)

- No instrument names, instrument-tuned numbers or `if instrument == ...`. Defaults come from
  theory (closure needs 3 or 4 antennas, Nyquist sampling) or from the data; instrument- or
  science-dependent limits are opt-in with no default.
- Variation goes through hooks, ClassVars, registries and `multimethod`, not `if` chains.

**Design** (`design.mdc`, `implementation.mdc`)

- At most about eight constructor parameters, all used; options grouped in frozen config objects.
- Known types dispatched with `isinstance`/`multimethod`, not attribute probing.
- Recipes: a new public class serialises (fields, or `to_recipe`/`from_recipe`) and round-trips.

**CASA and data** (`casa-io.mdc`, `implementation.mdc`)

- No I/O in constructors; no module-level CASA tools; tables closed on every path.
- No fixed file names in the working directory; outputs under the run's directory.
- Reads of a measurement set in samples when a statistic suffices: what happens on 10⁹ rows?

**Errors** (`error-handling.mdc`)

- Silent fallbacks that change the science are errors or explicit warnings; messages say how to
  fix the problem.

**Tests** (`testing.mdc`)

- Behaviour changes have tests; regression tests fail on the old code.
- Assertions about values, fixtures where the tested quantity is non-trivial and the simulated
  data are physically valid (Nyquist-sampled cells, enough S/N).
- Tolerances derived from the simulation's noise, and the degeneracies of self-cal respected.
- Unit tests stay CASA-free; patch targets are `wenuray.…`, never `src.wenuray.…`.

**Docs** (`documentation.mdc`)

- Docstrings, README, `docs/`, examples and recipes match the new behaviour; examples still build
  equal runs; comments say why.

## Output

Findings ranked most severe first: `file:line`, the defect in one sentence, the failure scenario,
the suggested fix. Then a short list of what was checked and found fine. A finding that needs a
run (results, speed) goes to `verify-ab` instead of a guess.
