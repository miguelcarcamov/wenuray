# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this project is

Wenuray is a Python framework for radio-interferometric **self-calibration** on top of CASA (`casatasks`/`casatools`), with pluggable imagers (`tclean` today; Pyralysis, which replaces the deprecated GPUvmem, is the target). Its purpose is to let astronomers describe a self-cal procedure **briefly and reproducibly** instead of rewriting a one-off script for every dataset. Every change should make that easier, and stay correct by self-cal theory.

Source lives in `src/wenuray/`: `selfcalibration/` (self-cal loop), `imaging/` (imager backends) and `utils/`. Examples are in `examples/<name>/`: a `recipe.toml` and a `selfcal.py` that build the same run.

## Agents, skills and hooks

`AGENTS.md` is the guide for every agent tool: the workflow, labels, environment and gotchas.
Skills live in `.cursor/skills/` and are linked from `.claude/skills/`, so Claude Code and Cursor
read the same files (`feature-dev`, `bug-fix`, `pr-review`, `convention-review`, `verify-ab`,
`staged-feature`, `qa-explorer`, `docs-maintainer`, `estimate-effort`). A `PreToolUse` hook in
`.claude/settings.json` (`tools/hooks/block-protected-push.sh`) blocks pushes to `development` and
`master`.

## Canonical conventions — read these first

The conventions live in **`.cursor/rules/*.mdc`** (see `.cursor/rules/README.md`) and are mirrored in **`CONVENTIONS.md`**. This file doesn't repeat them; it only adds operational notes. If the two disagree, the `.mdc` files win; regenerate the mirror with `python tools/sync_conventions.py`.

Highlights to keep in mind:

- **Architecture** (`design.mdc`):
  - One self-cal loop, in `Selfcal`.
  - Calibration types are frozen-dataclass subclasses of `CalibrationStep` (`Phasecal`, `Ampcal`, `AmpPhasecal`), in any valid order.
  - `CalStack` is the only place that builds `gaincal`/`applycal` table lists.
  - Theory rules are `multimethod` overloads of `check_transition`; an invalid schedule raises `ScheduleError` before any I/O.
- **Parameter budget:** at most ~8 constructor parameters, all of them used. Group parameters into frozen config objects (`Selection`, `SolveOptions`, `ApplyOptions`, `ImageGeometry`, `Weighting`). Prefer presets and data-derived defaults over new parameters.
- **Pluggable imagers:** subclass `Imager` and implement `_reconstruct` (or wrap a function with `CallableImager`). The final `image()` method enforces the contract: image `CORRECTED_DATA`, write `MODEL_DATA`.
- **Self-cal invariants** (`selfcal-domain.mdc`):
  - `gaincal` solves against `DATA`.
  - `MODEL_DATA` must hold the accepted model.
  - Pre-apply the accepted table stack; `spwmap`/`interp` belong to each table.
  - One selection for solve, apply and imaging.
  - Never use PSNR alone to accept a round.
- No I/O in constructors, no module-level CASA tool singletons, no `os.system`, no files with fixed names in the working directory (`casa-io.mdc`).
- Imagers pass backend options through with the backend's own parameter names; optional backends such as Pyralysis are extras imported lazily.
- Legacy code predates these rules. Follow them in new or changed code; don't refactor beyond the scope of the issue you're working on.

## Roadmap and issues

The redesign is tracked on GitHub (`miguelcarcamov/wenuray`), milestone **"v1.0 redesign"**:

- Epic: #22.
- Imager interface: #23.
- Pyralysis backend: #24 (the end goal).
- Science bugs: #25, #26, #27.
- Stopping rule: #28.
- Smaller bugs: #29.
- Packaging and Python version: #30.
- Docs and examples: #31.
- Tests: #32.
- CI: #33, #34.

Check the relevant issue before starting work, and reference it in commits and PRs (`Closes #NN`).

Python is `>=3.12,<3.14`: the overlap of the CASA wheels (cp312/cp313) and Pyralysis (`<3.14`) (#30).

## Environment

```bash
pip install -r requirements.txt -e ".[test]"   # pinned CI set; casatools wheels are Linux/macOS only
pre-commit install
```

- CASA wheels exist only for specific Python versions; check `requires-python` in `pyproject.toml`.
- CASA needs measures data (`casaconfig`) the first time `casatools` is imported. Without network access, provide a measures directory. The base image ships it in `/opt/casa/data`.
- `requirements.txt` installs casatools from its `manylinux_2_28` wheel: the `manylinux_2_34` wheel fails to import outside RHEL-like systems (`OPENSSL_3.0.1 not found`).
- The Docker images (`ghcr.io/miguelcarcamov/wenuray:base`, `:latest`) are optional and kept small: `python:3.12-slim`, the pinned dependencies, CASA measures data only, no CUDA (Pyralysis brings that in #24). Build locally with `docker build .`; `--build-arg MEASURES_SITE=<url>` picks another measures server.
- `reproject` is an optional extra (`wenuray[reproject]`), imported inside `wenuray.utils.reproject()` only.

## Tests and linting

```bash
pytest tests/unit/ tests/examples/     # CASA-free
pytest -m casa tests/integration/      # needs CASA
pre-commit run --all-files             # yapf, YAML/TOML formatting, markdownlint, workflow schema
python tools/sync_conventions.py --check
```

When you add behaviour, add the test under `tests/unit/` mirroring `src/wenuray/`; anything that needs CASA goes in `tests/integration/` on `wenuray.testing`'s simulated data. Run CASA tests in the Docker image where `casatools` can't import natively (`AGENTS.md`).

## Commits and branches

- `development` is the integration branch; `master` is the release branch.
- Work on `chg/<issue>-<slug>` or `fix/<issue>-<slug>` branches from `development`, with PRs into `development`. Never base a PR on another open PR.
- Commit prefixes: `CHG:` and `FIX:`.
- Don't push to `development` or `master` directly unless asked.
