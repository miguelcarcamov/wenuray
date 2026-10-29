# Agent-assisted development

How AI agents (Claude Code, Cursor, Codex, Aider, ...) work on Wenuray
(`github.com/miguelcarcamov/wenuray`). Humans follow the same rules; see
[CONTRIBUTING.md](CONTRIBUTING.md).

## Overview

| Component | Location | Role |
| --- | --- | --- |
| Conventions | `.cursor/rules/*.mdc` ([README](.cursor/rules/README.md)), mirrored in `CONVENTIONS.md` | The rules every change follows |
| Skills | `.cursor/skills/<name>/SKILL.md`; `.claude/skills/<name>` links to the same files | Playbooks for features, bugs, reviews, QA, docs, estimates |
| Hook | `tools/hooks/block-protected-push.sh`, wired in `.cursor/hooks.json` and `.claude/settings.json` | Blocks pushes to `development` and `master` |
| Templates | `.github/ISSUE_TEMPLATE/`, `.github/pull_request_template.md` | What an issue and a PR must say |
| Tool entry points | `CLAUDE.md` (Claude Code), `.cursor/rules` (Cursor), `.aider.conf.yml` (Aider), this file (Codex and others) | Each points back here and to the rules |

The `.mdc` files are the source of truth. After editing one, run
`python tools/sync_conventions.py` to regenerate `CONVENTIONS.md`; CI checks it.

## What Wenuray is, in one paragraph

A framework for radio-interferometric **self-calibration** on CASA, so astronomers describe a
procedure briefly and reproducibly (a TOML recipe or a few Python objects) instead of
rewriting a script per dataset. Every change must keep it **correct by self-calibration theory**
(`selfcal-domain.mdc`) and **generic across arrays**: limits come from theory or the data, never
from one instrument, and Wenuray ships no instrument presets.

Read the rules before the first change: start with `general.mdc` (what Wenuray is for) and
`selfcal-domain.mdc` (the self-calibration invariants no change may break).

## Workflow

```text
issue (enhancement | bug | documentation, effort:N, acceptance criteria)
  -> branch from development: chg/<issue>-<slug> or fix/<issue>-<slug>
  -> code + tests + docs, pre-commit, targeted pytest (CASA tests in Docker if needed)
  -> convention-review skill on the diff (and verify-ab when results or speed may change)
  -> PR into development: "Closes #N", what changed, commands run
  -> CI green -> the maintainer reviews and merges
  -> issues close when development is released to master
```

- **One issue, one PR, based on `development`.** Never base a PR on another unmerged PR's branch:
  squash merges strand the stacked one. Wait for the merge, then branch again.
- **Commits:** prefixes `CHG:` (change, feature) and `FIX:` (bug fix), the issue in the subject
  (`(#24)`), and the body explains why.
- **Never push to `development` or `master`.** The hook blocks it.
- **Science first.** A change to how gains are solved or applied, how the model is written or how
  a round is accepted explains the theory in the PR and adds or extends a simulated end-to-end
  test (`tests/integration/`, #32).
- **Ask when in doubt.** Decisions about API semantics, defaults, behaviour changes or scope
  belong to the maintainer: ask with a recommendation and record the answer in the issue or PR.
- **Upstream bugs** (CASA, Pyralysis) get an issue upstream with a minimal reproduction, and a
  workaround in Wenuray that names it.

## Labels

| Label | Meaning |
| --- | --- |
| `enhancement`, `bug`, `documentation` | Kind of work: `feature-dev`, `bug-fix` or `docs-maintainer` |
| `design` | Needs a design decision before code |
| `effort:1` ... `effort:13` | Size (Fibonacci), from the `estimate-effort` skill |
| `pyralysis`, `github_actions`, `dependencies` | Area |

Milestone **v1.0 redesign** holds the redesign (#22 and its parts).

## Skills

| Skill | When |
| --- | --- |
| `feature-dev` | Implement an `enhancement` issue |
| `bug-fix` | Fix a `bug` issue, test first |
| `pr-review` | Bring an open PR to mergeable: CI, review threads, shallow fixes, escalation |
| `convention-review` | Deep review of a diff against the rules, with a failure scenario per finding |
| `verify-ab` | Compare a self-calibration run on the branch with one on `development` |
| `staged-feature` | A change too big for one session: living plan, one verified stage at a time |
| `qa-explorer` | Find test gaps and encode self-calibration invariants as tests |
| `docs-maintainer` | README, Sphinx docs, examples and docstrings |
| `estimate-effort` | Set `effort:N` on an issue |

Ask for one by name, e.g. *"use the feature-dev skill for issue #65"*.

## Environment

- **Python** `>=3.11,<3.14`; `tox` tests each version (`tox.ini`). Install the pinned set: `pip install -r requirements.txt -e ".[test]"`,
  then `pre-commit install`.
- **CASA:** `casatools` comes from its `manylinux_2_28` wheel (the `manylinux_2_34` one fails
  outside RHEL-like systems with `OPENSSL_3.0.1 not found`). It needs measures data and a
  writable `HOME` on first import. Where it can't import natively, run CASA tests in the image:

  ```bash
  docker run --rm -v $PWD:/src:ro -e HOME=/tmp -e SETUPTOOLS_SCM_PRETEND_VERSION_FOR_WENURAY=0.0.dev0 \
    ghcr.io/miguelcarcamov/wenuray:base bash -c \
    "cp -r /src /tmp/wenuray && cd /tmp/wenuray && pip install -q -e '.[test]' && pytest tests -q"
  ```

- **Pyralysis** (optional extra) runs in a worker process; Numba there needs OpenMP
  (`libgomp1`) or TBB.
- **Tests:** `pytest tests/unit` needs no CASA (CASA modules are stubbed when missing);
  `tests/integration` needs CASA and simulates its data with `wenuray.testing`. The full suite takes
  about ten minutes. Run the narrowest paths first.
- **Lint:** `pre-commit run --files <changed files>`. It skips untracked files, so `git add`
  new files first, and re-add after yapf rewrites them.
- **Docs:** `pip install -r docs/requirements.txt && sphinx-build -W -b html docs docs/_build/html`
  (no CASA needed).

## Gotchas

- **Imaging cells must be Nyquist-sampled.** `Imager.image()` refuses a cell coarser than
  1 / (2 max(|u|, |v|)) of the selected data. The simulated test MS reaches 59 kλ: use 1″ cells.
- **gaincal solves against `DATA`**, imagers image `CORRECTED_DATA` and must write `MODEL_DATA`.
  A test that hand-edits columns must respect that.
- **Self-cal can't fix the absolute position:** compare recovered gains with injected ones after
  fitting out a source shift (see `tests/integration/test_selfcal_recovers_gains_casa.py`).
- **casacore twice in one process crashes:** casatools and python-casacore (Pyralysis) can't share
  a process; that is why the Pyralysis imager runs a worker.
