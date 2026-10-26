# Contributing to Wenuray

Thanks for helping. Wenuray exists so astronomers can describe a self-calibration procedure briefly
and reproducibly; contributions should make that easier and stay correct by self-calibration
theory.

## Ways to contribute

- Report a bug or propose a feature: [open an issue](https://github.com/miguelcarcamov/wenuray/issues/new/choose).
- Improve the documentation or the examples.
- Add tests, especially ones that encode self-calibration invariants.
- Fix a bug or implement a feature from the [issue list](https://github.com/miguelcarcamov/wenuray/issues).
- Use an AI agent: the same rules apply, see [AGENTS.md](AGENTS.md).

Search existing issues and pull requests first. Open an issue before a large change, so the design
can be agreed before the code.

## Development setup

Wenuray needs Python 3.12 or 3.13 (the CASA wheels) on Linux or macOS.

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt -e ".[test]"   # the pinned set CI tests
pre-commit install
```

CASA downloads its measures data the first time `casatools` is imported. Without network access,
or where `casatools` won't import, use the image `ghcr.io/miguelcarcamov/wenuray:base` (see
[AGENTS.md](AGENTS.md#environment)).

## Conventions

The rules are in [`.cursor/rules/`](.cursor/rules/README.md), mirrored in
[CONVENTIONS.md](CONVENTIONS.md). In short:

- One self-calibration loop, `Selfcal`; calibration types are `CalibrationStep` subclasses; imagers
  follow the contract in `wenuray.imaging.contract`.
- At most about eight constructor parameters; group options in frozen config objects.
- Defaults come from theory or the data, never from one instrument.
- Numpy-style docstrings; comments say why, not what.
- New behaviour has tests; a bug fix has a test that fails without it.

## Checks

```bash
pre-commit run --all-files                 # yapf, isort, TOML/YAML, markdownlint, workflows
pytest tests/unit tests/examples           # no CASA needed
pytest tests/integration                   # needs CASA; simulates its own data
python tools/sync_conventions.py --check   # if you edited .cursor/rules/
sphinx-build -W -b html docs docs/_build/html   # if you changed docs or docstrings
```

## Branches, commits and pull requests

- Branch from `development`: `chg/<issue>-<slug>` for changes and features, `fix/<issue>-<slug>`
  for bug fixes. Base every PR on `development`, not on another open PR.
- Commit prefixes: `CHG:` and `FIX:`, with the issue number in the subject, e.g.
  `FIX: refuse an imaging cell coarser than the Nyquist cell (#60)`.
- The PR says `Closes #N`, what changed and why, and the commands you ran. A change to how gains
  are solved or applied, how the model is written, or how a round is accepted explains the theory.
- `development` is the integration branch and `master` the release branch; releases publish to
  PyPI as `wenuray`.
