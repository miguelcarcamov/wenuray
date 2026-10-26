<!-- Base: development. One issue per PR; don't base it on another open PR. -->

Closes #

## What and why

<!-- What changed, and the reasoning. A change to how gains are solved or applied, how the model
is written, or how a round is accepted explains the theory. -->

## Tests

<!-- Tests added or changed, and the commands you ran with their results, e.g.
pytest tests/unit (N passed), the CASA tests in Docker, sphinx-build -W. -->

## Checklist

- [ ] Tests for the new behaviour; a bug fix has a test that fails without it
- [ ] `pre-commit run` passes; `python tools/sync_conventions.py --check` if rules changed
- [ ] README, docs, examples and recipes match the change
