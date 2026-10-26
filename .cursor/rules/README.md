# Cursor project rules

Conventions for Wenuray contributors and AI agents. Cursor loads every `.mdc` file in this directory according to its `description`, `globs` and `alwaysApply` frontmatter. `CONVENTIONS.md` at the repo root mirrors these files for other tools. Claude Code reads them through `CLAUDE.md`.

| File | Topic |
|------|-------|
| [`general.mdc`](general.mdc) | Purpose of Wenuray and general principles — read first |
| [`design.mdc`](design.mdc) | Parameter budget, config objects, rounds, policies, backends |
| [`selfcal-domain.mdc`](selfcal-domain.mdc) | Self-calibration theory invariants |
| [`casa-io.mdc`](casa-io.mdc) | CASA tools, Measurement Sets, files, subprocesses, FITS |
| [`implementation.mdc`](implementation.mdc) | Clear purpose, explicit code, known types, reading data in samples |
| [`naming.mdc`](naming.mdc) | Naming and method order |
| [`documentation.mdc`](documentation.mdc) | Docstrings, comments, README/examples |
| [`error-handling.mdc`](error-handling.mdc) | Validation, exceptions, logging |
| [`testing.mdc`](testing.mdc) | pytest layout, CASA-free unit tests, simulated end-to-end tests |
| [`cicd.mdc`](cicd.mdc) | GitHub Actions, packaging, containers, branches/commits |

Naming, OOP and policy conventions follow [Pyralysis](https://gitlab.com/clirai/pyralysis) so the two projects read the same. Pyralysis's array-backend, precision and UML rules have no counterpart here. How agents use these rules, with skills and hooks, is in [`AGENTS.md`](../../AGENTS.md).

## Keeping `CONVENTIONS.md` in sync

The `.mdc` files are the source of truth. After editing them, regenerate the mirror:

```bash
python tools/sync_conventions.py
```
