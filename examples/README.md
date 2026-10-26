# Examples

Each example is a self-calibration procedure in both of Wenuray's forms, which build the same run (a
test checks it):

- `recipe.toml`, run with `wenuray run recipe.toml`;
- `selfcal.py`, the same run written in Python, plus any data preparation around it.

| Example | Shows |
| --- | --- |
| [`alma/`](alma/) | Phase rounds with spectral windows combined, then amplitude-phase per spectral window, cleaned deeper; automasking for long baselines |
| [`vla/`](vla/) | Phase then amplitude-phase, `gaintype = "G"`, primary-beam correction |
| [`emerlin/`](emerlin/) | Continuing from an existing phase table (`initial_stack`), amplitude-phase rounds from long to short solints, residual RFI flagging in each round, `applymode = "calonly"`, multiscale clean; the script also splits the target and flags RFI before self-calibrating |

The values (cell sizes, automasking thresholds, reference antennas, solution intervals) belong to
each example's data and are not Wenuray defaults: adapt them to yours. Check a recipe before running
it with `wenuray validate recipe.toml`, and see every value it sets with `wenuray show --defaults`.
