# Examples

Each example is a self-calibration procedure in both of Wenuray's forms, which build the same run (a
test checks it):

- `recipe.toml`, run with `wenuray run recipe.toml`;
- `selfcal.py`, the same run written in Python, plus any data preparation around it.

| Example | Shows |
| --- | --- |
| [`alma/`](alma/) | Phase rounds with spectral windows combined, then amplitude-phase per spectral window, cleaned deeper; automasking for long baselines |
| [`vla/`](vla/) | Phase then amplitude-phase, `gaintype = "G"`, primary-beam correction |
| [`simulated/`](simulated/) | The [tutorial](https://wenuray.readthedocs.io/tutorial.html)'s simulated ALMA observation: `simulate.py` makes the data with Pyralysis (a disk, antenna gain errors, thermal noise), the recipe self-calibrates it with an `"auto"` cell and phase then amplitude-phase rounds every integration, and `make_figures.py` draws the results |
| [`gallery/`](gallery/) | The [gallery](https://wenuray.readthedocs.io/gallery.html): nine simulated scenarios (VLA and ALMA at low and high frequency, compact and extended, and MeerKAT at UHF), each with a target typical for its band and gain errors that drift as the atmosphere's, self-calibrated the same way with tclean and with Pyralysis; `make_gallery.py` simulates, self-calibrates and draws them all |
| [`emerlin/`](emerlin/) | Continuing from an existing phase table (`initial_stack`), amplitude-phase rounds from long to short solints, residual RFI flagging in each round, `applymode = "calonly"`, multiscale clean; the script also splits the target and flags RFI before self-calibrating |

The values (cell sizes, automasking thresholds, reference antennas, solution intervals) belong to
each example's data and are not Wenuray defaults: adapt them to yours. Check a recipe before running
it with `wenuray validate recipe.toml`, and see every value it sets with `wenuray show --defaults`.

`pyralysis_simulation.py` and `figures.py` sit next to the examples so others can share them: the
first simulates observations with Pyralysis and writes them as measurement
sets CASA reads, the second draws the figures in one style. The tutorial (`simulated/`) and the
gallery (`gallery/`) use both.
