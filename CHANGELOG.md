<!-- markdownlint-configure-file { "MD024": { "siblings_only": true } } -->

# Changelog

All notable changes to Wenuray. The entries are built with
[towncrier](https://towncrier.readthedocs.io/) from the news fragments in
[`changes/`](https://github.com/miguelcarcamov/wenuray/tree/development/changes); see its README
to add one.

<!-- towncrier release notes start -->

## [1.2.0](https://github.com/miguelcarcamov/wenuray/releases/tag/v1.2.0) (2026-11-04)

### Features

- An "auto" image grid: `ImageGeometry(cell="auto", imsize="auto")` derives the cell from the Nyquist cell of the selected data (over `oversampling`, default 2.5) and the size from the primary beam (`field_of_view` FWHMs, default 2.5), once when the run starts, and records them. ([#67](https://github.com/miguelcarcamov/wenuray/issues/67))
- Apply a run's accepted calibration to other data of the same observation (the line data, other spws, a split MS): `ApplyOptions(targets=...)` / `[[apply.targets]]` in a recipe, `SelfcalResult.apply_to()`, or `wenuray apply`. Spectral windows are matched to the tables' by frequency, and each application is recorded in `selfcal.toml`. ([#68](https://github.com/miguelcarcamov/wenuray/issues/68))
- A warning before an amplitude step that solves per spectral window, across a band wide enough for its gains to absorb the source's spectral index, against a model without a spectrum; imagers say whether they model one with `Imager.models_spectrum`. ([#118](https://github.com/miguelcarcamov/wenuray/issues/118))
- Each round records and logs its residual noise against the thermal noise estimated from the data (natural weighting; a floor for other weightings), and warns when the residual inside the source mask is below 0.9x thermal: the model and gains are fitting noise there. ([#119](https://github.com/miguelcarcamov/wenuray/issues/119))
- Each solved round records diagnostics of its gain table (`gains` in `selfcal.toml`): its phase and amplitude corrections against their noise (near 1: the round corrected nothing beyond noise), the antennas that scatter more than their distance from the array centre predicts, and X against Y for gaintype G. ([#126](https://github.com/miguelcarcamov/wenuray/issues/126))
- Pre-flight compares the data's measured noise with the noise their weights (SIGMA) state, per baseline and spectral window, and warns when the weights misstate it by more than 10% or a baseline stands out; the initial round records the median ratio as `noise_over_sigma`. ([#129](https://github.com/miguelcarcamov/wenuray/issues/129))
- The Pyralysis worker can run with another Python: set `WENURAY_PYRALYSIS_PYTHON` to one with Pyralysis; by default it runs with Wenuray's own (#170). The run record takes Pyralysis's version, and its environment's, from the worker. ([#148](https://github.com/miguelcarcamov/wenuray/issues/148))
- Conda environment files for Pyralysis on CPU and on CUDA 11, 12 and 13 (`envs/`), each with CUDA and CuPy from conda-forge; since #170 each holds Wenuray too. ([#155](https://github.com/miguelcarcamov/wenuray/issues/155))
- One environment for Wenuray, CASA and Pyralysis. `requirements.txt` pins Wenuray's dependencies and Pyralysis's together, and one conda or micromamba command installs it all: `environment.yml` on CPU, `envs/wenuray-cuda11.yml`, `-cuda12.yml` or `-cuda13.yml` on a GPU, with CUDA and CuPy from conda-forge. Each file stands alone, so it can be downloaded without a clone. The Docker images hold one environment too, and `WENURAY_PYRALYSIS_PYTHON` is no longer needed. The installation guide chooses the GPU file by both the GPU and the driver, and explains the errors of a driver too old for it and of casatools's `manylinux_2_34` wheel. ([#170](https://github.com/miguelcarcamov/wenuray/issues/170))
- `wenuray run` tells what it is doing: each phase (resetting the data, checking it, solving, applying, imaging) as it starts and how long it took, and Pyralysis's iterations as it works, instead of nothing until the round ends. Log lines carry their time; `--debug` adds the details, e.g. all of Pyralysis's output. Observers get the phases through a new hook, `SelfcalObserver.on_phase`. ([#176](https://github.com/miguelcarcamov/wenuray/issues/176))
- The log reads as a trace of the run: each round ends with a block of its solutions (how many failed minsnr and on which antennas, intervals with too few antennas to determine the gains), its gains against their noise ("a correction" or "consistent with noise", judged on as many solutions as the table has), its image against the thermal noise, and its decision; a round rejected for lack of S/N suggests a longer solint or combined spectral windows. The run ends with a table of every round. `selfcal.toml` records each round's solutions, and `wenuray run` reports steps accepted, not rounds. ([#179](https://github.com/miguelcarcamov/wenuray/issues/179))
- A table solved per spectral window that lacks windows is explained in the round's solutions (#179). Windows without unflagged data, e.g. those a split keeps by their original ids, make CASA's applycal print "The following MS spws have no corresponding cal spws", and Wenuray says that's harmless. A selected window with unflagged data but no solutions is left uncalibrated, not flagged, by applycal: Wenuray warns about it and records it (`uncalibrated_spws`). `wenuray run` keeps two routine CASA notices out of the terminal ("Forcing use of OLD VisibilityIterator", "Deleting version: wenuray_..."), except with `--debug`.

### Bug fixes

- The predicted solution S/N counts only the spectral windows that hold data when the spw selection is empty, instead of every `SPECTRAL_WINDOW` row; for ALMA splits it was off by about a factor 2. ([#115](https://github.com/miguelcarcamov/wenuray/issues/115))
- Declare `six`, which casatasks imports without declaring: since python-dateutil 4.0 no longer installs it, `import casatasks` failed in a fresh environment. ([#157](https://github.com/miguelcarcamov/wenuray/issues/157))
- The Pyralysis imager restores its residual and restored images, and fits their beam, with the weighting it reconstructs with; they were naturally weighted whatever the `Weighting`, so a round's noise and peak S/N described another image. ([#163](https://github.com/miguelcarcamov/wenuray/issues/163))
- `envs/wenuray-cuda11.yml` installs CUDA 11.8, the version its CuPy 13 is built against, instead of 11.4; the two mixed in one process and cuBLAS failed (`CUBLAS_STATUS_INVALID_VALUE`). Its minimum driver is now R520. Wenuray also runs Pyralysis with the conda environment's CUDA (`CUDA_PATH`), so a system CUDA on the host can't mix its libraries in. ([#173](https://github.com/miguelcarcamov/wenuray/issues/173))

### Documentation

- A gallery (docs/gallery.md, examples/gallery/): nine simulated scenarios, VLA and ALMA at low and high frequency in compact and extended configurations and MeerKAT at UHF, each observing a target typical for its band with gain errors that drift as the atmosphere's, self-calibrated the same way with tclean and with Pyralysis and drawn before and after.
- A tutorial (docs/tutorial.md, examples/simulated/): simulate an ALMA observation of a disk with Pyralysis, corrupt it with antenna gain errors and thermal noise, self-calibrate it with Wenuray, and compare the images and the recovered gains with the truth. The README shows its before/after figure, and links the documentation on Read the Docs.
- The Imagers page explains how Pyralysis predicts visibilities and grids its residual image, with options for accurate gridding (clirai/pyralysis#414), and that a weakly regularised model fits the noise inside its mask.
- The gallery's images, remade with natural weighting for both imagers: Pyralysis reconstructs with a Gaussian-sinc kernel on a grid padded by 2, and its residual and restored images are rebuilt from its models with the same kernel (examples/gallery/restore_pyralysis.py, until Pyralysis grids them with its own options); MeerKAT's are restored with tclean's beam, as Pyralysis's own, from the uv moments, is half its size there. The page states where the comparison isn't like for like: Pyralysis's mask comes from tclean, and Pyralysis fits part of the noise inside it.
- The installation guide gives Pyralysis's environment a section of its own, with how to create it without a clone of the repository, and links to the environment files on `master`.

### Packaging and maintenance

- Internal: the self-calibration loop is split into a run state, round 0 and one round at a time (`_RunState`, `_start`, `_run_round`), with the schedule consumed as an iterable; results are unchanged. ([#140](https://github.com/miguelcarcamov/wenuray/issues/140))
- Internal: the run's advisory checks (predicted S/N, absorbed spectrum, misstated weights, residual below thermal) are objects the loop runs at fixed points; their findings also reach observers (`SelfcalObserver.on_finding`) and are recorded with each round (`findings` in `selfcal.toml`). ([#141](https://github.com/miguelcarcamov/wenuray/issues/141))
- CI runs the build on pushes to `master` too, so release pull requests have a CASA data cache to fall back on when the measures server is slow. ([#184](https://github.com/miguelcarcamov/wenuray/issues/184))
- Docker images with Pyralysis (#148): `:latest` runs it on CPU, the new `:gpu` on CUDA 12 (CuPy and numba-cuda, with NVIDIA's pip wheels; `docker run --gpus all`), in the same environment as Wenuray (#170).

## [1.1.0](https://github.com/miguelcarcamov/wenuray/releases/tag/v1.1.0) (2026-10-29)

### Features

- Support Python 3.11 (with CASA 6.7.0, the newest CASA with 3.11 wheels), and test every supported Python version with tox. ([#99](https://github.com/miguelcarcamov/wenuray/issues/99))

### Bug fixes

- An empty spw selection now means the spectral windows that hold data, so combined-spw solutions are applied when the measurement set has unused `SPECTRAL_WINDOW` rows. ([#102](https://github.com/miguelcarcamov/wenuray/issues/102))
- Relative file options of an imager, e.g. Pyralysis's `mask_path`, are resolved against the recipe file that names them; given from Python, Pyralysis finds them from the caller's working directory. ([#103](https://github.com/miguelcarcamov/wenuray/issues/103))

### Documentation

- Document the release naming convention: each major version is named after a woman who shaped radio astronomy. ([#97](https://github.com/miguelcarcamov/wenuray/issues/97))
- A friendlier README, introducing self-calibration and why Wenuray exists, with new installation, guide and imagers pages, and a `CITATION.cff` for GitHub's "Cite this repository" and Zenodo. ([#106](https://github.com/miguelcarcamov/wenuray/issues/106))
- The README badge, the BibTeX entry and `CITATION.cff` give Wenuray's Zenodo DOI, [10.5281/zenodo.23775089](https://doi.org/10.5281/zenodo.23775089).

### Packaging and maintenance

- CI installs what the CASA data fetch needs before running it, so pull requests into `master` can run the tests, and a fallback to cached measures data is reported as a warning. ([#109](https://github.com/miguelcarcamov/wenuray/issues/109))
- Dependency updates: multimethod 2.1, python-dateutil 3.0, newer Sphinx extensions and theme, and newer GitHub Actions.
- Keep a changelog, `CHANGELOG.md`, built with towncrier from a news fragment per pull request.

## [1.0.0](https://github.com/miguelcarcamov/wenuray/releases/tag/v1.0.0) (2026-10-26)

**"Payne-Scott"**, after Ruby Payne-Scott (1912–1981), the first woman radio astronomer and a
pioneer of radio interferometry.

The first release as Wenuray, previously *snow*: a redesign around one self-calibration loop
(#22). See the [release notes](https://github.com/miguelcarcamov/wenuray/releases/tag/v1.0.0) and
the [migration guide](https://github.com/miguelcarcamov/wenuray/blob/development/docs/migration.md).
