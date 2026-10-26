# Wenuray

## Radio-interferometric self-calibration

**Wenuray** is a Python framework for radio-interferometric self-calibration. Instead of
rewriting a calibration script for each dataset, you describe the procedure once, as a short TOML
recipe or a few Python objects, and Wenuray runs it: imaging with CASA tclean, Pyralysis or your
own imager, solving phases and amplitudes round after round, and keeping only the rounds that
improve the image. It works with any radio interferometer.

### About the name

*Wenuray* (also written *Huenuray*) is a Mapuche woman's name. In Mapudungun it means **"flower of
the sky"**: *wenu*, the sky, and *rayen*, flower. The radio sky is what Wenuray brings into focus,
and its roots are in Chile, home of many of the world's radio telescopes.

Wenuray is dedicated to the women of radio astronomy. Each major release carries one of their
names, beginning with **v1.0 "Payne-Scott"**, after Ruby Payne-Scott (1912–1981), the first woman
radio astronomer and a pioneer of radio interferometry.

Wenuray was called *snow* (ca**S**a pytho**N** self-calibrati**O**n frame**W**ork) before
version 1.0.

### Features

- **Object-oriented design**: Clean, modular architecture for easy customization
- **Multiple imagers supported**: CASA tclean, Pyralysis, or your own
- **Flexible self-calibration**: Phase-only, amplitude-only, or combined amplitude-phase calibration
- **CASA integration**: Built on CASA (Common Astronomy Software Applications) for radio astronomy data processing
- **Reproducible workflows**: Version-controlled calibration pipelines
- **Docker support**: Pre-built container images for easy deployment

## Requirements

- **Python**: `>=3.12, <3.14`, the versions with both CASA wheels and Pyralysis support
- **Dependencies**: casatools, casatasks (CASA >= 6.7), numpy, astropy and reproject. The
  package declares lower bounds; [`requirements.txt`](https://github.com/miguelcarcamov/wenuray/blob/development/requirements.txt) pins the set tested in CI
  and used in the Docker images.
- **CASA data**: CASA downloads its measures data the first time `casatools` is imported
  (see [External Data](https://casadocs.readthedocs.io/en/stable/notebooks/external-data.html)).
  Without network access, point `measurespath` to an existing copy.

On Debian, Ubuntu or Arch, the `manylinux_2_34` casatools wheel that pip picks fails to import
(`OPENSSL_3.0.1 not found`). Install from `requirements.txt`, which uses the working
`manylinux_2_28` wheel, or install that wheel first.

## Installation

### Option 1: From PyPI (Recommended)

```bash
pip install wenuray
```

The package is imported as `wenuray`. (`pip install wenuray` installs an unrelated project.) Until
the first release is published, install from GitHub instead.

### Option 2: From GitHub

```bash
pip install -U git+https://github.com/miguelcarcamov/wenuray.git
```

### Option 3: Using Conda

Create a conda environment from the provided environment file:

```bash
conda env create -f environment.yml
conda activate wenuray-env
```

The environment provides Python 3.12 and the pinned dependencies from `requirements.txt`; then
install Wenuray itself with `pip install -e .` (from a clone) or `pip install wenuray`.

### Option 4: From Source

```bash
git clone https://github.com/miguelcarcamov/wenuray
cd wenuray
pip install .
```

### Option 5: Development Installation

For development work with editable installation:

```bash
git clone https://github.com/miguelcarcamov/wenuray
cd wenuray
pip install -e .
```

### Option 6: Docker Container

Pull the pre-built Docker image:

```bash
docker pull ghcr.io/miguelcarcamov/wenuray:latest
```

The image has Python 3.12, Wenuray with its pinned dependencies, and CASA's measures data and
observatory table in `/opt/casa/data`, readable by any user (also under Apptainer/Singularity). To stay small it has
no CUDA, and it skips the rest of casarundata (antenna responses and simulator files, used by
e.g. the `awproject` gridder); mount a full copy at `/opt/casa/data` if you need it:

```bash
docker run --rm -v "$PWD":/work -w /work ghcr.io/miguelcarcamov/wenuray:latest python main_ALMA.py target.ms out
```

CASA needs a writable home directory. Apptainer/Singularity provides yours; with `docker run -u`,
also pass `-e HOME=/tmp`.

`wenuray.utils.reproject()` needs the optional `reproject` package: `pip install "wenuray[reproject]"`.

## Quick Start

### Basic Usage Example

Phase, then amplitude-phase self-calibration of an ALMA dataset. This is
[`examples/alma/selfcal.py`](https://github.com/miguelcarcamov/wenuray/blob/development/examples/alma/selfcal.py); a test keeps the two identical.

```python
"""
Phase, then amplitude-phase self-calibration of an ALMA dataset imaged with tclean.

Usage: python selfcal.py <visfile> <output_directory> [--plot]
"""
import argparse
import logging

from casatasks import split

from wenuray.config import ImageGeometry, SolveOptions, Weighting
from wenuray.imaging import Tclean
from wenuray.selfcalibration import AmpPhasecal, GainPlotObserver, LoggingObserver, Phasecal, Selfcal


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("visfile", help="Measurement set with the calibrated data in DATA")
    parser.add_argument("output", help="Directory for the images, tables and the result record")
    parser.add_argument("--plot", action="store_true", help="Plot the gain solutions")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    # Wenuray logs each self-calibration round's metrics at INFO
    logging.basicConfig(level=logging.INFO)

    # tclean's own parameter names. Automasking values for long baselines:
    # https://casaguides.nrao.edu/index.php/Automasking_Guide
    imager = Tclean(
        ImageGeometry(imsize=1024, cell="0.005arcsec"),
        Weighting("briggs", robust=0.5),
        niter=100,
        threshold="0.1mJy",
        specmode="mfs",
        deconvolver="hogbom",
        gridder="standard",
        usemask="auto-multithresh",
        sidelobethreshold=3.0,
        noisethreshold=5.0,
        minbeamfrac=0.3,
        lownoisethreshold=1.5,
        negativethreshold=0.0,
        interactive=False  # True to edit the masks by hand
    )

    # Phases combined over spectral windows, then amplitudes and phases per spectral window,
    # cleaned deeper. Each spwmap is derived from combine.
    steps = [
        Phasecal("inf", solve={"combine": "spw"}),
        Phasecal("600s", solve={"combine": "spw"}),
        AmpPhasecal("inf", imaging={"threshold": "0.025mJy"}),
    ]
    selfcal = Selfcal(
        args.visfile,
        imager,
        steps,
        solve=SolveOptions(refant="DA51", minblperant=6, minsnr=3.0, gaintype="T"),
        output=args.output,
    )
    observers = [LoggingObserver()] + ([GainPlotObserver()] if args.plot else [])
    result = selfcal.run(observers)

    # The self-calibrated data, for final imaging
    split(vis=result.ms, outputvis=args.output + "/selfcal.ms", datacolumn="corrected")
```

Run the script:

```bash
python examples/alma/selfcal.py target.ms target_selfcal
```

### The same run as a TOML recipe

A recipe is the same objects written as TOML: each table is a constructor and its keys are that
class's arguments, so a recipe and a script build equal objects and raise the same errors. The
imager, the steps, the acceptance policies and the flagging name their class with `type`.

```toml
ms = "target.ms"           # paths are relative to the recipe
output = "target_selfcal"

[solve]
refant = "DA51"
minblperant = 6

[imager]
type = "tclean"
niter = 100
threshold = "0.1mJy"
deconvolver = "hogbom"
usemask = "auto-multithresh"
geometry = { imsize = 1024, cell = "0.005arcsec" }
weighting = { weighting = "briggs", robust = 0.5 }

[[steps]]
type = "phase"             # Phasecal; "amp" is Ampcal, "ampphase" AmpPhasecal
solint = "inf"
solve = { combine = "spw" }

[[steps]]
type = "phase"
solint = "600s"
solve = { combine = "spw" }

[[steps]]
type = "ampphase"
solint = "inf"
imaging = { threshold = "0.025mJy" }
```

```python
selfcal = Selfcal.from_toml("recipe.toml")  # an ordinary Selfcal: inspect it, change it, run it
selfcal.to_toml("copy.toml")                # and back; values equal to the defaults are left out
```

Or from the command line:

```bash
wenuray init > recipe.toml                 # a generic recipe to start from
wenuray validate recipe.toml               # builds the run and checks it, without touching the data
wenuray show --defaults recipe.toml        # the resolved recipe: presets laid in, every value shown
wenuray run recipe.toml --plots            # runs it, plotting the gain solutions
wenuray run recipe.toml --ms other.ms      # the same procedure on another dataset
```

Errors are the same as from Python, e.g. a misspelt `refant` or an amplitude step before any
phase step, and `wenuray` exits with status 1.

### Reproducing a run

Every run records how to repeat it in its output directory:

- **`recipe.toml`**: the run as a resolved recipe, written before anything runs, with presets laid
  in and every default written out, so later changes to Wenuray's defaults can't change it.
- **`selfcal.toml`**: the rounds and accepted tables, plus a `[provenance]` table with the start
  time, a fingerprint of the input measurement set (a hash of its metadata and of sampled `DATA`
  and `FLAG` rows) and the versions of Python, Wenuray, CASA and the imager's packages.

```bash
wenuray run target_selfcal/recipe.toml --ms fresh_copy.ms --output rerun
```

reruns it on a fresh copy of the input. A run Wenuray can't write as a recipe, e.g. one with a
`Check(name, lambda ...)` policy, still runs; it logs a warning and records no recipe.

`preset = "group.toml"` lays a recipe over another TOML file, e.g. the solve options and imager
settings your group shares: tables merge key by key and the recipe wins. Wenuray ships no presets,
since its defaults come from theory or the data rather than from an instrument. A class of your
own, e.g. an acceptance policy, is named as `type = "package.module:Class"`. Policies built with
`Check(name, function)` can't be written to a recipe; make them a class.

How a run is described:

- **One `Selfcal`, a list of steps.** Each step is one round: `Phasecal`, `Ampcal` or
  `AmpPhasecal` with its solution interval. A step can override the run's solve options
  (`solve={"minsnr": 5.0}`), the imager's options (`imaging={"nsigma": 2.0}`) or the acceptance
  policy (`acceptance=...`), and can flag residual outliers before imaging
  (`flagging=ResidualFlagging("rflag")`, off by default).
- **Invalid schedules fail before any I/O.** An amplitude step needs a phase solution before it,
  in the schedule or in `initial_stack`; otherwise `Selfcal(...)` raises `ScheduleError`.
- **Every accepted table stays applied.** Each solve pre-applies all accepted tables, each with
  its own `spwmap` (derived from `combine`) and `interp`.
- **A round is kept only if it improves the image.** By default the PSNR must rise and every
  solution interval must keep enough antennas (3 for phases, 4 for amplitudes), and amplitude
  rounds must not lose model flux. A rejected round is rolled back and stops the run.
- **Each solve is checked first.** Before `gaincal`, Wenuray predicts a solution's S/N for the
  step's solint and combine from the data: the median model amplitude over the noise of one
  visibility, times √(N_ant − 1) · √(n_int · n_chan · n_pol). It warns when the prediction is
  below `minsnr`, giving the shortest solint that would reach it, and records it as
  `predicted_snr`.
- **Continue later** with `Selfcal.resume(result_or_output_directory, imager, more_steps)`, or
  start from tables of your own with `initial_stack=CalStack([CalTable("pcal", "p")])`.
- **`result.final`** is the last accepted round; every round's metrics, decision and reasons
  are in `result.rounds` and in `<output>/selfcal.toml`.

What the input must look like:

- **Calibrated data in `DATA`.** `gaincal` solves against `DATA`, while the imager images
  `CORRECTED_DATA`. Split the calibrated target first
  (`split(vis=..., outputvis=..., datacolumn="corrected")`). A run resets `CORRECTED_DATA` to
  `DATA` (plus `initial_stack`) and warns if it held something else.
- **One selection.** `Selection(field=..., spw=..., uvrange=...)` is used by the solve, the
  apply and the imager.
- **`applymode`.** The default `ApplyOptions(applymode="calflag")` flags data whose solutions fail
  `minsnr`. `"calonly"` leaves those data uncalibrated but unflagged.

More examples, for the VLA and e-MERLIN too, are in [`examples/`](https://github.com/miguelcarcamov/wenuray/tree/development/examples), each as a recipe and
the script that builds the same run. Code written for snow, the name before
Wenuray 1.0, raises an error that shows the new call; see the [migration guide](https://github.com/miguelcarcamov/wenuray/blob/development/docs/migration.md).

## API Overview

### Self-calibration (`wenuray.selfcalibration`)

- **`Selfcal`**: the self-calibration loop; `run()` returns a `SelfcalResult`, and `resume()`
  continues one.
- **`Phasecal`, `Ampcal`, `AmpPhasecal`**: calibration steps.
- **Acceptance policies**: `PSNRImproves`, `EnoughAntennas`, `FluxNotLost`, `BeamStable`,
  `FlaggedFractionBelow`, `Check(name, predicate)` for your own, combined with `AllOf`/`AnyOf`.
- **Observers**: `LoggingObserver` (the default) and `GainPlotObserver`, passed to `run()`.

### Configuration (`wenuray.config`)

Frozen objects with CASA's parameter names: `Selection`, `SolveOptions`, `ApplyOptions`,
`ImageGeometry` and `Weighting` (natural by default).

### Imaging (`wenuray.imaging`)

An imager images `CORRECTED_DATA` and writes its model to `MODEL_DATA`; `Imager.image()` enforces
both and raises `ImagerContractError` otherwise. It also raises `ValueError`, before imaging, when
the cell is coarser than the Nyquist cell of the selected data: the image grid can't hold the
longest baselines, so the model written for them would mislead the next solve. The message gives
the largest cell that works, and the `Selection(uvrange=...)` that leaves those baselines out of
the solve, the apply and the image alike.

- **`Tclean(geometry, weighting, **tclean_options)`**: CASA tclean, with tclean's own parameter
  names.
- **`Pyralysis(geometry, weighting, terms=[...], optimizer=..., **pyralysis_options)`**:
  optimization-based reconstruction with [Pyralysis](https://gitlab.com/clirai/pyralysis), with
  its own names for terms (`"chi_squared"`, `"l1norm:1e-3"`), the optimizer (`"lbfgsb"`, `"fista"`)
  and options (`max_iter`, `mask_mode`, ...). Each reconstruction runs in a separate process,
  since Pyralysis's casacore and CASA's can't share one. **Give it a mask** (`mask_mode="file"`,
  `mask_path=...`): a model free in every pixel can absorb the gain errors self-calibration
  should solve for. Wenuray needs Pyralysis 3.2.1 or later, from PyPI; its
  `ska-ost-array-config` dependency comes from SKA's index:

  ```bash
  pip install "wenuray[pyralysis]" \
    --extra-index-url https://artefact.skao.int/repository/pypi-internal/simple
  ```

  A `Selection(uvrange=...)` works with Pyralysis in metres or km (e.g. `"<3090m"`, the form the
  Nyquist check suggests); a range in wavelengths is refused, since it depends on each channel's
  frequency.

  Numba needs OpenMP's runtime (`libgomp1`, present on most systems and in Wenuray's images) or TBB
  for Pyralysis's parallel kernels; Wenuray checks for it when the imager is created.

- **Your own imager**: subclass `Imager` and implement `_reconstruct`, wrap a function in
  `CallableImager`, or register it under the `wenuray.imagers` entry point to build it by name.
  `wenuray.testing.ImagerContractSuite` checks that it follows the contract.

## Project Structure

```
wenuray/
├── src/wenuray/
│   ├── config.py         # Selection, SolveOptions, ApplyOptions, ImageGeometry, Weighting
│   ├── imaging/          # Imager contract, tclean backend, image statistics
│   ├── selfcalibration/  # The Selfcal loop, steps, schedule rules, acceptance, observers
│   ├── testing.py        # Simulated data and the imager contract suite
│   └── utils/            # Utility functions
├── examples/             # Recipe + script pairs (ALMA, VLA, e-MERLIN)
├── docs/                 # Sphinx documentation and the migration guide
├── requirements.txt      # Pinned dependencies tested in CI
└── pyproject.toml        # Project metadata and build configuration
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the terms specified in the `LICENSE` file.

## Citation

If you use Wenuray in your research, please cite:

```bibtex
@software{wenuray,
  author = {Cárcamo, Miguel},
  title = {Wenuray: radio-interferometric self-calibration},
  url = {https://github.com/miguelcarcamov/wenuray},
  version = {<version>},
  year = {<year>}
}
```

## Support

- **Issues**: Report bugs or request features on [GitHub Issues](https://github.com/miguelcarcamov/wenuray/issues)
- **Documentation**: See the [CASA documentation](https://casadocs.readthedocs.io/) for CASA-specific questions
- **Contact**: <miguel.carcamo@usach.cl>

## Acknowledgments

Wenuray is built on top of [CASA](https://casa.nrao.edu/) (Common Astronomy Software Applications), developed by the National Radio Astronomy Observatory (NRAO).
