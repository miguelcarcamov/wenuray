# Wenuray

[![PyPI](https://img.shields.io/pypi/v/wenuray)](https://pypi.org/project/wenuray/)
[![Python](https://img.shields.io/pypi/pyversions/wenuray)](https://pypi.org/project/wenuray/)
[![build](https://github.com/miguelcarcamov/wenuray/actions/workflows/build.yaml/badge.svg?branch=development)](https://github.com/miguelcarcamov/wenuray/actions/workflows/build.yaml)
[![Docs](https://readthedocs.org/projects/wenuray/badge/?version=latest)](https://wenuray.readthedocs.io/)
[![License: GPL v3+](https://img.shields.io/badge/license-GPLv3%2B-blue)](https://github.com/miguelcarcamov/wenuray/blob/development/LICENSE)

**Self-calibration for radio interferometers, written once and run anywhere.**

Wenuray lets you describe a self-calibration procedure as a short recipe (a dozen lines of TOML or
a few Python objects) and runs it on any measurement set, from any array. It images with CASA
tclean, Pyralysis or your own imager, solves for phases and then amplitudes round after round, and
keeps a round only if it really improves the image. Every run records what it did, so you or a
referee can repeat it exactly.

## Why self-calibration?

An interferometer measures visibilities, and every antenna *i* multiplies what it sees by its own
complex gain *gᵢ(t, ν)*:

```text
V_ij(observed) = g_i · g_j* · V_ij(true) + noise
```

Calibrating against a calibrator source removes most of those gains, but not all of them. The
atmosphere and the ionosphere change faster than the calibrator cycle, and the calibrator is in
another part of the sky. What's left over shows up as stripes, sidelobes and a noise floor far
above the thermal limit.

Self-calibration uses the target itself. An array of *N* antennas has *N(N−1)/2* baselines but only
*N* unknown gains, so even an imperfect model of the sky constrains the gains well. The loop is:

1. **Image** the data and keep the model.
2. **Solve** for the antenna gains that best match the data to that model (`gaincal`).
3. **Apply** them and image again. If the image is better, keep the solution and go on with a
   shorter interval or with amplitudes; if not, roll it back.

Doing this by hand is a script full of choices: solution intervals, reference antennas, which
tables to pre-apply with which `spwmap`, when to stop. Wenuray turns those choices into a recipe
it checks against self-calibration theory before it touches your data. For example, it refuses an
amplitude round with no phase solution before it, and it warns when a solution interval is too
short to reach your `minsnr`.

## Installation

```bash
pip install wenuray
```

You need Python 3.11, 3.12 or 3.13 on Linux or macOS, the platforms CASA ships wheels for. Conda,
Docker, source installs and the optional extras (Pyralysis, plots) are in the
[installation guide](https://github.com/miguelcarcamov/wenuray/blob/development/docs/installation.md).

> **Debian, Ubuntu or Arch?** The casatools wheel pip picks by default fails to import there
> (`OPENSSL_3.0.1 not found`). From a clone, `pip install -r requirements.txt .` pins a wheel that
> works.

## Quick start

Your data should already be calibrated, and the target split out so that the calibrated
visibilities sit in `DATA`:

```python
split(vis="calibrated.ms", outputvis="target.ms", field="my_target", datacolumn="corrected")
```

### Your first recipe

Phase, then amplitude-phase self-calibration of an ALMA dataset. The values (cell size, threshold,
reference antenna) belong to this dataset; adapt them to yours.

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

Three rounds: a phase solution over the whole scan, combining spectral windows for S/N; a shorter
one, now that the model is better; and finally amplitudes and phases, cleaned deeper. Check it,
then run it:

```bash
wenuray validate recipe.toml    # builds the run and checks it, without touching the data
wenuray run recipe.toml         # images, solves, applies and decides, round by round
```

`wenuray init > recipe.toml` writes a generic recipe to start from, and
`wenuray run recipe.toml --ms other.ms` runs the same procedure on another dataset.

### The same run in Python

Each table in a recipe is a constructor and its keys are that class's arguments, so a recipe and a
script build equal objects. This is
[`examples/alma/selfcal.py`](https://github.com/miguelcarcamov/wenuray/blob/development/examples/alma/selfcal.py);
a test keeps the two identical.

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

`Selfcal.from_toml("recipe.toml")` gives you the same `Selfcal` from the recipe, to inspect,
change or run; `selfcal.to_toml(...)` writes it back.

## What you get

- **A decision for every round.** By default a round is kept only if the peak S/N of the image
  rises, every solution interval keeps enough antennas, and amplitude rounds don't lose model
  flux: PSNR alone can rise while the model goes wrong. A rejected round is rolled back and the
  run stops. You can combine your own acceptance rules with `AllOf`/`AnyOf`.
- **Checks before any I/O.** An invalid schedule raises `ScheduleError` while the run is being
  built. Before each `gaincal`, Wenuray predicts the solution's S/N from the data and tells you the
  shortest solint that would reach `minsnr`.
- **Bookkeeping you don't have to do.** Every accepted table stays applied in later solves, each
  with its own `spwmap` and `interp`, and one `Selection` is used by the solve, the apply and the
  imager.
- **Reproducibility.** Each output directory holds the resolved `recipe.toml`, with every default
  written out, and a `selfcal.toml` with each round's metrics, the accepted tables and the
  provenance: a fingerprint of the input measurement set and the versions of Python, CASA,
  Wenuray and the imager.
- **Any imager.** CASA tclean, [Pyralysis](https://gitlab.com/clirai/pyralysis) (optimization-based
  reconstruction), or your own: subclass `Imager` or wrap a function. Each one images
  `CORRECTED_DATA` and writes `MODEL_DATA`, and Wenuray checks that it does.
- **Any array.** The defaults come from theory and from the data, not from an instrument: there are
  examples for ALMA, the VLA and e-MERLIN.

## Learn more

- [How a run works](https://github.com/miguelcarcamov/wenuray/blob/development/docs/guide.md):
  preparing the data, schedules, acceptance, recipes and presets, resuming, and an API overview.
- [Imagers](https://github.com/miguelcarcamov/wenuray/blob/development/docs/imagers.md): the
  imager contract, the Nyquist-cell check, tclean, Pyralysis and writing your own.
- [Examples](https://github.com/miguelcarcamov/wenuray/tree/development/examples): ALMA, VLA and
  e-MERLIN, each as a recipe and the script that builds the same run.
- [Migrating from snow](https://github.com/miguelcarcamov/wenuray/blob/development/docs/migration.md):
  code written for snow, the name before Wenuray 1.0, raises an error that shows the new call.
- [API reference](https://wenuray.readthedocs.io/) on Read the Docs.

## About the name

*Wenuray* (also written *Huenuray*) is a Mapuche woman's name. In Mapudungun it means **"flower of
the sky"**: *wenu*, the sky, and *rayen*, flower. The radio sky is what Wenuray brings into focus,
and its roots are in Chile, home of many of the world's radio telescopes.

Wenuray is dedicated to the women of radio astronomy. Each major release carries one of their
names, beginning with **v1.0 "Payne-Scott"**, after Ruby Payne-Scott (1912–1981), the first woman
radio astronomer and a pioneer of radio interferometry.

Wenuray was called *snow* (ca**S**a pytho**N** self-calibrati**O**n frame**W**ork) before
version 1.0.

## Citing Wenuray

If Wenuray helps your research, please cite it:

```bibtex
@software{wenuray,
  author = {Cárcamo, Miguel},
  title = {Wenuray: radio-interferometric self-calibration},
  url = {https://github.com/miguelcarcamov/wenuray},
  version = {<version>},
  year = {<year>}
}
```

## Contributing and support

Bug reports, questions and pull requests are welcome on
[GitHub](https://github.com/miguelcarcamov/wenuray/issues); see
[`CONTRIBUTING.md`](https://github.com/miguelcarcamov/wenuray/blob/development/CONTRIBUTING.md).
For CASA itself, the [CASA documentation](https://casadocs.readthedocs.io/) is the place to look.
You can also write to <miguel.carcamo@usach.cl>.

Wenuray is free software under the
[GNU General Public License v3 or later](https://github.com/miguelcarcamov/wenuray/blob/development/LICENSE).
It is built on [CASA](https://casa.nrao.edu/) (Common Astronomy Software Applications), developed
by the National Radio Astronomy Observatory (NRAO).
