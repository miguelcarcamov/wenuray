"""
The wenuray command (#36), a thin layer over the recipes of wenuray.recipe:

    wenuray init > selfcal.toml         a generic recipe to start from
    wenuray validate selfcal.toml       builds the run and checks it, without touching the data
    wenuray show selfcal.toml           the recipe with its presets laid in (--defaults: all values)
    wenuray run selfcal.toml            runs it

--ms and --output reuse a recipe for another dataset. Errors are the constructors' own, as from
Python, and exit with status 1.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings
from typing import Optional, Sequence

import tomli_w

TEMPLATE = '''\
# A Wenuray self-calibration recipe. Each table is a constructor and its keys are that class's
# arguments; see https://github.com/miguelcarcamov/wenuray#the-same-run-as-a-toml-recipe
# Paths are relative to this file.

ms = "target.ms"               # the calibrated target data in DATA (split them first)
# output = "target_selfcal"    # default: <ms name>_selfcal, next to the measurement set
# preset = "group.toml"        # settings you share across datasets, laid underneath this recipe

[selection]                    # one selection for the solve, the apply and the imager
field = ""
spw = ""

[solve]                        # gaincal options shared by every step; steps may override them
refant = ""                    # e.g. a central antenna with good data
minblperant = 4
minsnr = 3.0
gaintype = "T"                 # "T": one solution for both polarizations, "G": one for each

[imager]
type = "tclean"                # or "pyralysis", or "package.module:Class" for your own
niter = 1000
nsigma = 3.0
# Your data's grid: the cell must be no coarser than the Nyquist cell of the selected data
geometry = { imsize = 1024, cell = "0.1arcsec" }
weighting = { weighting = "natural" }

[[steps]]                      # one round each, in order
type = "phase"                 # "phase", "amp" or "ampphase"
solint = "inf"

[[steps]]
type = "phase"
solint = "60s"

# [[steps]]
# type = "ampphase"            # amplitude steps need a phase solution before them
# solint = "inf"
# imaging = { nsigma = 2.0 }   # this round's imager options
'''


def _parser() -> argparse.ArgumentParser:
    from . import __version__

    parser = argparse.ArgumentParser(
        prog="wenuray", description="Radio-interferometric self-calibration from TOML recipes"
    )
    parser.add_argument("--version", action="version", version="wenuray " + __version__)
    parser.add_argument(
        "--traceback", action="store_true", help="show the full traceback of an error"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    def recipe_command(name: str, help: str) -> argparse.ArgumentParser:
        command = commands.add_parser(name, help=help, description=help)
        command.add_argument("recipe", help="TOML recipe")
        command.add_argument("--ms", help="measurement set to use instead of the recipe's")
        command.add_argument("--output", help="output directory to use instead of the recipe's")
        return command

    run = recipe_command("run", "run a recipe")
    run.add_argument("--plots", action="store_true", help="plot the gain solutions")
    run.add_argument("--quiet", action="store_true", help="log warnings only")
    recipe_command("validate", "build a recipe's run and check it, without touching the data")
    show = recipe_command("show", "print a recipe with its presets laid in")
    show.add_argument("--defaults", action="store_true", help="also print the default values")
    init = commands.add_parser("init", help="write a generic recipe to start from")
    init.add_argument("-o", "--output", help="file to write instead of the standard output")
    return parser


def _load(args: argparse.Namespace):
    from . import recipe
    from .selfcalibration import Selfcal

    table = recipe.read(args.recipe)
    if args.ms:
        table["ms"] = os.path.abspath(args.ms)
        if not args.output:
            table.pop("output", None)  # next to the new measurement set
    if args.output:
        table["output"] = os.path.abspath(args.output)
    return Selfcal.from_recipe(table)


def _run(args: argparse.Namespace) -> int:
    from .selfcalibration import GainPlotObserver, LoggingObserver

    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO)
    selfcal = _load(args)
    observers = [LoggingObserver()] + ([GainPlotObserver()] if args.plots else [])
    result = selfcal.run(observers)
    print(
        "{0} of {1} rounds accepted; final image {2}".format(
            len(result.accepted_rounds), len(result.rounds), result.final.image
        )
    )
    return 0


def _validate(args: argparse.Namespace) -> int:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        selfcal = _load(args)
    for warning in caught:
        print("warning: {0}".format(warning.message))
    if not os.path.exists(selfcal.ms):
        print("warning: {0} does not exist (yet)".format(selfcal.ms))
    print("{0}: valid".format(args.recipe))
    print("  ms:      {0}".format(selfcal.ms))
    print("  output:  {0}".format(selfcal.output))
    print("  imager:  {0}".format(type(selfcal.imager).__name__))
    for index, step in enumerate(selfcal.steps, start=1):
        print("  step {0}:  {1}({2!r})".format(index, type(step).__name__, step.solint))
    return 0


def _show(args: argparse.Namespace) -> int:
    selfcal = _load(args)
    directory = os.path.dirname(os.path.abspath(args.recipe))
    sys.stdout.write(
        tomli_w.dumps(selfcal.to_recipe(relative_to=directory, defaults=args.defaults))
    )
    return 0


def _init(args: argparse.Namespace) -> int:
    if args.output is None:
        sys.stdout.write(TEMPLATE)
        return 0
    if os.path.exists(args.output):
        raise FileExistsError("{0} already exists".format(args.output))
    with open(args.output, "w") as file:
        file.write(TEMPLATE)
    return 0


COMMANDS = {"run": _run, "validate": _validate, "show": _show, "init": _init}


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    Runs the wenuray command; returns its exit status
    """
    args = _parser().parse_args(argv)
    try:
        return COMMANDS[args.command](args)
    except (ValueError, TypeError, KeyError, OSError) as error:
        # Recipe, schedule and constructor errors (RecipeError and ScheduleError are ValueErrors)
        if args.traceback:
            raise
        print("wenuray: error: {0}".format(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
