"""
Self-calibration of the tutorial's simulated ALMA observation (docs/tutorial.md), imaged with
tclean: the same run as recipe.toml. Make the data first with simulate.py.

Usage: python selfcal.py <visfile> <output_directory> [--plot]
"""
import argparse
import logging

from wenuray.config import ImageGeometry, SolveOptions
from wenuray.imaging import Tclean
from wenuray.selfcalibration import AmpPhasecal, GainPlotObserver, LoggingObserver, Phasecal, Selfcal


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("visfile", help="Measurement set with the simulated data in DATA")
    parser.add_argument("output", help="Directory for the images, tables and the result record")
    parser.add_argument("--plot", action="store_true", help="Plot the gain solutions")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    # Wenuray logs each self-calibration round's metrics at INFO
    logging.basicConfig(level=logging.INFO)

    # The cell comes from the data's Nyquist cell; 256 pixels hold the disk and some sky around it
    imager = Tclean(
        ImageGeometry(imsize=256, cell="auto"), niter=2000, nsigma=3.0, usemask="auto-multithresh"
    )
    # The simulated gains change every integration: a solution over the whole observation can't
    # follow them, solutions every integration can
    steps = [Phasecal("inf"), Phasecal("int"), AmpPhasecal("int")]
    selfcal = Selfcal(
        args.visfile,
        imager,
        steps,
        solve=SolveOptions(refant="A042"),  # the antenna nearest the array's centre
        output=args.output,
    )
    observers = [LoggingObserver()] + ([GainPlotObserver()] if args.plot else [])
    selfcal.run(observers)
