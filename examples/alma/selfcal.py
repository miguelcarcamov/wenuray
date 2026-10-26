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
