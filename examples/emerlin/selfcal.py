"""
Amplitude-phase self-calibration of an e-MERLIN target, on top of existing phase solutions, imaged
with multiscale tclean. Optionally flags residual RFI before self-calibrating.

Usage: python selfcal.py <visfile> <output_directory> <field> [--phase-caltable pcal]
       [--phase-caltable-per-spw] [--mask MASK] [--uvrange UVRANGE] [--refant REFANT]
       [--flag [--rflag-spw SPW ...]] [--plot]
"""
import argparse
import logging
import os
from pathlib import Path

from casatasks import flagdata, flagmanager, mstransform, split

from wenuray.config import ApplyOptions, ImageGeometry, Selection, SolveOptions, Weighting
from wenuray.imaging import Tclean
from wenuray.selfcalibration import (
    AmpPhasecal, CalStack, CalTable, GainPlotObserver, LoggingObserver, ResidualFlagging, Selfcal
)
from wenuray.selfcalibration.caltable import derive_spwmap
from wenuray.utils import count_spectral_windows, get_spw_ids

# Amplitude-phase solution intervals, from long to short, cleaning deeper as calibration improves
SOLINTS = ["14min", "7min", "3.5min"]
NSIGMAS = [3.0, 2.5, 2.0]


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("visfile", help="Calibrated measurement set (CORRECTED_DATA is used)")
    parser.add_argument("output", help="Directory for the images, tables and the result record")
    parser.add_argument("field", help="Target field to self-calibrate")
    parser.add_argument(
        "--phase-caltable", default="pcal", help="Existing phase calibration table to pre-apply"
    )
    parser.add_argument(
        "--phase-caltable-per-spw",
        action="store_true",
        help="The phase table has one solution per spw (default: combined over spws)"
    )
    parser.add_argument("--mask", help="CASA mask image; automasking is used if omitted")
    parser.add_argument(
        "--uvrange", default="", help="uv-range to solve, apply and image, e.g. '>60klambda'"
    )
    parser.add_argument(
        "--refant", default="Kn,Cm,Mk2,Pi,De,Da", help="Reference antennas, in order of preference"
    )
    parser.add_argument(
        "--flag", action="store_true", help="Flag residual RFI with rflag before self-calibrating"
    )
    parser.add_argument(
        "--rflag-spw",
        action="append",
        default=None,
        help="Spectral window selection flagged together; repeat for several groups "
        "(default: all spws in one pass)"
    )
    parser.add_argument("--plot", action="store_true", help="Plot the gain solutions")
    return parser.parse_args()


def flag_residual_rfi(vis: str, spw_groups: list) -> None:
    flagmanager(vis=vis, mode='save', versionname='before_selfcal_flagging')
    for spw in spw_groups:
        flagdata(
            vis=vis,
            datacolumn="data",
            spw=spw,
            correlation="LL,RR",
            mode="rflag",
            action="apply",
            flagbackup=False
        )
        flagdata(
            vis=vis,
            datacolumn="data",
            spw=spw,
            mode="extend",
            extendflags=False,
            action="apply",
            flagbackup=False
        )
    flagmanager(vis=vis, mode='save', versionname='after_selfcal_flagging')


if __name__ == '__main__':
    args = parse_arguments()
    # Wenuray logs each self-calibration round's metrics at INFO
    logging.basicConfig(level=logging.INFO)

    # Self-calibration solves against DATA, so the calibrated target is split into DATA first
    selfcal_vis = Path(args.visfile).stem + ".flagged.ms"
    if not os.path.exists(selfcal_vis):
        mstransform(
            vis=args.visfile, outputvis=selfcal_vis, datacolumn="corrected", field=args.field
        )

    if args.flag:
        flag_residual_rfi(selfcal_vis, args.rflag_spw or [""])

    # tclean's own parameter names. Cell size and scales suit e-MERLIN at L-band; adjust them to
    # your band and target
    imager = Tclean(
        ImageGeometry(imsize=1024, cell="0.02arcsec"),
        Weighting("briggs", robust=2.0),
        niter=10000,
        scales=[0, 3, 5, 10, 15, 20, 30, 40, 50, 80, 100],
        specmode="mfs",
        deconvolver="multiscale",
        gridder="standard",
        pbcor=False,
        interactive=False,
        cycleniter=100,
        usemask="user" if args.mask else "auto-multithresh",
        mask=args.mask or ""
    )

    # The existing phase table is applied before every solve. Its spwmap: solutions combined over
    # spws are stored under the lowest one
    phase_spwmap = derive_spwmap(
        "" if args.phase_caltable_per_spw else "spw", get_spw_ids(selfcal_vis),
        count_spectral_windows(selfcal_vis)
    )
    phase_table = CalTable(args.phase_caltable, "p", tuple(phase_spwmap))

    # Each round flags residual RFI (CORRECTED_DATA - MODEL_DATA) before imaging, with flagdata's
    # names; a rejected round's flags are rolled back
    rflag = ResidualFlagging(
        "rflag",
        timedevscale=3.0,
        freqdevscale=3.0,
        spectralmax=500,
        extendflags=False,
        ntime="scan"
    )
    steps = [
        AmpPhasecal(solint, imaging={"nsigma": nsigma}, flagging=rflag)
        for solint, nsigma in zip(SOLINTS, NSIGMAS)
    ]
    selfcal = Selfcal(
        selfcal_vis,
        imager,
        steps,
        selection=Selection(uvrange=args.uvrange),
        solve=SolveOptions(
            refant=args.refant, minblperant=2, minsnr=3.0, gaintype="G", combine="spw"
        ),
        apply=ApplyOptions(applymode="calonly"),
        output=args.output,
        initial_stack=CalStack([phase_table]),
    )
    observers = [LoggingObserver()] + ([GainPlotObserver()] if args.plot else [])
    result = selfcal.run(observers)

    # The self-calibrated data, for final imaging
    split(vis=result.ms, outputvis=args.output + "/selfcal.ms", datacolumn="corrected")
