"""
Applying a run's accepted calibration to other data (#68), on simulated data with a second spectral
window 300 MHz above the first. Spectral windows are matched by frequency, so a split measurement
set, which renumbers them, takes the right solutions.
"""
import numpy as np
import pytest
from casatasks import split
from casatools import table

from wenuray.config import ApplyOptions, ApplyTarget, ImageGeometry, Selection, SolveOptions
from wenuray.imaging.backends import Tclean
from wenuray.selfcalibration.loop import Selfcal
from wenuray.selfcalibration.steps import Phasecal
from wenuray.testing import simulate_point_source_ms
from wenuray.utils.selfcal_utils import open_table

IMAGER = Tclean(ImageGeometry(imsize=128, cell="1arcsec"), niter=200, nsigma=3.0)
SHIFT = 300e6


def add_a_second_spw(ms: str, seed: int = 7) -> None:
    """
    Copies every row into a new spectral window 300 MHz higher, whose data get antenna phases of
    their own, so its gain solutions differ from spw 0's
    """
    tool = table()
    for name in ("SPECTRAL_WINDOW", "DATA_DESCRIPTION"):
        tool.open(ms + "/" + name, nomodify=False)
        try:
            tool.copyrows(ms + "/" + name, startrowin=0, nrow=1)
        finally:
            tool.close()
    tool.open(ms + "/SPECTRAL_WINDOW", nomodify=False)
    try:
        tool.putcell("CHAN_FREQ", 1, tool.getcell("CHAN_FREQ", 0) + SHIFT)
        tool.putcell("REF_FREQUENCY", 1, tool.getcell("REF_FREQUENCY", 0) + SHIFT)
    finally:
        tool.close()
    tool.open(ms + "/DATA_DESCRIPTION", nomodify=False)
    try:
        tool.putcell("SPECTRAL_WINDOW_ID", 1, 1)
    finally:
        tool.close()
    tool.open(ms, nomodify=False)
    try:
        nrows = tool.nrows()
        tool.copyrows(ms, startrowin=0, nrow=nrows)
        descriptions = tool.getcol("DATA_DESC_ID")
        descriptions[nrows:] = 1
        tool.putcol("DATA_DESC_ID", descriptions)
        phases = np.random.default_rng(seed).uniform(-1.0, 1.0, 64)
        antenna1, antenna2 = tool.getcol("ANTENNA1")[nrows:], tool.getcol("ANTENNA2")[nrows:]
        factor = np.exp(1j * (phases[antenna1] - phases[antenna2]))
        for column in ("DATA", "CORRECTED_DATA"):
            data = tool.getcol(column, startrow=nrows, nrow=nrows)
            tool.putcol(column, data * factor, startrow=nrows, nrow=nrows)
    finally:
        tool.close()
    tool.done()


def column(ms: str, name: str, ddid: int) -> np.ndarray:
    with open_table(ms) as ms_table:
        rows = ms_table.query(
            "DATA_DESC_ID == {0}".format(ddid), sortlist="TIME, ANTENNA1, ANTENNA2"
        )
        try:
            return rows.getcol(name)
        finally:
            rows.close()


@pytest.fixture
def ms(tmp_path):
    path = simulate_point_source_ms(str(tmp_path / "target.ms"))
    add_a_second_spw(path)
    return path


def test_a_combined_solution_calibrates_another_spw_of_the_run(ms):
    solve = SolveOptions(refant="A0", minblperant=3, combine="spw")
    apply = ApplyOptions(targets=(ApplyTarget(ms, Selection(spw="1")), ))
    # Self-calibrated on spw 0 only; spw 1 is calibrated when the run ends
    result = Selfcal(
        ms, IMAGER, [Phasecal("inf")], selection=Selection(spw="0"), solve=solve, apply=apply
    ).run()

    assert result.rounds[1].accepted
    (application, ) = result.applications
    assert application.stack.tables[0].spwmap == (0, 0)
    assert not np.allclose(column(ms, "DATA", 1), column(ms, "CORRECTED_DATA", 1))


def test_per_spw_solutions_reach_a_split_ms_by_frequency(ms, tmp_path):
    # Solved per spw on both; spw 1 split into line.ms becomes its spw 0
    result = Selfcal(ms, IMAGER, [Phasecal("inf")], solve=SolveOptions(refant="A0",
                                                                       minblperant=3)).run()
    line = str(tmp_path / "line.ms")
    split(vis=ms, outputvis=line, spw="1", datacolumn="data")

    result = result.apply_to(line)

    assert result.applications[0].stack.tables[0].spwmap == (1, )
    # By id, spw 0's solutions would leave spw 1's own antenna phases in
    np.testing.assert_allclose(
        column(line, "CORRECTED_DATA", 0), column(ms, "CORRECTED_DATA", 1), atol=1e-4
    )


def test_per_spw_solutions_at_other_frequencies_raise(ms):
    result = Selfcal(
        ms,
        IMAGER, [Phasecal("inf")],
        selection=Selection(spw="0"),
        solve=SolveOptions(refant="A0", minblperant=3)
    ).run()

    with pytest.raises(ValueError, match=r"Spectral window 1 .* nearest is 0, .* combine='spw'"):
        result.apply_to(ms, Selection(spw="1"))
