"""
An empty spw selection means the spectral windows that hold data. Data split from a larger
measurement set keep their original ids (ALMA's 13, 15, 17, 19 of 21 windows), so a spw-combined
solution is written under the lowest of those, and the derived spwmap must point there.
"""
import numpy as np
import pytest
from casatasks import flagdata
from casatools import table

from wenuray.config import ImageGeometry, Selection, SolveOptions
from wenuray.imaging.backends import Tclean
from wenuray.imaging.contract import ImagingRequest
from wenuray.selfcalibration.loop import Selfcal
from wenuray.selfcalibration.preflight import read_sample
from wenuray.selfcalibration.steps import Phasecal
from wenuray.testing import simulate_point_source_ms
from wenuray.utils.selfcal_utils import get_spw_ids, open_table

IMAGER = Tclean(ImageGeometry(imsize=128, cell="1arcsec"), niter=200, nsigma=3.0)
SOLVE = SolveOptions(refant="A0", minblperant=3, combine="spw")


def move_data_to_a_second_spw(ms: str) -> None:
    """
    Adds a spectral window and its data description, and moves every row to them: spw 0 is left
    without data, as in a split that keeps the original ids
    """
    tool = table()
    for name in ("SPECTRAL_WINDOW", "DATA_DESCRIPTION"):
        tool.open(ms + "/" + name, nomodify=False)
        try:
            tool.copyrows(ms + "/" + name, startrowin=0, nrow=1)
        finally:
            tool.close()
    tool.open(ms + "/DATA_DESCRIPTION", nomodify=False)
    try:
        tool.putcell("SPECTRAL_WINDOW_ID", 1, 1)
    finally:
        tool.close()
    tool.open(ms, nomodify=False)
    try:
        tool.putcol("DATA_DESC_ID", np.ones(tool.nrows(), dtype=np.int32))
    finally:
        tool.close()
    tool.done()


def copy_data_to_a_third_spw(ms: str) -> None:
    """
    Adds a third spectral window and its data description, and copies every row of spw 1 to it
    """
    tool = table()
    for name in ("SPECTRAL_WINDOW", "DATA_DESCRIPTION"):
        tool.open(ms + "/" + name, nomodify=False)
        try:
            tool.copyrows(ms + "/" + name, startrowin=1, nrow=1)
        finally:
            tool.close()
    tool.open(ms + "/DATA_DESCRIPTION", nomodify=False)
    try:
        tool.putcell("SPECTRAL_WINDOW_ID", 2, 2)
    finally:
        tool.close()
    tool.open(ms, nomodify=False)
    try:
        nrows = tool.nrows()
        tool.copyrows(ms, startrowin=0, nrow=nrows)
        data_descriptions = tool.getcol("DATA_DESC_ID")
        data_descriptions[nrows:] = 2
        tool.putcol("DATA_DESC_ID", data_descriptions)
    finally:
        tool.close()
    tool.done()


@pytest.fixture(scope="module")
def ms(tmp_path_factory):
    path = simulate_point_source_ms(str(tmp_path_factory.mktemp("spw") / "target.ms"))
    move_data_to_a_second_spw(path)
    return path


@pytest.fixture(scope="module")
def ms_with_a_flagged_spw(tmp_path_factory):
    """Spw 0 without data, spw 1 fully flagged, spw 2 with the data"""
    path = simulate_point_source_ms(str(tmp_path_factory.mktemp("flagged") / "target.ms"))
    move_data_to_a_second_spw(path)
    copy_data_to_a_third_spw(path)
    flagdata(vis=path, mode="manual", spw="1", flagbackup=False)  # sets FLAG, not FLAG_ROW
    tool = table()
    tool.open(path, nomodify=False)
    try:
        tool.putcol("FLAG_ROW", tool.getcol("DATA_DESC_ID") == 1)
    finally:
        tool.close()
    tool.done()
    return path


def test_an_empty_selection_is_the_spectral_windows_with_data(ms):
    with open_table(ms + "/SPECTRAL_WINDOW") as spw_table:
        assert spw_table.nrows() == 2

    assert get_spw_ids(ms, "") == [1]


def test_spw_combined_solutions_are_applied(ms):
    result = Selfcal(ms, IMAGER, [Phasecal("inf")], solve=SOLVE).run()

    # Before the fix the solutions were mapped to spw 0, applycal applied nothing and the round,
    # identical to the initial one, was rejected
    assert [record.accepted for record in result.rounds] == [True, True]
    assert result.stack.tables[0].spwmap == (1, 1)
    with open_table(ms) as ms_table:
        assert not np.allclose(ms_table.getcell("DATA", 0), ms_table.getcell("CORRECTED_DATA", 0))


def test_a_flagged_spw_still_holds_the_combined_solutions(ms_with_a_flagged_spw):
    # gaincal writes spw-combined solutions under the lowest spw with data even when that spw is
    # fully flagged, so the spws must not be filtered by flags or the spwmap points to spw 2,
    # where there are no solutions, and nothing is applied
    ms = ms_with_a_flagged_spw
    assert get_spw_ids(ms, "") == [1, 2]

    result = Selfcal(ms, IMAGER, [Phasecal("inf")], solve=SOLVE).run()

    assert [record.accepted for record in result.rounds] == [True, True]
    assert result.stack.tables[0].spwmap == (1, 1, 1)
    with open_table(ms) as ms_table:
        unflagged = ms_table.query("DATA_DESC_ID == 2")
        try:
            assert not np.allclose(unflagged.getcol("DATA"), unflagged.getcol("CORRECTED_DATA"))
        finally:
            unflagged.close()


def test_the_solution_snr_counts_the_channels_of_the_spectral_windows_with_data(tmp_path):
    # Before the fix spw 0, which holds no data, was counted too: two windows of one channel, so
    # a spw-combined prediction was sqrt(2) too high
    ms = simulate_point_source_ms(str(tmp_path / "target.ms"), phase_errors_in=())
    move_data_to_a_second_spw(ms)
    IMAGER.image(ImagingRequest(ms, str(tmp_path / "model")))

    assert read_sample(ms, Selection()).channels == (1, )
