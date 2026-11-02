"""Applying the accepted calibration to other data (#68): spectral windows matched by frequency."""
import os

import pytest

from wenuray.config import Selection
from wenuray.selfcalibration.application import (Application, map_spectral_windows, spwmap_list)
from wenuray.selfcalibration.caltable import CalStack, CalTable
from wenuray.selfcalibration.result import SelfcalResult

GHZ = 1e9
# Four continuum windows solved on, as an ALMA split keeps them: ids 13, 15, 17, 19
SOLVED = {
    13: (335.8 * GHZ, 337.7 * GHZ),
    15: (337.8 * GHZ, 339.7 * GHZ),
    17: (347.8 * GHZ, 349.7 * GHZ),
    19: (349.8 * GHZ, 351.7 * GHZ),
}


def test_a_combined_table_calibrates_every_window_with_its_one_solution():
    line = {0: (345.7 * GHZ, 345.9 * GHZ), 1: (330.5 * GHZ, 330.7 * GHZ)}

    assert map_spectral_windows(line, {13: SOLVED[13]}, combined=True) == {0: 13, 1: 13}


def test_per_spw_tables_are_matched_by_frequency_not_by_id():
    # The same windows split into another measurement set, renumbered 0 to 3
    renumbered = {index: band for index, band in enumerate(SOLVED.values())}

    assert map_spectral_windows(renumbered, SOLVED, combined=False) == {0: 13, 1: 15, 2: 17, 3: 19}


def test_a_narrower_window_takes_the_solved_window_it_lies_in():
    line = {0: (338.0 * GHZ, 338.2 * GHZ)}  # a line inside spw 15's band

    assert map_spectral_windows(line, SOLVED, combined=False) == {0: 15}


def test_the_largest_overlap_wins():
    straddling = {0: (337.6 * GHZ, 338.5 * GHZ)}  # 0.1 GHz of 13, 0.7 GHz of 15

    assert map_spectral_windows(straddling, SOLVED, combined=False) == {0: 15}


def test_a_window_without_solutions_raises_naming_the_nearest():
    line = {4: (345.7 * GHZ, 345.9 * GHZ)}  # between 15 and 17

    with pytest.raises(ValueError, match=r"Spectral window 4 .* nearest is 17, .* combine='spw'"):
        map_spectral_windows(line, SOLVED, combined=False)


def test_no_solutions_raise():
    with pytest.raises(ValueError, match="no solutions"):
        map_spectral_windows({0: SOLVED[13]}, {}, combined=True)


def test_the_spwmap_lists_every_window_up_to_the_highest_calibrated():
    # Windows not calibrated map to themselves
    assert spwmap_list({1: 13, 3: 13}) == (0, 13, 2, 13)


def test_applications_round_trip_in_the_result_record(tmp_path):
    stack = CalStack([CalTable(str(tmp_path / "round1_p.cal"), "p", (0, 0))])
    application = Application(
        str(tmp_path / "line.ms"), Selection(spw="0"), "calonly",
        CalStack([CalTable(str(tmp_path / "round1_p.cal"), "p", (0, ))]), "sha256:abc"
    )
    result = SelfcalResult(str(tmp_path / "target.ms"), stack, applications=(application, ))

    result.to_toml(str(tmp_path / "selfcal.toml"))

    text = (tmp_path / "selfcal.toml").read_text()
    assert '[[applications]]' in text and 'ms = "line.ms"' in text
    assert SelfcalResult.from_toml(str(tmp_path / "selfcal.toml")) == result


def test_apply_to_records_the_application(tmp_path, monkeypatch):
    from wenuray.selfcalibration import result as result_module
    calls = []

    def apply_to(stack, ms, selection, applymode):
        calls.append((ms, selection, applymode))
        return Application(ms, selection, applymode, stack, "sha256:abc")

    monkeypatch.setattr(result_module, "apply_to", apply_to)
    result = SelfcalResult(str(tmp_path / "target.ms"), CalStack())

    applied = result.apply_to("line.ms", Selection(spw="1"))

    assert calls == [(os.path.abspath("line.ms"), Selection(spw="1"), "calflag")]
    assert applied.applications[0].ms.endswith("line.ms") and result.applications == ()
