"""The new self-calibration loop (#22c) end to end with real CASA."""
import os

import numpy as np
import pytest
from casatasks import flagmanager
from casatools import table

from wenuray.config import ImageGeometry, SolveOptions
from wenuray.imaging.backends import Tclean
from wenuray.selfcalibration.acceptance import AllOf, PSNRImproves
from wenuray.selfcalibration.flagging import ResidualFlagging
from wenuray.selfcalibration.loop import Selfcal
from wenuray.selfcalibration.observers import GainPlotObserver, LoggingObserver
from wenuray.selfcalibration.result import SelfcalResult
from wenuray.selfcalibration.steps import AmpPhasecal, Phasecal
from wenuray.testing import simulate_point_source_ms
from wenuray.utils.selfcal_utils import model_column_problem, open_table

IMAGER = Tclean(ImageGeometry(imsize=128, cell="1arcsec"), niter=200, nsigma=3.0)
SOLVE = SolveOptions(refant="A0", minblperant=3)


def wenuray_flag_versions(ms: str) -> list:
    versions = flagmanager(vis=ms, mode="list")
    return [
        v["name"] for k, v in versions.items() if k != "MS" and v["name"].startswith("wenuray_")
    ]


@pytest.fixture(scope="module")
def accepted(tmp_path_factory):
    ms = simulate_point_source_ms(str(tmp_path_factory.mktemp("loop") / "target.ms"))
    observers = [LoggingObserver(), GainPlotObserver()]
    return Selfcal(ms, IMAGER, [Phasecal("inf"), Phasecal("10min")], solve=SOLVE).run(observers)


def test_every_round_improves_and_is_kept(accepted):
    psnr = [record.psnr for record in accepted.rounds]

    assert all(record.accepted for record in accepted.rounds)
    assert psnr[0] < psnr[1] < psnr[2]
    assert psnr[2] > 2 * psnr[0]
    assert len(accepted.stack) == 2


def test_the_measurement_set_holds_the_accepted_state(accepted):
    assert model_column_problem(accepted.ms) == ""
    with open_table(accepted.ms) as ms_table:
        assert not np.allclose(ms_table.getcell("DATA", 0), ms_table.getcell("CORRECTED_DATA", 0))
    assert wenuray_flag_versions(accepted.ms) == []


def test_the_result_record_round_trips(accepted):
    path = os.path.join(os.path.dirname(accepted.final.image), "selfcal.toml")

    assert SelfcalResult.from_toml(path) == accepted


def test_a_rejected_round_is_rolled_back(tmp_path):
    ms = simulate_point_source_ms(str(tmp_path / "target.ms"))
    impossible = AllOf(PSNRImproves(min_ratio=100.0))

    result = Selfcal(ms, IMAGER, [Phasecal("inf", acceptance=impossible)], solve=SOLVE).run()

    assert [record.accepted for record in result.rounds] == [True, False]
    assert len(result.stack) == 0
    with open_table(ms) as ms_table:
        # No calibration accepted: CORRECTED_DATA is DATA again, and the round-0 model is back
        assert np.allclose(ms_table.getcell("DATA", 0), ms_table.getcell("CORRECTED_DATA", 0))
    assert model_column_problem(ms) == ""
    assert wenuray_flag_versions(ms) == []


def test_gain_plots_are_written_next_to_the_tables(accepted):
    assert all(os.path.getsize(table.path + ".png") > 0 for table in accepted.stack)


def test_resume_continues_from_the_accepted_state(accepted):
    resumed = Selfcal.resume(accepted, IMAGER, [AmpPhasecal("inf")], solve=SOLVE)

    result = resumed.run()

    # The resumed run starts where the first ended
    assert result.rounds[0].psnr == pytest.approx(accepted.final.psnr, rel=0.05)
    assert [t.path for t in result.stack][:2] == [t.path for t in accepted.stack]
    assert resumed.output != os.path.dirname(accepted.final.image)


OUTLIER_ROWS = list(range(100, 3000, 300))


def with_outliers(ms: str) -> str:
    """A few rows with 100 Jy of interference, far above the 1 Jy source."""
    tb = table()
    tb.open(ms, nomodify=False)
    try:
        for row in OUTLIER_ROWS:
            tb.putcell("DATA", row, tb.getcell("DATA", row) + 100.0)
    finally:
        tb.close()
    return ms


def flagged_rows(ms: str) -> list:
    with open_table(ms) as ms_table:
        return [bool(np.all(ms_table.getcell("FLAG", row))) for row in OUTLIER_ROWS]


@pytest.mark.parametrize(
    "policy, kept", [(None, True), (AllOf(PSNRImproves(min_ratio=100.0)), False)]
)
def test_residual_flagging(tmp_path, policy, kept):
    ms = with_outliers(simulate_point_source_ms(str(tmp_path / "target.ms")))
    clip = ResidualFlagging("clip", clipminmax=[0.0, 50.0])

    result = Selfcal(
        ms, IMAGER, [Phasecal("10min", flagging=clip, acceptance=policy)], solve=SOLVE
    ).run()

    before, after = result.rounds[1].data_flagged
    assert after > before
    assert result.rounds[1].accepted is kept
    # Kept: the interference stays flagged. Rejected: the round's flags are rolled back
    assert flagged_rows(ms) == [kept] * len(OUTLIER_ROWS)


def test_the_solution_snr_sample_reads_the_simulated_noise(tmp_path):
    """0.05 Jy noise on a 1 Jy source, 8 antennas, 60 s integrations over 2 h, RR and LL"""
    from wenuray.imaging.contract import ImagingRequest
    from wenuray.selfcalibration.preflight import read_sample
    from wenuray.config import Selection

    ms = simulate_point_source_ms(str(tmp_path / "target.ms"), phase_errors_in=())
    IMAGER.image(ImagingRequest(ms, str(tmp_path / "model")))

    sample = read_sample(ms, Selection())

    assert sample.noise == pytest.approx(0.05, rel=0.1)
    assert sample.signal == pytest.approx(1.0, rel=0.1)
    assert (sample.nantennas, sample.interval, sample.channels, sample.parallel_hands) == \
        (8, 60.0, (1, ), 2)
    assert sample.span == pytest.approx(7200.0, abs=60.0)


def test_each_round_records_its_predicted_snr(accepted):
    assert accepted.rounds[0].predicted_snr is None
    assert all(record.predicted_snr > 3.0 for record in accepted.rounds[1:])


def test_the_recorded_recipe_reproduces_the_run(accepted, tmp_path):
    """#36: wenuray run <output>/recipe.toml on the same input data gives the same rounds."""
    from wenuray import cli
    from wenuray.utils.selfcal_utils import ms_fingerprint

    output = os.path.dirname(accepted.final.image)
    fresh = simulate_point_source_ms(str(tmp_path / "fresh.ms"))
    assert ms_fingerprint(fresh) == accepted.provenance["ms_fingerprint"]

    status = cli.main(
        [
            "run", "--quiet",
            os.path.join(output, accepted.provenance["recipe"]), "--ms", fresh, "--output",
            str(tmp_path / "rerun")
        ]
    )

    rerun = SelfcalResult.from_toml(str(tmp_path / "rerun" / "selfcal.toml"))
    assert status == 0
    assert [r.psnr
            for r in rerun.rounds] == pytest.approx([r.psnr for r in accepted.rounds], rel=1e-6)
    assert [r.accepted for r in rerun.rounds] == [r.accepted for r in accepted.rounds]


def test_the_fingerprint_changes_with_the_data(tmp_path):
    from wenuray.utils.selfcal_utils import ms_fingerprint

    ms = simulate_point_source_ms(str(tmp_path / "target.ms"))
    before = ms_fingerprint(ms)
    tb = table()
    tb.open(ms, nomodify=False)
    try:
        tb.putcell("DATA", 0, tb.getcell("DATA", 0) * 2)
    finally:
        tb.close()

    assert ms_fingerprint(ms) != before
