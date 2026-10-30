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
    assert sample.band == pytest.approx((5.0e9 - 64e6, 5.0e9 + 64e6))  # one 128 MHz channel


def test_the_thermal_noise_is_the_noise_of_a_naturally_weighted_residual(tmp_path):
    """
    The thermal noise predicted from the data matches the MAD std of tclean's residual of the
    same data, with natural weighting (#119)
    """
    from wenuray.imaging.contract import ImagingRequest
    from wenuray.selfcalibration.preflight import read_sample
    from wenuray.config import Selection

    ms = simulate_point_source_ms(str(tmp_path / "target.ms"), phase_errors_in=())
    result = IMAGER.image(ImagingRequest(ms, str(tmp_path / "model")))

    thermal = read_sample(ms, Selection()).thermal_noise

    assert result.statistics().rms == pytest.approx(thermal, rel=0.15)


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


@pytest.mark.parametrize("sigma, expected", [(0.05, 1.0), (0.05 / np.sqrt(2), np.sqrt(2))])
def test_the_noise_over_sigma_measures_how_wrong_the_weights_are(tmp_path, sigma, expected):
    """0.05 Jy noise per part of each visibility; SIGMA set right, then sqrt(2) too small (#129)"""
    from casatools import table
    from wenuray.config import Selection
    from wenuray.imaging.contract import ImagingRequest
    from wenuray.selfcalibration.preflight import read_sample, weight_anomalies

    ms = simulate_point_source_ms(str(tmp_path / "target.ms"), phase_errors_in=())
    tool = table()
    tool.open(ms, nomodify=False)
    try:
        tool.putcol("SIGMA", np.full_like(tool.getcol("SIGMA"), sigma))
    finally:
        tool.close()
    tool.done()
    IMAGER.image(ImagingRequest(ms, str(tmp_path / "model")))

    sample = read_sample(ms, Selection())

    ratios = [ratio for *_, ratio in sample.noise_over_sigma]
    assert len(ratios) == 28  # every baseline of the 8 antennas
    assert np.median(ratios) == pytest.approx(expected, rel=0.05)
    assert bool(weight_anomalies(sample)) == bool(expected > 1.1)


def test_an_auto_grid_is_derived_from_the_data(tmp_path):
    """
    The simulated data reach 59.1 klambda (a 1.745 arcsec Nyquist cell) with 25 m dishes at 4.936
    to 5.064 GHz (#67)
    """
    from wenuray.config import ImageGeometry, Selection
    from wenuray.imaging.contract import resolve_geometry

    ms = simulate_point_source_ms(str(tmp_path / "target.ms"), phase_errors_in=())

    grid = resolve_geometry(ImageGeometry(imsize="auto", cell="auto"), ms, Selection())

    cell = float(grid.cell[:-len("arcsec")])
    assert cell == pytest.approx(1.745 / 2.5, rel=0.01)
    fwhm = np.degrees(1.02 * 299792458.0 / 4.936e9 / 25.0) * 3600
    needed = 2.5 * fwhm / cell
    assert grid.imsize[0] == grid.imsize[1]
    assert needed <= grid.imsize[0] < 1.1 * needed


def test_a_run_with_an_auto_grid_records_it(tmp_path):
    from wenuray.config import ImageGeometry
    from wenuray.selfcalibration.loop import Selfcal
    from wenuray.selfcalibration.steps import Phasecal

    ms = simulate_point_source_ms(str(tmp_path / "target.ms"))
    # A small field keeps the test fast: a tenth of the primary beam
    imager = Tclean(
        ImageGeometry(imsize="auto", cell="auto", field_of_view=0.25), niter=200, nsigma=3.0
    )

    result = Selfcal(ms, imager, [Phasecal("inf")], solve=SolveOptions(refant="A0")).run()

    cell = result.provenance["geometry"]["cell"]
    assert float(cell[:-len("arcsec")]) == pytest.approx(1.745 / 2.5, rel=0.01)
    assert result.rounds[1].accepted


@pytest.mark.parametrize("gaintype, calmode", [("T", "p"), ("T", "ap"), ("G", "ap"), ("G", "p")])
def test_gain_diagnostics_read_one_at_the_noise(tmp_path, gaintype, calmode):
    """
    Solutions against the true model of data without gain errors are pure noise: every ratio of
    scatter to noise reads about 1, whatever the gaintype and calmode (#126). This pins how CASA's
    SNR column relates to the noise of the solutions
    """
    from casatasks import clearcal, gaincal
    from wenuray.selfcalibration.diagnostics import gain_diagnostics

    ms = simulate_point_source_ms(str(tmp_path / "target.ms"), phase_errors_in=())
    clearcal(vis=ms, addmodel=True)
    tool = table()
    tool.open(ms, nomodify=False)
    try:
        tool.putcol("MODEL_DATA", np.ones_like(tool.getcol("MODEL_DATA")))  # the 1 Jy source
    finally:
        tool.close()
    tool.done()
    caltable = str(tmp_path / "noise.cal")
    gaincal(
        vis=ms,
        caltable=caltable,
        solint="int",
        refant="A0",
        minblperant=3,
        gaintype=gaintype,
        calmode=calmode
    )

    diagnostics = gain_diagnostics(caltable, solves_amplitude="a" in calmode)

    ratios = [diagnostics.phase_over_noise]
    ratios += [diagnostics.amplitude_over_noise] if "a" in calmode else []
    ratios += [diagnostics.xy_phase_over_noise] if gaintype == "G" else []
    assert all(0.8 < ratio < 1.2 for ratio in ratios), ratios


def test_gain_diagnostics_fall_as_the_run_converges(tmp_path):
    """
    The first round corrects the injected phase errors, far above the noise; the second, against
    the improved model, corrects less (#126)
    """
    from wenuray.selfcalibration.loop import Selfcal
    from wenuray.selfcalibration.steps import Phasecal

    ms = simulate_point_source_ms(str(tmp_path / "target.ms"))

    result = Selfcal(
        ms,
        IMAGER, [Phasecal("int"), Phasecal("int")],
        solve=SolveOptions(refant="A0", minblperant=3)
    ).run()

    first, second = result.rounds[1].gains, result.rounds[2].gains
    assert first.phase_over_noise > 3
    assert second.phase_over_noise < first.phase_over_noise
    assert first.amplitude_over_noise is None  # phase-only
