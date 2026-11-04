"""Gain diagnostics per round (#126): corrections against their noise, outlying antennas, X-Y."""
import numpy as np
import pytest

from wenuray.selfcalibration.diagnostics import GainDiagnostics, SolutionCoverage, cover, diagnose

SNR = 50.0
# The noise of a phase relative to the reference antenna, and of an amplitude, at that SNR, for a
# phase-only table (see the module docstring)
PHASE_NOISE = np.sqrt(2) / SNR
NANT = 10
NAMES = np.array(["A{0}".format(i) for i in range(NANT)])
# Antennas along a line, 100 m to 1 km from the centre
POSITIONS = np.column_stack([np.linspace(-1000, 1000, NANT), np.zeros(NANT), np.zeros(NANT)])


def table(phase_sigma, ntimes=200, npol=1, amplitude_sigma=0.0, seed=0, scale=None):
    """
    A gain table of NANT antennas over ntimes solutions, antenna 0 the reference: phases and
    amplitude offsets drawn with the given scatter (per antenna, scale multiplies it)
    """
    rng = np.random.default_rng(seed)
    antenna1 = np.tile(np.arange(NANT), ntimes)
    antenna2 = np.zeros_like(antenna1)
    factor = np.ones(NANT) if scale is None else scale
    shape = (npol, 1, len(antenna1))
    phases = rng.normal(0, 1, shape) * phase_sigma * factor[antenna1]
    amplitudes = 1 + rng.normal(0, 1, shape) * amplitude_sigma
    phases[..., antenna1 == 0] = 0  # the reference antenna's own solutions
    gains = amplitudes * np.exp(1j * phases)
    return dict(
        gains=gains,
        flags=np.zeros(shape, dtype=bool),
        snr=np.full(shape, SNR),
        antenna1=antenna1,
        antenna2=antenna2,
        names=NAMES,
        positions=POSITIONS,
    )


def test_corrections_at_the_noise_read_about_one():
    diagnostics = diagnose(**table(PHASE_NOISE), solves_amplitude=False)

    assert diagnostics.phase_over_noise == pytest.approx(1.0, rel=0.1)
    assert diagnostics.amplitude_over_noise is None


def test_corrections_above_the_noise_read_their_ratio():
    # 12x the noise, as round 2 on HD135344B; an amplitude table's SNR is its S/N over sqrt(2)
    own = SNR * np.sqrt(2)
    diagnostics = diagnose(
        **table(12 * np.sqrt(2) / own, amplitude_sigma=3 / own), solves_amplitude=True
    )

    assert diagnostics.phase_over_noise == pytest.approx(12.0, rel=0.1)
    assert diagnostics.amplitude_over_noise == pytest.approx(3.0, rel=0.1)


def test_the_reference_antenna_is_left_out():
    # Its zero phases would pull the scatter down by a tenth
    columns = table(12 / SNR)
    without = diagnose(**columns, solves_amplitude=False)
    columns["antenna2"] = np.full_like(columns["antenna2"], -1)  # no reference: zeros count

    assert diagnose(**columns, solves_amplitude=False).phase_over_noise < without.phase_over_noise


def test_outer_antennas_are_expected_to_scatter_more():
    distance = np.abs(np.linspace(-1000, 1000, NANT))
    scale = (distance / distance.min())**0.5  # a structure-function-like growth
    diagnostics = diagnose(**table(10 / SNR, scale=scale), solves_amplitude=False)

    assert diagnostics.outlying_antennas == ()


def test_an_antenna_that_scatters_more_than_its_distance_predicts_is_named():
    distance = np.abs(np.linspace(-1000, 1000, NANT))
    scale = (distance / distance.min())**0.5
    scale[4] *= 5

    diagnostics = diagnose(**table(10 / SNR, scale=scale), solves_amplitude=False)

    assert diagnostics.outlying_antennas == ("A4", )


def test_the_polarizations_are_compared():
    columns = table(10 * PHASE_NOISE, npol=2, amplitude_sigma=0.02)

    diagnostics = diagnose(**columns, solves_amplitude=False)

    # Independent X and Y phases differ by sqrt(2) times their scatter, against 2/S noise
    assert diagnostics.xy_phase_over_noise == pytest.approx(10.0, rel=0.15)
    diagnostics = diagnose(**columns, solves_amplitude=True)
    assert diagnostics.xy_amplitude == pytest.approx(0.0191, rel=0.25)  # median |N(0, 0.028)|


def test_x_and_y_amplitudes_are_only_compared_when_solved():
    diagnostics = diagnose(**table(10 / SNR, npol=2), solves_amplitude=False)

    assert diagnostics.xy_amplitude is None and diagnostics.xy_phase_over_noise is not None


def test_a_fully_flagged_table_has_no_diagnostics():
    columns = table(PHASE_NOISE)
    columns["flags"][:] = True

    assert diagnose(**columns, solves_amplitude=False) is None


def test_diagnostics_are_read_back_from_their_record():
    from wenuray.selfcalibration.result import RoundRecord

    record = RoundRecord(
        1,
        "Phasecal('inf')",
        "t.cal",
        "i.image",
        10.0,
        1.0,
        0.1,
        1.0,
        gains={
            "phase_over_noise": 3.9,
            "outlying_antennas": ["DV15"]
        }
    )

    assert record.gains == GainDiagnostics(3.9, outlying_antennas=("DV15", ))


class TestAboveNoise:
    """#179: whether a scatter over its noise is a correction, on as many solutions as it had."""

    @pytest.mark.parametrize("seed", range(20))
    def test_noise_alone_reads_as_noise(self, seed):
        diagnostics = diagnose(**table(PHASE_NOISE, ntimes=20, seed=seed), solves_amplitude=False)

        assert diagnostics.solutions == 20 * (NANT - 1)
        assert diagnostics.above_noise(diagnostics.phase_over_noise) is False

    def test_a_small_correction_on_many_solutions_reads_as_one(self):
        diagnostics = diagnose(**table(1.5 * PHASE_NOISE, ntimes=50), solves_amplitude=False)

        assert diagnostics.above_noise(diagnostics.phase_over_noise) is True

    def test_the_same_ratio_on_few_solutions_reads_as_noise(self):
        # 1.5 x noise on 9 solutions is within 3 standard errors (1.167 / 3) of 1
        assert GainDiagnostics(1.5, solutions=9).above_noise(1.5) is False
        assert GainDiagnostics(1.5, solutions=900).above_noise(1.5) is True

    def test_records_without_the_count_are_not_judged(self):
        assert GainDiagnostics(12.0).above_noise(12.0) is None


def coverage_columns(nant=4, ntimes=3):
    flags = np.zeros((1, 1, nant * ntimes), dtype=bool)
    times = np.repeat(np.arange(ntimes, dtype=float), nant)
    antennas = np.tile(np.arange(nant), ntimes)
    return flags, times, np.zeros_like(antennas), antennas


class TestCoverage:
    """#179: how many solutions failed, on which antennas, and which intervals lack antennas."""

    def test_a_complete_table(self):
        flags, times, spws, antennas = coverage_columns()

        assert cover(flags, times, spws, antennas, NAMES, 3, 3.0) == SolutionCoverage(
            solutions=12,
            failed=0,
            intervals=3,
            short_intervals=0,
            min_antennas=3,
            minsnr=3.0,
            spws=(0, )
        )

    def test_failures_are_counted_by_antenna_and_interval(self):
        flags, times, spws, antennas = coverage_columns()
        # A2 fails in every interval, A3 in the first: that interval keeps 2 of 4 antennas
        flags[..., antennas == 2] = True
        flags[..., (antennas == 3) & (times == 0)] = True

        coverage = cover(flags, times, spws, antennas, NAMES, 3, 3.0)

        assert (coverage.failed, coverage.short_intervals) == (4, 1)
        assert coverage.failed_antennas == ("A2", "A3")

    def test_each_spectral_window_is_its_own_interval(self):
        flags, times, spws, antennas = coverage_columns()

        coverage = cover(flags, times, antennas % 2, antennas, NAMES, 3, 3.0)

        # Two antennas per spectral window and time: fewer than the 3 phases need
        assert (coverage.intervals, coverage.short_intervals) == (6, 6)


def test_coverage_is_read_back_from_its_record():
    from wenuray.selfcalibration.result import RoundRecord

    record = RoundRecord(
        1,
        "Phasecal('inf')",
        "t.cal",
        "i.image",
        10.0,
        1.0,
        0.1,
        1.0,
        solutions={
            "solutions": 12,
            "failed": 1,
            "intervals": 3,
            "short_intervals": 0,
            "min_antennas": 3,
            "minsnr": 3.0,
            "failed_antennas": ["A2"]
        }
    )

    assert record.solutions == SolutionCoverage(12, 1, 3, 0, 3, 3.0, ("A2", ))
