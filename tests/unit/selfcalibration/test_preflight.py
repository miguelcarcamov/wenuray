"""The pre-flight solution S/N prediction (#28), from a sample of the data."""
import dataclasses
import math

import pytest

from wenuray.config import SolveOptions
import numpy as np

from wenuray.selfcalibration.preflight import (
    VisibilitySample, _noise_over_sigma, _selected_channels, absorbed_spectrum, predict,
    weight_anomalies
)
from wenuray.selfcalibration.steps import Ampcal, AmpPhasecal, Phasecal

# 1 Jy over 0.1 Jy noise with 10 antennas: 10 * 3 = 30 for one integration, channel and hand
SAMPLE = VisibilitySample(
    signal=1.0,
    noise=0.1,
    nantennas=10,
    interval=10.0,
    scan_length=600.0,
    span=3600.0,
    channels=(4, 4, 8),
    parallel_hands=2,
    band=(336e9, 351e9),
    weights=(1.0e4, 1.0e4),
)
ONE = 30.0


@pytest.mark.parametrize(
    "solint, combine, gaintype, expected",
    [
        ("int", "", "G", ONE * 2),  # the narrowest spectral window has 4 channels
        ("int", "", "T", ONE * math.sqrt(8)),  # and two hands
        ("60s", "", "G", ONE * math.sqrt(4 * 6)),
        ("inf", "", "G", ONE * math.sqrt(4 * 60)),  # one scan
        ("inf", "scan", "G", ONE * math.sqrt(4 * 360)),  # the whole observation
        ("int", "spw", "G", ONE * math.sqrt(16)),  # every channel
        (7200.0, "", "G", ONE * math.sqrt(4 * 60)),  # no longer than a scan
    ],
)
def test_the_counts_follow_the_solint_and_combine(solint, combine, gaintype, expected):
    options = SolveOptions(combine=combine, gaintype=gaintype)

    assert predict(SAMPLE, Phasecal(solint), options).snr == pytest.approx(expected)


def test_amplitude_predictions_are_casas_snr():
    # gaincal reports the antenna's S/N over sqrt(2) when it solves amplitudes too
    options = SolveOptions(gaintype="G")

    assert predict(SAMPLE, AmpPhasecal("int"), options).snr == pytest.approx(ONE * 2 / math.sqrt(2))
    assert predict(SAMPLE, Phasecal("int"), options).snr == pytest.approx(ONE * 2)


def test_step_overrides_count():
    prediction = predict(SAMPLE, Ampcal("int", solve={"minsnr": 100.0}), SolveOptions(gaintype="G"))

    assert prediction.minsnr == 100.0
    assert not prediction.enough


def test_the_advice_is_the_shortest_solint_that_reaches_minsnr():
    # One integration gives 60 with G: 3 integrations reach 100
    prediction = predict(SAMPLE, Phasecal("int"), SolveOptions(gaintype="G", minsnr=100.0))

    assert prediction.advice == "a solint of 30s or longer would reach it"
    assert predict(SAMPLE, Phasecal("30s"), SolveOptions(gaintype="G", minsnr=100.0)).enough


def test_when_no_solint_is_enough_combining_is_advised():
    prediction = predict(SAMPLE, Phasecal("int"), SolveOptions(gaintype="G", minsnr=1e4))

    assert prediction.advice == "even a solint of a whole scan wouldn't; combine spectral windows"


# An ALMA split: 21 SPECTRAL_WINDOW rows, data in spws 13, 15, 17 and 19 of 4 channels each
DD_SPW = np.arange(21)
ALL_CHANNELS = np.array([1, 1, 1, 1, 1, 8, 1, 8, 1, 8, 1, 8, 1, 4, 1, 4, 1, 4, 1, 4, 1])
DATA_DDIDS = np.array([13, 15, 17, 19])


def test_an_empty_spw_selection_counts_the_spectral_windows_with_data():
    nothing_named = {"channel": np.zeros((0, 4), dtype=int), "dd": np.zeros(0, dtype=int)}

    assert _selected_channels(nothing_named, DD_SPW, ALL_CHANNELS, DATA_DDIDS) == [4, 4, 4, 4]


def test_a_spw_selection_counts_its_channel_ranges():
    two_spws = {"channel": np.array([[13, 0, 3, 1], [15, 0, 1, 1]]), "dd": np.array([13, 15])}

    assert _selected_channels(two_spws, DD_SPW, ALL_CHANNELS, DATA_DDIDS) == [4, 2]


class TestAbsorbedSpectrum:
    """Amplitudes per spw against a flat model take up the source's spectrum (#118)."""
    OPTIONS = SolveOptions()

    def check(self, step, snr=100.0, models_spectrum=False, sample=SAMPLE):
        return absorbed_spectrum(sample, step, self.OPTIONS, snr, models_spectrum)

    def test_amplitudes_per_spw_across_a_wide_band_warn(self):
        # 4.4% band; solutions precise to 1%
        message = self.check(AmpPhasecal("inf"))

        assert "across a 4.4% band" in message and "precise to 1.0%" in message
        assert self.check(Ampcal("inf"))

    @pytest.mark.parametrize(
        "step, snr, models_spectrum",
        [
            (Phasecal("inf"), 100.0, False),  # phases don't take up flux
            (AmpPhasecal("inf", solve={"combine": "spw"}), 100.0, False),  # one gain for the band
            (AmpPhasecal("inf"), 100.0, True),  # the model has the spectrum
            (AmpPhasecal("inf"), 20.0, False),  # solutions too noisy (5%) to resolve alpha = 1
        ],
    )
    def test_no_warning(self, step, snr, models_spectrum):
        assert self.check(step, snr, models_spectrum) == ""

    def test_one_spectral_window_keeps_the_spectrum(self):
        one = dataclasses.replace(SAMPLE, channels=(8, ))

        assert self.check(AmpPhasecal("inf"), sample=one) == ""


class TestThermalNoise:
    """The image's thermal noise from the data (#119): noise * sqrt(sum w^2) / sum w."""

    def test_equal_weights_average_the_noise_down(self):
        # 10000 visibilities of 0.1 Jy
        assert SAMPLE.thermal_noise == pytest.approx(1e-3)

    def test_unequal_weights_count_less_than_their_number(self):
        # Half the visibilities with twice the weight: 10000 of 1 and 10000 of 2
        sample = dataclasses.replace(SAMPLE, weights=(3.0e4, 5.0e4))

        assert sample.thermal_noise == pytest.approx(0.1 * math.sqrt(5.0e4) / 3.0e4)
        assert sample.thermal_noise > 0.1 / math.sqrt(2.0e4)


class TestWeightAnomalies:
    """The data's measured noise against the noise their weights state (#129)."""

    @staticmethod
    def sample(*ratios):
        entries = tuple(
            ("A{0}".format(i), "A{0}".format(i + 1), 0, r) for i, r in enumerate(ratios)
        )
        return dataclasses.replace(SAMPLE, noise_over_sigma=entries)

    def test_differences_measure_the_noise_sigma_states(self):
        # Visibilities with 0.1 Jy noise differ by sqrt(2) * 0.1 in each part
        rng = np.random.default_rng(0)
        delta = math.sqrt(2) * 0.1 * (rng.standard_normal(4000) + 1j * rng.standard_normal(4000))
        per_baseline = {(0, 1, 0): [(delta, np.full(4000, 0.1))]}

        ((first, second, spw,
          ratio), ) = _noise_over_sigma(per_baseline, np.array(["A0", "A1"]), np.array([3]))

        assert (first, second, spw) == ("A0", "A1", 3)
        assert ratio == pytest.approx(1.0, rel=0.03)

    def test_right_weights_raise_nothing(self):
        assert weight_anomalies(self.sample(1.02, 0.97, 1.01, 0.99, 1.0)) == ""
        assert weight_anomalies(SAMPLE) == ""  # nothing measured

    def test_weights_that_understate_the_noise(self):
        message = weight_anomalies(self.sample(1.40, 1.43, 1.41, 1.42, 1.39))

        assert "1.41 times the noise" in message and "understate" in message
        assert "statwt" in message

    def test_a_baseline_that_stands_out_is_named(self):
        message = weight_anomalies(self.sample(1.0, 1.01, 0.99, 1.0, 1.02, 0.98, 2.5))

        assert "1 of 7 baselines" in message and "A6-A7 spw 0 (2.50)" in message
        assert "times the noise" not in message  # the median is right


class TestNoiseFromTheHands:
    """
    #153: consecutive integrations cancel the source only if the gains hold over an integration;
    the parallel hands cancel an unpolarized source and the gains both hands share
    """
    SIGMA = 0.05

    def block(self, phase_per_integration=0.0, polarized=0.0, nant=8, ntimes=40, seed=0):
        """A block of rows: a 1 Jy source, antenna phases drawn every integration, noise SIGMA"""
        rng = np.random.default_rng(seed)
        pairs = [(a, b) for a in range(nant) for b in range(a + 1, nant)]
        phases = rng.normal(0.0, phase_per_integration, (ntimes, nant))
        rows = [(t, a, b) for t in range(ntimes) for a, b in pairs]
        time = np.array([60.0 * t for t, _, _ in rows])
        antenna1 = np.array([a for _, a, _ in rows])
        antenna2 = np.array([b for _, _, b in rows])
        gain = np.exp(
            1j *
            (phases[time.astype(int) // 60, antenna1] - phases[time.astype(int) // 60, antenna2])
        )
        source = np.stack(
            [np.full(len(rows), 1.0 + polarized),
             np.full(len(rows), 1.0 - polarized)]
        )
        noise = rng.normal(0, self.SIGMA,
                           (2, len(rows))) + 1j * rng.normal(0, self.SIGMA, (2, len(rows)))
        data = (source * gain + noise)[:, None, :]
        return {
            "data": data,
            "flags": np.zeros(data.shape, dtype=bool),
            "sigma": np.full((2, len(rows)), self.SIGMA),
            "baseline": antenna1 * 100000 + antenna2,
            "ddid": np.zeros(len(rows), dtype=int),
            "time": time,
            "parallel_hands": 2,
        }

    def estimates(self, block):
        from wenuray.selfcalibration.preflight import (
            _difference_noise, _differences, _hand_differences
        )
        found = {}
        for name, differences in (
            ("consecutive integrations", _differences(block, 60.0)),
            ("the parallel hands", _hand_differences(block)),
        ):
            per_baseline = {key: [pair] for key, pair in differences.items()}
            noise, count = _difference_noise(per_baseline)
            found[name] = (noise, count, per_baseline)
        return found

    def test_gains_that_change_every_integration_raise_only_the_time_differences(self):
        from wenuray.selfcalibration.preflight import _least_raised
        # 40 degrees of phase drawn anew every integration, as in the issue's simulation
        estimates = self.estimates(self.block(np.radians(40.0)))

        assert estimates["consecutive integrations"][0] > 2 * self.SIGMA
        assert _least_raised(estimates) == "the parallel hands"
        assert estimates["the parallel hands"][0] == pytest.approx(self.SIGMA, rel=0.05)

    def test_steady_gains_keep_the_default(self):
        from wenuray.selfcalibration.preflight import _least_raised
        estimates = self.estimates(self.block())

        # Both read the noise; the lower is only trusted when significantly lower
        assert estimates["the parallel hands"][0] == pytest.approx(self.SIGMA, rel=0.05)
        assert estimates["consecutive integrations"][0] == pytest.approx(self.SIGMA, rel=0.05)
        assert _least_raised(estimates) == "consecutive integrations"

    def test_a_polarized_source_raises_the_hands_and_keeps_the_default(self):
        from wenuray.selfcalibration.preflight import _least_raised
        # Stokes Q of 10%: XX - YY holds 0.2 Jy of source
        estimates = self.estimates(self.block(polarized=0.1))

        assert estimates["the parallel hands"][0] > 2 * self.SIGMA
        assert _least_raised(estimates) == "consecutive integrations"

    def test_one_hand_has_no_hand_differences(self):
        from wenuray.selfcalibration.preflight import _hand_differences, _least_raised
        block = self.block()
        block["parallel_hands"] = 1

        assert _hand_differences(block) == {}
        assert _least_raised({"consecutive integrations": (0.05, 1000, {})}) == \
            "consecutive integrations"


class TestShortestSolint:
    """#66: solint="auto" is the shortest solint whose predicted S/N reaches 2 x minsnr."""

    @staticmethod
    def resolve(noise, combine=""):
        from wenuray.selfcalibration.preflight import shortest_solint
        return shortest_solint(
            dataclasses.replace(SAMPLE, noise=noise), Phasecal("auto"),
            SolveOptions(gaintype="G", combine=combine)
        )[0]

    @pytest.mark.parametrize(
        "noise, solint",
        [
            # S/N in one integration: 60, 6 (exactly 2 x minsnr 3), 3
            (0.1, "int"),
            (1.0, "int"),
            (2.0, "40s"),  # 4 integrations of 10 s: 3 x sqrt(4) = 6
            (10.0, "inf"),  # 1000 s is longer than a scan, whose S/N 4.6 still reaches minsnr
            (20.0, None),  # a whole scan gives 2.3, below minsnr
        ]
    )
    def test_the_solint_follows_the_noise(self, noise, solint):
        assert self.resolve(noise) == solint

    def test_combining_scans_allows_solints_longer_than_a_scan(self):
        # 0.6 per integration: (6 / 0.6)^2 = 100 integrations, 1000 s, within the 3600 s span
        assert self.resolve(10.0, combine="scan") == "1000s"

    def test_no_scan_ends_in_a_solution_too_short_for_minsnr(self):
        from wenuray.selfcalibration.preflight import shortest_solint
        # 7 integrations of 10 s are needed (S/N 6 / sqrt(7) each); a 120-integration scan would
        # end in a 1-integration solution, below minsnr. 8 integrations divide it into 15
        sample = dataclasses.replace(SAMPLE, noise=math.sqrt(7), scan_lengths=(1200.0, ))

        solint, why = shortest_solint(sample, Phasecal("auto"), SolveOptions(gaintype="G"))

        assert solint == "80s"
        assert "8 integrations rather than 7" in why

    def test_what_no_length_can_save_is_reported(self):
        from wenuray.selfcalibration.preflight import shortest_solint
        # A 1-integration scan ends short whatever the solint: 1 of 121 integrations
        sample = dataclasses.replace(SAMPLE, noise=math.sqrt(7), scan_lengths=(1200.0, 10.0))

        solint, why = shortest_solint(sample, Phasecal("auto"), SolveOptions(gaintype="G"))

        assert solint == "80s"
        assert "0.8% of the integrations still do" in why

    def test_amplitude_solutions_need_sqrt2_more(self):
        # CASA's S/N for calmode "ap" is the antenna's over sqrt(2): 60 / sqrt(2) in one
        # integration, so 2 x minsnr 30 = 60 takes 2 integrations
        assert self.resolve_step(AmpPhasecal("auto", solve={"minsnr": 30.0})) == "20s"
        assert self.resolve_step(Phasecal("auto", solve={"minsnr": 30.0})) == "int"

    @staticmethod
    def resolve_step(step):
        from wenuray.selfcalibration.preflight import shortest_solint
        return shortest_solint(SAMPLE, step, SolveOptions(gaintype="G"))[0]

    def test_the_steps_minsnr_sets_the_target(self):
        from wenuray.selfcalibration.preflight import shortest_solint
        solint, why = shortest_solint(
            dataclasses.replace(SAMPLE, noise=1.0), Phasecal("auto", solve={"minsnr": 6.0}),
            SolveOptions(gaintype="G")
        )

        # Target 12 from 6 per integration: 4 integrations
        assert solint == "40s"
        assert "12 (2 x minsnr)" in why


class TestAntennaFlagged:
    """#190: each antenna's flagged fraction, from the blocks the sample reads"""

    def test_counts_both_ends_of_each_baseline(self):
        from wenuray.selfcalibration.preflight import _antenna_flagged

        # Two hands, two channels, three rows: 0-1 all flagged, 0-2 half, 1-2 none
        flags = np.zeros((2, 2, 3), dtype=bool)
        flags[:, :, 0] = True
        flags[0, :, 1] = True
        block = {"flags": flags, "antenna1": np.array([0, 0, 1]), "antenna2": np.array([1, 2, 2])}

        fractions = _antenna_flagged([block], 4)

        assert fractions == pytest.approx((0.75, 0.5, 0.25, 1.0))
