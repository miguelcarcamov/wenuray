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
