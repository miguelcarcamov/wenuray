"""The pre-flight solution S/N prediction (#28), from a sample of the data."""
import math

import pytest

from wenuray.config import SolveOptions
import numpy as np

from wenuray.selfcalibration.preflight import VisibilitySample, _selected_channels, predict
from wenuray.selfcalibration.steps import Ampcal, Phasecal

# 1 Jy over 0.1 Jy noise with 10 antennas: 10 * 3 = 30 for one integration, channel and hand
SAMPLE = VisibilitySample(
    signal=1.0,
    noise=0.1,
    nantennas=10,
    interval=10.0,
    scan_length=600.0,
    span=3600.0,
    channels=(4, 4, 8),
    parallel_hands=2
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
