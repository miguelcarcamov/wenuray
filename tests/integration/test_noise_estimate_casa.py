"""
#153 on simulated data: antenna phases drawn anew every integration, the same in both hands, as
the atmosphere's are. Consecutive integrations then keep part of the source in their
differences, and read the noise too high; the parallel hands cancel it. The thermal noise and the
weights check must agree with those of the same data without gain errors.
"""
import shutil
from statistics import median

import numpy as np
import pytest
from casatools import table

from wenuray.config import ImageGeometry, Selection
from wenuray.imaging.backends import Tclean
from wenuray.imaging.contract import ImagingRequest
from wenuray.selfcalibration.preflight import read_sample
from wenuray.testing import simulate_point_source_ms

IMAGER = Tclean(ImageGeometry(imsize=128, cell="1arcsec"), niter=200, nsigma=3.0)


def corrupt_every_integration(ms: str, degrees: float, seed: int = 3) -> None:
    """Antenna phases drawn anew for every integration, the same in every correlation"""
    tool = table()
    tool.open(ms, nomodify=False)
    try:
        times = tool.getcol("TIME")
        antenna1, antenna2 = tool.getcol("ANTENNA1"), tool.getcol("ANTENNA2")
        unique, at = np.unique(times, return_inverse=True)
        rng = np.random.default_rng(seed)
        phases = rng.normal(0.0, np.radians(degrees), (len(unique), antenna2.max() + 1))
        gain = np.exp(1j * (phases[at, antenna1] - phases[at, antenna2]))
        for column in ("DATA", "CORRECTED_DATA"):
            tool.putcol(column, tool.getcol(column) * gain[None, None, :])
    finally:
        tool.close()
        tool.done()


@pytest.fixture(scope="module")
def samples(tmp_path_factory):
    """The sample of the ideal data and of the same data with gains drawn every integration"""
    directory = tmp_path_factory.mktemp("noise")
    ideal = simulate_point_source_ms(str(directory / "ideal.ms"), phase_errors_in=())
    corrupted = str(directory / "corrupted.ms")
    shutil.copytree(ideal, corrupted)
    corrupt_every_integration(corrupted, 40.0)
    found = {}
    for name, ms in (("ideal", ideal), ("corrupted", corrupted)):
        IMAGER.image(ImagingRequest(ms, str(directory / name), Selection()))
        found[name] = read_sample(ms, Selection())
    return found


def test_the_thermal_noise_agrees_with_the_ideal_datas(samples):
    ideal, corrupted = samples["ideal"], samples["corrupted"]

    assert corrupted.noise_from == "the parallel hands"
    assert corrupted.thermal_noise == pytest.approx(ideal.thermal_noise, rel=0.05)
    # The time differences hold the source the gains moved
    assert corrupted.other_noise[0] == "consecutive integrations"
    assert corrupted.other_noise[1] > 1.5 * corrupted.noise


def test_the_weights_check_agrees_with_the_ideal_datas(samples):
    ratios = {
        name: median(ratio for *_, ratio in sample.noise_over_sigma)
        for name, sample in samples.items()
    }

    assert ratios["corrupted"] == pytest.approx(ratios["ideal"], rel=0.05)


def test_steady_data_keep_the_consecutive_integrations(samples):
    assert samples["ideal"].noise_from == "consecutive integrations"
