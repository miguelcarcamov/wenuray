"""
The scientific regression test of #32: self-calibration recovers the antenna phase errors that
were injected into simulated data, with each imager, and the image improves.

Self-calibration recovers the phases up to the reference antenna's phase (A0 has no error) and up
to the source position: the first model sits where the corrupted image put the source, and a
shifted model is a phase gradient across the array, l u + m v, that the gains take on. The test
fits that shift out, and checks that it is small.
"""
import importlib.util

import numpy as np
import pytest

from wenuray.config import ImageGeometry, SolveOptions
from wenuray.imaging.backends import Pyralysis, Tclean
from wenuray.selfcalibration import Phasecal, Selfcal
from wenuray.testing import injected_phases, simulate_point_source_ms
from wenuray.utils.selfcal_utils import SPEED_OF_LIGHT, open_table

GEOMETRY = ImageGeometry(imsize=128, cell="1arcsec")


def box_mask(path: str) -> str:
    """A 9x9-pixel mask on the source, as Pyralysis's mask_path wants it"""
    from astropy.io import fits

    mask = np.zeros((128, 128), dtype=np.float32)
    mask[60:69, 60:69] = 1.0
    fits.PrimaryHDU(mask).writeto(path)
    return path


def tclean(tmp_path):
    return Tclean(GEOMETRY, niter=200, nsigma=3.0)


def pyralysis(tmp_path):
    return Pyralysis(
        GEOMETRY,
        terms=["chi_squared"],
        optimizer="lbfgsb",
        max_iter=50,
        mask_mode="file",
        mask_path=box_mask(str(tmp_path / "mask.fits")),
    )


IMAGERS = [
    pytest.param(tclean, id="tclean"),
    pytest.param(
        pyralysis,
        id="pyralysis",
        marks=pytest.mark.skipif(
            importlib.util.find_spec("pyralysis") is None, reason="needs Pyralysis"
        ),
    ),
]


def solved_phases(caltable: str, times: np.ndarray, nantennas: int) -> np.ndarray:
    """A table's phase solutions at the given times (nearest solution), shaped (antenna, time)"""
    with open_table(caltable) as table:
        antenna, time = table.getcol("ANTENNA1"), table.getcol("TIME")
        gains, flags = table.getcol("CPARAM"), table.getcol("FLAG")  # (correlation, channel, row)
    phases = np.full((nantennas, len(times)), np.nan)
    for index in range(nantennas):
        rows = np.flatnonzero((antenna == index) & ~flags[0, 0])
        nearest = rows[np.abs(time[rows][None, :] - times[:, None]).argmin(axis=1)]
        phases[index] = np.angle(gains[0, 0, nearest])
    return phases


def antenna_uv(ms: str, times: np.ndarray, nantennas: int) -> np.ndarray:
    """Each antenna's u and v relative to A0, in wavelengths, shaped (2, antenna, time)"""
    with open_table(ms) as table:
        antenna1, antenna2 = table.getcol("ANTENNA1"), table.getcol("ANTENNA2")
        time, uvw = table.getcol("TIME"), table.getcol("UVW")
    with open_table(ms + "/SPECTRAL_WINDOW") as table:
        wavelength = SPEED_OF_LIGHT / table.getcell("REF_FREQUENCY", 0)
    uv = np.zeros((2, nantennas, len(times)))
    rows = np.flatnonzero(antenna1 == 0)
    uv[:, antenna2[rows], np.searchsorted(times, time[rows])] = uvw[:2, rows] / wavelength
    return uv


def without_a_shift(error: np.ndarray, uv: np.ndarray) -> tuple:
    """The phase error less the best-fitting source shift, and the shift (l, m) in arcsec"""
    design = 2 * np.pi * uv.reshape(2, -1).T
    shift, *_ = np.linalg.lstsq(design, error.ravel(), rcond=None)
    residual = np.angle(np.exp(1j * (error.ravel() - design @ shift)))
    return residual.reshape(error.shape), np.degrees(shift) * 3600


@pytest.mark.parametrize("make_imager", IMAGERS)
def test_phase_selfcal_recovers_the_injected_phases(tmp_path, make_imager):
    ms = simulate_point_source_ms(str(tmp_path / "target.ms"))
    times, injected = injected_phases(ms)

    result = Selfcal(
        ms,
        make_imager(tmp_path),
        [Phasecal("inf"), Phasecal("int"),
         Phasecal("int"), Phasecal("int")],
        solve=SolveOptions(refant="A0", minblperant=3),
    ).run()

    assert all(record.accepted for record in result.rounds)
    assert result.final.psnr > 4 * result.rounds[0].psnr
    # The applied correction is the sum of the accepted tables' phases
    solved = sum(solved_phases(table.path, times, len(injected)) for table in result.stack)
    error, shift = without_a_shift(
        np.angle(np.exp(1j * (solved - injected))), antenna_uv(ms, times, len(injected))
    )
    # The longest baselines, 59 klambda, give a 3.5 arcsec beam
    assert np.hypot(*shift) < 1.2
    # 0.6 rad of injected errors (rms); one solution's noise is 0.013 rad
    assert np.sqrt(np.mean(injected[1:]**2)) > 0.5
    assert np.sqrt(np.mean(error**2)) < 0.1
    assert np.abs(error).max() < 0.3
