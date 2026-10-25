"""
The Pyralysis imager (#24) with real CASA and Pyralysis: the imager contract, and self-calibration
end to end. Skipped where Pyralysis isn't installed. Pyralysis is never imported here: it runs in
its own process, since it can't share one with casatools.
"""
import importlib.util
import re
import warnings

import numpy as np
import pytest
from astropy.io import fits

from wenuray.config import ImageGeometry, Selection, SolveOptions
from wenuray.imaging import Pyralysis
from wenuray.imaging.contract import ImagingRequest
from wenuray.selfcalibration import Phasecal, Selfcal
from wenuray.testing import ImagerContractSuite, simulate_point_source_ms
from wenuray.utils.selfcal_utils import open_table

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("pyralysis") is None, reason="Pyralysis isn't installed"
)

# Nyquist sampling: the longest baselines (~59 klambda) need cells of 1.74 arcsec or less. With
# coarser cells they fall outside the uv grid, which Pyralysis's fit doesn't represent
GEOMETRY = ImageGeometry(imsize=128, cell="1arcsec")


def pyralysis_imager(**options) -> Pyralysis:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # no mask: the contract doesn't need one
        return Pyralysis(
            GEOMETRY, terms=["chi_squared"], optimizer="lbfgsb", max_iter=200, **options
        )


class TestPyralysisContract(ImagerContractSuite):
    imager = pyralysis_imager() if importlib.util.find_spec("pyralysis") else None


def test_options_are_checked_by_pyralysis():
    with pytest.raises(TypeError, match="Invalid Pyralysis options"):
        pyralysis_imager(linesearch="no-such-linesearch")
    with pytest.raises(TypeError, match="not a Pyralysis option"):
        pyralysis_imager(maxiter=10)


def test_settings_wenuray_controls_are_rejected():
    with pytest.raises(TypeError, match="ImageGeometry"):
        pyralysis_imager(imsize=256)


def box_mask(path) -> str:
    """A box around the source, as for CLEAN: a model free in every pixel absorbs the gain errors"""
    mask = np.zeros((128, 128), dtype=np.int16)
    mask[60:69, 60:69] = 1
    fits.PrimaryHDU(mask).writeto(path)
    return str(path)


def test_the_masked_model_holds_the_source_flux(tmp_path):
    ms = simulate_point_source_ms(str(tmp_path / "target.ms"), phase_errors_in=("DATA", ))
    imager = pyralysis_imager(mask_mode="file", mask_path=box_mask(tmp_path / "mask.fits"))

    statistics = imager.image(ImagingRequest(ms, str(tmp_path / "masked"))).statistics()

    assert statistics.model_flux == pytest.approx(1.0, rel=0.05)


def test_a_measurement_set_read_in_several_chunks(tmp_path):
    """Dask runs the chunks in parallel threads, which Numba allows with OpenMP or TBB only"""
    ms = simulate_point_source_ms(str(tmp_path / "target.ms"), phase_errors_in=("DATA", ))
    mask = box_mask(tmp_path / "mask.fits")

    whole, chunked = (
        pyralysis_imager(mask_mode="file", mask_path=mask,
                         **options).image(ImagingRequest(ms, str(tmp_path / name))).statistics()
        for name, options in (("whole", {}), ("chunked", {
            "row_chunks": 37
        }))
    )

    assert chunked.peak == pytest.approx(whole.peak, rel=0.01)
    assert chunked.model_flux == pytest.approx(whole.model_flux, rel=0.01)


def test_self_calibration_with_pyralysis(tmp_path):
    ms = simulate_point_source_ms(str(tmp_path / "target.ms"))
    imager = pyralysis_imager(mask_mode="file", mask_path=box_mask(tmp_path / "mask.fits"))

    result = Selfcal(
        ms,
        imager, [Phasecal("inf"), Phasecal("10min")],
        solve=SolveOptions(refant="A0", minblperant=3)
    ).run()

    psnr = [record.psnr for record in result.rounds]
    assert all(record.accepted for record in result.rounds)
    assert psnr[0] < psnr[1] < psnr[2] and psnr[2] > 2 * psnr[0]
    # gaincal found the phase errors: the model didn't absorb them
    with open_table(result.stack.tables[0].path) as cal_table:
        phases = np.degrees(np.angle(cal_table.getcol("CPARAM")))
    assert np.max(np.abs(phases)) > 10.0


# Pyralysis 3.2.1 (#80): its own LoadData and CheckpointModelResiduals, and row selections


def test_the_installed_pyralysis_is_one_wenuray_supports():
    from importlib.metadata import version

    from wenuray.imaging.backends._pyralysis_worker import MINIMUM_PYRALYSIS

    found = tuple(int(part) for part in version("pyralysis").split(".")[:3])
    assert found >= MINIMUM_PYRALYSIS >= (3, 2, 1)


def baseline_lengths(ms: str) -> np.ndarray:
    with open_table(ms) as ms_table:
        uvw = ms_table.getcol("UVW")
    return np.hypot(uvw[0], uvw[1])


def model_written(ms: str) -> np.ndarray:
    """Which rows have a non-zero model"""
    with open_table(ms) as ms_table:
        return np.any(ms_table.getcol("MODEL_DATA") != 0, axis=(0, 1))


def test_only_the_model_column_is_written(tmp_path):
    ms = simulate_point_source_ms(str(tmp_path / "target.ms"), phase_errors_in=("DATA", ))

    pyralysis_imager(max_iter=20).image(ImagingRequest(ms, str(tmp_path / "image")))

    with open_table(ms) as ms_table:
        columns = ms_table.colnames()
    assert "MODEL_DATA" in columns and "RESIDUAL_DATA" not in columns
    assert model_written(ms).all()


def test_a_uvrange_selects_the_rows_pyralysis_fits_and_models(tmp_path):
    """The model is written for the selected baselines only, so gaincal solves on those"""
    ms = simulate_point_source_ms(str(tmp_path / "target.ms"), phase_errors_in=("DATA", ))
    lengths = baseline_lengths(ms)
    cut = float(np.median(lengths))

    pyralysis_imager(max_iter=20).image(
        ImagingRequest(ms, str(tmp_path / "image"), Selection(uvrange="<{0}m".format(cut)))
    )

    written = model_written(ms)
    assert written[lengths < cut].all()
    assert not written[lengths > cut].any()


def test_a_coarse_cell_images_with_the_uvrange_the_nyquist_check_suggests(tmp_path):
    """#60 refuses a 2 arcsec cell for these data and suggests a uvrange; Pyralysis can use it"""
    ms = simulate_point_source_ms(str(tmp_path / "target.ms"), phase_errors_in=("DATA", ))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        coarse = Pyralysis(
            ImageGeometry(imsize=64, cell="2arcsec"),
            terms=["chi_squared"],
            optimizer="lbfgsb",
            max_iter=50,
            mask_mode="file",
            mask_path=box_mask_64(tmp_path / "mask.fits"),
        )

    with pytest.raises(ValueError, match="Nyquist") as error:
        coarse.image(ImagingRequest(ms, str(tmp_path / "refused")))
    uvrange = re.search(r'uvrange="([^"]+)"', str(error.value)).group(1)
    statistics = coarse.image(
        ImagingRequest(ms, str(tmp_path / "cut"), Selection(uvrange=uvrange))
    ).statistics()

    assert statistics.model_flux == pytest.approx(1.0, rel=0.05)


def box_mask_64(path) -> str:
    mask = np.zeros((64, 64), dtype=np.int16)
    mask[28:37, 28:37] = 1
    fits.PrimaryHDU(mask).writeto(path)
    return str(path)
