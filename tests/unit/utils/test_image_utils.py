"""FITS utilities: reproject onto non-square grids, HDUs usable after the file is closed (#29)."""
import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from wenuray.utils import get_hdu, reproject

NAXIS1, NAXIS2 = 100, 60  # non-square: columns != rows


def celestial_wcs() -> WCS:
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---SIN", "DEC--SIN"]
    wcs.wcs.crval = [180.0, 45.0]
    wcs.wcs.crpix = [NAXIS1 / 2, NAXIS2 / 2]
    wcs.wcs.cdelt = [-1e-5, 1e-5]
    return wcs


def write_model_image(path) -> str:
    """A 4D (RA, Dec, frequency, Stokes) image like those written by CASA."""
    wcs = WCS(naxis=4)
    wcs.wcs.ctype = ["RA---SIN", "DEC--SIN", "FREQ", "STOKES"]
    wcs.wcs.crval = [180.0, 45.0, 1.4e9, 1.0]
    wcs.wcs.crpix = [NAXIS1 / 2, NAXIS2 / 2, 1.0, 1.0]
    wcs.wcs.cdelt = [-1e-5, 1e-5, 1e6, 1.0]
    fits.writeto(path, np.zeros((1, 1, NAXIS2, NAXIS1), np.float32), wcs.to_header())
    return str(path)


def write_mask(path) -> np.ndarray:
    mask = np.zeros((NAXIS2, NAXIS1), np.float32)
    mask[10:30, 20:70] = 1.0
    fits.writeto(path, mask, celestial_wcs().to_header())
    return mask


class TestReproject:

    @pytest.fixture
    def resampled(self, tmp_path):
        model = write_model_image(tmp_path / "model.fits")
        mask = write_mask(tmp_path / "mask.fits")
        output = reproject(str(tmp_path / "mask.fits"), model, order="nearest-neighbor")
        return mask, output

    def test_output_follows_the_model_shape_rows_by_columns(self, resampled):
        mask, output = resampled

        assert fits.getdata(output).shape == (NAXIS2, NAXIS1)

    def test_mask_on_the_same_grid_is_unchanged_and_stays_binary(self, resampled):
        mask, output = resampled

        np.testing.assert_array_equal(fits.getdata(output), mask)

    def test_output_header_is_2d_like_the_data(self, resampled):
        _, output = resampled
        header = fits.getheader(output)

        assert header["NAXIS"] == 2
        assert WCS(header).naxis == 2

    def test_missing_input_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="missing.fits"):
            reproject(str(tmp_path / "missing.fits"), str(tmp_path / "model.fits"))


def test_get_hdu_data_is_usable_after_the_file_is_closed(tmp_path):
    path = tmp_path / "image.fits"
    fits.writeto(path, np.arange(6, dtype=np.float32).reshape(2, 3))

    hdu = get_hdu(str(path))

    np.testing.assert_array_equal(hdu.data, np.arange(6).reshape(2, 3))
