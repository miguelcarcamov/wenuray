"""The built-in imagers pass the public imager contract suite (#23) with real CASA."""
import os
import re

import pytest

from wenuray.config import ImageGeometry, Selection
from wenuray.imaging.backends import Tclean
from wenuray.imaging.contract import CallableImager, ImagerContractError, ImagingRequest, ImagingResult
from wenuray.testing import ImagerContractSuite, simulate_point_source_ms
from wenuray.utils.selfcal_utils import SPEED_OF_LIGHT, uv_extent


class TestTcleanContract(ImagerContractSuite):
    imager = Tclean(ImageGeometry(imsize=128, cell="1arcsec"), niter=200, nsigma=3.0)


def test_an_imager_that_writes_no_model_is_caught(tmp_path):
    """With real CASA: the model tclean wrote earlier is removed, so it can't hide the omission."""
    ms = simulate_point_source_ms(str(tmp_path / "target.ms"))
    tclean_imager = Tclean(ImageGeometry(imsize=64, cell="1arcsec"), niter=10)
    tclean_imager.image(ImagingRequest(ms, str(tmp_path / "first")))
    lazy = CallableImager(lambda request: ImagingResult("x", "x", "x"))

    with pytest.raises(ImagerContractError, match="MODEL_DATA does not exist"):
        lazy.image(ImagingRequest(ms, str(tmp_path / "second")))


def test_a_cell_coarser_than_nyquist_raises_and_the_suggested_uvrange_images(tmp_path):
    """The simulated baselines reach 59.1 klambda, so Nyquist needs cells of 1.74 arcsec or less."""
    ms = simulate_point_source_ms(str(tmp_path / "target.ms"))
    coarse = Tclean(ImageGeometry(imsize=64, cell="2arcsec"), niter=10)

    with pytest.raises(ValueError, match="Nyquist cell of the selected data, 1.74") as error:
        coarse.image(ImagingRequest(ms, str(tmp_path / "coarse")))
    assert not os.path.exists(str(tmp_path / "coarse.image"))

    uvrange = re.search(r'uvrange="([^"]+)"', str(error.value)).group(1)
    result = coarse.image(ImagingRequest(ms, str(tmp_path / "cut"), Selection(uvrange=uvrange)))
    assert os.path.exists(result.restored)


def test_the_uv_extent_of_a_selection(tmp_path):
    ms = simulate_point_source_ms(str(tmp_path / "target.ms"))

    metres, frequency = uv_extent(ms)
    klambda = metres * frequency / SPEED_OF_LIGHT / 1e3

    assert klambda.max() == pytest.approx(59.1, abs=0.1)
    assert uv_extent(ms, uvrange="<30klambda")[0].max() * frequency.max() / SPEED_OF_LIGHT < 3e4
    assert len(uv_extent(ms, field="no such field")[0]) == 0
