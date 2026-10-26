"""The imager contract (#23): image() enforces CORRECTED_DATA in and MODEL_DATA out."""
import dataclasses
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
from astropy.io import fits

from wenuray.config import ImageGeometry, Selection
from wenuray.imaging import contract
from wenuray.utils.selfcal_utils import SPEED_OF_LIGHT
from wenuray.imaging.contract import (
    CallableImager, Imager, ImagerContractError, ImagingRequest, ImagingResult
)

RESULT = ImagingResult("out.image", "out.residual", "out.model")


@pytest.fixture
def ms(monkeypatch):
    """A measurement set with CORRECTED_DATA whose MODEL_DATA the imager may or may not write."""
    state = SimpleNamespace(columns={"DATA", "CORRECTED_DATA"}, model_written=False, calls=[])

    def delmod(**kwargs):
        state.calls.append(("delmod", kwargs))
        state.model_written = False

    def model_column_problem(ms_name, field, spw):
        state.calls.append(("check", (field, spw)))
        return "" if state.model_written else "MODEL_DATA does not exist"

    monkeypatch.setattr(contract, "delmod", delmod)
    monkeypatch.setattr(
        contract, "is_column_in_ms", lambda ms_name, column: column in state.columns
    )
    monkeypatch.setattr(contract, "model_column_problem", model_column_problem)
    return state


@dataclasses.dataclass(frozen=True)
class FakeImager(Imager):
    writes_model: bool = True
    niter: int = 100
    state: object = None

    def _reconstruct(self, request):
        self.state.calls.append(("reconstruct", request))
        self.state.model_written = self.writes_model
        return RESULT


def request(selection=Selection()) -> ImagingRequest:
    return ImagingRequest("target.ms", "out", selection)


class TestImageEnforcesTheContract:

    def test_a_conforming_imager_returns_its_result(self, ms):
        assert FakeImager(state=ms).image(request()) is RESULT

    def test_the_model_is_removed_before_imaging(self, ms):
        FakeImager(state=ms).image(request())

        names = [name for name, _ in ms.calls]
        assert names == ["delmod", "reconstruct", "check"]
        assert ms.calls[0][1] == {"vis": "target.ms", "otf": True, "scr": True}

    def test_an_imager_that_writes_no_model_raises(self, ms):
        with pytest.raises(
            ImagerContractError, match="FakeImager broke the self-calibration contract"
        ):
            FakeImager(writes_model=False, state=ms).image(request())

    def test_a_stale_model_cannot_pass(self, ms):
        """A model written before the call is removed, so it can't hide an imager that writes none."""
        ms.model_written = True

        with pytest.raises(ImagerContractError):
            FakeImager(writes_model=False, state=ms).image(request())

    def test_the_model_is_checked_on_the_selection(self, ms):
        FakeImager(state=ms).image(request(Selection(field="target", spw="1~3")))

        assert ("check", ("target", "1~3")) in ms.calls

    def test_a_measurement_set_without_corrected_data_raises_before_imaging(self, ms):
        ms.columns = {"DATA"}

        with pytest.raises(ImagerContractError, match="CORRECTED_DATA"):
            FakeImager(state=ms).image(request())
        assert ms.calls == []

    def test_image_cannot_be_overridden(self):
        with pytest.raises(TypeError, match="_reconstruct"):

            class Bypass(Imager):

                def image(self, request):
                    return RESULT

                def _reconstruct(self, request):
                    return RESULT


ARCSEC = np.pi / (180 * 3600)


@dataclasses.dataclass(frozen=True)
class GriddedImager(FakeImager):
    geometry: ImageGeometry = ImageGeometry(imsize=128, cell="2arcsec")


class TestTheCellMustSampleTheData:
    """A 2 arcsec grid holds |u|, |v| up to 1 / (2 * 2 arcsec) = 51.6 klambda (#60)."""

    LIMIT = 1 / (4 * ARCSEC)

    @pytest.fixture
    def uv(self, ms, monkeypatch):
        """Rows at 1 GHz, so an extent of x klambda is 0.3 x km"""
        state = SimpleNamespace(klambda=[10.0, 40.0], selections=[])

        def uv_extent(ms_name, field, spw, uvrange):
            state.selections.append((field, spw, uvrange))
            metres = np.array(state.klambda) * 1e3 * SPEED_OF_LIGHT / 1e9
            return metres, np.full(len(metres), 1e9)

        monkeypatch.setattr(contract, "uv_extent", uv_extent)
        monkeypatch.setattr(contract, "_radians", lambda cell: float(cell[:-6]) * ARCSEC)
        return state

    def test_a_grid_that_holds_every_row_images(self, ms, uv):
        assert GriddedImager(state=ms).image(request()) is RESULT

    def test_a_coarser_cell_raises_before_any_model_is_touched(self, ms, uv):
        uv.klambda = [10.0, 40.0, 59.1, 60.0]

        with pytest.raises(ValueError) as error:
            GriddedImager(state=ms).image(request())

        assert ms.calls == []
        message = str(error.value)
        assert "Nyquist cell of the selected data, 1.719arcsec" in message
        assert "2 of 4 rows" in message
        assert "at most 1.71arcsec" in message
        # 51.6 klambda at 1 GHz is 15.4 km
        assert 'Selection(uvrange="<15400m")' in message

    def test_the_check_uses_the_request_selection(self, ms, uv):
        GriddedImager(state=ms).image(request(Selection("target", "1~3", "<50klambda")))

        assert uv.selections == [("target", "1~3", "<50klambda")]

    def test_a_row_on_the_grid_edge_is_held(self, ms, uv):
        uv.klambda = [self.LIMIT / 1e3]

        assert GriddedImager(state=ms).image(request()) is RESULT

    def test_an_empty_selection_is_left_to_the_model_check(self, ms, uv):
        uv.klambda = []

        with pytest.raises(ImagerContractError, match="MODEL_DATA"):
            GriddedImager(writes_model=False, state=ms).image(request())

    def test_imagers_without_a_grid_are_not_checked(self, ms, uv):
        FakeImager(state=ms).image(request())

        assert uv.selections == []


def test_requests_always_image_corrected_data():
    assert request().data_column == "CORRECTED_DATA"
    with pytest.raises(TypeError):
        ImagingRequest("target.ms", "out", data_column="DATA")


class TestWithOverrides:

    def test_dataclass_imagers_use_replace(self):
        assert FakeImager().with_overrides(niter=500) == FakeImager(niter=500)

    def test_unknown_names_raise(self):
        with pytest.raises(TypeError):
            FakeImager().with_overrides(nitre=500)

    def test_other_imagers_must_implement_it(self):

        class Plain(Imager):

            def _reconstruct(self, request):
                return RESULT

        with pytest.raises(NotImplementedError):
            Plain().with_overrides(niter=1)


class TestCallableImager:

    def test_the_function_gets_the_request_and_the_options(self, ms):

        def reconstruct(request, **options):
            ms.model_written = True
            return RESULT

        function = mock.Mock(side_effect=reconstruct)

        assert CallableImager(function, {"niter": 10}).image(request()) is RESULT
        function.assert_called_once_with(request(), niter=10)

    def test_the_contract_is_enforced(self, ms):
        with pytest.raises(ImagerContractError, match="CallableImager"):
            CallableImager(lambda request: RESULT).image(request())

    def test_overrides_merge_into_the_options(self):
        imager = CallableImager(print, {"niter": 10, "gain": 0.1}).with_overrides(niter=50)

        assert dict(imager.options) == {"niter": 50, "gain": 0.1}

    def test_the_function_must_be_callable(self):
        with pytest.raises(TypeError, match="callable"):
            CallableImager("tclean")


class TestImagingResultStatistics:

    @staticmethod
    def write(path, data, beam=True):
        header = fits.Header()
        if beam:
            header.update(BMAJ=2.0 / 3600, BMIN=1.0 / 3600, BPA=45.0)
        fits.PrimaryHDU(data=data, header=header).writeto(path)
        return str(path)

    def test_fits_images_with_a_mask(self, tmp_path):
        rng = np.random.default_rng(0)
        residual = rng.normal(0.0, 1e-3, (64, 64))
        residual[30:34, 30:34] += 0.1  # source residual, inside the mask
        restored = residual.copy()
        restored[32, 32] += 1.0
        model = np.zeros((64, 64))
        model[32, 32] = 1.0
        mask = np.zeros((64, 64))
        mask[28:36, 28:36] = 1.0
        result = ImagingResult(
            self.write(tmp_path / "r.fits", restored),
            self.write(tmp_path / "res.fits", residual, beam=False),
            self.write(tmp_path / "m.fits", model, beam=False),
            self.write(tmp_path / "mask.fits", mask, beam=False),
        )

        statistics = result.statistics()

        assert statistics.rms == pytest.approx(1e-3, rel=0.1)
        assert statistics.model_flux == pytest.approx(1.0)
        assert statistics.beam == pytest.approx((2.0, 1.0, 45.0))

    def test_a_missing_mask_is_ignored(self, tmp_path):
        rng = np.random.default_rng(0)
        image = rng.normal(0.0, 1e-3, (32, 32))
        path = self.write(tmp_path / "i.fits", image)

        statistics = ImagingResult(path, path, path, str(tmp_path / "no.mask")).statistics()

        assert statistics.rms == pytest.approx(1e-3, rel=0.15)
