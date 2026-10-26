"""The tclean backend (#23): tclean's own parameter names, and Wenuray's settings in one place."""
import os
from unittest import mock

import pytest

from wenuray.config import ImageGeometry, Selection, Weighting
from wenuray.imaging.backends import tclean as backend
from wenuray.imaging.backends.tclean import Tclean
from wenuray.imaging.contract import ImagingRequest

GEOMETRY = ImageGeometry(imsize=256, cell="0.1arcsec")
TCLEAN_PARAMETERS = frozenset(
    {
        "vis", "imagename", "field", "spw", "uvrange", "datacolumn", "savemodel", "imsize", "cell",
        "phasecenter", "stokes", "weighting", "robust", "uvtaper", "niter", "deconvolver",
        "usemask", "nsigma", "threshold", "gridder"
    }
)


@pytest.fixture(autouse=True)
def tclean_signature(monkeypatch):
    monkeypatch.setattr(backend, "_tclean_parameters", lambda: TCLEAN_PARAMETERS)


@pytest.fixture
def tclean(monkeypatch):
    """tclean that writes the image directories it would create."""

    def run(**kwargs):
        suffix = ".tt0" if kwargs.get("deconvolver") == "mtmfs" else ""
        for extension in ("image", "residual", "model"):
            os.makedirs(kwargs["imagename"] + "." + extension + suffix)

    task = mock.Mock(side_effect=run)
    monkeypatch.setattr(backend, "tclean", task)
    return task


class TestOptions:

    def test_options_keep_tcleans_names(self):
        imager = Tclean(GEOMETRY, deconvolver="hogbom", niter=500, usemask="auto-multithresh")

        assert dict(imager.options) == {
            "deconvolver": "hogbom",
            "niter": 500,
            "usemask": "auto-multithresh"
        }

    @pytest.mark.parametrize(
        "name, source", [
            ("savemodel", "contract"),
            ("datacolumn", "contract"),
            ("field", "Selection"),
            ("imsize", "ImageGeometry"),
            ("robust", "Weighting"),
            ("vis", "request"),
        ]
    )
    def test_settings_wenuray_controls_are_rejected(self, name, source):
        with pytest.raises(TypeError, match=source):
            Tclean(GEOMETRY, **{name: "x"})

    def test_names_that_are_not_tclean_parameters_are_rejected(self):
        with pytest.raises(TypeError, match="use_mask"):
            Tclean(GEOMETRY, use_mask="auto-multithresh")

    def test_imsize_and_cell_are_required(self):
        with pytest.raises(ValueError, match="imsize"):
            Tclean(ImageGeometry(cell="0.1arcsec"))

    def test_with_overrides_merges_options_and_replaces_config(self):
        imager = Tclean(GEOMETRY, niter=100, nsigma=3.0)

        changed = imager.with_overrides(niter=1000, weighting=Weighting("briggs", robust=0.0))

        assert dict(changed.options) == {"niter": 1000, "nsigma": 3.0}
        assert changed.weighting == Weighting("briggs", robust=0.0)
        assert changed.geometry == GEOMETRY
        assert dict(imager.options) == {"niter": 100, "nsigma": 3.0}  # unchanged

    def test_overrides_are_validated(self):
        with pytest.raises(TypeError, match="Selection"):
            Tclean(GEOMETRY).with_overrides(spw="0")

    def test_imagers_compare_by_value(self):
        assert Tclean(GEOMETRY, niter=10) == Tclean(GEOMETRY, niter=10)


class TestReconstruct:

    def test_contract_selection_geometry_and_weighting_reach_tclean(self, tclean, tmp_path):
        imager = Tclean(
            ImageGeometry(imsize=(512, 256), cell="0.1arcsec", phasecenter="J2000 10h +30d"),
            Weighting("briggs", robust=0.5, uvtaper=("0.2arcsec", )),
            niter=100,
        )
        name = str(tmp_path / "out")

        result = imager._reconstruct(
            ImagingRequest("target.ms", name, Selection(field="t", spw="1~3"))
        )

        kwargs = tclean.call_args.kwargs
        assert kwargs["vis"] == "target.ms" and kwargs["imagename"] == name
        assert (kwargs["datacolumn"], kwargs["savemodel"]) == ("corrected", "modelcolumn")
        assert (kwargs["field"], kwargs["spw"], kwargs["uvrange"]) == ("t", "1~3", "")
        assert (kwargs["imsize"], kwargs["cell"],
                kwargs["phasecenter"]) == ([512, 256], "0.1arcsec", "J2000 10h +30d")
        assert (kwargs["weighting"], kwargs["robust"],
                kwargs["uvtaper"]) == ("briggs", 0.5, ["0.2arcsec"])
        assert kwargs["niter"] == 100
        assert result.restored == name + ".image" and result.mask == name + ".mask"

    def test_mtmfs_images_are_the_zeroth_taylor_term(self, tclean, tmp_path):
        name = str(tmp_path / "out")

        result = Tclean(GEOMETRY,
                        deconvolver="mtmfs")._reconstruct(ImagingRequest("target.ms", name))

        assert result.restored == name + ".image.tt0"
        assert result.model == name + ".model.tt0"

    def test_missing_output_raises(self, monkeypatch, tmp_path):
        monkeypatch.setattr(backend, "tclean", mock.Mock())

        with pytest.raises(RuntimeError, match="did not write"):
            Tclean(GEOMETRY)._reconstruct(ImagingRequest("target.ms", str(tmp_path / "out")))


class TestLegacyOptions:
    """Tclean calls written for snow < 1.0 are told where each option went."""

    @pytest.mark.parametrize(
        "options, hint", [
            (
                {
                    "inputvis": "target.ms",
                    "M": 1024,
                    "N": 1024
                }, r"M -> ImageGeometry\(imsize=\(M, N\)\)"
            ),
            ({
                "use_mask": "auto-multithresh"
            }, "use_mask -> usemask"),
            ({
                "sidelobe_threshold": 3.0
            }, "sidelobe_threshold -> sidelobethreshold"),
            ({
                "save_model": True
            }, "MODEL_DATA is always written"),
        ]
    )
    def test_legacy_options_name_their_replacement(self, options, hint):
        with pytest.raises(TypeError, match=hint):
            Tclean(**options)

    def test_tclean_stand_ins_without_a_signature_skip_the_name_check(self, monkeypatch):
        monkeypatch.setattr(backend, "_tclean_parameters", lambda: None)

        assert Tclean(GEOMETRY, anything=1).options["anything"] == 1
