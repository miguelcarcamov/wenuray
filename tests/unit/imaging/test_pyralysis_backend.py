"""The Pyralysis imager (#24) without CASA or Pyralysis: the worker process is replaced."""
import json
import warnings
from types import SimpleNamespace
from unittest import mock

import pytest

from wenuray.config import ImageGeometry, Selection, Weighting
from wenuray.imaging.backends import pyralysis as backend
from wenuray.imaging.backends.pyralysis import Pyralysis
from wenuray.imaging.contract import ImagingRequest

GEOMETRY = ImageGeometry(imsize=128, cell="2arcsec")


@pytest.fixture(autouse=True)
def worker(monkeypatch):
    """The worker process: records requests and replies with success unless told otherwise."""
    state = SimpleNamespace(requests=[], reply={"error": ""}, returncode=0, stderr="")

    def run(command, input, **kwargs):
        state.requests.append(SimpleNamespace(command=command, request=json.loads(input), **kwargs))
        return SimpleNamespace(
            returncode=state.returncode,
            stdout="progress\n" + json.dumps(state.reply),
            stderr=state.stderr
        )

    monkeypatch.setattr(backend.subprocess, "run", run)
    backend._check.cache_clear()
    return state


def imager(**options) -> Pyralysis:
    options.setdefault("mask_mode", "file")
    return Pyralysis(GEOMETRY, terms=["chi_squared", "l1norm:1e-3"], optimizer="lbfgsb", **options)


class TestConstruction:

    def test_options_are_checked_by_pyralysis_in_the_worker(self, worker):
        imager(max_iter=100)

        (call, ) = worker.requests
        assert call.request["check"] is True
        assert call.request["options"]["max_iter"] == 100
        assert call.command[1] == "-P"  # the worker's directory would shadow pyralysis

    def test_checks_are_cached_per_option_set(self, worker):
        imager(max_iter=100)
        imager(max_iter=100)
        imager(max_iter=200)

        assert len(worker.requests) == 2

    def test_invalid_options_raise(self, worker):
        worker.reply, worker.returncode = {
            "error": "ValueError: 'maxiter' is not a Pyralysis option"
        }, 1

        with pytest.raises(TypeError, match="maxiter"):
            imager(maxiter=100)

    @pytest.mark.parametrize(
        "name, source",
        [("imsize", "ImageGeometry"), ("term", "terms"), ("hermitian_symmetry", "folds")]
    )
    def test_settings_wenuray_controls_are_rejected(self, name, source):
        with pytest.raises(TypeError, match=source):
            imager(**{name: 1})

    def test_terms_and_optimizer_are_required(self):
        with pytest.raises(TypeError):
            Pyralysis(GEOMETRY)
        with pytest.raises(ValueError, match="terms"):
            Pyralysis(GEOMETRY, terms=[], optimizer="lbfgsb")

    @pytest.mark.parametrize(
        "geometry, weighting, match", [
            (ImageGeometry(cell="1arcsec"), Weighting(), "image size"),
            (
                ImageGeometry(imsize=64, cell="1arcsec",
                              phasecenter="J2000 10h +30d"), Weighting(), "phasecenter"
            ),
            (GEOMETRY, Weighting("superuniform"), "weighting"),
            (GEOMETRY, Weighting(uvtaper=("1arcsec", )), "uvtaper"),
        ]
    )
    def test_unsupported_settings_raise(self, geometry, weighting, match):
        with pytest.raises(ValueError, match=match):
            Pyralysis(geometry, weighting, terms=["chi_squared"], optimizer="lbfgsb")

    def test_no_mask_warns(self):
        with pytest.warns(UserWarning, match="mask"):
            Pyralysis(GEOMETRY, terms=["chi_squared"], optimizer="lbfgsb")

    def test_a_mask_does_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            imager(mask_path="mask.fits")


class TestOptions:

    def test_config_objects_map_to_pyralysis_names(self):
        options = Pyralysis(
            ImageGeometry(imsize=(256, 128), cell="0.1arcsec", stokes="I"),
            Weighting("briggs", robust=-0.5),
            terms="chi_squared",
            optimizer="fista",
            mask_mode="file",
        )._cli_options()

        assert options == {
            "imsize": "256,128",
            "cellsize": "0.1arcsec",
            "stokes": "I",
            "weighting": "robust",
            "robust_parameter": -0.5,
            "term": ["chi_squared"],
            "optimizer": "fista",
            "mask_mode": "file"
        }

    def test_with_overrides(self):
        changed = imager(max_iter=100).with_overrides(max_iter=300, terms=["chi_squared"])

        assert changed.terms == ("chi_squared", ) and changed.options["max_iter"] == 300
        assert changed.optimizer == "lbfgsb"


class TestReconstruct:

    def test_the_worker_gets_the_request(self, worker, monkeypatch, tmp_path):
        monkeypatch.setattr(backend, "_taql", lambda ms, selection: "FIELD_ID IN [0]")
        name = str(tmp_path / "round1")

        result = imager()._reconstruct(ImagingRequest("target.ms", name, Selection(field="t")))

        call = worker.requests[-1]
        assert call.request["taql"] == "FIELD_ID IN [0]"
        assert call.request["outputs"]["model"] == name + ".model.fits"
        assert call.cwd == str(tmp_path)  # scratch files stay out of the working directory
        assert (result.restored, result.model) == (name + ".image.fits", name + ".model.fits")

    def test_failures_raise_with_the_workers_message(self, worker, monkeypatch, tmp_path):
        monkeypatch.setattr(backend, "_taql", lambda ms, selection: "")
        built = imager()
        worker.reply, worker.returncode = {"error": "RuntimeError: out of memory"}, 1

        with pytest.raises(RuntimeError, match="exit code 1.*out of memory"):
            built._reconstruct(ImagingRequest("target.ms", str(tmp_path / "r")))

    def test_a_missing_pyralysis_says_how_to_install_it(self, worker):
        worker.reply, worker.returncode = {
            "error": "ModuleNotFoundError: No module named 'pyralysis'"
        }, 1

        with pytest.raises(ImportError, match=r"wenuray\[pyralysis\]"):
            imager()

    def test_a_missing_numba_threading_layer_says_what_to_install(self, worker):
        worker.reply, worker.returncode = {
            "error": "ThreadingLayerError: Pyralysis needs Numba's OpenMP or TBB threading layer "
            "(...). Install OpenMP's runtime (e.g. apt install libgomp1)"
        }, 1

        with pytest.raises(RuntimeError, match="^Pyralysis needs.*libgomp1"):
            imager()

    def test_an_old_pyralysis_says_which_version_is_needed(self, worker):
        worker.reply, worker.returncode = {
            "error": "PyralysisVersionError: The Pyralysis imager needs Pyralysis 3.2.1 or later "
            "(found 3.1.0): pip install 'wenuray[pyralysis]'"
        }, 1

        with pytest.raises(ImportError, match=r"^The Pyralysis imager needs Pyralysis 3\.2\.1"):
            imager()

    def test_no_selection_selects_everything(self):
        assert backend._taql("target.ms", Selection()) == ""


LENGTH = backend.BASELINE_LENGTH


class TestUvrangeSelection:
    """A uvrange in metres becomes TaQL on UVW, which Pyralysis's LoadData applies (#80)."""

    @pytest.mark.parametrize(
        "uvrange, taql", [
            ("<3090m", LENGTH + " < 3090.0"),
            (">0.5km", LENGTH + " > 500.0"),
            ("100~2000", "(" + LENGTH + " >= 100.0 AND " + LENGTH + " <= 2000.0)"),
            (
                "<100m, >2km",
                "(" + LENGTH + " < 100.0 OR " + LENGTH + " > 2000.0)",
            ),
        ]
    )
    def test_ranges_in_metres(self, uvrange, taql):
        assert backend.uvrange_taql(uvrange) == taql

    @pytest.mark.parametrize("uvrange", [">10klambda", "<5Mlambda", "0~3lambda"])
    def test_wavelengths_are_refused_with_the_reason(self, uvrange):
        with pytest.raises(ValueError, match="metres.*frequency"):
            backend.uvrange_taql(uvrange)

    @pytest.mark.parametrize("uvrange", ["3090m", "<1~2m", "", " , "])
    def test_invalid_ranges(self, uvrange):
        with pytest.raises(ValueError):
            backend.uvrange_taql(uvrange)

    def test_it_joins_the_field_and_spw_selection(self, monkeypatch):
        tool = mock.Mock()
        tool.msseltoindex.return_value = {"field": [0], "dd": [1, 2]}
        monkeypatch.setattr(backend, "ms", lambda: tool)

        taql = backend._taql("target.ms", Selection(field="t", spw="1~2", uvrange="<3090m"))

        assert taql == "FIELD_ID IN [0] AND DATA_DESC_ID IN [1,2] AND " + LENGTH + " < 3090.0"

    def test_a_uvrange_alone_needs_no_measurement_set(self, monkeypatch):
        monkeypatch.setattr(backend, "ms", mock.Mock(side_effect=AssertionError("not needed")))

        assert backend._taql("target.ms", Selection(uvrange=">1km")) == LENGTH + " > 1000.0"
