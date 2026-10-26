"""Imagers by name through the wenuray.imagers entry point group (#23)."""
from types import SimpleNamespace

import pytest

from wenuray.config import ImageGeometry
from wenuray.imaging import registry
from wenuray.imaging.backends import tclean as backend
from wenuray.imaging.backends.tclean import Tclean


def entry_point(name, loaded):
    return SimpleNamespace(name=name, load=lambda: loaded)


@pytest.fixture
def registered(monkeypatch):
    """Registers tclean, a third-party imager and something that is not an imager."""
    points = [entry_point("tclean", Tclean), entry_point("notanimager", dict)]

    def entry_points(group, name=None):
        assert group == "wenuray.imagers"
        return [point for point in points if name is None or point.name == name]

    monkeypatch.setattr(registry, "entry_points", entry_points)
    monkeypatch.setattr(backend, "_tclean_parameters", lambda: frozenset({"niter"}))


def test_tclean_is_registered_by_the_package():
    """Reads the installed package metadata, so it checks pyproject's entry point."""
    assert "tclean" in registry.available_imagers()


def test_imager_from_name_passes_the_arguments(registered):
    imager = registry.imager_from_name("tclean", ImageGeometry(imsize=64, cell="1arcsec"), niter=5)

    assert imager == Tclean(ImageGeometry(imsize=64, cell="1arcsec"), niter=5)


def test_unknown_names_list_the_available_imagers(registered):
    with pytest.raises(KeyError, match="'notanimager', 'tclean'"):
        registry.imager_from_name("wsclean")


def test_registered_objects_must_be_imagers(registered):
    with pytest.raises(TypeError, match="not an Imager"):
        registry.imager_from_name("notanimager")
