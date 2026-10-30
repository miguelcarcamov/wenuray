"""The Pyralysis worker script (#24), without Pyralysis: the checks it makes before importing it."""
from importlib import metadata

import pytest

from wenuray.imaging.backends import _pyralysis_worker as worker


@pytest.mark.parametrize("found", ["3.2.1", "3.2.2", "3.10.0", "4.0.0", "3.2.1.dev4+g1a2b3c"])
def test_a_recent_enough_pyralysis_passes(monkeypatch, found):
    monkeypatch.setattr(metadata, "version", lambda name: found)

    worker.require_pyralysis()


@pytest.mark.parametrize("found", ["3.1.0", "3.2.0", "2.9.0"])
def test_an_older_pyralysis_is_refused_with_the_version_it_needs(monkeypatch, found):
    monkeypatch.setattr(metadata, "version", lambda name: found)

    with pytest.raises(worker.PyralysisVersionError, match="3.2.1 or later.*found " + found):
        worker.require_pyralysis()


def test_a_missing_pyralysis_says_how_to_install_it(monkeypatch):

    def missing(name):
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(metadata, "version", missing)

    with pytest.raises(worker.PyralysisVersionError, match=r"not installed.*wenuray\[pyralysis\]"):
        worker.require_pyralysis()


def test_the_versions_request_names_what_the_worker_runs_with(monkeypatch, capsys):
    installed = {"pyralysis": "3.3.0", "numba": "0.61.2", "cupy-cuda12x": "13.6.0"}

    def version(name):
        if name not in installed:
            raise metadata.PackageNotFoundError(name)
        return installed[name]

    monkeypatch.setattr(metadata, "version", version)
    monkeypatch.setattr(worker, "require_pyralysis", lambda: None)
    monkeypatch.setattr(worker, "require_thread_safe_numba", lambda: None)
    monkeypatch.setattr(worker.sys, "stdin", __import__("io").StringIO('{"versions": true}'))

    assert worker.main() == 0
    reply = __import__("json").loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert reply == {"error": "", "versions": installed}
