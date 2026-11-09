"""
wenuray suggest (#121) on simulated data: a 1 Jy point source with antenna phase errors in DATA.
The recipe it writes must build and self-calibrate the data, and the signal it starts from must
be the source's flux, which the phase errors don't lower.
"""
import os

import pytest

from wenuray import recipe
from wenuray.config import AUTO, ImageGeometry, Selection
from wenuray.imaging.contract import resolve_geometry
from wenuray.selfcalibration import Selfcal
from wenuray.selfcalibration.preflight import read_sample
from wenuray.suggest import suggest
from wenuray.testing import simulate_point_source_ms


@pytest.fixture(scope="module")
def ms(tmp_path_factory):
    return simulate_point_source_ms(str(tmp_path_factory.mktemp("suggest") / "target.ms"))


def write(tmp_path, text: str) -> str:
    path = tmp_path / "selfcal.toml"
    path.write_text(text)
    return str(path)


def test_the_signal_from_the_data_is_the_flux_despite_phase_errors(ms):
    sample = read_sample(ms, Selection(), signal_from="data")

    # 1 Jy; the debiased amplitude of one baseline averages many visibilities, so it is far
    # better than the noise of one (0.05 Jy)
    assert sample.signal == pytest.approx(1.0, abs=0.02)
    assert sample.noise == pytest.approx(0.05, rel=0.1)


def test_the_recipe_builds_and_names_its_choices(ms, tmp_path):
    run = Selfcal.from_recipe(
        recipe.read(write(tmp_path, suggest(ms, "tclean", relative_to=str(tmp_path))))
    )

    assert run.ms == ms
    assert run.selection.field == "target"
    assert run.solve.refant in {"A{0}".format(i) for i in range(8)}
    assert run.solve.combine == ""  # one window: never combined
    assert run.imager.geometry.auto


def test_resolve_writes_what_auto_resolves_to(ms, tmp_path):
    run = Selfcal.from_recipe(
        recipe.read(
            write(tmp_path, suggest(ms, "tclean", resolve=True, relative_to=str(tmp_path)))
        )
    )

    expected = resolve_geometry(
        ImageGeometry(imsize=AUTO, cell=AUTO), ms, Selection(field="target")
    )
    assert run.imager.geometry.cell == expected.cell
    assert run.imager.geometry.imsize == expected.imsize
    assert all(step.solint != AUTO for step in run.steps)


def test_the_suggested_recipe_self_calibrates_the_data(ms, tmp_path):
    path = write(tmp_path, suggest(ms, "tclean", relative_to=str(tmp_path)))
    table = recipe.read(path)
    table["output"] = str(tmp_path / "selfcal")
    # The suggested selection, solve options and schedule, on a small grid: "auto" spans 2.5
    # primary beams, 1920 x 1920 pixels on this simulation, too slow for a test (the grid itself
    # is test_resolve_writes_what_auto_resolves_to's)
    table["imager"]["geometry"] = {"imsize": 128, "cell": "1arcsec"}

    result = Selfcal.from_recipe(table).run()

    first, last = result.rounds[0], result.final
    assert result.rounds[1].accepted
    assert last.psnr > 2 * first.psnr
    assert os.path.exists(os.path.join(str(tmp_path / "selfcal"), "selfcal.toml"))
