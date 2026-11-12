"""
#190 on simulated data: `wenuray suggest --image` makes a quick model without touching the
measurement set, and its S/N decides the recipe's steps.
"""
import numpy as np
import pytest

from wenuray import cli, suggest
from wenuray.selfcalibration import AmpPhasecal, Selfcal
from wenuray.testing import simulate_point_source_ms
from wenuray.utils.selfcal_utils import open_table

pytestmark = pytest.mark.casa


def columns(ms):
    with open_table(ms) as table:
        return {name: table.getcol(name) for name in sorted(table.colnames()) if "DATA" in name}


def test_the_quick_model_holds_the_source_and_leaves_the_data(tmp_path):
    ms = simulate_point_source_ms(str(tmp_path / "target.ms"))
    before = columns(ms)

    summary = suggest.read_summary(ms, image=True)

    after = columns(ms)
    assert sorted(after) == sorted(before)
    assert all(np.array_equal(after[name], before[name]) for name in before)
    # A 1 Jy point source with phase errors in DATA. The model's total flux also holds the
    # gain-error artefacts, which stand above any noise threshold (#198), but their visibilities
    # scatter in phase, so the median amplitude, which predicts the S/N, is the source's
    model = summary.model
    assert model.signal == pytest.approx(1.0, abs=0.2)
    assert model.nsigma == pytest.approx(np.sqrt(2 * np.log(model.beams)))


def test_the_suggested_recipe_keeps_the_amplitude_step(tmp_path):
    ms = simulate_point_source_ms(str(tmp_path / "target.ms"))
    recipe = tmp_path / "selfcal.toml"

    assert cli.main(["suggest", ms, "--image", "-o", str(recipe)]) == 0

    text = recipe.read_text()
    assert "--image: a quick tclean model of DATA" in text
    assert Selfcal.from_toml(str(recipe)).steps[-1] == AmpPhasecal("auto")
