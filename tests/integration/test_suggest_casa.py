"""
#190 on simulated data: `wenuray suggest` reads the measurement set, writes a recipe that
`wenuray validate` accepts and `wenuray run` runs, and every derived value says why.
"""
import pytest

from wenuray import cli, suggest
from wenuray.selfcalibration import SelfcalResult
from wenuray.testing import simulate_point_source_ms

pytestmark = pytest.mark.casa


@pytest.fixture(scope="module")
def ms(tmp_path_factory):
    return simulate_point_source_ms(str(tmp_path_factory.mktemp("suggest") / "target.ms"))


def test_reads_the_data(ms):
    summary = suggest.read_summary(ms)

    assert summary.spws == (0, ) and summary.spws_flagged == ()
    assert len(summary.antennas) == 8
    assert all(antenna.flagged == 0.0 for antenna in summary.antennas)
    # The simulation's noise: 0.05 Jy per visibility
    assert summary.sample.noise == pytest.approx(0.05, rel=0.1)
    assert summary.geometry.cell.endswith("arcsec")


def test_the_suggested_recipe_validates_and_runs(ms, tmp_path):
    recipe = tmp_path / "selfcal.toml"

    assert cli.main(["suggest", ms, "-o", str(recipe)]) == 0
    assert cli.main(["validate", str(recipe)]) == 0
    assert cli.main(["run", str(recipe), "--output", str(tmp_path / "run"), "--quiet"]) == 0

    result = SelfcalResult.from_toml(str(tmp_path / "run" / "selfcal.toml"))
    assert all(record.accepted for record in result.rounds)
    assert result.rounds[-1].psnr > result.rounds[0].psnr
    # The amplitude step's "auto" resolved: the data support it
    assert result.rounds[-1].step.startswith("AmpPhasecal('auto' -> ")
