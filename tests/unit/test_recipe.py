"""TOML recipes (#36): a recipe and a Python script build equal objects and raise the same errors."""
import dataclasses
import os
import textwrap

import pytest

from wenuray import recipe
from wenuray.config import ImageGeometry, Selection, SolveOptions, Weighting
from wenuray.imaging.backends import Tclean
from wenuray.imaging.backends import pyralysis as pyralysis_backend
from wenuray.imaging.backends.pyralysis import Pyralysis
from wenuray.recipe import RecipeError
from wenuray.selfcalibration import (
    AllOf, AmpPhasecal, CalStack, CalTable, Check, EnoughAntennas, FluxNotLost, Phasecal,
    PSNRImproves, ResidualFlagging, ScheduleError, Selfcal
)
from wenuray.selfcalibration.acceptance import AcceptancePolicy, Decision

GEOMETRY = ImageGeometry(imsize=1024, cell="0.05arcsec")


@pytest.fixture(autouse=True)
def no_pyralysis_worker(monkeypatch):
    """Pyralysis checks its options in a worker process; accept them here."""
    monkeypatch.setattr(pyralysis_backend, "_check", lambda options: None)


def write(path, text: str) -> str:
    path.write_text(textwrap.dedent(text))
    return str(path)


RECIPE = """
    ms = "target.ms"

    [selection]
    spw = "0~3"

    [solve]
    refant = "DA51"
    combine = "spw"

    [imager]
    type = "tclean"
    niter = 1000
    deconvolver = "hogbom"
    geometry = { imsize = [1024, 1024], cell = "0.05arcsec" }
    weighting = { weighting = "briggs", robust = 0.5 }

    [[steps]]
    type = "phase"
    solint = "inf"

    [[steps]]
    type = "phase"
    solint = "60s"
    flagging = { type = "residual", mode = "rflag", timedevscale = 4.0 }

    [[steps]]
    type = "ampphase"
    solint = "inf"
    imaging = { threshold = "0.025mJy" }
    acceptance = { type = "all_of", policies = [
        { type = "psnr_improves", min_ratio = 1.05 },
        { type = "flux_not_lost", max_loss = 0.2 },
    ] }
"""


def script(directory) -> Selfcal:
    return Selfcal(
        ms=os.path.join(directory, "target.ms"),
        selection=Selection(spw="0~3"),
        solve=SolveOptions(refant="DA51", combine="spw"),
        imager=Tclean(GEOMETRY, Weighting("briggs", robust=0.5), niter=1000, deconvolver="hogbom"),
        steps=[
            Phasecal("inf"),
            Phasecal("60s", flagging=ResidualFlagging("rflag", timedevscale=4.0)),
            AmpPhasecal(
                "inf",
                imaging={"threshold": "0.025mJy"},
                acceptance=AllOf(PSNRImproves(1.05), FluxNotLost(0.2))
            ),
        ],
    )


class TestRecipeAndScriptAgree:

    def test_they_build_equal_runs(self, tmp_path):
        assert Selfcal.from_toml(write(tmp_path / "recipe.toml", RECIPE)) == script(tmp_path)

    def test_a_run_written_to_toml_builds_back(self, tmp_path):
        run = script(tmp_path)
        run.to_toml(tmp_path / "written.toml")

        assert Selfcal.from_toml(tmp_path / "written.toml") == run

    def test_the_written_recipe_names_the_imager_and_leaves_defaults_out(self, tmp_path):
        script(tmp_path).to_toml(tmp_path / "written.toml")

        text = (tmp_path / "written.toml").read_text()
        assert 'ms = "target.ms"' in text
        assert 'type = "tclean"' in text
        assert "minsnr" not in text and "output" not in text
        assert text.index("[solve]") < text.index("[imager]") < text.index("[[steps]]")

    def test_everything_a_run_can_hold_round_trips(self, tmp_path):
        run = Selfcal(
            ms=str(tmp_path / "data" / "target.ms"),
            output=str(tmp_path / "elsewhere"),
            imager=Pyralysis(
                GEOMETRY,
                terms=["chi_squared", "l1norm:1e-3"],
                optimizer="lbfgsb",
                mask_mode="file",
                max_iter=50
            ),
            steps=[
                Phasecal(30.0, solve={"minsnr": 5.0}),
                AmpPhasecal("inf", solnorm=False, imaging={"geometry": {
                    "cell": "0.02arcsec"
                }}),
            ],
            acceptance=AllOf(PSNRImproves(), EnoughAntennas(max_short_fraction=0.1)),
            initial_stack=CalStack([CalTable(str(tmp_path / "data" / "p.cal"), "p", (0, 0), "")]),
        )
        os.makedirs(tmp_path / "recipes")
        run.to_toml(tmp_path / "recipes" / "run.toml")

        text = (tmp_path / "recipes" / "run.toml").read_text()
        assert 'path = "../data/p.cal"' in text
        assert Selfcal.from_toml(tmp_path / "recipes" / "run.toml") == run


class TestErrorsAreTheConstructors:

    def test_a_typo_in_a_table(self, tmp_path):
        path = write(tmp_path / "recipe.toml", RECIPE.replace("refant =", "refnat ="))

        with pytest.raises(TypeError) as from_recipe:
            Selfcal.from_toml(path)
        with pytest.raises(TypeError) as from_script:
            SolveOptions(refnat="DA51", combine="spw")
        assert str(from_recipe.value) == str(from_script.value)

    def test_an_invalid_schedule(self, tmp_path):
        steps = """
            [[steps]]
            type = "ampphase"
            solint = "inf"
        """
        path = write(tmp_path / "recipe.toml", RECIPE.split("[[steps]]")[0] + steps)

        with pytest.raises(ScheduleError) as from_recipe:
            Selfcal.from_toml(path)
        with pytest.raises(ScheduleError) as from_script:
            dataclasses.replace(script(tmp_path), steps=[AmpPhasecal("inf")])
        assert str(from_recipe.value) == str(from_script.value)

    def test_an_unknown_top_level_key(self, tmp_path):
        path = write(tmp_path / "recipe.toml", "nrounds = 3\n" + RECIPE)

        with pytest.raises(TypeError, match="nrounds"):
            Selfcal.from_toml(path)

    def test_an_unknown_type_lists_the_built_in_ones(self, tmp_path):
        path = write(tmp_path / "recipe.toml", RECIPE.replace('"ampphase"', '"amplitude"'))

        with pytest.raises(RecipeError, match=r"step type 'amplitude'.*'amp', 'ampphase'"):
            Selfcal.from_toml(path)

    def test_a_table_without_a_type(self):
        with pytest.raises(RecipeError, match="needs a type"):
            recipe.decode("step", {"solint": "inf"})

    def test_a_recipe_without_steps(self, tmp_path):
        path = write(tmp_path / "recipe.toml", RECIPE.split("[[steps]]")[0])

        with pytest.raises(RecipeError, match="needs steps"):
            Selfcal.from_toml(path)

    def test_invalid_toml(self, tmp_path):
        path = write(tmp_path / "recipe.toml", "ms = target.ms\n")

        with pytest.raises(RecipeError, match="not valid TOML"):
            Selfcal.from_toml(path)


@dataclasses.dataclass(frozen=True)
class ClosurePhasesImprove(AcceptancePolicy):
    """A policy of the user's own, importable from this module"""
    factor: float = 1.0

    def decide(self, previous, current):
        return Decision.accept("closure phases")


class TestClassesOfYourOwn:

    def test_are_named_by_module_and_class(self, tmp_path):
        run = dataclasses.replace(
            script(tmp_path), acceptance=AllOf(PSNRImproves(), ClosurePhasesImprove(0.9))
        )
        run.to_toml(tmp_path / "recipe.toml")

        assert '"tests.unit.test_recipe:ClosurePhasesImprove"' in (tmp_path /
                                                                   "recipe.toml").read_text()
        assert Selfcal.from_toml(tmp_path / "recipe.toml") == run

    def test_a_class_of_another_family_is_refused(self):
        with pytest.raises(RecipeError, match="not a step class"):
            recipe.decode("step", {"type": "tests.unit.test_recipe:ClosurePhasesImprove"})

    def test_a_function_can_not_be_written(self, tmp_path):
        run = dataclasses.replace(script(tmp_path), acceptance=Check("x", lambda p, c: True))

        with pytest.raises(RecipeError, match="can't hold"):
            run.to_toml(tmp_path / "recipe.toml")

    def test_a_class_defined_in_a_script_can_not_be_written(self):
        Local = type("Local", (PSNRImproves, ), {"__module__": "__main__"})

        with pytest.raises(RecipeError, match="defined in a script"):
            recipe.encode(Local())


class TestPresets:

    def test_the_recipe_is_laid_over_its_preset(self, tmp_path):
        os.makedirs(tmp_path / "shared")
        write(
            tmp_path / "shared" / "group.toml", """
            [solve]
            refant = "DA41"
            minsnr = 5.0

            [imager]
            type = "tclean"
            niter = 500
            nsigma = 3.0
            geometry = { imsize = 512, cell = "0.1arcsec" }
        """
        )
        path = write(
            tmp_path / "recipe.toml", """
            preset = "shared/group.toml"
            ms = "target.ms"

            [solve]
            refant = "DA51"

            [imager]
            niter = 1000
            geometry = { cell = "0.05arcsec" }

            [[steps]]
            type = "phase"
            solint = "inf"
        """
        )

        run = Selfcal.from_toml(path)

        assert run.solve == SolveOptions(refant="DA51", minsnr=5.0)
        assert run.imager == Tclean(
            ImageGeometry(imsize=512, cell="0.05arcsec"), niter=1000, nsigma=3.0
        )
        assert run.ms == str(tmp_path / "target.ms")

    def test_presets_that_include_each_other(self, tmp_path):
        write(tmp_path / "a.toml", 'preset = "b.toml"\n')
        write(tmp_path / "b.toml", 'preset = "a.toml"\n')

        with pytest.raises(RecipeError, match="include each other"):
            recipe.read(str(tmp_path / "a.toml"))

    def test_steps_are_replaced_not_merged(self):
        merged = recipe.merge(
            {
                "steps": [{
                    "type": "phase"
                }],
                "solve": {
                    "refant": "A"
                }
            }, {
                "steps": [{
                    "type": "amp"
                }],
                "solve": {
                    "minsnr": 2.0
                }
            }
        )

        assert merged == {"steps": [{"type": "amp"}], "solve": {"refant": "A", "minsnr": 2.0}}


def test_imaging_overrides_can_change_part_of_the_geometry():
    imager = Tclean(GEOMETRY, niter=10)

    finer = imager.with_overrides(
        geometry={"cell": "0.02arcsec"}, weighting={"weighting": "uniform"}
    )

    assert finer.geometry == ImageGeometry(imsize=1024, cell="0.02arcsec")
    assert finer.weighting == Weighting("uniform")
