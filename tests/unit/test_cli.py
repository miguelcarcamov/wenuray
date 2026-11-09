"""The wenuray command (#36): init, validate, show and run over TOML recipes."""
import logging
import os
import textwrap
import tomllib
from types import SimpleNamespace

import pytest

from wenuray import cli
from wenuray.config import SolveOptions
from wenuray.selfcalibration import Selfcal

RECIPE = """
    ms = "target.ms"

    [solve]
    refant = "DA51"

    [imager]
    type = "tclean"
    niter = 100
    geometry = { imsize = 256, cell = "0.1arcsec" }

    [[steps]]
    type = "phase"
    solint = "inf"

    [[steps]]
    type = "ampphase"
    solint = "inf"
"""


@pytest.fixture
def recipe(tmp_path):
    path = tmp_path / "selfcal.toml"
    path.write_text(textwrap.dedent(RECIPE))
    return str(path)


def wenuray(capsys, *argv):
    status = cli.main(list(argv))
    output = capsys.readouterr()
    return SimpleNamespace(status=status, out=output.out, err=output.err)


class TestInit:

    def test_writes_a_recipe_that_validates(self, capsys, tmp_path):
        result = wenuray(capsys, "init")
        (tmp_path / "selfcal.toml").write_text(result.out)

        assert result.status == 0
        assert wenuray(capsys, "validate", str(tmp_path / "selfcal.toml")).status == 0

    def test_to_a_file_that_it_does_not_overwrite(self, capsys, tmp_path):
        path = str(tmp_path / "selfcal.toml")

        assert wenuray(capsys, "init", "-o", path).status == 0
        result = wenuray(capsys, "init", "-o", path)
        assert result.status == 1
        assert "already exists" in result.err

    def test_ships_no_instrument_values(self, capsys):
        """Wenuray's defaults come from theory or the data (#36)"""
        assert "preset = " not in wenuray(capsys, "init").out.replace("# preset = ", "")


class TestValidate:

    def test_lists_the_run(self, capsys, recipe, tmp_path):
        result = wenuray(capsys, "validate", recipe)

        assert result.status == 0
        assert "valid" in result.out
        assert str(tmp_path / "target.ms") in result.out
        assert "step 2:  AmpPhasecal('inf')" in result.out
        assert "does not exist" in result.out  # a note, not an error

    def test_a_typo_is_the_constructor_error(self, capsys, recipe):
        with open(recipe) as file:
            text = file.read()
        with open(recipe, "w") as file:
            file.write(text.replace("refant", "refnat"))

        result = wenuray(capsys, "validate", recipe)

        assert result.status == 1
        assert result.err.startswith("wenuray: error:") and "refnat" in result.err

    def test_an_invalid_schedule(self, capsys, recipe):
        with open(recipe) as file:
            text = file.read()
        with open(recipe, "w") as file:
            file.write(text.replace('type = "phase"', 'type = "amp"'))

        result = wenuray(capsys, "validate", recipe)

        assert result.status == 1
        assert "phase" in result.err

    def test_traceback_reraises(self, recipe):
        with pytest.raises(FileNotFoundError):
            cli.main(["--traceback", "validate", recipe + ".missing"])


class TestShow:

    def test_prints_the_recipe_with_its_preset(self, capsys, recipe, tmp_path):
        (tmp_path / "group.toml").write_text('[solve]\nminsnr = 5.0\n')
        with open(recipe) as file:
            text = file.read()
        with open(recipe, "w") as file:
            file.write('preset = "group.toml"\n' + text)

        shown = tomllib.loads(wenuray(capsys, "show", recipe).out)

        assert shown["solve"] == {"refant": "DA51", "minsnr": 5.0}
        assert shown["ms"] == "target.ms"

    def test_defaults_prints_every_value(self, capsys, recipe):
        shown = tomllib.loads(wenuray(capsys, "show", "--defaults", recipe).out)

        assert shown["solve"]["minblperant"] == 4
        assert shown["output"] == "target_selfcal"
        assert shown["imager"]["weighting"] == {
            "weighting": "natural",
            "robust": 0.5,
            "uvtaper": []
        }
        assert shown["steps"][0]["flagging"] == {"type": "none"}

    def test_the_resolved_recipe_builds_the_same_run(self, capsys, recipe, tmp_path):
        resolved = tmp_path / "resolved.toml"
        resolved.write_text(wenuray(capsys, "show", "--defaults", recipe).out)

        assert Selfcal.from_toml(resolved) == Selfcal.from_toml(recipe)

    def test_ms_and_output_overrides(self, capsys, recipe, tmp_path):
        other = str(tmp_path / "data" / "other.ms")

        shown = tomllib.loads(wenuray(capsys, "show", "--defaults", recipe, "--ms", other).out)

        # Without --output, the output follows the new measurement set
        assert shown["ms"] == os.path.join("data", "other.ms")
        assert shown["output"] == os.path.join("data", "other_selfcal")


def test_run_runs_the_recipe(capsys, recipe, monkeypatch):
    runs = []

    def run(self, observers):
        runs.append((self, observers))
        record = SimpleNamespace(image="round2.image")
        return SimpleNamespace(rounds=[record] * 3, accepted_rounds=[record] * 2, final=record)

    monkeypatch.setattr(Selfcal, "run", run)

    result = wenuray(capsys, "run", "--quiet", "--plots", recipe)

    assert result.status == 0
    assert "1 of 2 steps accepted; final image round2.image" in result.out
    (selfcal, observers), = runs
    assert selfcal.solve == SolveOptions(refant="DA51")
    assert [type(observer).__name__ for observer in observers] == \
        ["LoggingObserver", "GainPlotObserver"]


def test_apply_records_the_application_in_the_run(capsys, tmp_path, monkeypatch):
    from wenuray.config import Selection
    from wenuray.selfcalibration import SelfcalResult
    from wenuray.selfcalibration import result as result_module
    from wenuray.selfcalibration.application import Application
    from wenuray.selfcalibration.caltable import CalStack, CalTable

    monkeypatch.setattr(
        result_module, "apply_to",
        lambda stack, ms, selection, applymode: Application(ms, selection, applymode, stack, "")
    )
    run = tmp_path / "target_selfcal"
    run.mkdir()
    SelfcalResult(str(tmp_path / "target.ms"),
                  CalStack([CalTable(str(run / "p.cal"), "p")])).to_toml(str(run / "selfcal.toml"))

    result = wenuray(capsys, "apply", str(run), "--ms", str(tmp_path / "line.ms"), "--spw", "1")

    assert result.status == 0 and "1 tables applied" in result.out
    (application, ) = SelfcalResult.from_toml(str(run / "selfcal.toml")).applications
    assert application.ms == str(tmp_path / "line.ms"
                                 ) and application.selection == Selection(spw="1")


@pytest.mark.parametrize(
    "flags, level",
    [((), logging.INFO), (("--quiet", ), logging.WARNING), (("--debug", ), logging.DEBUG)]
)
def test_run_logs_wenuray_at_the_level_asked(capsys, recipe, monkeypatch, flags, level):
    """#176: --debug adds the details, e.g. Pyralysis's output; --quiet keeps warnings"""
    record = SimpleNamespace(image="round1.image")
    monkeypatch.setattr(
        Selfcal, "run", lambda self, observers:
        SimpleNamespace(rounds=[record], accepted_rounds=[record], final=record)
    )
    logger = logging.getLogger("wenuray")
    monkeypatch.setattr(logger, "level", logger.level)

    wenuray(capsys, "run", *flags, recipe)

    assert logger.level == level


def test_quiet_and_debug_exclude_each_other(capsys, recipe):
    with pytest.raises(SystemExit):
        cli.main(["run", "--quiet", "--debug", recipe])


@pytest.mark.parametrize("flags, filtered", [((), True), (("--debug", ), False)])
def test_run_keeps_casas_routine_notices_quiet_but_with_debug(
    capsys, recipe, monkeypatch, flags, filtered
):
    """#179: notices every run gets from CASA, which say nothing about the data"""
    calls = []
    monkeypatch.setattr(cli, "_quiet_casa_notices", lambda: calls.append(True))
    record = SimpleNamespace(image="round1.image")
    monkeypatch.setattr(
        Selfcal, "run", lambda self, observers:
        SimpleNamespace(rounds=[record] * 2, accepted_rounds=[record] * 2, final=record)
    )

    wenuray(capsys, "run", *flags, recipe)

    assert bool(calls) is filtered


class TestSuggest:
    """#121: the command over wenuray.suggest, with the data read mocked"""

    @pytest.fixture
    def suggested(self, monkeypatch):
        calls = []

        def fake(ms, imager, resolve=False, relative_to=None):
            calls.append((ms, imager, resolve, relative_to))
            return "# suggested\n"

        monkeypatch.setattr("wenuray.suggest.suggest", fake)
        return calls

    def test_writes_the_suggestion(self, capsys, suggested):
        result = wenuray(capsys, "suggest", "target.ms", "--imager", "tclean")

        assert result.status == 0
        assert result.out == "# suggested\n"
        assert suggested == [("target.ms", "tclean", False, None)]

    def test_to_a_file_with_paths_relative_to_it(self, capsys, suggested, tmp_path):
        path = tmp_path / "selfcal.toml"

        result = wenuray(
            capsys, "suggest", "target.ms", "--imager", "pyralysis", "--resolve", "-o", str(path)
        )

        assert result.status == 0
        assert path.read_text() == "# suggested\n"
        assert suggested == [("target.ms", "pyralysis", True, str(tmp_path))]

    def test_does_not_overwrite_a_file(self, capsys, suggested, tmp_path):
        path = tmp_path / "selfcal.toml"
        path.write_text("mine")

        result = wenuray(capsys, "suggest", "target.ms", "--imager", "tclean", "-o", str(path))

        assert result.status == 1 and "already exists" in result.err
        assert path.read_text() == "mine"

    def test_the_imager_is_required(self, capsys):
        with pytest.raises(SystemExit):
            cli.main(["suggest", "target.ms"])
