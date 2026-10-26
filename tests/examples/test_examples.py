"""
The README and the examples must run against the current API (#31). Each examples/<name>/selfcal.py
builds its Selfcal for real, so the schedule, the step overrides and the tclean option names are
validated; only run() is replaced, so no data is needed. Each recipe.toml must build the same run
as its script.
"""
import dataclasses
import os
import re
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

import wenuray.utils
from wenuray.selfcalibration import (
    AmpPhasecal, CalStack, GainPlotObserver, LoggingObserver, Phasecal, Selfcal, SelfcalResult
)

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"


def readme_example() -> str:
    readme = (ROOT / "README.md").read_text()
    section = readme[readme.index("### Basic Usage Example"):]
    return re.search(r"```python\n(.*?)```", section, re.DOTALL).group(1)


def run_example(name: str, *args: str) -> None:
    script = EXAMPLES / name / "selfcal.py"
    with mock.patch.object(sys, "argv", [str(script), *args]):
        runpy.run_path(str(script), run_name="__main__")


@pytest.fixture
def runs(monkeypatch, tmp_path):
    """Records each Selfcal the example runs, instead of running it."""
    monkeypatch.chdir(tmp_path)
    recorded = []

    def run(selfcal, observers=()):
        recorded.append(SimpleNamespace(selfcal=selfcal, observers=list(observers)))
        return SelfcalResult(selfcal.ms, CalStack(), ())

    monkeypatch.setattr(Selfcal, "run", run)
    return recorded


@pytest.fixture
def casatasks_module(monkeypatch):
    """casatasks as imported by the example scripts themselves (split, flagdata, mstransform)."""
    module = mock.MagicMock(name="casatasks")
    monkeypatch.setitem(sys.modules, "casatasks", module)
    return module


def describe(steps) -> list:
    return [(type(step), step.solint, dict(step.solve), dict(step.imaging)) for step in steps]


def test_readme_example_is_the_alma_script():
    assert readme_example() == (EXAMPLES / "alma" / "selfcal.py").read_text()


def test_alma_example(runs, casatasks_module):
    run_example("alma", "target.ms", "out")

    (run, ) = runs
    assert describe(run.selfcal.steps) == [
        (Phasecal, "inf", {
            "combine": "spw"
        }, {}),
        (Phasecal, "600s", {
            "combine": "spw"
        }, {}),
        (AmpPhasecal, "inf", {}, {
            "threshold": "0.025mJy"
        }),
    ]
    assert run.selfcal.solve.refant == "DA51" and run.selfcal.output == "out"
    assert run.selfcal.imager.options["usemask"] == "auto-multithresh"
    casatasks_module.split.assert_called_once_with(
        vis="target.ms", outputvis="out/selfcal.ms", datacolumn="corrected"
    )


@pytest.mark.parametrize("script", ["alma", "vla"])
@pytest.mark.parametrize(
    "plot, observers", [([], [LoggingObserver]), (["--plot"], [LoggingObserver, GainPlotObserver])]
)
def test_plot_option_adds_gain_plots(script, plot, observers, runs, casatasks_module):
    run_example(script, "target.ms", "out", *plot)

    assert [type(observer) for observer in runs[0].observers] == observers


def test_vla_example(runs, casatasks_module):
    run_example("vla", "target.ms", "out")

    assert describe(runs[0].selfcal.steps) == [
        (Phasecal, "inf", {
            "combine": "spw"
        }, {}),
        (AmpPhasecal, "inf", {}, {}),
    ]
    assert runs[0].selfcal.imager.options["pbcor"] is True


class TestEmerlinExample:

    @pytest.fixture(autouse=True)
    def eight_spws(self, monkeypatch):
        monkeypatch.setattr(wenuray.utils, "get_spw_ids", lambda ms, spw="": list(range(8)))
        monkeypatch.setattr(wenuray.utils, "count_spectral_windows", lambda ms: 8)

    def run(self, *args):
        run_example("emerlin", "target.ms", "out", "1252+5634", *args)

    @pytest.mark.parametrize(
        "flag_args, flagged_spws", [
            ([], []), (["--flag"], [""]),
            (["--flag", "--rflag-spw", "0", "--rflag-spw", "1~7"], ["0", "1~7"])
        ]
    )
    def test_flags_before_selfcal(self, flag_args, flagged_spws, runs, casatasks_module):
        self.run(*flag_args)

        rflag = [
            c.kwargs["spw"] for c in casatasks_module.flagdata.call_args_list
            if c.kwargs["mode"] == "rflag"
        ]
        assert rflag == flagged_spws

    @pytest.mark.parametrize(
        "phase_args, spwmap", [([], (0, ) * 8), (["--phase-caltable-per-spw"], ())]
    )
    def test_continues_from_the_phase_table(self, phase_args, spwmap, runs, casatasks_module):
        self.run(*phase_args)

        (table, ) = runs[0].selfcal.initial_stack
        assert (table.path, table.calmode, table.spwmap) == ("pcal", "p", spwmap)

    def test_cleans_deeper_each_round(self, runs, casatasks_module):
        self.run("--uvrange", ">60klambda")

        selfcal = runs[0].selfcal
        assert describe(selfcal.steps) == [
            (AmpPhasecal, "14min", {}, {
                "nsigma": 3.0
            }),
            (AmpPhasecal, "7min", {}, {
                "nsigma": 2.5
            }),
            (AmpPhasecal, "3.5min", {}, {
                "nsigma": 2.0
            }),
        ]
        assert all(step.flagging.mode == "rflag" for step in selfcal.steps)
        assert selfcal.selection.uvrange == ">60klambda"
        assert selfcal.apply.applymode == "calonly"
        assert selfcal.solve.combine == "spw" and selfcal.solve.gaintype == "G"


def test_the_readme_recipe_builds_the_readme_schedule(tmp_path):
    readme = (ROOT / "README.md").read_text()
    section = readme[readme.index("### The same run as a TOML recipe"):]
    (tmp_path /
     "recipe.toml").write_text(re.search(r"```toml\n(.*?)```", section, re.DOTALL).group(1))

    selfcal = Selfcal.from_toml(tmp_path / "recipe.toml")

    assert selfcal.steps == (
        Phasecal("inf", solve={"combine": "spw"}),
        Phasecal("600s", solve={"combine": "spw"}),
        AmpPhasecal("inf", imaging={"threshold": "0.025mJy"}),
    )
    assert selfcal.output == str(tmp_path / "target_selfcal")


def by_name(selfcal: Selfcal) -> Selfcal:
    """The run with its paths reduced to file names: the script's are relative to the working
    directory, the recipe's to its own"""
    tables = [dataclasses.replace(t, path=os.path.basename(t.path)) for t in selfcal.initial_stack]
    return dataclasses.replace(
        selfcal,
        ms=os.path.basename(selfcal.ms),
        output=os.path.basename(selfcal.output),
        initial_stack=CalStack(tables)
    )


@pytest.mark.parametrize(
    "name, args", [
        ("alma", ["target.ms", "target_selfcal"]),
        ("vla", ["target.ms", "target_selfcal"]),
        ("emerlin", ["target.ms", "target_selfcal", "1252+5634"]),
    ]
)
def test_each_recipe_builds_its_script_run(name, args, runs, casatasks_module, monkeypatch):
    monkeypatch.setattr(wenuray.utils, "get_spw_ids", lambda ms, spw="": list(range(8)))
    monkeypatch.setattr(wenuray.utils, "count_spectral_windows", lambda ms: 8)
    run_example(name, *args)

    from_recipe = Selfcal.from_toml(EXAMPLES / name / "recipe.toml")

    assert by_name(from_recipe) == by_name(runs[0].selfcal)
