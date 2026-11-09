"""wenuray suggest (#121): each value of the recipe from the data, with its reason."""
import dataclasses
import tomllib

import numpy as np
import pytest

from wenuray import recipe, suggest
from wenuray.imaging.backends import pyralysis as pyralysis_backend
from wenuray.selfcalibration import Selfcal
from wenuray.selfcalibration.preflight import VisibilitySample, predict
from wenuray.selfcalibration.steps import AmpPhasecal, Phasecal
from wenuray.config import SolveOptions

# Four windows of four channels, 46 antennas, 6 s integrations in 60 s scans
SAMPLE = VisibilitySample(
    signal=0.1,
    noise=0.075,
    nantennas=46,
    interval=6.0,
    scan_length=60.0,
    span=2400.0,
    channels=(4, 4, 4, 4),
    parallel_hands=2,
    band=(335.5e9, 351.5e9),
    weights=(1.0e6, 1.0e6),
)


@pytest.fixture(autouse=True)
def no_pyralysis_worker(monkeypatch):
    """Pyralysis checks its options in a worker process; accept them here."""
    monkeypatch.setattr(pyralysis_backend, "_check", lambda options: None)


def with_signal(signal: float) -> VisibilitySample:
    return dataclasses.replace(SAMPLE, signal=signal)


class TestField:

    def test_the_only_field(self):
        assert suggest.choose_field([("HD135344B", 743728)]).value == "HD135344B"

    def test_several_fields_are_all_selected_and_listed(self):
        choice = suggest.choose_field([("A", 10), ("B", 20)])

        assert choice.value == ""
        assert "A (10 rows)" in choice.reason and "B (20 rows)" in choice.reason


class TestRefant:
    POSITIONS = np.array([[0.0, 0, 0], [10, 0, 0], [500, 0, 0], [-400, 0, 0], [0, 300, 0]])
    NAMES = np.array(["A0", "A1", "A2", "A3", "A4"])

    def test_the_antenna_nearest_the_centre(self):
        flagged = np.zeros(5)

        choice = suggest.choose_refant(self.NAMES, self.POSITIONS, flagged)

        # The median position is (0, 0, 0)
        assert choice.value == "A0"
        assert "0 m from the array's centre" in choice.reason

    def test_skips_an_antenna_flagged_more_than_most(self):
        flagged = np.array([0.5, 0.02, 0.0, 0.01, 0.0])

        assert suggest.choose_refant(self.NAMES, self.POSITIONS, flagged).value == "A1"

    def test_antennas_without_data_are_not_candidates_and_do_not_move_the_centre(self):
        flagged = np.array([np.nan, 0.0, 0.0, 0.0, 0.0])

        # The centre of A1-A4 is (5, 0, 0): A1 is 5 m from it, A4 300 m
        choice = suggest.choose_refant(self.NAMES, self.POSITIONS, flagged)

        assert choice.value == "A1"
        assert "5 m from the array's centre" in choice.reason

    def test_no_data_raises(self):
        with pytest.raises(ValueError, match="No antenna"):
            suggest.choose_refant(self.NAMES, self.POSITIONS, np.full(5, np.nan))


class TestCombine:

    def test_each_window_its_own_solutions_when_one_reaches_twice_minsnr(self):
        choice = suggest.choose_combine(SAMPLE, 4)

        assert choice.value == ""
        snr = predict(SAMPLE, Phasecal("inf"), SolveOptions()).snr
        assert "{0:.1f}".format(snr) in choice.reason

    def test_windows_combined_when_one_falls_short(self):
        # S/N scales with the signal: a hundredth of it leaves one window below 6
        faint = with_signal(0.001)

        choice = suggest.choose_combine(faint, 4)

        assert predict(faint, Phasecal("inf"), SolveOptions()).snr < 6
        assert choice.value == "spw"
        assert "the 4 windows combined" in choice.reason

    def test_a_single_window_is_never_combined(self):
        assert suggest.choose_combine(with_signal(0.001), 1).value == ""


class TestAmplitudeGaintype:

    def test_one_solution_per_polarization_when_it_reaches_twice_minsnr(self):
        assert suggest.choose_amplitude_gaintype(SAMPLE, "").value == "G"

    def test_shared_when_one_polarization_falls_short(self):
        faint = with_signal(0.002)
        assert predict(faint, AmpPhasecal("inf"), SolveOptions(gaintype="G")).snr < 6

        assert suggest.choose_amplitude_gaintype(faint, "").value == "T"


class TestSteps:

    def test_phases_per_scan_then_repeated_shorter_then_amplitudes(self):
        steps = suggest.choose_steps(SAMPLE, "", "G", resolve=False)

        assert [step.value for step in steps] == [
            {
                "type": "phase",
                "solint": "inf"
            },
            {
                "type": "repeat",
                "step": {
                    "type": "phase",
                    "solint": "auto"
                }
            },
            {
                "type": "ampphase",
                "solint": "auto",
                "solve": {
                    "gaintype": "G"
                }
            },
        ]

    def test_amplitudes_per_window_after_combined_phases(self):
        steps = suggest.choose_steps(SAMPLE, "spw", "T", resolve=False)

        assert steps[-1].value["solve"] == {"gaintype": "T", "combine": ""}

    def test_resolved_solints_are_what_auto_resolves_to_now(self):
        steps = suggest.choose_steps(SAMPLE, "", "G", resolve=True)

        assert [step.value["solint"] for step in steps] == ["int", "int"]
        assert "S/N" in steps[0].reason

    def test_no_supported_solint_raises(self):
        with pytest.raises(ValueError, match="no solint"):
            suggest.choose_steps(with_signal(1e-6), "", "T", resolve=True)


def rendered(imager: str, resolve: bool = False) -> str:
    geometry = suggest.Choice(
        {
            "imsize": 1000,
            "cell": "0.04arcsec"
        } if resolve else {
            "imsize": "auto",
            "cell": "auto"
        }, "grid"
    )
    return suggest.render(
        ms="target.ms",
        field=suggest.Choice("HD135344B", "the only field"),
        spws=[13, 15, 17, 19],
        refant=suggest.Choice("DA51", "central"),
        combine=suggest.Choice("", "enough S/N"),
        steps=suggest.choose_steps(SAMPLE, "", "G", resolve),
        imager=imager,
        geometry=geometry,
        sample=SAMPLE,
    )


class TestRender:

    @pytest.mark.parametrize("imager", suggest.IMAGERS)
    @pytest.mark.parametrize("resolve", [False, True])
    def test_the_recipe_builds_a_run(self, tmp_path, imager, resolve):
        path = tmp_path / "selfcal.toml"
        path.write_text(rendered(imager, resolve))

        with pytest.warns(UserWarning, match="mask") if imager == "pyralysis" else _no_warning():
            run = Selfcal.from_recipe(recipe.read(str(path)))

        assert run.ms == str(tmp_path / "target.ms")
        assert run.solve.refant == "DA51"
        assert run.selection.field == "HD135344B"

    def test_every_value_has_its_reason(self):
        text = rendered("tclean")
        lines = [line for line in text.splitlines() if " = " in line and not line.startswith("#")]
        commented = {line.split(" = ")[0] for line in lines if "  # " in line}

        assert {"ms", "field", "spw", "refant", "combine", "geometry"} <= commented

    def test_notes_the_spectral_index_for_the_imager(self):
        assert "deconvolver = \"mtmfs\"" in rendered("tclean")
        assert "spectral_index" in rendered("pyralysis")
        assert "4.7%" in rendered("tclean")  # 2 (351.5 - 335.5) / (351.5 + 335.5)

    def test_pyralysis_says_to_add_a_mask(self):
        assert "mask_mode = \"file\"" in rendered("pyralysis")

    @pytest.mark.parametrize(
        "value, text", [
            ("int", '"int"'),
            (3.0, "3.0"),
            (["chi_squared", "l1norm:1e-4"], '["chi_squared", "l1norm:1e-4"]'),
            ({
                "imsize": "auto",
                "cell": "auto"
            }, '{ imsize = "auto", cell = "auto" }'),
        ]
    )
    def test_values_are_inline_toml(self, value, text):
        assert suggest._toml(value) == text
        assert tomllib.loads("x = " + text)["x"] == value


class TestRecipePath:

    def test_relative_inside_the_recipe_directory(self, tmp_path):
        ms = tmp_path / "data" / "target.ms"

        assert suggest.recipe_path(str(ms), str(tmp_path)) == "data/target.ms"

    def test_absolute_outside_it(self, tmp_path):
        ms = tmp_path / "data" / "target.ms"

        assert suggest.recipe_path(str(ms), str(tmp_path / "recipes")) == str(ms)


class TestArguments:

    def test_an_unknown_imager_raises(self, tmp_path):
        with pytest.raises(ValueError, match="imager"):
            suggest.suggest(str(tmp_path), "wsclean")

    def test_a_missing_measurement_set_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            suggest.suggest(str(tmp_path / "missing.ms"), "tclean")


class _no_warning:

    def __enter__(self):
        import warnings
        self._catcher = warnings.catch_warnings()
        self._catcher.__enter__()
        warnings.simplefilter("error")

    def __exit__(self, *exc):
        self._catcher.__exit__(*exc)
