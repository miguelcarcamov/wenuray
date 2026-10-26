"""Calibration steps (#22): frozen dataclasses whose behaviour differs by class, not by calmode."""
import dataclasses
import math

import pytest

from wenuray.config import Selection, SolveOptions
from wenuray.selfcalibration.steps import (
    AmpPhasecal, Ampcal, CalibrationStep, Phasecal, Severity, _AmplitudeStep, solint_seconds
)


class TestSolintSeconds:

    @pytest.mark.parametrize(
        "solint, seconds", [
            ("inf", math.inf),
            ("int", 0.0),
            ("60s", 60.0),
            ("30sec", 30.0),
            ("10min", 600.0),
            ("1.5h", 5400.0),
            ("45", 45.0),
            (90, 90.0),
            (2.5, 2.5),
        ]
    )
    def test_casa_solution_intervals(self, solint, seconds):
        assert solint_seconds(solint) == seconds

    @pytest.mark.parametrize("solint", ["", "fast", "10 parsec", "-5s", 0, -1, True])
    def test_invalid_intervals_raise(self, solint):
        with pytest.raises(ValueError, match="solint"):
            solint_seconds(solint)


class TestCalibrationSteps:

    @pytest.mark.parametrize(
        "step, calmode, phase, amplitude", [
            (Phasecal("inf"), "p", True, False),
            (Ampcal("inf"), "a", False, True),
            (AmpPhasecal("inf"), "ap", True, True),
        ]
    )
    def test_what_each_type_solves(self, step, calmode, phase, amplitude):
        assert (step.calmode, step.solves_phase,
                step.solves_amplitude) == (calmode, phase, amplitude)

    @pytest.mark.parametrize("abstract", [CalibrationStep, _AmplitudeStep])
    def test_abstract_steps_cannot_be_created(self, abstract):
        with pytest.raises(TypeError, match="abstract"):
            abstract("inf")

    def test_solint_is_positional_and_the_rest_keyword_only(self):
        with pytest.raises(TypeError):
            AmpPhasecal("inf", {"minsnr": 5.0})

    def test_steps_are_frozen_and_hashable(self):
        step = AmpPhasecal("inf", solve={"minsnr": 5.0})
        with pytest.raises(dataclasses.FrozenInstanceError):
            step.solint = "60s"
        with pytest.raises(TypeError):
            step.solve["minsnr"] = 1.0
        assert hash(step) == hash(AmpPhasecal("inf", solve={"minsnr": 5.0}))

    def test_unknown_solve_override_raises_at_construction(self):
        with pytest.raises(TypeError, match="refnt"):
            Phasecal("inf", solve={"refnt": "A0"})

    def test_invalid_solve_override_value_raises_at_construction(self):
        with pytest.raises(ValueError, match="minsnr"):
            Phasecal("inf", solve={"minsnr": -1})

    def test_invalid_solint_raises_at_construction(self):
        with pytest.raises(ValueError, match="solint"):
            Phasecal("soon")


class TestGaincalKwargs:

    def test_selection_options_and_step_are_combined(self):
        kwargs = Phasecal("60s").gaincal_kwargs(
            SolveOptions(refant="A0", combine="spw"), Selection(field="target", spw="1~3")
        )

        assert kwargs == {
            "field": "target",
            "spw": "1~3",
            "uvrange": "",
            "refant": "A0",
            "minblperant": 4,
            "minsnr": 3.0,
            "gaintype": "T",
            "combine": "spw",
            "solint": "60s",
            "calmode": "p",
        }

    def test_step_overrides_win_over_the_run_options(self):
        kwargs = Phasecal("inf", solve={
            "minsnr": 5.0
        }).gaincal_kwargs(SolveOptions(minsnr=3.0), Selection())

        assert kwargs["minsnr"] == 5.0

    def test_numeric_solint_is_passed_in_seconds(self):
        assert Phasecal(90).gaincal_kwargs(SolveOptions(), Selection())["solint"] == "90s"

    @pytest.mark.parametrize("step", [Ampcal("inf"), AmpPhasecal("inf")])
    def test_amplitude_steps_pass_solnorm(self, step):
        assert step.gaincal_kwargs(SolveOptions(), Selection())["solnorm"] is True

    def test_phase_steps_do_not_pass_solnorm(self):
        assert "solnorm" not in Phasecal("inf").gaincal_kwargs(SolveOptions(), Selection())


class TestStepChecks:

    def test_ampcal_without_normalisation_warns(self):
        (finding, ) = Ampcal("inf", solnorm=False).check()

        assert finding.severity is Severity.WARNING
        assert "solnorm=False" in finding.message

    @pytest.mark.parametrize(
        "step", [Ampcal("inf"), AmpPhasecal("inf", solnorm=False),
                 Phasecal("inf")]
    )
    def test_other_steps_have_no_findings(self, step):
        assert step.check() == []


class TestLegacyCalls:
    """Code written for snow < 1.0 gets a TypeError that shows the new call."""

    @pytest.mark.parametrize(
        "build", [
            lambda: Phasecal(visfile="target.ms", imager=None, solint=["inf", "60s"]),
            lambda: AmpPhasecal(previous_selfcal=None, solint=["inf"]),
            lambda: Ampcal(["inf"]),
        ]
    )
    def test_legacy_arguments_show_the_new_call(self, build):
        with pytest.raises(TypeError, match=r"Selfcal\(ms, imager, \[Phasecal"):
            build()
