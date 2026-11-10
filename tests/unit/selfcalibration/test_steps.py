"""Calibration steps (#22): frozen dataclasses whose behaviour differs by class, not by calmode."""
import dataclasses
import math
from types import SimpleNamespace

import pytest

from wenuray.config import Selection, SolveOptions
from wenuray.selfcalibration.diagnostics import GainDiagnostics
from wenuray.selfcalibration.steps import (
    AmpPhasecal, Ampcal, CalibrationStep, Phasecal, Repeat, Severity, _AmplitudeStep, solint_seconds
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


class TestAutoSolint:
    """#66: a solint resolved from the data when its round starts."""

    def test_is_a_valid_solint_whose_length_is_unknown(self):
        step = Phasecal("auto")

        assert step.is_auto
        assert math.isnan(step.solint_seconds)

    def test_must_be_resolved_before_it_is_solved(self):
        with pytest.raises(ValueError, match="resolved from the data"):
            Phasecal("auto").gaincal_kwargs(SolveOptions(), Selection())

    def test_resolves_to_a_step_with_the_same_options(self):
        step = AmpPhasecal("auto", solve={"minsnr": 5.0})

        assert step.with_solint("96s") == AmpPhasecal("96s", solve={"minsnr": 5.0})


class TestRepeat:
    """#66: a step repeated at shorter solints until its solutions are at the noise."""

    def test_its_step_must_be_auto(self):
        with pytest.raises(TypeError, match="must be 'auto'"):
            Repeat(Phasecal("60s"))

    def test_it_takes_a_calibration_step(self):
        with pytest.raises(TypeError, match="takes a calibration step"):
            Repeat("phase")

    def test_runs_its_step(self):
        assert Repeat(Phasecal("auto")).steps == (Phasecal("auto"), )

    @pytest.mark.parametrize("value", [-0.1, 1.0, "a", True])
    def test_min_shortening_is_a_fraction(self, value):
        with pytest.raises(ValueError, match="min_shortening"):
            Repeat(Phasecal("auto"), min_shortening=value)

    @pytest.mark.parametrize("value", [0.25, 0.1, 1.5, "a", True])
    def test_max_shortening_is_a_fraction_above_min_shortening(self, value):
        with pytest.raises(ValueError, match="max_shortening"):
            Repeat(Phasecal("auto"), max_shortening=value)

    @pytest.mark.parametrize(
        "last, window",
        [
            (None, (600.0, math.inf)),  # a lone repeat starts at the longest, a scan here
            ("inf", (300.0, 450.0)),  # "inf" is the scan: half of it, up to 3/4
            ("120s", (60.0, 90.0)),
            ("1200s", (300.0, 450.0)),  # longer than a scan counts as a scan
        ]
    )
    def test_each_round_takes_at_least_half_the_last(self, last, window):
        previous = None if last is None else Phasecal(last)

        assert Repeat(Phasecal("auto")).window(previous, 600.0) == window

    @pytest.mark.parametrize("last", [None, "inf", "120s"])
    def test_max_shortening_1_sets_no_bounds(self, last):
        repeat = Repeat(Phasecal("auto"), max_shortening=1.0)
        previous = None if last is None else Phasecal(last)

        assert repeat.window(previous, 600.0) == (0.0, math.inf)

    @pytest.mark.parametrize(
        "last, solint, goes_on",
        [
            (None, "inf", True),  # the first round
            ("inf", "inf", False),
            ("inf", "600s", True),
            ("120s", "90s", True),  # 25% shorter
            ("120s", "100s", False),  # 17%
            ("60s", "int", True),
        ]
    )
    def test_each_round_must_be_shorter_by_min_shortening(self, last, solint, goes_on):
        repeat = Repeat(Phasecal("auto"))
        previous = None if last is None else Phasecal(last)

        assert (repeat.ends_before(previous, Phasecal(solint)) == "") is goes_on

    @pytest.mark.parametrize(
        "accepted, gains, solint, reason",
        [
            (False, None, "60s", "rejected"),
            (True, GainDiagnostics(1.0, solutions=100), "60s", "consistent with their noise"),
            (True, GainDiagnostics(10.0, 1.0, solutions=100), "60s", ""),  # phases still correct
            (True, GainDiagnostics(10.0, solutions=100), "int", "one integration"),
            (True, GainDiagnostics(1.0), "60s", ""),  # unknown without the count
            (True, None, "60s", ""),
        ]
    )
    def test_ends_after_a_round_for_its_reasons(self, accepted, gains, solint, reason):
        record = SimpleNamespace(accepted=accepted, gains=gains)

        ends = Repeat(Phasecal("auto")).ends_after(record, Phasecal(solint))

        assert (reason in ends) if reason else ends == ""
