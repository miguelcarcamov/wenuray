"""Schedule validation (#22): one test per transition rule, all without CASA."""
import pytest

from wenuray.selfcalibration.caltable import CalStack, CalTable
from wenuray.selfcalibration.schedule import (
    Finding, ScheduleError, ScheduleWarning, Severity, check_transition, validate_schedule
)
from wenuray.selfcalibration.steps import AmpPhasecal, Ampcal, Phasecal

PHASE_STACK = CalStack([CalTable("pcal", "p")])


def severities(findings) -> list:
    return [finding.severity for finding in findings]


class TestAmplitudeNeedsPhase:

    @pytest.mark.parametrize("step", [Ampcal("inf"), AmpPhasecal("inf")])
    def test_amplitude_first_without_phase_is_an_error(self, step):
        assert severities(check_transition(None, step, CalStack())) == [Severity.ERROR]

    @pytest.mark.parametrize("step", [Ampcal("inf"), AmpPhasecal("inf")])
    def test_phase_table_in_the_initial_stack_is_enough(self, step):
        assert check_transition(None, step, PHASE_STACK) == []

    def test_amplitude_after_an_amplitude_only_stack_is_an_error(self):
        stack = CalStack([CalTable("acal", "a")])

        assert severities(check_transition(Ampcal("inf"), Ampcal("inf"), stack)) == [Severity.ERROR]

    def test_amplitude_after_amp_phase_is_allowed(self):
        stack = CalStack([CalTable("pcal", "p"), CalTable("apcal", "ap")])

        assert check_transition(AmpPhasecal("inf"), Ampcal("inf"), stack) == []


class TestAmplitudeSolint:

    def test_amplitude_solint_shorter_than_the_phase_solint_warns(self):
        findings = check_transition(Phasecal("10min"), AmpPhasecal("60s"), PHASE_STACK)

        assert severities(findings) == [Severity.WARNING]
        assert "solint" in findings[0].message

    @pytest.mark.parametrize("solint", ["10min", "1h", "inf"])
    def test_amplitude_solint_as_long_or_longer_is_fine(self, solint):
        assert check_transition(Phasecal("10min"), AmpPhasecal(solint), PHASE_STACK) == []

    def test_int_counts_as_the_shortest_interval(self):
        findings = check_transition(Phasecal("60s"), Ampcal("int"), PHASE_STACK)

        assert severities(findings) == [Severity.WARNING]


class TestAllowedTransitions:

    def test_phase_after_amp_phase_refines_the_phases(self):
        stack = CalStack([CalTable("pcal", "p"), CalTable("apcal", "ap")])

        assert check_transition(AmpPhasecal("inf"), Phasecal("int"), stack) == []

    @pytest.mark.parametrize("previous", [None, Phasecal("inf")])
    def test_phase_steps_need_nothing(self, previous):
        assert check_transition(previous, Phasecal("60s"), CalStack()) == []


class TestValidateSchedule:

    def test_a_typical_schedule_is_valid(self):
        steps = [Phasecal("inf"), Phasecal("60s"), Phasecal("int"), AmpPhasecal("inf")]

        assert validate_schedule(steps) == []

    def test_amplitude_first_raises_before_anything_runs(self):
        with pytest.raises(ScheduleError, match=r"step 0: Ampcal\('inf'\)"):
            validate_schedule([Ampcal("inf"), Phasecal("60s")])

    def test_a_resumed_run_with_phase_tables_may_start_with_amplitude(self):
        assert validate_schedule([Ampcal("1h")], PHASE_STACK) == []

    def test_the_stack_is_simulated_through_the_schedule(self):
        # The phase solution of step 0 satisfies the amplitude step two steps later
        assert validate_schedule([Phasecal("inf"), AmpPhasecal("inf"), Ampcal("inf")]) == []

    def test_every_error_is_reported_at_once(self):
        stack = CalStack([CalTable("acal", "a")])

        with pytest.raises(ScheduleError) as error:
            validate_schedule([Ampcal("inf"), AmpPhasecal("inf")], stack)

        assert "step 0" in str(error.value) and "step 1" in str(error.value)

    def test_warnings_are_emitted_and_returned(self):
        with pytest.warns(ScheduleWarning, match="step 1"):
            found = validate_schedule([Phasecal("10min"), Ampcal("60s", solnorm=False)])

        assert len(found) == 2  # the short solint and solnorm=False

    def test_empty_schedule_raises(self):
        with pytest.raises(ScheduleError, match="no calibration steps"):
            validate_schedule([])

    def test_entries_must_be_steps(self):
        with pytest.raises(ScheduleError, match="calibration steps"):
            validate_schedule(["p"])


def test_projects_can_add_their_own_rules():

    class Strict(Phasecal):
        pass

    @check_transition.register
    def _(previous: Strict, step: Phasecal, stack: CalStack) -> list:
        return [Finding.error("no phase steps after Strict")]

    with pytest.raises(ScheduleError, match="no phase steps after Strict"):
        validate_schedule([Strict("inf"), Phasecal("60s")])
    # The rule is specific to Strict: other transitions are unaffected
    assert validate_schedule([Phasecal("inf"), Phasecal("60s")]) == []
