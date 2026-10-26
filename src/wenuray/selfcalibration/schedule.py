"""
Schedule validation (#22): a schedule that breaks self-calibration theory raises ScheduleError
before anything touches the data. The calibration stack is simulated through the schedule and each
transition is checked by check_transition, dispatched on the types of the previous and the next
step. Each rule is one small overload; projects can add their own with check_transition.register.
"""
from __future__ import annotations

import warnings
from typing import Sequence

from multimethod import multimethod

from .caltable import CalStack, CalTable
from .steps import CalibrationStep, Finding, Phasecal, Severity, _AmplitudeStep

__all__ = [
    "Finding", "ScheduleError", "ScheduleWarning", "Severity", "check_transition",
    "validate_schedule"
]


class ScheduleError(ValueError):
    """
    The schedule breaks self-calibration theory; raised before any I/O
    """


class ScheduleWarning(UserWarning):
    """
    The schedule runs, but probably not as intended
    """


def _amplitude_needs_phase(step: _AmplitudeStep, stack: CalStack) -> list:
    if stack.solves_phase:
        return []
    return [
        Finding.error(
            "{0}({1!r}) solves amplitudes with no phase solution before it: amplitude solutions "
            "need phase-calibrated data. Start with a Phasecal step or pass an initial table "
            "with phase solutions.".format(type(step).__name__, step.solint)
        )
    ]


@multimethod
def check_transition(previous: object, step: CalibrationStep, stack: CalStack) -> list:
    """
    Findings about running step after previous (None for the first step), with stack holding the
    tables accepted so far. Transitions without a rule are allowed, e.g. a Phasecal after an
    AmpPhasecal refines the phases.
    """
    return []


@check_transition.register
def _(previous: object, step: _AmplitudeStep, stack: CalStack) -> list:
    return _amplitude_needs_phase(step, stack)


@check_transition.register
def _(previous: Phasecal, step: _AmplitudeStep, stack: CalStack) -> list:
    findings = _amplitude_needs_phase(step, stack)
    if step.solint_seconds < previous.solint_seconds:
        findings.append(
            Finding.warning(
                "{0}({1!r}) after Phasecal({2!r}): amplitude solutions need more signal-to-noise "
                "than phases, so their solint is usually at least as long".format(
                    type(step).__name__, step.solint, previous.solint
                )
            )
        )
    return findings


def validate_schedule(steps: Sequence[CalibrationStep], stack: CalStack = CalStack()) -> list:
    """
    Simulates the calibration stack through the schedule and checks every step and transition.
    Errors raise ScheduleError, listing all of them; warnings are emitted as ScheduleWarning.

    Parameters
    ----------
    steps :
        The calibration steps, in the order they run
    stack :
        Tables accepted before the schedule starts, e.g. from a previous run

    Returns
    -------
    The warnings found
    """
    if not steps:
        raise ScheduleError("The schedule has no calibration steps")
    for step in steps:
        if not isinstance(step, CalibrationStep):
            raise ScheduleError(
                "Schedule entries must be calibration steps such as Phasecal('60s'), not "
                "{0!r}".format(step)
            )
    findings = []
    previous = None
    for index, step in enumerate(steps):
        findings += [
            Finding(finding.severity, "step {0}: {1}".format(index, finding.message))
            for finding in step.check() + check_transition(previous, step, stack)
        ]
        # A rejected round stops the run, so each step runs after all earlier ones were accepted
        stack = stack.push(CalTable("<step {0}>".format(index), step.calmode))
        previous = step
    errors = [finding.message for finding in findings if finding.severity is Severity.ERROR]
    if errors:
        raise ScheduleError("Invalid self-calibration schedule:\n- " + "\n- ".join(errors))
    found_warnings = [finding for finding in findings if finding.severity is Severity.WARNING]
    for finding in found_warnings:
        warnings.warn(finding.message, ScheduleWarning, stacklevel=2)
    return found_warnings
