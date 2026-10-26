"""
Self-calibration (#22): one loop, Selfcal, runs a schedule of calibration steps.

    result = Selfcal("target.ms", imager, [Phasecal("inf"), Phasecal("60s"), AmpPhasecal("inf")]).run()
"""
from .acceptance import (
    AcceptancePolicy, AllOf, AnyOf, BeamStable, Check, Decision, EnoughAntennas,
    FlaggedFractionBelow, FluxNotLost, PSNRImproves, RoundMetrics, solution_snr
)
from .caltable import CalStack, CalTable
from .flagging import Flagging, NoFlagging, ResidualFlagging
from .loop import Selfcal
from .observers import GainPlotObserver, LoggingObserver, SelfcalObserver
from .result import RoundRecord, SelfcalResult
from .schedule import ScheduleError, ScheduleWarning, check_transition, validate_schedule
from .steps import AmpPhasecal, Ampcal, CalibrationStep, Phasecal
