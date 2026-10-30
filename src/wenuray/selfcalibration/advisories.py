"""
Advisories (#141): checks that warn about a run, its data or a round, and never decide. Acceptance
policies decide whether a round is kept; advisories say what deserves a look. Each one is an object
with hooks the loop calls at fixed points, returning Findings, which the loop logs, sends to the
observers (SelfcalObserver.on_finding) and records with the round. A new check is a new class here;
the loop doesn't change.

    at_start      once, after the initial image, with a sample of the data
    before_solve  before each step's gaincal, with the predicted solution S/N
    after_round   after each round is judged, with its record
"""
from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple

from ..config import SolveOptions
from . import preflight
from .steps import CalibrationStep, Finding

if TYPE_CHECKING:
    from ..imaging.contract import Imager
    from .result import RoundRecord

# A residual inside the source mask below this fraction of the thermal noise means the model and
# the gains are fitting noise there (#119)
OVERFITTING_FRACTION = 0.9


@dataclass(frozen=True)
class StartContext:
    """
    What the run-wide checks see: the measurement set and a sample of its data, if one could be
    read
    """
    ms: str
    sample: Optional[preflight.VisibilitySample]


@dataclass(frozen=True)
class SolveContext:
    """
    What the checks before a solve see

    Parameters
    ----------
    step, description :
        The step about to solve, and how it is described in messages
    options :
        The run's solve options (the step's overrides are applied by the checks that need them)
    sample :
        A sample of the data, against the accepted model in MODEL_DATA
    prediction :
        The step's predicted solution S/N (#28)
    model_imager :
        The imager that made the accepted model the step solves against
    """
    step: CalibrationStep
    description: str
    options: SolveOptions
    sample: preflight.VisibilitySample
    prediction: preflight.SolutionSNR
    model_imager: "Imager"


@dataclass(frozen=True)
class RoundContext:
    """
    What the checks after a round see: its record
    """
    record: "RoundRecord"


class Advisory(ABC):
    """
    A check that warns and never decides; override the hooks it needs
    """

    def at_start(self, context: StartContext) -> List[Finding]:
        return []

    def before_solve(self, context: SolveContext) -> List[Finding]:
        return []

    def after_round(self, context: RoundContext) -> List[Finding]:
        return []


class WeightsMisstateNoise(Advisory):
    """
    The data weights misstate the noise the data show (#129)
    """

    def at_start(self, context: StartContext) -> List[Finding]:
        if context.sample is None:
            return []
        anomalies = preflight.weight_anomalies(context.sample)
        return [Finding.warning("{0}: {1}".format(context.ms, anomalies))] if anomalies else []


class SolutionSNRBelowMinimum(Advisory):
    """
    A step's predicted solution S/N is below its minsnr, so gaincal may flag most solutions (#28)
    """

    def before_solve(self, context: SolveContext) -> List[Finding]:
        prediction = context.prediction
        if prediction.enough:
            return []
        return [
            Finding.warning(
                "{0}: predicted solution S/N {1:.1f} is below minsnr {2:g}, so gaincal may flag "
                "most solutions; {3}".format(
                    context.description, prediction.snr, prediction.minsnr, prediction.advice
                )
            )
        ]


class SpectrumAbsorbed(Advisory):
    """
    Amplitude gains solved per spectral window against a model without a spectrum would absorb
    the source's spectral index (#118)
    """

    def before_solve(self, context: SolveContext) -> List[Finding]:
        absorbed = preflight.absorbed_spectrum(
            context.sample, context.step, context.options, context.prediction.snr,
            context.model_imager.models_spectrum
        )
        return [Finding.warning("{0}: {1}".format(context.description, absorbed))
                ] if absorbed else []


class ResidualBelowThermal(Advisory):
    """
    A round's residual inside the source mask is below the thermal noise: the model and the gains
    fit noise there (#119)
    """

    def after_round(self, context: RoundContext) -> List[Finding]:
        record = context.record
        if record.rms_in_mask is None or record.thermal_noise is None:
            return []
        if record.rms_in_mask >= OVERFITTING_FRACTION * record.thermal_noise:
            return []
        return [
            Finding.warning(
                "Round {0} {1}: the residual inside the source mask, {2:.3g} mJy/beam, is below "
                "the thermal noise, {3:.3g} mJy/beam: the model and the gains may be fitting noise "
                "there".format(
                    record.index, record.step, record.rms_in_mask * 1e3, record.thermal_noise * 1e3
                )
            )
        ]


# The advisories every run makes
ADVISORIES: Tuple[Advisory, ...] = (
    WeightsMisstateNoise(), SolutionSNRBelowMinimum(), SpectrumAbsorbed(), ResidualBelowThermal()
)
