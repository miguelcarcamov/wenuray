"""
Acceptance policies (#28): whether a self-calibration round is kept or rolled back. A higher PSNR
alone is not enough: amplitude gains can absorb noise and unmodelled flux, which lowers the
residual rms while the image gets worse.

The defaults hold for any interferometer, because their limits come from theory or the data, not
from an instrument: each step type provides its default policy (CalibrationStep.default_acceptance),
with EnoughAntennas counting antennas, which adapts to the array size. Checks whose limits depend on
the instrument or the science (BeamStable, FlaggedFractionBelow) have no default value, and Check
turns any function of the round metrics into a policy.

Policies decide and never act: the loop (#22c) restores the data when a round is rejected.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Callable, Mapping, Optional, Tuple

from .. import recipe
from ..imaging.statistics import ImageStatistics

if TYPE_CHECKING:
    from .steps import CalibrationStep


def _require_in_range(
    policy: object, name: str, low: float, high: float, include_low: bool = True
) -> None:
    value = getattr(policy, name)
    above_low = value >= low if include_low else value > low
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not above_low \
            or not value <= high:
        raise ValueError(
            "{0}.{1} must be in {2}{3}, {4}], not {5!r}".format(
                type(policy).__name__, name, "[" if include_low else "(", low, high, value
            )
        )


@dataclass(frozen=True)
class RoundMetrics:
    """
    What a round is judged on

    Parameters
    ----------
    statistics :
        Statistics of the round's image
    step :
        The step that solved the round's gains; None for the initial image
    flagged_fraction :
        Fraction of the round's gain solutions that were flagged (0 for the initial image)
    antennas_per_solution :
        Number of antennas with unflagged solutions in each solution interval
    extra :
        Further metrics by name, for checks of your own (e.g. closure-phase residuals in VLBI)
    """
    statistics: ImageStatistics
    step: Optional[CalibrationStep] = None
    flagged_fraction: float = 0.0
    antennas_per_solution: Tuple[int, ...] = ()
    extra: Mapping = field(default_factory=dict, hash=False)

    def __post_init__(self):
        if not 0.0 <= self.flagged_fraction <= 1.0:
            raise ValueError(
                "flagged_fraction must be between 0 and 1, not {0!r}".format(self.flagged_fraction)
            )
        object.__setattr__(
            self, "antennas_per_solution", tuple(int(n) for n in self.antennas_per_solution)
        )
        object.__setattr__(self, "extra", MappingProxyType(dict(self.extra)))


@dataclass(frozen=True)
class Decision:
    """
    Whether to keep a round, and why
    """
    accepted: bool
    reasons: Tuple[str, ...] = ()

    @classmethod
    def accept(cls, reason: str) -> Decision:
        return cls(True, (reason, ))

    @classmethod
    def reject(cls, reason: str) -> Decision:
        return cls(False, (reason, ))


class AcceptancePolicy(ABC):
    """
    Decides whether the current round is kept, comparing it with the last accepted one
    """

    @abstractmethod
    def decide(self, previous: RoundMetrics, current: RoundMetrics) -> Decision:
        ...


@dataclass(frozen=True)
class PSNRImproves(AcceptancePolicy):
    """
    The peak signal-to-noise ratio improves by more than a margin

    Parameters
    ----------
    min_ratio :
        The current PSNR must be above min_ratio times the previous one
    """
    min_ratio: float = 1.0

    def __post_init__(self):
        _require_in_range(self, "min_ratio", 0.0, math.inf, include_low=False)

    def decide(self, previous: RoundMetrics, current: RoundMetrics) -> Decision:
        before, after = previous.statistics.psnr, current.statistics.psnr
        message = "PSNR {0:.3f} -> {1:.3f} (needs > {2:.3f})".format(
            before, after, before * self.min_ratio
        )
        if after > before * self.min_ratio:
            return Decision.accept(message)
        return Decision.reject(message)


@dataclass(frozen=True)
class FluxNotLost(AcceptancePolicy):
    """
    The model keeps its flux. Phase self-calibration usually raises it (less decorrelation); a
    drop is the sign of amplitude gains absorbing flux

    Parameters
    ----------
    max_loss :
        Largest allowed fractional drop of the model flux
    """
    max_loss: float = 0.1

    def __post_init__(self):
        _require_in_range(self, "max_loss", 0.0, 1.0)

    def decide(self, previous: RoundMetrics, current: RoundMetrics) -> Decision:
        before, after = previous.statistics.model_flux, current.statistics.model_flux
        message = "model flux {0:.4g} -> {1:.4g} Jy (max loss {2:.0%})".format(
            before, after, self.max_loss
        )
        if after >= (1.0 - self.max_loss) * before:
            return Decision.accept(message)
        return Decision.reject(message)


@dataclass(frozen=True)
class EnoughAntennas(AcceptancePolicy):
    """
    Every solution interval keeps enough antennas for the gains to be determined: closure phases
    need at least 3 antennas and closure amplitudes at least 4 (Thompson, Moran & Swenson). The
    limit is the step's min_antennas unless given, so it holds for any array size.

    Parameters
    ----------
    min_antennas :
        Antennas needed per solution interval; None takes the step's (3 phase, 4 amplitude)
    max_short_fraction :
        Largest allowed fraction of solution intervals with fewer antennas
    """
    min_antennas: Optional[int] = None
    max_short_fraction: float = 0.0

    def __post_init__(self):
        if self.min_antennas is not None:
            _require_in_range(self, "min_antennas", 2, math.inf)
        _require_in_range(self, "max_short_fraction", 0.0, 1.0)

    def decide(self, previous: RoundMetrics, current: RoundMetrics) -> Decision:
        counts = current.antennas_per_solution
        needed = self.min_antennas or getattr(current.step, "min_antennas", None)
        if not counts or needed is None:
            return Decision.accept("no solutions to count antennas in")
        short = sum(count < needed for count in counts) / len(counts)
        message = "{0:.1%} of the solution intervals have fewer than {1} antennas (max {2:.0%})".format(
            short, needed, self.max_short_fraction
        )
        if short <= self.max_short_fraction:
            return Decision.accept(message)
        return Decision.reject(message)


@dataclass(frozen=True)
class BeamStable(AcceptancePolicy):
    """
    The restoring beam axes change by at most a fraction, e.g. because flagging removed antennas.
    How much change is acceptable depends on the array and the science, so there is no default.
    Rounds without beam information pass

    Parameters
    ----------
    tolerance :
        Largest allowed fractional change of the major and minor axes
    """
    tolerance: float

    def __post_init__(self):
        _require_in_range(self, "tolerance", 0.0, math.inf)

    def decide(self, previous: RoundMetrics, current: RoundMetrics) -> Decision:
        before, after = previous.statistics.beam, current.statistics.beam
        if before is None or after is None:
            return Decision.accept("no beam to compare")
        change = max(abs(a - b) / b for a, b in zip(after[:2], before[:2]))
        message = "beam axes change {0:.1%} (max {1:.0%})".format(change, self.tolerance)
        if change <= self.tolerance:
            return Decision.accept(message)
        return Decision.reject(message)


@dataclass(frozen=True)
class FlaggedFractionBelow(AcceptancePolicy):
    """
    Few of the round's gain solutions were flagged (e.g. below minsnr). A fraction means different
    things for 6 and for 64 antennas, so there is no default; EnoughAntennas holds for any array

    Parameters
    ----------
    max_fraction :
        Largest allowed fraction of flagged solutions
    """
    max_fraction: float

    def __post_init__(self):
        _require_in_range(self, "max_fraction", 0.0, 1.0)

    def decide(self, previous: RoundMetrics, current: RoundMetrics) -> Decision:
        message = "{0:.1%} of the solutions flagged (max {1:.0%})".format(
            current.flagged_fraction, self.max_fraction
        )
        if current.flagged_fraction <= self.max_fraction:
            return Decision.accept(message)
        return Decision.reject(message)


@dataclass(frozen=True, init=False)
class AllOf(AcceptancePolicy):
    """
    Accepts when every policy accepts. All policies are evaluated, so the decision lists every
    reason
    """
    policies: Tuple[AcceptancePolicy, ...]

    def __init__(self, *policies: AcceptancePolicy):
        if not policies:
            raise ValueError("AllOf needs at least one policy")
        object.__setattr__(self, "policies", policies)

    @classmethod
    def from_recipe(cls, table: Mapping) -> AllOf:
        return cls(*(recipe.decode("acceptance", policy) for policy in table["policies"]))

    def decide(self, previous: RoundMetrics, current: RoundMetrics) -> Decision:
        decisions = [policy.decide(previous, current) for policy in self.policies]
        return Decision(
            all(decision.accepted for decision in decisions),
            tuple(reason for decision in decisions for reason in decision.reasons)
        )


@dataclass(frozen=True, init=False)
class AnyOf(AcceptancePolicy):
    """
    Accepts when at least one policy accepts. All policies are evaluated, so the decision lists
    every reason
    """
    policies: Tuple[AcceptancePolicy, ...]

    def __init__(self, *policies: AcceptancePolicy):
        if not policies:
            raise ValueError("AnyOf needs at least one policy")
        object.__setattr__(self, "policies", policies)

    @classmethod
    def from_recipe(cls, table: Mapping) -> AnyOf:
        return cls(*(recipe.decode("acceptance", policy) for policy in table["policies"]))

    def decide(self, previous: RoundMetrics, current: RoundMetrics) -> Decision:
        decisions = [policy.decide(previous, current) for policy in self.policies]
        return Decision(
            any(decision.accepted for decision in decisions),
            tuple(reason for decision in decisions for reason in decision.reasons)
        )


@dataclass(frozen=True)
class Check(AcceptancePolicy):
    """
    Turns any function of the previous and current round metrics into a policy, e.g.

        Check("closure phases improve",
              lambda previous, current: current.extra["closure_rms"] < previous.extra["closure_rms"])

    Parameters
    ----------
    name :
        What the check tests, reported in the decision
    predicate :
        predicate(previous, current) returns True to accept the round
    """
    name: str
    predicate: Callable[[RoundMetrics, RoundMetrics], bool]

    def decide(self, previous: RoundMetrics, current: RoundMetrics) -> Decision:
        if self.predicate(previous, current):
            return Decision.accept("{0}: passed".format(self.name))
        return Decision.reject("{0}: failed".format(self.name))


def solution_snr(
    flux: float,
    sigma_visibility: float,
    nantennas: int,
    nintegrations: int,
    nchannels: int = 1,
    npolarizations: int = 2
) -> float:
    """
    Pre-flight estimate of the signal-to-noise ratio of one antenna's gain solution: each antenna
    is solved from its nantennas - 1 baselines, averaged over the integrations, channels and
    polarizations combined in the solution interval

        S/N = (S / sigma_vis) * sqrt(nantennas - 1) * sqrt(nintegrations * nchannels * npolarizations)

    Compare it with minsnr to check that a solint can be supported before solving.

    Parameters
    ----------
    flux :
        Flux of the model, in Jy
    sigma_visibility :
        Noise of one visibility (one integration, channel and polarization), in Jy
    nantennas :
        Number of antennas
    nintegrations :
        Integrations in one solution interval
    nchannels :
        Channels combined in one solution
    npolarizations :
        Polarizations combined in one solution (2 for gaintype "T", 1 for "G")
    """
    if sigma_visibility <= 0:
        raise ValueError("sigma_visibility must be positive")
    if nantennas < 2 or min(nintegrations, nchannels, npolarizations) < 1:
        raise ValueError("Need at least 2 antennas and 1 integration, channel and polarization")
    return flux / sigma_visibility * math.sqrt(nantennas - 1) * math.sqrt(
        nintegrations * nchannels * npolarizations
    )
