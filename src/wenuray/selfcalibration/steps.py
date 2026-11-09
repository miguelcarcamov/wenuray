"""
Calibration steps of a self-calibration schedule (#22). Each step type is a frozen dataclass that
holds data and small hooks; what differs between calibration types lives in its class, never in
calmode branches. The self-calibration loop that runs them comes with #22c.

    Phasecal("inf"), Phasecal("60s"), AmpPhasecal("inf", solve={"minsnr": 5.0})

A solint of "auto" is resolved when its round starts, from the data (#66), and Repeat runs a step
again at shorter solints until its solutions are at the noise:

    Phasecal("inf"), Repeat(Phasecal("auto")), AmpPhasecal("auto")
"""
from __future__ import annotations

import dataclasses
import math
import re
from abc import ABC
from dataclasses import KW_ONLY, dataclass, field
from types import MappingProxyType
from enum import Enum
from typing import ClassVar, Mapping, Optional, Union

from multimethod import multimethod

from .. import recipe
from ..config import Selection, SolveOptions
from .acceptance import AcceptancePolicy, AllOf, EnoughAntennas, FluxNotLost, PSNRImproves
from .flagging import Flagging, NoFlagging

# Constructor arguments of the classes in snow, before Wenuray 1.0, which now belong to Selfcal or
# its options
LEGACY_ARGUMENTS = frozenset(
    {
        "visfile", "imager", "previous_selfcal", "input_caltable", "refant", "spwmap",
        "minblperant", "want_plot", "interp", "gaintype", "uvrange", "field", "spw",
        "varchange_imager", "varchange_selfcal", "output_caltables", "minsnr", "applymode",
        "flag_mode", "combine", "flag_dataset", "restore_psnr", "subtract_source", "incremental"
    }
)

# The solint resolved from the data when its round starts (#66)
AUTO = "auto"

# Units of a CASA solution interval, in seconds
SOLINT_UNITS = {"s": 1.0, "sec": 1.0, "min": 60.0, "m": 60.0, "h": 3600.0, "hr": 3600.0}
SOLINT_PATTERN = re.compile(r"^\s*(\d+(?:\.\d*)?|\.\d+)\s*([a-z]*)\s*$")


class Severity(Enum):
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True)
class Finding:
    """
    A problem found in a schedule. Errors stop the run before any I/O; warnings are reported.
    """
    severity: Severity
    message: str

    @classmethod
    def error(cls, message: str) -> Finding:
        return cls(Severity.ERROR, message)

    @classmethod
    def warning(cls, message: str) -> Finding:
        return cls(Severity.WARNING, message)


@multimethod
def solint_seconds(solint: str) -> float:
    """
    Converts a CASA solution interval to seconds: "inf" is infinite and "int" (one integration)
    is 0, the shortest possible. Numbers without a unit are seconds. "auto" is not known until its
    round starts, so it is NaN, which compares false with every length
    """
    if solint == AUTO:
        return math.nan
    if solint == "inf":
        return math.inf
    if solint == "int":
        return 0.0
    match = SOLINT_PATTERN.match(solint.lower())
    if match is None or match.group(2) not in SOLINT_UNITS and match.group(2) != "":
        raise ValueError(
            "Invalid solint {0!r}: use 'inf', 'int', 'auto', a number of seconds or a number with a "
            "unit ({1})".format(solint, ", ".join(SOLINT_UNITS))
        )
    return float(match.group(1)) * SOLINT_UNITS.get(match.group(2), 1.0)


@solint_seconds.register
def _(solint: Union[int, float]) -> float:
    if isinstance(solint, bool) or not solint > 0:
        raise ValueError(
            "A numeric solint must be a positive number of seconds, not {0!r}".format(solint)
        )
    return float(solint)


class ScheduleEntry(ABC):
    """
    An entry of a self-calibration schedule: a calibration step, one round, or a Repeat of one
    """

    @property
    def steps(self) -> tuple:
        """
        The calibration steps the entry runs, in order, as far as is known before the run
        """
        return (self, )


@dataclass(frozen=True)
class CalibrationStep(ScheduleEntry):
    """
    One self-calibration round: solve the gains with the step's calibration mode, apply them and
    image the result

    Parameters
    ----------
    solint :
        Solution interval: "inf", "int", "60s", "10min", a number of seconds, or "auto": the
        shortest solint whose predicted solution S/N reaches twice minsnr, resolved from the data
        when the round starts (#66)
    solve :
        Overrides of the run's SolveOptions for this step only, e.g. {"minsnr": 5.0}
    imaging :
        Overrides of the imager's options for this step only, e.g. {"threshold": "0.025mJy"}.
        The imager checks the names (#23)
    acceptance :
        Whether this step's round is kept. None uses the run's policy if one is given, otherwise
        the step type's default_acceptance()
    flagging :
        What to flag after applying this step's solutions and before imaging, e.g.
        ResidualFlagging("rflag"); nothing by default
    """
    calmode: ClassVar[str]
    solves_phase: ClassVar[bool]
    solves_amplitude: ClassVar[bool]
    # Default axis of the gain plots
    plot_yaxis: ClassVar[str]
    # Antennas needed per solution interval: closure phases need 3
    min_antennas: ClassVar[int] = 3

    solint: Union[str, float]
    _: KW_ONLY
    solve: Mapping = field(default_factory=dict, hash=False)
    imaging: Mapping = field(default_factory=dict, hash=False)
    acceptance: Optional[AcceptancePolicy] = None
    flagging: Flagging = NoFlagging()

    def __new__(cls, *args, **kwargs):
        legacy = sorted(set(kwargs) & LEGACY_ARGUMENTS)
        if legacy or (args and isinstance(args[0], (list, tuple))):
            raise TypeError(
                "{0} is a calibration step since Wenuray 1.0; {1} belong to the run. One step per "
                "solution interval, run by Selfcal:\n"
                "    Selfcal(ms, imager, [Phasecal(\"inf\"), Phasecal(\"60s\"), AmpPhasecal(\"inf\")],\n"
                "            solve=SolveOptions(refant=...)).run()\n"
                "See docs/migration.md".format(
                    cls.__name__, legacy or "lists of solution intervals"
                )
            )
        return super().__new__(cls)

    def __post_init__(self):
        if not hasattr(type(self), "calmode"):
            raise TypeError(
                "{0} is abstract: use Phasecal, Ampcal or AmpPhasecal".format(type(self).__name__)
            )
        solint_seconds(self.solint)
        unknown = set(self.solve) - {f.name for f in dataclasses.fields(SolveOptions)}
        if unknown:
            raise TypeError(
                "{0}: unknown solve override(s) {1}; valid names are the SolveOptions fields".
                format(type(self).__name__, sorted(unknown))
            )
        dataclasses.replace(SolveOptions(), **self.solve)  # validates the values
        if not isinstance(self.flagging, Flagging):
            raise TypeError(
                "flagging must be a Flagging such as ResidualFlagging(...), not {0!r}".format(
                    self.flagging
                )
            )
        if self.acceptance is not None and not isinstance(self.acceptance, AcceptancePolicy):
            raise TypeError(
                "acceptance must be an AcceptancePolicy, not {0!r}".format(self.acceptance)
            )
        object.__setattr__(self, "solve", MappingProxyType(dict(self.solve)))
        object.__setattr__(self, "imaging", MappingProxyType(dict(self.imaging)))

    @classmethod
    def from_recipe(cls, table: Mapping) -> CalibrationStep:
        """
        Builds the step from its recipe table (wenuray.recipe), whose acceptance and flagging are
        typed tables
        """
        table = dict(table)
        if "acceptance" in table:
            table["acceptance"] = recipe.decode("acceptance", table["acceptance"])
        if "flagging" in table:
            table["flagging"] = recipe.decode("flagging", table["flagging"])
        return cls(**table)

    @property
    def solint_seconds(self) -> float:
        return solint_seconds(self.solint)

    @property
    def is_auto(self) -> bool:
        """
        Whether the solint is resolved when the round starts
        """
        return self.solint == AUTO

    def with_solint(self, solint: str) -> CalibrationStep:
        """
        The step with another solint, e.g. the one "auto" resolved to
        """
        return dataclasses.replace(self, solint=solint)

    @property
    def casa_solint(self) -> str:
        """
        The solint as gaincal takes it, e.g. "60.0s" for 60.0; "auto" until it is resolved
        """
        return self.solint if isinstance(self.solint, str) else "{0}s".format(self.solint)

    @classmethod
    def default_acceptance(cls) -> AcceptancePolicy:
        """
        The policy used when neither the step nor the run sets one. Its limits come from theory,
        not from an instrument, so it holds for any interferometer: the PSNR improves, and every
        solution interval keeps enough antennas
        """
        return AllOf(PSNRImproves(), EnoughAntennas())

    def acceptance_policy(self, run_policy: Optional[AcceptancePolicy] = None) -> AcceptancePolicy:
        """
        The policy that judges this step's round: the step's own, else the run's, else the step
        type's default
        """
        if self.acceptance is not None:
            return self.acceptance
        if run_policy is not None:
            return run_policy
        return self.default_acceptance()

    def solve_options(self, options: SolveOptions) -> SolveOptions:
        """
        The run's solve options with this step's overrides
        """
        return dataclasses.replace(options, **self.solve)

    def gaincal_kwargs(self, options: SolveOptions, selection: Selection) -> dict:
        """
        The gaincal arguments of this step, except the measurement set, the output table and the
        pre-applied tables, which the loop and CalStack provide

        Parameters
        ----------
        options :
            The run's solve options; this step's overrides are applied on top
        selection :
            The data selection shared by the solve, the apply and the imager
        """
        if self.is_auto:
            raise ValueError(
                "{0}('auto') must be resolved from the data before it is solved".format(
                    type(self).__name__
                )
            )
        return {
            **dataclasses.asdict(selection),
            **dataclasses.asdict(self.solve_options(options)),
            "solint": self.casa_solint,
            "calmode": self.calmode,
            **self._extra_gaincal_kwargs(),
        }

    def _extra_gaincal_kwargs(self) -> dict:
        """
        Hook for gaincal arguments that only some calibration types use
        """
        return {}

    def check(self) -> list:
        """
        Findings about this step on its own; transitions between steps are checked by
        wenuray.selfcalibration.schedule.check_transition
        """
        return []


@dataclass(frozen=True)
class Phasecal(CalibrationStep):
    """
    Phase-only self-calibration (calmode "p")
    """
    calmode: ClassVar[str] = "p"
    solves_phase: ClassVar[bool] = True
    solves_amplitude: ClassVar[bool] = False
    plot_yaxis: ClassVar[str] = "phase"


@dataclass(frozen=True)
class _AmplitudeStep(CalibrationStep):
    """
    Behaviour shared by the steps that solve amplitudes: normalisation, and needing a phase
    solution first (see schedule.check_transition)

    Parameters
    ----------
    solnorm :
        Normalise the amplitude solutions, so self-calibration doesn't change the flux scale
    """
    solves_amplitude: ClassVar[bool] = True
    plot_yaxis: ClassVar[str] = "amp"
    # Closure amplitudes need 4 antennas
    min_antennas: ClassVar[int] = 4

    _: KW_ONLY
    solnorm: bool = True

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.solnorm, bool):
            raise TypeError("solnorm must be True or False, not {0!r}".format(self.solnorm))

    @classmethod
    def default_acceptance(cls) -> AcceptancePolicy:
        """
        Amplitude gains can absorb unmodelled flux, so the model must also keep its flux
        """
        return AllOf(PSNRImproves(), EnoughAntennas(), FluxNotLost())

    def _extra_gaincal_kwargs(self) -> dict:
        return {"solnorm": self.solnorm}


@dataclass(frozen=True)
class Ampcal(_AmplitudeStep):
    """
    Amplitude-only self-calibration (calmode "a")
    """
    calmode: ClassVar[str] = "a"
    solves_phase: ClassVar[bool] = False

    def check(self) -> list:
        if not self.solnorm:
            return [
                Finding.warning(
                    "Ampcal({0!r}, solnorm=False): without normalisation the amplitude solutions "
                    "can drift the flux scale".format(self.solint)
                )
            ]
        return []


@dataclass(frozen=True)
class AmpPhasecal(_AmplitudeStep):
    """
    Amplitude and phase self-calibration (calmode "ap")
    """
    calmode: ClassVar[str] = "ap"
    solves_phase: ClassVar[bool] = True


@dataclass(frozen=True)
class Repeat(ScheduleEntry):
    """
    Runs a step again and again, each round at a shorter solint than the last, as long as the
    solutions still correct more than their noise (#66). The step's solint is "auto", so each
    round takes the shortest solint the data support then, against the model the last round left.

    The repeat ends, and the run goes on with the next entry, when:

    - a round's gains are consistent with their noise: shorter solints would only fit noise;
    - the data support no solint at least min_shortening shorter than the last round's, or the
      solint is one integration;
    - a round is rejected, which is rolled back as any other.

    Parameters
    ----------
    step :
        The step to repeat, with solint "auto"
    min_shortening :
        How much shorter each round's solint must be than the last, as a fraction: 0.25 means at
        most 3/4 of it. A better model raises the solution S/N, and the solint the data support
        shortens as the square of it, so a model a few percent brighter would otherwise buy a round,
        a whole imaging run, for a solint a few percent shorter. 0 only asks for a shorter one
    """
    step: CalibrationStep
    min_shortening: float = 0.25

    def __post_init__(self):
        if not isinstance(self.step, CalibrationStep):
            raise TypeError("Repeat takes a calibration step, not {0!r}".format(self.step))
        if not self.step.is_auto:
            raise TypeError(
                "Repeat shortens the solint round after round, so its step's solint must be "
                "'auto', not {0!r}".format(self.step.solint)
            )
        if not isinstance(self.min_shortening, (int, float)) or isinstance(
            self.min_shortening, bool
        ) or not 0.0 <= self.min_shortening < 1.0:
            raise ValueError(
                "min_shortening is a fraction from 0 to below 1, not {0!r}".format(
                    self.min_shortening
                )
            )

    @classmethod
    def from_recipe(cls, table: Mapping) -> Repeat:
        """
        Builds the repeat from its recipe table, whose step is a typed table
        """
        table = dict(table)
        if "step" not in table:
            raise recipe.RecipeError("A repeat needs a step, e.g. step = {type = \"phase\", ...}")
        table["step"] = recipe.decode("step", table["step"])
        return cls(**table)

    @property
    def steps(self) -> tuple:
        return (self.step, )

    def ends_before(self, last: Optional[CalibrationStep], step: CalibrationStep) -> str:
        """
        Why the repeat ends rather than run step after its last round, last; "" if it goes on
        """
        if last is None:
            return ""
        longest = (1.0 - self.min_shortening) * last.solint_seconds
        if step.solint_seconds < last.solint_seconds and step.solint_seconds <= longest:
            return ""
        return "the data support {0!r}, not a solint at least {1:.0%} shorter than {2!r}".format(
            step.solint, self.min_shortening, last.solint
        )

    def ends_after(self, record, step: CalibrationStep) -> str:
        """
        Why the repeat ends after the round of step that record describes; "" if it goes on
        """
        if not record.accepted:
            return "its last round was rejected"
        if _at_noise(record.gains):
            return "its last round's gains are consistent with their noise"
        if step.solint == "int":
            return "its solint is one integration"
        return ""


def _at_noise(gains) -> bool:
    """
    Whether a round's gains are consistent with their noise: their phase scatter, and their
    amplitude scatter if solved, within noise for as many solutions as they have (#181). Unknown
    when the round has no diagnostics, which counts as not at the noise
    """
    if gains is None:
        return False
    ratios = [gains.phase_over_noise
              ] + ([gains.amplitude_over_noise] if gains.amplitude_over_noise is not None else [])
    return all(gains.above_noise(ratio) is False for ratio in ratios)
