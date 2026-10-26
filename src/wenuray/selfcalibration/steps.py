"""
Calibration steps of a self-calibration schedule (#22). Each step type is a frozen dataclass that
holds data and small hooks; what differs between calibration types lives in its class, never in
calmode branches. The self-calibration loop that runs them comes with #22c.

    Phasecal("inf"), Phasecal("60s"), AmpPhasecal("inf", solve={"minsnr": 5.0})
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
    is 0, the shortest possible. Numbers without a unit are seconds.
    """
    if solint == "inf":
        return math.inf
    if solint == "int":
        return 0.0
    match = SOLINT_PATTERN.match(solint.lower())
    if match is None or match.group(2) not in SOLINT_UNITS and match.group(2) != "":
        raise ValueError(
            "Invalid solint {0!r}: use 'inf', 'int', a number of seconds or a number with a unit "
            "({1})".format(solint, ", ".join(SOLINT_UNITS))
        )
    return float(match.group(1)) * SOLINT_UNITS.get(match.group(2), 1.0)


@solint_seconds.register
def _(solint: Union[int, float]) -> float:
    if isinstance(solint, bool) or not solint > 0:
        raise ValueError(
            "A numeric solint must be a positive number of seconds, not {0!r}".format(solint)
        )
    return float(solint)


@dataclass(frozen=True)
class CalibrationStep(ABC):
    """
    One self-calibration round: solve the gains with the step's calibration mode, apply them and
    image the result

    Parameters
    ----------
    solint :
        Solution interval: "inf", "int", "60s", "10min", or a number of seconds
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
        solint = self.solint if isinstance(self.solint, str) else "{0}s".format(self.solint)
        return {
            **dataclasses.asdict(selection),
            **dataclasses.asdict(self.solve_options(options)),
            "solint": solint,
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
