"""
Optional flagging of residual outliers between self-calibration rounds (#57). A step flags only if
given a ResidualFlagging; the default, NoFlagging, does nothing, so the loop never branches on it.

    AmpPhasecal("7min", flagging=ResidualFlagging("rflag", timedevscale=4.0, freqdevscale=4.0))

The step's round flags after applying its solutions and before imaging, on the residuals
CORRECTED_DATA - MODEL_DATA against the model the gains were solved with. A rejected round's flags
are restored with the rest of the round.
"""
from __future__ import annotations

import dataclasses
import inspect
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Optional, Tuple

from casatasks import flagdata

from .. import recipe
from ..config import Selection

logger = logging.getLogger(__name__)

# flagdata modes that find outliers in the residuals
RESIDUAL_MODES = ("rflag", "tfcrop", "clip")
# flagdata parameters set by Wenuray, and where each comes from
CONTROLLED = {
    "vis": "the run",
    "field": "the Selection",
    "spw": "the Selection",
    "uvrange": "the Selection",
    "mode": "ResidualFlagging(mode)",
    "datacolumn": "residual flagging (residual)",
    "action": "residual flagging (apply)",
    "flagbackup": "the run's flag snapshots",
    "savepars": "residual flagging",
    "display": "residual flagging",
}


class Flagging(ABC):
    """
    What a step flags after applying its solutions and before imaging
    """

    @abstractmethod
    def flag(self, ms: str, selection: Selection) -> Optional[Tuple[float, float]]:
        """
        Flags the selected data, returning the flagged fraction before and after, or None if
        nothing was flagged
        """


@dataclass(frozen=True)
class NoFlagging(Flagging):
    """
    The default: a step flags nothing
    """

    def flag(self, ms: str, selection: Selection) -> Optional[Tuple[float, float]]:
        return None


@dataclass(frozen=True, init=False)
class ResidualFlagging(Flagging):
    """
    Flags outliers of the residuals with flagdata. Options use flagdata's own names and defaults,
    since a good threshold depends on the data; Wenuray sets only the data column, the selection and
    the action.

    Parameters
    ----------
    mode :
        "rflag", "tfcrop" or "clip"
    flagdata_options :
        Any other flagdata parameter of that mode, e.g. timedevscale, freqdevscale, ntime
    """
    mode: str
    options: Mapping = field(hash=False)

    def __init__(self, mode: str = "rflag", **flagdata_options):
        if mode not in RESIDUAL_MODES:
            raise ValueError(
                "Residual flagging needs one of {0}, not {1!r}".format(RESIDUAL_MODES, mode)
            )
        _check_options(flagdata_options)
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "options", MappingProxyType(dict(flagdata_options)))

    def to_recipe(self) -> dict:
        return {"mode": self.mode, **recipe.encode(dict(self.options))}

    def __repr__(self) -> str:
        options = ", ".join("{0}={1!r}".format(name, value) for name, value in self.options.items())
        return "ResidualFlagging({0!r}{1})".format(self.mode, ", " + options if options else "")

    def flag(self, ms: str, selection: Selection) -> Optional[Tuple[float, float]]:
        before = flagged_fraction(ms, selection)
        flagdata(
            vis=ms,
            mode=self.mode,
            datacolumn="residual",
            **dataclasses.asdict(selection),
            action="apply",
            flagbackup=False,
            **self.options
        )
        after = flagged_fraction(ms, selection)
        logger.info(
            "Residual flagging (%s): %.2f%% -> %.2f%% flagged", self.mode, 100 * before, 100 * after
        )
        return before, after


def flagged_fraction(ms: str, selection: Selection) -> float:
    """
    Fraction of the selected data that is flagged
    """
    summary = flagdata(vis=ms, mode="summary", **dataclasses.asdict(selection))
    return summary["flagged"] / summary["total"] if summary["total"] else 0.0


def _flagdata_parameters() -> Optional[frozenset]:
    """
    flagdata's parameter names, or None if its signature doesn't list them (e.g. a stand-in)
    """
    parameters = inspect.signature(flagdata).parameters
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return None
    return frozenset(parameters)


def _check_options(options: Mapping) -> None:
    controlled = sorted(set(options) & set(CONTROLLED))
    if controlled:
        raise TypeError(
            "Residual flagging sets {0} itself; they come from {1}".format(
                controlled, ", ".join(sorted({CONTROLLED[name]
                                              for name in controlled}))
            )
        )
    parameters = _flagdata_parameters()
    unknown = sorted(set(options) - parameters) if parameters is not None else []
    if unknown:
        raise TypeError("{0} are not flagdata parameters".format(unknown))
