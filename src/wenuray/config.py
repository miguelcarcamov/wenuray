"""
Frozen configuration objects shared by the self-calibration loop and the imagers (#22). Fields use
CASA's own parameter names. Values are validated once, at construction, and derived configurations
use dataclasses.replace, which rejects unknown names.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Mapping, Tuple, Union

# applycal applymode values
APPLY_MODES = ("calflag", "calflagstrict", "trial", "flagonly", "flagonlystrict", "calonly")
# tclean weighting values
WEIGHTINGS = (
    "natural", "uniform", "briggs", "briggsabs", "briggsbwtaper", "superuniform", "radial"
)


def _require_str(config: object, *names: str) -> None:
    for name in names:
        value = getattr(config, name)
        if not isinstance(value, str):
            raise TypeError(
                "{0}.{1} must be a string, not {2!r}".format(type(config).__name__, name, value)
            )


def to_dict(config: object) -> dict:
    """
    Returns a configuration object as a dictionary that TOML can hold: tuples become lists and
    unset (None) values are left out. ``type(config)(**to_dict(config))`` rebuilds an equal object.
    """
    return {
        name: list(value) if isinstance(value, tuple) else value
        for name, value in dataclasses.asdict(config).items() if value is not None
    }


def updated(current: object, value: object) -> object:
    """
    A config object changed by an override: a mapping changes some of its fields, e.g.
    updated(geometry, {"cell": "0.5arcsec"}); any other value replaces it
    """
    if isinstance(value, Mapping):
        return dataclasses.replace(current, **value)
    return value


@dataclass(frozen=True)
class Selection:
    """
    Data selection used by the solve, the apply and the imager alike (#27)

    Parameters
    ----------
    field :
        Field selection
    spw :
        Spectral window selection
    uvrange :
        uv-range selection. A restriction for the solve must be explicit, so it applies everywhere
    """
    field: str = ""
    spw: str = ""
    uvrange: str = ""

    def __post_init__(self):
        _require_str(self, "field", "spw", "uvrange")


@dataclass(frozen=True)
class SolveOptions:
    """
    gaincal options shared by every self-calibration step. The solution interval and the
    calibration mode belong to each step.

    Parameters
    ----------
    refant :
        Reference antenna
    minblperant :
        Minimum number of baselines per antenna to solve for it
    minsnr :
        Solutions below this signal-to-noise ratio are flagged
    gaintype :
        "G" solves each polarization, "T" one solution for all polarizations
    combine :
        Data axes combined when solving, e.g. "spw" or "scan,spw"
    """
    refant: str = ""
    minblperant: int = 4
    minsnr: float = 3.0
    gaintype: str = "T"
    combine: str = ""

    def __post_init__(self):
        _require_str(self, "refant", "gaintype", "combine")
        if isinstance(self.minblperant, bool) or not isinstance(self.minblperant, int) \
                or self.minblperant < 1:
            raise ValueError(
                "minblperant must be a positive integer, not {0!r}".format(self.minblperant)
            )
        if not isinstance(self.minsnr, (int, float)) or self.minsnr < 0:
            raise ValueError("minsnr must be a non-negative number, not {0!r}".format(self.minsnr))
        if self.gaintype not in ("G", "T"):
            raise ValueError("gaintype must be 'G' or 'T', not {0!r}".format(self.gaintype))


@dataclass(frozen=True)
class ApplyOptions:
    """
    applycal options. Each accepted table records the interpolation it is applied with.

    Parameters
    ----------
    applymode :
        "calflag" also flags the data whose solutions failed minsnr
    interp :
        Temporal interpolation of the solutions, e.g. "linear" or "nearest"
    """
    applymode: str = "calflag"
    interp: str = "linear"

    def __post_init__(self):
        _require_str(self, "applymode", "interp")
        if self.applymode not in APPLY_MODES:
            raise ValueError(
                "applymode must be one of {0}, not {1!r}".format(APPLY_MODES, self.applymode)
            )
        if not self.interp:
            raise ValueError("interp must not be empty")


@dataclass(frozen=True)
class ImageGeometry:
    """
    Image grid. The imsize and cell left unset can be derived from the data.

    Parameters
    ----------
    imsize :
        Number of pixels, (nx, ny), or a single number for a square image
    cell :
        Pixel size, e.g. "0.05arcsec"
    phasecenter :
        Image center; empty for the phase center of the field
    stokes :
        Stokes planes to image
    """
    imsize: Union[int, Tuple[int, int], None] = None
    cell: str = ""
    phasecenter: str = ""
    stokes: str = "I"

    def __post_init__(self):
        _require_str(self, "cell", "phasecenter", "stokes")
        if self.imsize is not None:
            if isinstance(self.imsize, int):
                imsize = (self.imsize, self.imsize)
            else:
                imsize = tuple(self.imsize) if isinstance(self.imsize, (tuple, list)) else ()
            if len(imsize) != 2 or not all(
                isinstance(n, int) and not isinstance(n, bool) and n > 0 for n in imsize
            ):
                raise ValueError(
                    "imsize must be a positive integer or a pair of them, not {0!r}".format(
                        self.imsize
                    )
                )
            object.__setattr__(self, "imsize", imsize)


@dataclass(frozen=True)
class Weighting:
    """
    Visibility weighting. Natural weighting, the default, maximises sensitivity, which helps the
    self-calibration models.

    Parameters
    ----------
    weighting :
        Weighting scheme, as tclean names it
    robust :
        Briggs robustness, from -2 (uniform-like) to 2 (natural-like). Only the briggs schemes use it
    uvtaper :
        Gaussian taper, e.g. ("0.1arcsec", ) or ("0.1arcsec", "0.05arcsec", "45deg")
    """
    weighting: str = "natural"
    robust: float = 0.5
    uvtaper: Tuple[str, ...] = ()

    def __post_init__(self):
        _require_str(self, "weighting")
        if self.weighting not in WEIGHTINGS:
            raise ValueError(
                "weighting must be one of {0}, not {1!r}".format(WEIGHTINGS, self.weighting)
            )
        if not isinstance(self.robust, (int, float)) or not -2.0 <= self.robust <= 2.0:
            raise ValueError("robust must be between -2 and 2, not {0!r}".format(self.robust))
        uvtaper = (self.uvtaper, ) if isinstance(self.uvtaper, str) else tuple(self.uvtaper)
        if len(uvtaper) > 3 or not all(isinstance(value, str) for value in uvtaper):
            raise ValueError(
                "uvtaper must hold up to three strings, not {0!r}".format(self.uvtaper)
            )
        object.__setattr__(self, "uvtaper", uvtaper)
