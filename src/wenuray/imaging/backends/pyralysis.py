"""
Pyralysis as a self-calibration imager (#24). Install it with Wenuray's "pyralysis" extra.

    Pyralysis(ImageGeometry(imsize=512, cell="0.1arcsec"), Weighting("briggs", robust=0.5),
              terms=["chi_squared", "l1norm:1e-3"], optimizer="lbfgsb", max_iter=200)

The reconstruction is described with Pyralysis's own names: terms and the optimizer as its CLI
takes them (--term, --optimizer), and any other CLI option by its name (max_iter, linesearch,
estimator, ...). They are checked by Pyralysis's parser when the imager is created. There is no
default reconstruction: which terms and optimizer suit the data is the user's choice.

Constrain where the model may have flux, e.g. with mask_mode="file" and a mask_path around the
sources, as CLEAN boxes do: with every pixel free, the model can absorb antenna-based gain errors
instead of leaving them for gaincal, so self-calibration corrects little while the PSNR still
rises. Wenuray warns when no mask is set.

Each reconstruction runs in a separate Python process (_pyralysis_worker.py), because casatools
and python-casacore, which Pyralysis reads measurement sets with, crash in one process.
"""
from __future__ import annotations

import functools
import json
import logging
import os
import re
import subprocess
import sys
import warnings
from dataclasses import dataclass, field
from importlib import metadata
from types import MappingProxyType
from typing import ClassVar, Mapping, Sequence, Tuple

from casatools import ms

from ... import recipe
from ...config import ImageGeometry, Selection, Weighting, updated
from ...recipe import grid_tables
from ..contract import Imager, ImagingRequest, ImagingResult

logger = logging.getLogger(__name__)

WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_pyralysis_worker.py")
# Pyralysis CLI options that Wenuray sets, and where each comes from
CONTROLLED = {
    "input_path": "the request",
    "input_format": "the request",
    "output_model": "the request",
    "output_residual": "the request",
    "output_restored": "the request",
    "output_ms": "the contract (MODEL_DATA of the request's MS)",
    "restore": "the imager (images are always restored)",
    "restore_only": "the imager",
    "imsize": "ImageGeometry",
    "cellsize": "ImageGeometry",
    "stokes": "ImageGeometry",
    "weighting": "Weighting",
    "robust_parameter": "Weighting",
    "term": "Pyralysis(terms=...)",
    "optimizer": "Pyralysis(optimizer=...)",
    "hermitian_symmetry": "the imager (off: it folds the visibilities in place)",
}

# Wenuray's weighting schemes and their Pyralysis names
WEIGHTINGS = {"natural": "natural", "uniform": "uniform", "briggs": "robust"}


@dataclass(frozen=True, init=False)
class Pyralysis(Imager):
    """
    Parameters
    ----------
    geometry :
        Image grid; imsize and cell are required. phasecenter isn't supported yet
    weighting :
        natural, uniform or briggs (robust); uvtaper isn't supported yet
    terms :
        Objective terms, e.g. ["chi_squared", "l1norm:1e-3"]
    optimizer :
        Pyralysis optimizer, e.g. "lbfgsb" or "fista"
    pyralysis_options :
        Any other Pyralysis CLI option, by its name: max_iter, linesearch, estimator, ...
    """
    geometry: ImageGeometry
    weighting: Weighting
    terms: Tuple[str, ...]
    optimizer: str
    options: Mapping = field(hash=False)
    file_options: ClassVar[Tuple[str, ...]] = ("mask_path", "initial_model")

    def __init__(
        self,
        geometry: ImageGeometry,
        weighting: Weighting = Weighting(),
        *,
        terms: Sequence[str],
        optimizer: str,
        **pyralysis_options
    ):
        if not isinstance(geometry, ImageGeometry) or not isinstance(weighting, Weighting):
            raise TypeError("Pyralysis needs an ImageGeometry and a Weighting")
        if geometry.imsize is None or not geometry.cell:
            raise ValueError(
                "Pyralysis needs the image size and the cell: ImageGeometry(imsize=..., cell=...)"
            )
        if geometry.phasecenter:
            raise ValueError("The Pyralysis imager doesn't support phasecenter yet")
        if weighting.weighting not in WEIGHTINGS:
            raise ValueError(
                "The Pyralysis imager supports {0} weighting, not {1!r}".format(
                    sorted(WEIGHTINGS), weighting.weighting
                )
            )
        if weighting.uvtaper:
            raise ValueError("The Pyralysis imager doesn't support uvtaper yet")
        controlled = sorted(set(pyralysis_options) & set(CONTROLLED))
        if controlled:
            raise TypeError(
                "The Pyralysis imager sets {0} itself; they come from {1}".format(
                    controlled, ", ".join(sorted({CONTROLLED[name]
                                                  for name in controlled}))
                )
            )
        terms = (terms, ) if isinstance(terms, str) else tuple(terms)
        if not terms or not optimizer:
            raise ValueError("Pyralysis needs the objective terms and the optimizer")
        object.__setattr__(self, "geometry", geometry)
        object.__setattr__(self, "weighting", weighting)
        object.__setattr__(self, "terms", terms)
        object.__setattr__(self, "optimizer", optimizer)
        object.__setattr__(self, "options", MappingProxyType(dict(pyralysis_options)))
        _check(json.dumps(self._cli_options(), sort_keys=True))
        if self.options.get("mask_mode", "none") == "none":
            warnings.warn(
                "Pyralysis without a mask: a model free in every pixel can absorb the gain errors "
                "self-calibration should solve for. Pass mask_mode/mask_path to limit the model "
                "to the sources.",
                UserWarning,
                stacklevel=2
            )

    @property
    def models_spectrum(self) -> bool:
        # A spectral index, a number or the path of an alpha map; 0 is a flat spectrum
        index = self.options.get("spectral_index")
        return index is not None and not (isinstance(index, (int, float)) and index == 0)

    def with_overrides(self, **overrides) -> Pyralysis:
        """
        A copy with some options changed: geometry, weighting, terms and optimizer replace the
        imager's, any other name is a Pyralysis option
        """
        return Pyralysis(
            updated(self.geometry, overrides.pop("geometry", self.geometry)),
            updated(self.weighting, overrides.pop("weighting", self.weighting)),
            terms=overrides.pop("terms", self.terms),
            optimizer=overrides.pop("optimizer", self.optimizer),
            **{
                **self.options,
                **overrides
            }
        )

    def provenance(self) -> dict:
        versions = super().provenance()
        try:
            versions["pyralysis"] = metadata.version("pyralysis")
        except metadata.PackageNotFoundError:
            pass
        return versions

    def to_recipe(self) -> dict:
        return grid_tables(self, terms=list(self.terms), optimizer=self.optimizer)

    @classmethod
    def from_recipe(cls, table: Mapping) -> Pyralysis:
        return cls(**recipe.decode_configs(table, geometry=ImageGeometry, weighting=Weighting))

    def _cli_options(self) -> dict:
        """
        The Pyralysis CLI options, by destination name
        """
        nx, ny = self.geometry.imsize
        options = {
            "imsize": nx if nx == ny else "{0},{1}".format(nx, ny),
            "cellsize": self.geometry.cell,
            "stokes": self.geometry.stokes,
            "weighting": WEIGHTINGS[self.weighting.weighting],
            "term": list(self.terms),
            "optimizer": self.optimizer,
        }
        if self.weighting.weighting == "briggs":
            options["robust_parameter"] = self.weighting.robust
        return {**dict(self.options), **options}

    def _reconstruct(self, request: ImagingRequest) -> ImagingResult:
        outputs = {
            "model": request.name + ".model.fits",
            "residual": request.name + ".residual.fits",
            "restored": request.name + ".image.fits",
        }
        options = self._cli_options()
        # The worker runs in the output directory: files are found from the caller's
        for name in self.file_options:
            if isinstance(options.get(name), str):
                options[name] = os.path.abspath(options[name])
        _run(
            {
                "ms": os.path.abspath(request.ms),
                "taql": _taql(request.ms, request.selection),
                "options": options,
                "outputs": {
                    name: os.path.abspath(path)
                    for name, path in outputs.items()
                },
            },
            # Pyralysis and Dask may write scratch files in the working directory
            cwd=os.path.dirname(os.path.abspath(request.name)),
        )
        return ImagingResult(outputs["restored"], outputs["residual"], outputs["model"])


@functools.lru_cache(maxsize=256)
def _check(options: str) -> None:
    """
    Pyralysis's parser checks the options, in the worker process; cached per option set
    """
    _run({"check": True, "options": json.loads(options)})


def _run(request: dict, cwd: str = None) -> None:
    # -P: the worker's directory holds this module, which would shadow the pyralysis package
    command = [sys.executable, "-P", WORKER]
    logger.debug("Running %s", " ".join(command))
    process = subprocess.run(
        command, input=json.dumps(request), capture_output=True, text=True, cwd=cwd
    )
    lines = process.stdout.strip().splitlines()
    for line in lines[:-1]:
        logger.debug("pyralysis: %s", line)
    try:
        reply = json.loads(lines[-1]) if lines else {}
    except json.JSONDecodeError:
        reply = {}
    if process.returncode == 0 and reply.get("error") == "":
        return
    if reply.get("error", "").startswith("ModuleNotFoundError"):
        raise ImportError(
            "The Pyralysis imager needs Pyralysis: pip install 'wenuray[pyralysis]' ({0})".format(
                reply["error"]
            )
        )
    message = reply.get("error") or process.stderr.strip()[-2000:]
    if reply.get("error", "").startswith("PyralysisVersionError"):
        raise ImportError(message.split(": ", 1)[1])
    if reply.get("error", "").startswith("ThreadingLayerError"):
        raise RuntimeError(message.split(": ", 1)[1])
    if request.get("check"):
        raise TypeError("Invalid Pyralysis options: {0}".format(message))
    raise RuntimeError("Pyralysis failed (exit code {0}): {1}".format(process.returncode, message))


def _taql(ms_name: str, selection: Selection) -> str:
    """
    The TaQL selection of the request's field, spectral windows and uv range
    """
    conditions = []
    if selection.field or selection.spw:
        ms_tool = ms()
        try:
            indices = ms_tool.msseltoindex(vis=ms_name, field=selection.field, spw=selection.spw)
        finally:
            ms_tool.done()
        conditions += [
            "{0} IN [{1}]".format(column, ",".join(str(int(i)) for i in ids))
            for column, ids in (("FIELD_ID", indices["field"]), ("DATA_DESC_ID", indices["dd"]))
            if len(ids)
        ]
    if selection.uvrange:
        conditions.append(uvrange_taql(selection.uvrange))
    return " AND ".join(conditions)


# Units of a uv range in metres
UVRANGE_UNITS = {"": 1.0, "m": 1.0, "km": 1e3}
UVRANGE_PATTERN = re.compile(
    r"^\s*(?P<below><)?(?P<above>>)?\s*(?P<low>[0-9.eE+-]+)\s*(?:~\s*(?P<high>[0-9.eE+-]+))?"
    r"\s*(?P<unit>[a-zA-Z]*)\s*$"
)
# The projected baseline length, sqrt(u² + v²). Pyralysis reads with python-casacore, whose TaQL
# indexes arrays from 0
BASELINE_LENGTH = "SQRT(SQUARE(UVW[0]) + SQUARE(UVW[1]))"


def uvrange_taql(uvrange: str) -> str:
    """
    A CASA uv range in metres ("<3090m", ">0.5km", "100~2000m", or several separated by commas)
    as a TaQL condition on UVW

    Raises
    ------
    ValueError
        If the range isn't in metres: a range in wavelengths depends on each channel's frequency,
        which a selection of rows can't apply
    """
    ranges = []
    for part in filter(str.strip, uvrange.split(",")):
        match = UVRANGE_PATTERN.match(part)
        unit = match.group("unit") if match else ""
        if match is None or unit not in UVRANGE_UNITS:
            raise ValueError(
                "The Pyralysis imager takes a uvrange in metres or km (e.g. '<3090m', '100~2000m'), "
                "not {0!r}: a range in wavelengths depends on each channel's frequency, which a "
                "selection of rows can't apply".format(part.strip())
            )
        scale = UVRANGE_UNITS[unit]
        low = float(match.group("low")) * scale
        if match.group("high") is not None:
            if match.group("below") or match.group("above"):
                raise ValueError("Invalid uvrange {0!r}".format(part.strip()))
            high = float(match.group("high")) * scale
            ranges.append("({0} >= {1!r} AND {0} <= {2!r})".format(BASELINE_LENGTH, low, high))
        elif match.group("below"):
            ranges.append("{0} < {1!r}".format(BASELINE_LENGTH, low))
        elif match.group("above"):
            ranges.append("{0} > {1!r}".format(BASELINE_LENGTH, low))
        else:
            raise ValueError(
                "Invalid uvrange {0!r}: use '<x', '>x' or 'a~b', in m or km".format(part.strip())
            )
    if not ranges:
        raise ValueError("Empty uvrange {0!r}".format(uvrange))
    return ranges[0] if len(ranges) == 1 else "(" + " OR ".join(ranges) + ")"
