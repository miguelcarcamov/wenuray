"""
CASA tclean as a self-calibration imager (#23). Options pass through with tclean's own names and are
checked against its signature; the ones that self-calibration controls come from the request, the
image geometry and the weighting, so each is set in exactly one place.
"""
from __future__ import annotations

import dataclasses
import inspect
import os
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Optional

from casatasks import tclean

from ... import recipe
from ...config import ImageGeometry, Weighting, updated
from ...recipe import grid_tables
from ..contract import Imager, ImagingRequest, ImagingResult

# tclean parameters set by Wenuray, and where each comes from
CONTROLLED = {
    "vis": "the request",
    "imagename": "the request",
    "field": "the Selection",
    "spw": "the Selection",
    "uvrange": "the Selection",
    "datacolumn": "the contract (corrected)",
    "savemodel": "the contract (modelcolumn)",
    "imsize": "ImageGeometry",
    "cell": "ImageGeometry",
    "phasecenter": "ImageGeometry",
    "stokes": "ImageGeometry",
    "weighting": "Weighting",
    "robust": "Weighting",
    "uvtaper": "Weighting",
}

# Options of the Tclean class in snow, before Wenuray 1.0, and where they went
LEGACY_OPTIONS = {
    "inputvis": "Selfcal(ms=...)",
    "output": "Selfcal(output=...)",
    "M": "ImageGeometry(imsize=(M, N))",
    "N": "ImageGeometry(imsize=(M, N))",
    "phase_center": "ImageGeometry(phasecenter=...)",
    "data_column": "nothing: imagers always image CORRECTED_DATA",
    "save_model": "nothing: MODEL_DATA is always written",
    "use_mask": "usemask",
    "negative_threshold": "negativethreshold",
    "low_noise_threshold": "lownoisethreshold",
    "noise_threshold": "noisethreshold",
    "sidelobe_threshold": "sidelobethreshold",
    "min_beam_frac": "minbeamfrac",
    "grow_iterations": "growiterations",
    "cycle_niter": "cycleniter",
    "wproj_planes": "wprojplanes",
    "reference_freq": "reffreq",
    "noise_pixels": "nothing: the noise comes from ImageStatistics",
    "verbose": "nothing: logging is configured with the logging module",
}


def _tclean_parameters() -> Optional[frozenset]:
    """
    tclean's parameter names, or None if its signature doesn't list them (e.g. a stand-in)
    """
    parameters = inspect.signature(tclean).parameters
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return None
    return frozenset(parameters)


@dataclass(frozen=True, init=False)
class Tclean(Imager):
    """
    Images with CASA tclean, e.g.

        Tclean(ImageGeometry(imsize=1024, cell="0.05arcsec"), Weighting("briggs", robust=0.5),
               deconvolver="hogbom", niter=1000, usemask="auto-multithresh", nsigma=3.0)

    Parameters
    ----------
    geometry :
        Image grid; imsize and cell are required
    weighting :
        Visibility weighting
    tclean_options :
        Any other tclean parameter, with tclean's name
    """
    geometry: ImageGeometry
    weighting: Weighting
    options: Mapping = field(hash=False)

    def __init__(
        self, geometry: ImageGeometry = None, weighting: Weighting = Weighting(), **tclean_options
    ):
        # Options first, so a call written for snow < 1.0 gets the migration message
        _check_options(tclean_options)
        if not isinstance(geometry, ImageGeometry) or not isinstance(weighting, Weighting):
            raise TypeError("Tclean needs an ImageGeometry and a Weighting")
        if geometry.imsize is None or not geometry.cell:
            raise ValueError(
                "Tclean needs the image size and the cell: ImageGeometry(imsize=..., cell=...)"
            )
        object.__setattr__(self, "geometry", geometry)
        object.__setattr__(self, "weighting", weighting)
        object.__setattr__(self, "options", MappingProxyType(dict(tclean_options)))

    def with_overrides(self, **overrides) -> Tclean:
        """
        A copy with some options changed: geometry and weighting replace the config objects, any
        other name is a tclean parameter
        """
        geometry = updated(self.geometry, overrides.pop("geometry", self.geometry))
        weighting = updated(self.weighting, overrides.pop("weighting", self.weighting))
        return Tclean(geometry, weighting, **{**self.options, **overrides})

    def to_recipe(self) -> dict:
        return grid_tables(self)

    @classmethod
    def from_recipe(cls, table: Mapping) -> Tclean:
        return cls(**recipe.decode_configs(table, geometry=ImageGeometry, weighting=Weighting))

    def _image_suffix(self) -> str:
        # Multi-term deconvolution writes one image per Taylor term; the zeroth is the intensity
        return ".tt0" if self.options.get("deconvolver") == "mtmfs" else ""

    def _reconstruct(self, request: ImagingRequest) -> ImagingResult:
        tclean(
            vis=request.ms,
            imagename=request.name,
            **dataclasses.asdict(request.selection),
            datacolumn="corrected",
            savemodel="modelcolumn",
            imsize=list(self.geometry.imsize),
            cell=self.geometry.cell,
            phasecenter=self.geometry.phasecenter,
            stokes=self.geometry.stokes,
            weighting=self.weighting.weighting,
            robust=self.weighting.robust,
            uvtaper=list(self.weighting.uvtaper),
            **self.options
        )
        suffix = self._image_suffix()
        result = ImagingResult(
            restored=request.name + ".image" + suffix,
            residual=request.name + ".residual" + suffix,
            model=request.name + ".model" + suffix,
            mask=request.name + ".mask",
        )
        for path in (result.restored, result.residual, result.model):
            if not os.path.exists(path):
                raise RuntimeError("tclean did not write {0}".format(path))
        return result


def _check_options(options: Mapping) -> None:
    legacy = sorted(set(options) & set(LEGACY_OPTIONS))
    if legacy:
        raise TypeError(
            "Tclean options use tclean's own names since Wenuray 1.0: " +
            "; ".join("{0} -> {1}".format(name, LEGACY_OPTIONS[name])
                      for name in legacy) + ". See docs/migration.md"
        )
    controlled = sorted(set(options) & set(CONTROLLED))
    if controlled:
        raise TypeError(
            "Tclean sets {0} itself; they come from {1}".format(
                controlled, ", ".join(sorted({CONTROLLED[name]
                                              for name in controlled}))
            )
        )
    parameters = _tclean_parameters()
    unknown = sorted(set(options) - parameters) if parameters is not None else []
    if unknown:
        raise TypeError("{0} are not tclean parameters".format(unknown))
