"""
Image statistics computed one way for every imager (#28), so rounds are compared like for like
whatever produced the images. The imagers start using it with the imager contract (#23).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from astropy.stats import mad_std, sigma_clipped_stats

# Fewer pixels than this inside the source mask give no useful noise estimate there
MIN_PIXELS_IN_MASK = 25


@dataclass(frozen=True)
class ImageStatistics:
    """
    Statistics of one self-calibration image

    Parameters
    ----------
    peak :
        Peak of the restored image, in Jy/beam
    rms :
        Noise of the residual image with the source masked, in Jy/beam
    model_flux :
        Total flux of the model image, in Jy. A drop flags flux absorbed by amplitude gains
    beam :
        Restoring beam (major, minor) in arcsec and position angle in degrees, if known
    rms_in_mask :
        Noise of the residual image inside the source mask, in Jy/beam, if there is a mask. Below
        the thermal noise, the model and the gains are fitting noise there (#119)
    """
    peak: float
    rms: float
    model_flux: float
    beam: Optional[Tuple[float, float, float]] = None
    rms_in_mask: Optional[float] = None

    def __post_init__(self):
        if not self.rms > 0:
            raise ValueError("rms must be positive, not {0!r}".format(self.rms))
        if self.beam is not None:
            beam = tuple(float(value) for value in self.beam)
            if len(beam) != 3 or not beam[0] > 0 or not beam[1] > 0:
                raise ValueError(
                    "beam must be (major, minor, position angle) with positive axes, not "
                    "{0!r}".format(self.beam)
                )
            object.__setattr__(self, "beam", beam)

    @property
    def psnr(self) -> float:
        return self.peak / self.rms

    @classmethod
    def from_arrays(
        cls,
        restored: np.ndarray,
        residual: np.ndarray,
        model: np.ndarray,
        source_mask: Optional[np.ndarray] = None,
        beam: Optional[Tuple[float, float, float]] = None,
        sigma: float = 3.0
    ) -> ImageStatistics:
        """
        Computes the statistics from the image arrays. NaN pixels (e.g. outside the primary beam)
        are ignored. Noise is the MAD standard deviation: image noise is correlated by the beam and
        isn't Gaussian, so a plain standard deviation follows the outliers.

        Parameters
        ----------
        restored :
            Restored image, in Jy/beam
        residual :
            Residual image, in Jy/beam
        model :
            Model image, in Jy/pixel
        source_mask :
            True where there is emission. Those pixels are left out of the noise; whatever source
            residual remains outside the mask is removed by sigma clipping
        beam :
            Restoring beam (major, minor) in arcsec and position angle in degrees
        sigma :
            Clipping threshold, in standard deviations, of the noise estimate
        """
        residual = np.asarray(residual, dtype=float)
        excluded = ~np.isfinite(residual)
        rms_in_mask = None
        if source_mask is not None:
            source = np.broadcast_to(np.asarray(source_mask, dtype=bool), residual.shape)
            inside = residual[source & np.isfinite(residual)]
            # The residual left where the model is fitted; not clipped, since that is what is
            # measured
            if inside.size >= MIN_PIXELS_IN_MASK:
                rms_in_mask = float(mad_std(inside))
            excluded |= source
        if excluded.all():
            raise ValueError("The residual image has no finite noise outside the source mask")
        _, _, rms = sigma_clipped_stats(
            np.where(excluded, 0.0, residual), mask=excluded, sigma=sigma, stdfunc="mad_std"
        )
        if not math.isfinite(rms) or rms <= 0:
            raise ValueError("The residual image has no finite noise outside the source mask")
        return cls(
            peak=float(np.nanmax(restored)),
            rms=float(rms),
            model_flux=float(np.nansum(model)),
            beam=beam,
            rms_in_mask=rms_in_mask
        )
