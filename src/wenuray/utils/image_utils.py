import logging
import os
from typing import Optional, Tuple, Union
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
from casatasks import exportfits
from casatools import image

logger = logging.getLogger(__name__)


def nanrms(x, axis=None) -> Union[np.ndarray, float]:
    """
    Function that calculates the root-mean-squared of a numpy array discarding nan values.

    Parameters
    ----------
    x :
        Input numpy array
    axis :
        Axis or axes along which the calculation is performed. The default, axis=None, will calculate the RMS using
    all the elements of the input array. If axis is negative it counts from the last to the first axis.

    Returns
    -------
    The RMS of the input numpy array discarding nan values
    """
    return np.sqrt(np.nanmean(x**2, axis=axis))


def rms(x: np.ndarray, axis=None) -> Union[np.ndarray, float]:
    """
    Function that calculates the root-mean-squared of a numpy array.

    Parameters
    ----------
    x : np.ndarray
        Input numpy array
    axis :
        Axis or axes along which the calculation is performed. The default, axis=None, will calculate the RMS using
        all the elements of the input array. If axis is negative it counts from the last to the first axis.

    Returns
    -------
    The RMS of the input numpy array
    """
    return np.sqrt(np.mean(x**2, axis=axis))


def get_header(fits_name: str = "") -> dict:
    """
    Function that returns the FITS header from a FITS file.

    Parameters
    ----------
    fits_name : str
        Absolute path to the FITS file

    Returns
    -------
    The FITS header as a dictionary
    """
    header = fits.getheader(fits_name)
    return header


def get_hdu(fits_name: str = "") -> fits.PrimaryHDU:
    """
    Function that returns the Primary HDU at position 0 from a FITS file

    Parameters
    ----------
    fits_name : str
        Absolute path to the FITS file

    Returns
    -------
    The Primary HDU of the fits file, with header and data loaded in memory
    """
    with fits.open(fits_name) as hdul:
        # Copy while the file is open: the HDU data is read lazily from the file
        return fits.PrimaryHDU(data=hdul[0].data.copy(), header=hdul[0].header.copy())


def get_hdul(fits_name: str = "") -> fits.HDUList:
    """
    Function that returns the HDU List from a FITS file

    Parameters
    ----------
    fits_name :
        Absolute path to the FITS file

    Returns
    -------
    The HDU List
    """
    hdul = fits.open(fits_name)
    return hdul


def get_data(fits_name: str = "") -> np.ndarray:
    """
    Function that gets the data as a numpy array from a FITS file

    Parameters
    ----------
    fits_name :
        Absolute path to the FITS file

    Returns
    -------
    A numpy array with the FITS file data
    """
    image_data = fits.getdata(fits_name)
    return image_data.squeeze()


def get_header_and_data(fits_name: str = "") -> Tuple[dict, np.ndarray]:
    """
    Function that gets the header and the data from a FITS file

    Parameters
    ----------
    fits_name :
        Absolute path to the FITS file

    Returns
    -------
    A tuple with the header and the data
    """
    with fits.open(fits_name) as hdul:
        header = hdul[0].header
        data = hdul[0].data
    return header, data


def export_ms_to_fits(msname: str = "") -> str:
    """
    Function that export a CASA image file to a FITS image

    Parameters
    ----------
    msname :
        Absolute path to the CASA image file

    Returns
    -------
    The absolute path to the FITS file
    """
    fitsfile_name = msname + ".fits"
    exportfits(msname, fitsfile_name)
    return fitsfile_name


def reproject(fits_file_to_resamp: str = "",
              fits_file_model: str = "",
              order: str = "bilinear") -> Union[str, None]:
    """
    Function that reprojects an image onto the celestial grid (size, pixel size and coordinates)
    of a model image. For 0/1 masks use order="nearest-neighbor", so the result stays 0/1.

    Parameters
    ----------
    fits_file_to_resamp :
        Absolute path to the FITS file image that you want to resample
    fits_file_model :
        The model FITS file image
    order :
        The order of the interpolation when reprojecting. This can be any of the following strings:
            ‘nearest-neighbor’
            ‘bilinear’
            ‘biquadratic’
            ‘bicubic’
    Returns
    -------
    str:
        A string with the absolute path of the reproject image file
    """
    if os.path.exists(fits_file_to_resamp) and os.path.exists(fits_file_model):
        header_mask = get_header(fits_file_to_resamp)
        data_mask = get_data(fits_file_to_resamp)
        header_model = get_header(fits_file_model)
        model_WCS = WCS(header=header_model).celestial
        mask_WCS = WCS(header=header_mask).celestial

        logger.info("Resampling %s onto the grid of %s", fits_file_to_resamp, fits_file_model)
        # Optional dependency (wenuray[reproject]), imported here so the core stays light
        try:
            from reproject import reproject_interp
        except ImportError as error:
            raise ImportError(
                "reproject() needs the reproject package: pip install 'wenuray[reproject]'"
            ) from error
        reprojected_array = reproject_interp(
            (data_mask, mask_WCS),
            model_WCS,
            return_footprint=False,
            order=order,
            # NumPy arrays are (rows, columns) = (NAXIS2, NAXIS1)
            shape_out=(header_model['NAXIS2'], header_model['NAXIS1'])
        )
        path_object = Path(fits_file_to_resamp)
        resampled_mask_name = "{0}_{2}{1}".format(
            Path.joinpath(path_object.parent, path_object.stem), path_object.suffix, "resampled"
        )
        # The output is 2D, so it gets the 2D celestial header of the model
        fits.writeto(resampled_mask_name, reprojected_array, model_WCS.to_header(), overwrite=True)

        return resampled_mask_name

    else:
        raise FileNotFoundError(
            "Either the FITS file to reproject ({0}) or the model ({1}) does not exist".format(
                fits_file_to_resamp, fits_file_model
            )
        )


def read_image(path: str) -> Tuple[np.ndarray, Optional[Tuple[float, float, float]]]:
    """
    Function that reads the pixels and the restoring beam of a CASA image (a directory) or a FITS
    file

    Parameters
    ----------
    path :
        Path to the image

    Returns
    -------
    The pixel array and the beam as (major, minor) in arcsec and position angle in degrees, or
    None when the image has no single restoring beam (e.g. a model image)
    """
    if os.path.isdir(path):
        image_tool = image()
        image_tool.open(path)
        try:
            data = image_tool.getchunk()
            beam = image_tool.restoringbeam()
        finally:
            image_tool.close()
            image_tool.done()
        if "major" not in beam:  # no beam, or one per plane
            return data, None
        return data, (
            u.Quantity(beam["major"]["value"], beam["major"]["unit"]).to_value(u.arcsec),
            u.Quantity(beam["minor"]["value"], beam["minor"]["unit"]).to_value(u.arcsec),
            u.Quantity(beam["positionangle"]["value"],
                       beam["positionangle"]["unit"]).to_value(u.deg),
        )
    with fits.open(path, memmap=False) as hdul:
        header, data = hdul[0].header, np.array(hdul[0].data)
    if "BMAJ" not in header or "BMIN" not in header:
        return data, None
    return data, (header["BMAJ"] * 3600.0, header["BMIN"] * 3600.0, header.get("BPA", 0.0))
