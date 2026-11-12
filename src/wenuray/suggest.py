"""
A recipe suggested from the data (#190): ``wenuray suggest target.ms -o recipe.toml``.

Wenuray derives most of a recipe from the data when it runs ("auto" solints and grids). Suggesting
the recipe writes those values out before anything runs, so the user sees, edits and keeps them,
and each value carries a comment saying where it comes from. By default nothing is imaged: the
source model, and with it the solution S/N, is unknown, so solints stay "auto" and the run resolves
them round by round against its models (#66).

With image=True (`--image`), one quick tclean image of DATA gives a model (estimate_model), cleaned
only to the largest peak the noise alone would give over the image, so that noise doesn't enter it
(#198). The model's visibilities are computed here from its clean components, so the measurement
set is left untouched, and the solution S/N they predict decides the amplitude step and whether
the spectral windows are combined (amplitude_step, spw_combination).

The data are read once, into a DataSummary (read_summary); recipe_text() turns a summary into the
recipe and reads nothing, so every choice can be tested without CASA. Every value comes from
theory or the data: nothing depends on the instrument.
"""
from __future__ import annotations

import dataclasses
import math
import os
import tempfile
import textwrap
from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import tomli_w

from .config import AUTO, ImageGeometry, Selection, SolveOptions
from .selfcalibration import preflight
from .selfcalibration.steps import AmpPhasecal, Phasecal
from .utils.selfcal_utils import (get_spw_ids, open_table, selected_rows, spws_with_unflagged_data)

IMAGERS = ("tclean", "pyralysis")
# The comments start at this column, as in `wenuray init`'s template, and lines stop at this width
COMMENT_COLUMN = 31
WIDTH = 100
# Each round cleans to this many times its residual rms (tclean's nsigma), which stays above the
# gain-error artefacts of the early rounds, where a fixed threshold near the thermal noise wouldn't
NSIGMA = 3.0
# The quick model (--image): a Gaussian beam covers 1.133 FWHM^2, and the "auto" cell puts
# 2 x oversampling pixels across the FWHM (config.ImageGeometry), which gives the image's
# independent beams; the rows its visibilities are computed on, and components per batch, which
# bound the memory of that computation; and a cap on clean's iterations, which the threshold ends
GAUSSIAN_BEAM_AREA = 1.133
MODEL_ROWS = 4000
COMPONENT_BATCH = 2000
MAX_ITERATIONS = 100000
SPEED_OF_LIGHT = 299792458.0


@dataclass(frozen=True)
class Antenna:
    """
    An antenna as the refant choice sees it: its name, ITRF position in m, and the flagged
    fraction of its sampled data (NaN if the data gave no sample)
    """
    name: str
    position: Tuple[float, float, float]
    flagged: float


@dataclass(frozen=True)
class DataSummary:
    """
    What the suggestion reads from a measurement set

    Parameters
    ----------
    ms :
        The measurement set
    field :
        The field self-calibrated, by name
    spws :
        The spectral windows that hold unflagged data
    spws_flagged :
        The spectral windows with data, all of it flagged
    antennas :
        The antennas with unflagged data
    sample :
        A sample of the data without a model, for the noise and the weights; None if the data
        can't give one
    geometry :
        The grid derived from the data (#67); None if it couldn't be
    geometry_problem :
        Why the grid couldn't be derived
    model :
        The quick model, with --image; None without it, or if it couldn't be made
    model_problem :
        Why the quick model couldn't be made
    """
    ms: str
    field: str
    spws: Tuple[int, ...]
    spws_flagged: Tuple[int, ...]
    antennas: Tuple[Antenna, ...]
    sample: Optional[preflight.VisibilitySample]
    geometry: Optional[ImageGeometry]
    geometry_problem: str = ""
    model: Optional[ModelEstimate] = None
    model_problem: str = ""


@dataclass(frozen=True)
class ModelEstimate:
    """
    A quick model of the source (--image)

    Parameters
    ----------
    signal :
        Median amplitude of the model's visibilities on the sampled rows, in Jy: the S that
        predicts the solution S/N (preflight)
    flux :
        The model's total flux, in Jy
    components :
        Its clean components
    nsigma :
        How deep it was cleaned, in units of the residual's robust rms
    beams :
        The image's independent beams, which set nsigma
    """
    signal: float
    flux: float
    components: int
    nsigma: float
    beams: float


@dataclass(frozen=True)
class Choice:
    """
    A suggested value and why it was chosen
    """
    value: Any
    why: str = ""


def read_summary(ms: str, field: str = "", image: bool = False) -> DataSummary:
    """
    Reads what the suggestion needs from the measurement set: its fields, spectral windows and
    antennas, a sample of its data, and the grid; with image, a quick model too (estimate_model).
    The measurement set isn't changed

    Raises
    ------
    ValueError
        If the data hold several fields and field doesn't choose one, or no field matches it
    """
    from .imaging.contract import resolve_geometry

    field = _field(ms, field)
    with_data = get_spw_ids(ms, "")
    spws = tuple(spws_with_unflagged_data(ms, with_data))
    if not spws:
        raise ValueError("{0}: every spectral window's data are flagged".format(ms))
    selection = Selection(field=field, spw=_spw_selection(spws, with_data))
    sample = preflight.read_sample(ms, selection, model=False)
    antennas = _antennas(ms, sample)
    try:
        geometry, problem = resolve_geometry(
            ImageGeometry(imsize=AUTO, cell=AUTO), ms, selection
        ), ""
    except ValueError as error:
        geometry, problem = None, str(error)
    model, model_problem = None, ""
    if image:
        if geometry is None:
            model_problem = "it needs the grid, which couldn't be derived"
        else:
            try:
                model = estimate_model(ms, selection, geometry)
            except RuntimeError as error:
                # CASA's tasks and tools raise RuntimeError; the recipe is still worth writing
                model_problem = "tclean couldn't image the data: {0}".format(error)
    return DataSummary(
        ms, field, spws, tuple(sorted(set(with_data) - set(spws))), antennas, sample, geometry,
        problem, model, model_problem
    )


def _antennas(ms: str, sample: Optional[preflight.VisibilitySample]) -> Tuple[Antenna, ...]:
    """
    The antennas with unflagged data: their names, positions and sampled flagged fractions
    """
    with open_table(os.path.join(ms, "ANTENNA")) as antenna_table:
        names = antenna_table.getcol("NAME")
        positions = antenna_table.getcol("POSITION").T
        flag_row = antenna_table.getcol("FLAG_ROW")
    flagged = sample.antenna_flagged if sample is not None else (math.nan, ) * len(names)
    return tuple(
        Antenna(str(name), tuple(float(x) for x in position), float(fraction))
        for name, position, fraction, row_flagged in zip(names, positions, flagged, flag_row)
        if not row_flagged and not fraction >= 1.0
    )


def independent_beams(geometry: ImageGeometry) -> float:
    """
    About how many independent beams an image of the grid holds: its pixels over the beam's, with
    the beam 2 x oversampling pixels across (its FWHM) for the "auto" cell. A natural-weighting
    beam is wider, so this errs high, and the threshold it gives a little deep
    """
    nx, ny = geometry.imsize
    return nx * ny / (GAUSSIAN_BEAM_AREA * (2 * geometry.oversampling)**2)


def noise_maximum(beams: float) -> float:
    """
    The largest peak, in units of the noise, that Gaussian noise alone is expected to reach over
    beams independent beams: sqrt(2 ln N). Clean stops there, so its model holds emission rather
    than noise peaks (#198); never below the run's NSIGMA
    """
    return max(NSIGMA, math.sqrt(2 * math.log(max(beams, 1.0))))


def model_amplitudes(
    flux: np.ndarray, l: np.ndarray, m: np.ndarray, u: np.ndarray, v: np.ndarray
) -> np.ndarray:
    """
    The amplitudes of a model of point components (flux in Jy at direction cosines l, m) at the
    (u, v), in wavelengths, of each row: abs(sum_k flux_k exp(-2 pi i (u l_k + v m_k))). The
    amplitude doesn't depend on the sign convention of the exponent. Computed in batches of
    components, so the memory stays bounded
    """
    visibilities = np.zeros(len(u), dtype=complex)
    for start in range(0, len(flux), COMPONENT_BATCH):
        batch = slice(start, start + COMPONENT_BATCH)
        phase = -2 * np.pi * (np.outer(u, l[batch]) + np.outer(v, m[batch]))
        visibilities += np.exp(1j * phase) @ flux[batch]
    return np.abs(visibilities)


def estimate_model(ms: str, selection: Selection, geometry: ImageGeometry) -> ModelEstimate:
    """
    A quick model of the source: tclean images DATA on the grid, with natural weighting, cleaned
    to noise_maximum() times the residual's robust rms (tclean's nsigma), into a temporary
    directory it deletes, saving no model to the measurement set. The robust rms holds the
    gain-error artefacts too, which a threshold from the thermal noise would clean into the model
    """
    from casatasks import tclean

    beams = independent_beams(geometry)
    nsigma = noise_maximum(beams)
    with tempfile.TemporaryDirectory(prefix="wenuray-suggest-") as directory:
        name = os.path.join(directory, "model")
        tclean(
            vis=ms,
            imagename=name,
            field=selection.field,
            spw=selection.spw,
            datacolumn="data",
            savemodel="none",
            imsize=list(geometry.imsize),
            cell=geometry.cell,
            specmode="mfs",
            deconvolver="hogbom",
            weighting="natural",
            niter=MAX_ITERATIONS,
            nsigma=nsigma,
        )
        flux, l, m = _components(name + ".model")
    u, v = _sampled_uv(ms, selection)
    amplitudes = model_amplitudes(flux, l, m, u, v) if len(flux) and len(u) else np.zeros(0)
    return ModelEstimate(
        float(np.median(amplitudes)) if amplitudes.size else 0.0, float(flux.sum()), len(flux),
        nsigma, beams
    )


def _components(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    The clean components of a model image: flux, and l and m in radians from the reference
    pixel, signed by the image's own increments (RA grows to the left)
    """
    from casatools import image

    image_tool = image()
    image_tool.open(path)
    try:
        pixels = image_tool.getchunk()
        coordinates = image_tool.coordsys()
        increment = coordinates.increment(type="direction")["numeric"]
        reference = coordinates.referencepixel(type="direction")["numeric"]
        coordinates.done()
    finally:
        image_tool.close()
        image_tool.done()
    plane = pixels.reshape(pixels.shape[0], pixels.shape[1], -1).sum(axis=2)
    x, y = np.nonzero(plane)
    return (
        plane[x,
              y].astype(float), (x - reference[0]) * increment[0], (y - reference[1]) * increment[1]
    )


def _sampled_uv(ms: str, selection: Selection) -> Tuple[np.ndarray, np.ndarray]:
    """
    (u, v) in wavelengths of a sample of the selected cross-correlations that aren't flagged, at
    the centre frequency of each row's spectral window
    """
    rows = selected_rows(ms, selection.field, selection.spw, samples=MODEL_ROWS)
    if not rows:
        return np.zeros(0), np.zeros(0)
    with open_table(os.path.join(ms, "DATA_DESCRIPTION")) as dd_table:
        dd_spw = np.asarray(dd_table.getcol("SPECTRAL_WINDOW_ID"))
    with open_table(os.path.join(ms, "SPECTRAL_WINDOW")) as spw_table:
        centres = np.array(
            [
                float(np.mean(spw_table.getcell("CHAN_FREQ", spw)))
                for spw in range(spw_table.nrows())
            ]
        )
    with open_table(ms) as ms_table:
        sampled = ms_table.selectrows(rows)
        try:
            uvw = sampled.getcol("UVW")
            ddid = sampled.getcol("DATA_DESC_ID")
            keep = (sampled.getcol("ANTENNA1") != sampled.getcol("ANTENNA2")) \
                & ~sampled.getcol("FLAG_ROW")
        finally:
            sampled.close()
    wavelengths = centres[dd_spw[ddid]] / SPEED_OF_LIGHT
    return (uvw[0] * wavelengths)[keep], (uvw[1] * wavelengths)[keep]


def amplitude_step(sample: preflight.VisibilitySample, options: SolveOptions) -> Choice:
    """
    Whether the amplitude step is worth it, from a sample whose signal comes from a model: its
    predicted S/N (CASA's, the antenna's over sqrt(2)) at solint "inf" must reach
    AUTO_SNR_FACTOR x minsnr, the margin solint "auto" asks for. Below it, even its longest
    solint would leave many solutions flagged
    """
    snr = preflight.predict(sample, AmpPhasecal("inf"), options).snr
    target = preflight.AUTO_SNR_FACTOR * options.minsnr
    over = "over a whole observation" if "scan" in options.combine else "over a scan"
    if snr >= target:
        return Choice(
            True, "the amplitude solutions' predicted S/N {0}, {1:.1f} (CASA's, the antenna's "
            "over sqrt(2)), reaches {2:g} (2 x minsnr)".format(over, snr, target)
        )
    return Choice(
        False, "the amplitude solutions' predicted S/N {0}, {1:.1f} (CASA's, the antenna's over "
        "sqrt(2)), is below {2:g} (2 x minsnr)".format(over, snr, target)
    )


def spw_combination(sample: preflight.VisibilitySample, options: SolveOptions) -> Choice:
    """
    combine: "spw" when one spectral window's phase S/N at solint "inf" falls short of
    AUTO_SNR_FACTOR x minsnr but the windows combined reach it (or come closer: there is nothing
    better), "" when one window is enough or there is one window
    """
    if len(sample.channels) < 2:
        return Choice("", "")
    target = preflight.AUTO_SNR_FACTOR * options.minsnr

    def snr(combine: str) -> float:
        return preflight.predict(
            sample, Phasecal("inf"), dataclasses.replace(options, combine=combine)
        ).snr

    separate, combined = snr(""), snr("spw")
    if separate >= target:
        return Choice(
            "", "per spectral window, the phase solutions' predicted S/N over a scan, {0:.1f}, "
            "reaches {1:g} (2 x minsnr), so the {2} windows are solved one by one".format(
                separate, target, len(sample.channels)
            )
        )
    return Choice(
        "spw", "per spectral window, the phase solutions' predicted S/N over a scan is {0:.1f}, "
        "below {1:g} (2 x minsnr); the {2} windows combined give {3:.1f}{4}".format(
            separate, target, len(sample.channels), combined,
            "" if combined >= target else ", still short of it"
        )
    )


def _field(ms: str, field: str) -> str:
    """
    The field to self-calibrate, by name: the one given, or the only one with data
    """
    with open_table(ms) as ms_table:
        used = ms_table.query(query="", columns="DISTINCT FIELD_ID")
        try:
            ids = sorted(int(i) for i in used.getcol("FIELD_ID"))
        finally:
            used.close()
    with open_table(os.path.join(ms, "FIELD")) as field_table:
        names = [str(name) for name in field_table.getcol("NAME")]
    with_data = [names[i] for i in ids]
    if field:
        if field not in with_data and not (field.isdigit() and int(field) in ids):
            raise ValueError(
                "{0} has no data of field {1!r}; its fields with data are {2}".format(
                    ms, field, ", ".join(with_data)
                )
            )
        return names[int(field)] if field.isdigit() and field not in names else field
    if len(with_data) > 1:
        raise ValueError(
            "{0} holds data of {1} fields, {2}: choose the one to self-calibrate with --field".
            format(ms, len(with_data), ", ".join(with_data))
        )
    return with_data[0]


def _spw_selection(spws: Sequence[int], with_data: Sequence[int]) -> str:
    """
    The spw selection of the windows with unflagged data: empty when that is all of them
    """
    return "" if list(spws) == list(with_data) else ",".join(str(spw) for spw in spws)


def reference_antenna(antennas: Sequence[Antenna]) -> Choice:
    """
    The reference antenna: near the centre of the array, where its baselines to the others are
    short and its phases change least, and with little flagged data, so it is in every solution.
    Each antenna is ranked by its distance from the array centre (the median of the positions) and
    by its flagged fraction, and the best sum of the two ranks wins, the nearer one on a tie; the
    approach of the ALMA pipeline's hif_refant, with no instrument in it
    """
    if not antennas:
        return Choice("", "no antenna has unflagged data")
    positions = np.array([antenna.position for antenna in antennas])
    distances = np.linalg.norm(positions - np.median(positions, axis=0), axis=1)
    flagged = np.array([antenna.flagged for antenna in antennas])
    if np.any(np.isnan(flagged)):
        best = int(np.argmin(distances))
        return Choice(
            antennas[best].name, "{0} from the array centre, the nearest of {1} antennas; the "
            "data gave no sample, so their flagging is unknown".format(
                _length(distances[best]), len(antennas)
            )
        )
    score = _ranks(distances) + _ranks(flagged)
    best = min(range(len(antennas)), key=lambda i: (score[i], distances[i]))
    return Choice(
        antennas[best].name, "{0} from the array centre, {1:.1%} of its sampled data flagged: the "
        "best of {2} antennas by distance and flagging".format(
            _length(distances[best]), flagged[best], len(antennas)
        )
    )


def _ranks(values: np.ndarray) -> np.ndarray:
    """
    How many values are smaller than each: equal values share a rank
    """
    return np.array([np.sum(values < value) for value in values])


def _length(metres: float) -> str:
    return "{0:.2g} km".format(metres / 1000) if metres >= 1000 else "{0:.0f} m".format(metres)


def baselines_per_antenna(nantennas: int, amplitudes: bool) -> Choice:
    """
    minblperant: the baselines an antenna needs for its solution to be constrained by closure, one
    fewer than the antennas of the closure quantity (Phasecal.min_antennas for closure phases,
    AmpPhasecal.min_antennas for closure amplitudes). An antenna can't have more baselines than
    there are other antennas
    """
    step = AmpPhasecal if amplitudes else Phasecal
    needed = step.min_antennas - 1
    value = min(needed, nantennas - 1)
    why = "closure {0} need {1} antennas, so each antenna's solution needs baselines to {2} " \
        "others".format("amplitudes" if amplitudes else "phases", step.min_antennas, needed)
    if value < needed:
        why += "; only {0} antennas have data".format(nantennas)
    return Choice(value, why)


def recipe_text(summary: DataSummary, imager: str = "tclean", ms_path: Optional[str] = None) -> str:
    """
    The suggested recipe as TOML, every derived value with a comment saying why. It loads with
    Selfcal.from_toml

    Parameters
    ----------
    summary :
        What read_summary read from the data
    imager :
        The imager's recipe type, one of IMAGERS
    ms_path :
        How the recipe names the measurement set; summary.ms by default
    """
    from . import __version__

    if imager not in IMAGERS:
        raise ValueError("imager must be one of {0}, not {1!r}".format(IMAGERS, imager))
    nantennas = len(summary.antennas)
    if nantennas < Phasecal.min_antennas:
        raise ValueError(
            "{0}: {1} antennas have unflagged data, and self-calibration needs at least {2}, the "
            "antennas of a closure phase".format(summary.ms, nantennas, Phasecal.min_antennas)
        )
    amplitudes = nantennas >= AmpPhasecal.min_antennas
    combine, amplitude = _decisions(summary, amplitudes)
    if amplitude is not None:
        amplitudes = amplitude.value
    lines = _header(summary, __version__)
    lines += [_line("ms", ms_path or summary.ms, "the calibrated target data in DATA"), ""]
    lines += [
        "[selection]" + _pad("[selection]") + "# one selection for the solve, the apply "
        "and the imager"
    ]
    lines += [
        _line("field", summary.field, "the field self-calibrated"),
        _line("spw", *_spws(summary)), ""
    ]
    lines += _solve(summary, amplitudes, combine)
    lines += _imager(summary, imager)
    lines += _steps(amplitudes, nantennas, amplitude)
    return "\n".join(lines) + "\n"


def _decisions(summary: DataSummary, amplitudes: bool) -> Tuple[Optional[Choice], Optional[Choice]]:
    """
    With a quick model whose signal predicts an S/N: whether the windows are combined, and, if the
    antennas allow one, whether there is an amplitude step; None for what isn't decided
    """
    if summary.model is None or summary.sample is None or not summary.model.signal > 0:
        return None, None
    sample = dataclasses.replace(summary.sample, signal=summary.model.signal)
    combine = spw_combination(sample, SolveOptions())
    if not amplitudes:
        return combine, None
    return combine, amplitude_step(sample, SolveOptions(combine=combine.value))


def _header(summary: DataSummary, version: str) -> List[str]:
    lines = _comment(
        "A Wenuray self-calibration recipe suggested from {0} by `wenuray suggest` (Wenuray "
        "{1}). Each value derived from the data says why. Check it with `wenuray validate`, then "
        "`wenuray run` it. Paths are relative to this file.".format(
            os.path.basename(os.path.normpath(summary.ms)), version
        )
    )
    findings = []
    if summary.sample is not None:
        anomalies = preflight.weight_anomalies(summary.sample)
        if anomalies:
            findings.append(anomalies)
    else:
        findings.append(
            "the data gave no noise estimate (no consecutive integrations or parallel hands to "
            "difference), so the thermal noise is unknown"
        )
    if summary.geometry is None:
        findings.append("the grid couldn't be derived: " + summary.geometry_problem)
    model = summary.model
    if summary.model_problem:
        findings.append("no quick model was made: " + summary.model_problem)
    elif model is not None and not model.signal > 0:
        findings.append(
            "the quick image found no emission above the noise's expected maximum ({0:.1f} x "
            "its robust rms over {1:.2g} beams): there may be too little signal to "
            "self-calibrate".format(model.nsigma, model.beams)
        )
    for finding in findings:
        lines += _comment("Warning: " + finding)
    if model is not None and model.signal > 0:
        lines += _comment(
            "--image: a quick tclean model of DATA, cleaned to {0:.1f} x the residual's robust "
            "rms, the largest peak noise alone reaches over the image's {1:.2g} beams, holds "
            "{2} in {3} components; its visibilities' median amplitude, {4}, predicts the "
            "solution S/N below.".format(
                model.nsigma, model.beams, _flux(model.flux, "Jy"), model.components,
                _flux(model.signal, "Jy")
            )
        )
    return lines + [""]


def _spws(summary: DataSummary) -> Tuple[str, str]:
    every = ", ".join(str(spw) for spw in summary.spws)
    if not summary.spws_flagged:
        return "", "every spectral window with data: {0}".format(every)
    return ",".join(str(spw) for spw in summary.spws), "the windows with unflagged data; {0} " \
        "{1} all flagged".format(
            ", ".join(str(spw) for spw in summary.spws_flagged),
            "is" if len(summary.spws_flagged) == 1 else "are"
        )


def _solve(summary: DataSummary, amplitudes: bool, combine: Optional[Choice]) -> List[str]:
    refant = reference_antenna(summary.antennas)
    minblperant = baselines_per_antenna(len(summary.antennas), amplitudes)
    lines = ["[solve]" + _pad("[solve]") + "# gaincal options shared by every step"]
    lines += [
        _line("refant", refant.value, refant.why),
        _line("minblperant", minblperant.value, minblperant.why),
        _line("minsnr", 3.0, "gaincal's default; solint \"auto\" aims at twice it"),
        _line("gaintype", "T", "one solution for both polarizations, with the S/N of both"),
    ]
    if combine is not None and combine.why:
        lines += [_line("combine", combine.value, combine.why)]
    elif len(summary.spws) > 1:
        lines += _comment(
            "combine = \"spw\" isn't set, so the {0} spectral windows are solved one by one. If "
            "one window's S/N is too low for short solints, combining them adds their signal; "
            "deciding it needs the S/N, and so a model: `wenuray suggest --image` makes one".format(
                len(summary.spws)
            )
        )
    return lines + [""]


def _imager(summary: DataSummary, imager: str) -> List[str]:
    lines = ["[imager]", _line("type", imager, "or \"pyralysis\"" if imager == "tclean" else "")]
    thermal = summary.sample.thermal_noise if summary.sample is not None else None
    if imager == "tclean":
        lines += [
            _line("niter", 1000),
            _line(
                "nsigma", NSIGMA, "each round cleans to {0:g} x its residual rms{1}".format(
                    NSIGMA, "" if thermal is None else
                    ", which approaches the thermal noise from the data, {0}, as the gains "
                    "improve".format(_flux(thermal))
                )
            ),
        ]
    else:
        lines += [
            _line(
                "terms", ["chi_squared", "l1norm:1e-3"],
                "the regularisation's weight isn't derived from the data: tune it"
            ),
            _line("optimizer", "lbfgsb"),
            _line(
                "mask_mode", "auto", "a mask keeps the model from absorbing the gain errors; "
                "\"file\" and mask_path for your own"
            ),
            _line(
                "estimator", "degridding",
                "this and the next three: gridding within 0.06% of the exact transform"
            ),
            _line("ckernel", "gaussian-sinc"),
            _line("ckernel_oversampling", 8),
            _line("padding_factor", 2.0),
        ]
        if thermal is not None:
            lines += _comment("The thermal noise from the data is {0}.".format(_flux(thermal)))
    if summary.geometry is None:
        geometry = {"imsize": AUTO, "cell": AUTO}
        why = "derived when the run starts"
    else:
        geometry = {"imsize": list(summary.geometry.imsize), "cell": summary.geometry.cell}
        why = "the cell: the Nyquist cell over {0:g}; the size: {1:g} primary-beam FWHMs " \
            "(#67)".format(summary.geometry.oversampling, summary.geometry.field_of_view)
    lines += [
        _line("geometry", geometry, why),
        _line(
            "weighting", {"weighting": "natural"},
            "the most sensitive: self-calibration needs the signal"
        ), ""
    ]
    return lines


def _steps(amplitudes: bool, nantennas: int, amplitude: Optional[Choice] = None) -> List[str]:
    lines = _comment("A first phase solution per scan, against the initial model.")
    lines += ["[[steps]]", _line("type", "phase"), _line("solint", "inf"), ""]
    lines += _comment(
        "The phases again, round by round, each at the shortest solint the S/N supports against "
        "the last round's model, until the gains are at their noise (#66)."
    )
    lines += [
        "[[steps]]",
        _line("type", "repeat"),
        _line("step", {
            "type": "phase",
            "solint": AUTO
        })
    ]
    if amplitudes:
        lines += [""] + _comment(
            "Amplitudes last: {0}. If the run's models can't support an amplitude solution, "
            "\"auto\" finds no solint and the run ends there, with the phases kept.".format(
                amplitude.why if amplitude is not None else "the S/N isn't known without a model"
            )
        )
        lines += ["[[steps]]", _line("type", "ampphase"), _line("solint", AUTO)]
    elif amplitude is not None:
        lines += _comment("No amplitude step: {0}.".format(amplitude.why))
    else:
        lines += _comment(
            "No amplitude step: closure amplitudes need {0} antennas, and {1} have data.".format(
                AmpPhasecal.min_antennas, nantennas
            )
        )
    return lines


def _flux(jansky: float, unit: str = "Jy/beam") -> str:
    # A no-break space keeps the number with its unit when the comment is wrapped (_comment)
    if jansky >= 1:
        return "{0:.3g}\u00a0{1}".format(jansky, unit)
    if jansky >= 1e-3:
        return "{0:.3g}\u00a0m{1}".format(jansky * 1e3, unit)
    return "{0:.3g}\u00a0u{1}".format(jansky * 1e6, unit)


def _line(key: str, value: Any, why: str = "") -> str:
    """
    key = value, with why as a comment after it, or above it when the line would be too long
    """
    text = "{0} = {1}".format(key, _value(value))
    if not why:
        return text
    padded = text + _pad(text) + "# " + why.replace("\u00a0", " ")
    if len(padded) <= WIDTH:
        return padded
    return "\n".join(_comment(why) + [text])


def _pad(text: str) -> str:
    return " " * max(1, COMMENT_COLUMN - len(text))


def _comment(text: str) -> List[str]:
    """
    text as comment lines; a no-break space joins words that stay on one line
    """
    return [
        line.replace("\u00a0", " ") for line in textwrap.
        wrap(text, WIDTH, initial_indent="# ", subsequent_indent="# ", break_on_hyphens=False)
    ]


def _value(value: Any) -> str:
    """
    A value as TOML writes it, tables inline
    """
    if isinstance(value, Mapping):
        return "{ " + ", ".join(
            "{0} = {1}".format(key, _value(item)) for key, item in value.items()
        ) + " }"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_value(item) for item in value) + "]"
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("A recipe can't hold {0!r}".format(value))
    return tomli_w.dumps({"x": value})[len("x = "):].rstrip("\n")
