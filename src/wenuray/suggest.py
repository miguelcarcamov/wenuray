"""
A recipe suggested from the data (#190): ``wenuray suggest target.ms -o recipe.toml``.

Wenuray derives most of a recipe from the data when it runs ("auto" solints and grids). Suggesting
the recipe writes those values out before anything runs, so the user sees, edits and keeps them,
and each value carries a comment saying where it comes from. Nothing here is imaged: the source
model, and with it the solution S/N, is unknown, so solints stay "auto" and the run resolves them
round by round against its models (#66).

The data are read once, into a DataSummary (read_summary); recipe_text() turns a summary into the
recipe and reads nothing, so every choice can be tested without CASA. Every value comes from
theory or the data: nothing depends on the instrument.
"""
from __future__ import annotations

import math
import os
import textwrap
from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import tomli_w

from .config import AUTO, ImageGeometry, Selection
from .selfcalibration import preflight
from .selfcalibration.steps import AmpPhasecal, Phasecal
from .utils.selfcal_utils import get_spw_ids, open_table, spws_with_unflagged_data

IMAGERS = ("tclean", "pyralysis")
# The comments start at this column, as in `wenuray init`'s template, and lines stop at this width
COMMENT_COLUMN = 31
WIDTH = 100
# Each round cleans to this many times its residual rms (tclean's nsigma), which stays above the
# gain-error artefacts of the early rounds, where a fixed threshold near the thermal noise wouldn't
NSIGMA = 3.0


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
    """
    ms: str
    field: str
    spws: Tuple[int, ...]
    spws_flagged: Tuple[int, ...]
    antennas: Tuple[Antenna, ...]
    sample: Optional[preflight.VisibilitySample]
    geometry: Optional[ImageGeometry]
    geometry_problem: str = ""


@dataclass(frozen=True)
class Choice:
    """
    A suggested value and why it was chosen
    """
    value: Any
    why: str = ""


def read_summary(ms: str, field: str = "") -> DataSummary:
    """
    Reads what the suggestion needs from the measurement set: its fields, spectral windows and
    antennas, a sample of its data, and the grid

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
    with open_table(os.path.join(ms, "ANTENNA")) as antenna_table:
        names = antenna_table.getcol("NAME")
        positions = antenna_table.getcol("POSITION").T
        flag_row = antenna_table.getcol("FLAG_ROW")
    flagged = sample.antenna_flagged if sample is not None else (math.nan, ) * len(names)
    antennas = tuple(
        Antenna(str(name), tuple(float(x) for x in position), float(fraction))
        for name, position, fraction, row_flagged in zip(names, positions, flagged, flag_row)
        if not row_flagged and not fraction >= 1.0
    )
    try:
        geometry, problem = resolve_geometry(
            ImageGeometry(imsize=AUTO, cell=AUTO), ms, selection
        ), ""
    except ValueError as error:
        geometry, problem = None, str(error)
    return DataSummary(
        ms, field, spws, tuple(sorted(set(with_data) - set(spws))), antennas, sample, geometry,
        problem
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
    lines += _solve(summary, amplitudes)
    lines += _imager(summary, imager)
    lines += _steps(amplitudes, nantennas)
    return "\n".join(lines) + "\n"


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
    for finding in findings:
        lines += _comment("Warning: " + finding)
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


def _solve(summary: DataSummary, amplitudes: bool) -> List[str]:
    refant = reference_antenna(summary.antennas)
    minblperant = baselines_per_antenna(len(summary.antennas), amplitudes)
    lines = ["[solve]" + _pad("[solve]") + "# gaincal options shared by every step"]
    lines += [
        _line("refant", refant.value, refant.why),
        _line("minblperant", minblperant.value, minblperant.why),
        _line("minsnr", 3.0, "gaincal's default; solint \"auto\" aims at twice it"),
        _line("gaintype", "T", "one solution for both polarizations, with the S/N of both"),
    ]
    if len(summary.spws) > 1:
        lines += _comment(
            "combine = \"spw\" isn't set, so the {0} spectral windows are solved one by one. If "
            "one window's S/N is too low for short solints, combining them adds their signal; "
            "deciding it needs the S/N, and so a model, which this suggestion doesn't make".format(
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


def _steps(amplitudes: bool, nantennas: int) -> List[str]:
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
            "Amplitudes last: if the S/N can't support an amplitude solution, \"auto\" finds no "
            "solint and the run ends there, with the phases kept."
        )
        lines += ["[[steps]]", _line("type", "ampphase"), _line("solint", AUTO)]
    else:
        lines += _comment(
            "No amplitude step: closure amplitudes need {0} antennas, and {1} have data.".format(
                AmpPhasecal.min_antennas, nantennas
            )
        )
    return lines


def _flux(jansky: float) -> str:
    # A no-break space keeps the number with its unit when the comment is wrapped (_comment)
    if jansky >= 1:
        return "{0:.3g}\u00a0Jy/beam".format(jansky)
    if jansky >= 1e-3:
        return "{0:.3g}\u00a0mJy/beam".format(jansky * 1e3)
    return "{0:.3g}\u00a0uJy/beam".format(jansky * 1e6)


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
