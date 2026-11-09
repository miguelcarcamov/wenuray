"""
A recipe suggested from the data (#121): ``wenuray suggest target.ms --imager tclean`` writes a
recipe with every value taken from the measurement set, each with its reason as a comment.

Each decision comes from the data or from theory, never from an instrument:

- the selection: the fields and spectral windows that hold data;
- the reference antenna: the one nearest the array's centre among those with no more flagged data
  than most;
- ``combine``: spectral windows are combined only if one window's phase solution over a whole scan
  can't reach twice minsnr;
- ``gaintype``: "T" for phases (the atmosphere delays both polarizations alike, and one solution
  for both has sqrt(2) the S/N); "G" for amplitudes when one polarization's solution over a scan
  reaches twice minsnr;
- the schedule: phases per scan, then repeated at shorter "auto" solints until they reach their
  noise, then amplitudes. Starting per scan keeps the first gains from following the least
  reliable model, the initial one: on ALMA Band 7 data, starting at the shortest supported solint
  left the residual at 2.2 x thermal, against 1.2 x for a ladder from a scan down;
- the grid and the solints: "auto", resolved when the run starts against its model (#66, #67), or,
  with ``resolve``, the values they resolve to now;
- the imager: the one asked for, with a note on what to add (a mask for Pyralysis) and on the
  bandwidth over which a spectral index matters (#117).

Before any model exists, the signal of the S/N estimates is the visibility amplitude of DATA with
the noise taken out (preflight.read_sample(signal_from="data")), which phase errors don't lower.
"""
from __future__ import annotations

import datetime
import os
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import tomli_w

from .config import AUTO, ImageGeometry, Selection, SolveOptions
from .utils.selfcal_utils import get_spw_ids, open_table

IMAGERS = ("tclean", "pyralysis")
# Rows sampled for each antenna's flagged fraction
FLAG_SAMPLES = 20000
# An antenna can be the reference if its flagged fraction is at most this much above the median
FLAGGED_TOLERANCE = 0.05


@dataclass(frozen=True)
class Choice:
    """
    A value of the recipe and why

    Parameters
    ----------
    value :
        The value, as TOML holds it
    reason :
        Why, written as a comment next to it
    """
    value: Any
    reason: str = ""


def suggest(ms: str, imager: str, resolve: bool = False, relative_to: Optional[str] = None) -> str:
    """
    The recipe, as TOML text with a comment on each value

    Parameters
    ----------
    ms :
        Measurement set with the calibrated target data in DATA
    imager :
        "tclean" or "pyralysis"
    resolve :
        Write the grid and the solints the data support now instead of "auto"
    relative_to :
        Directory the recipe will be in, so that its ms path is relative to it; None: as given

    Raises
    ------
    FileNotFoundError
        If the measurement set doesn't exist
    ValueError
        If the imager is unknown, or the data can't support an S/N estimate
    """
    from .selfcalibration import preflight

    if imager not in IMAGERS:
        raise ValueError("imager must be one of {0}, not {1!r}".format(IMAGERS, imager))
    if not os.path.isdir(ms):
        raise FileNotFoundError("{0} doesn't exist".format(ms))
    fields = field_rows(ms)
    spws = get_spw_ids(ms, "")
    field = choose_field(fields)
    selection = Selection(field=field.value)
    sample = preflight.read_sample(ms, selection, signal_from="data")
    if sample is None:
        raise ValueError(
            "{0}: the selected data can't support an S/N estimate (no unflagged cross-correlations "
            "or no consecutive integrations)".format(ms)
        )
    positions, names, flagged = antenna_flagging(ms, selection)
    refant = choose_refant(names, positions, flagged)
    combine = choose_combine(sample, len(spws))
    amplitude_gaintype = choose_amplitude_gaintype(sample, combine.value)
    geometry = choose_geometry(ms, selection, resolve)
    steps = choose_steps(sample, combine.value, amplitude_gaintype.value, resolve)
    return render(
        ms=recipe_path(ms, relative_to),
        field=field,
        spws=spws,
        refant=refant,
        combine=combine,
        steps=steps,
        imager=imager,
        geometry=geometry,
        sample=sample,
    )


def recipe_path(ms: str, relative_to: Optional[str]) -> str:
    """
    The measurement set's path as the recipe holds it: relative to the recipe's directory when the
    set is inside it, so the two move together; absolute otherwise; as given with no directory
    """
    if relative_to is None:
        return ms
    relative = os.path.relpath(os.path.abspath(ms), relative_to)
    return os.path.abspath(ms) if relative.startswith(os.pardir) else relative


def field_rows(ms: str) -> List[Tuple[str, int]]:
    """
    The fields that hold data, with their numbers of rows, from the main table's FIELD_ID
    """
    with open_table(ms) as ms_table:
        query = ms_table.query(query="", columns="FIELD_ID")
        try:
            ids = query.getcol("FIELD_ID")
        finally:
            query.close()
    with open_table(os.path.join(ms, "FIELD")) as field_table:
        names = field_table.getcol("NAME")
    used, counts = np.unique(ids, return_counts=True)
    return [(str(names[i]), int(n)) for i, n in zip(used, counts)]


def choose_field(fields: Sequence[Tuple[str, int]]) -> Choice:
    """
    The field to self-calibrate: the only one with data; with several, all of them, saying which
    """
    if len(fields) == 1:
        return Choice(fields[0][0], "the only field with data")
    listed = ", ".join("{0} ({1} rows)".format(name, rows) for name, rows in fields)
    return Choice(
        "", "every field with data: {0}; select one to self-calibrate it alone".format(listed)
    )


def antenna_flagging(ms: str, selection: Selection) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Each antenna's position (ITRF, m), name and flagged fraction over a sample of the selected rows;
    NaN for an antenna with no sampled data
    """
    from .utils.selfcal_utils import selected_rows

    with open_table(os.path.join(ms, "ANTENNA")) as antenna_table:
        positions = np.asarray(antenna_table.getcol("POSITION")).T
        names = np.asarray(antenna_table.getcol("NAME"))
    flagged, total = np.zeros(len(names)), np.zeros(len(names))
    rows = selected_rows(ms, selection.field, selection.spw, samples=FLAG_SAMPLES)
    with open_table(ms) as ms_table:
        for row in rows:
            antenna1, antenna2 = ms_table.getcell("ANTENNA1",
                                                  row), ms_table.getcell("ANTENNA2", row)
            if antenna1 == antenna2:
                continue
            fraction = float(np.mean(ms_table.getcell("FLAG", row)))
            for antenna in (antenna1, antenna2):
                flagged[antenna] += fraction
                total[antenna] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        return positions, names, np.where(total > 0, flagged / total, np.nan)


def choose_refant(names: np.ndarray, positions: np.ndarray, flagged: np.ndarray) -> Choice:
    """
    The antenna nearest the array's centre (the median of the positions with data) among those
    whose flagged fraction is at most FLAGGED_TOLERANCE above the median antenna's: a central
    reference has the shortest baselines to the others, so the least atmospheric phase on them

    Raises
    ------
    ValueError
        If no antenna has data
    """
    has_data = np.isfinite(flagged)
    if not np.any(has_data):
        raise ValueError("No antenna has data in the selection")
    typical = float(np.median(flagged[has_data]))
    candidates = has_data & (flagged <= typical + FLAGGED_TOLERANCE)
    centre = np.median(positions[has_data], axis=0)
    distance = np.linalg.norm(positions - centre, axis=1)
    best = int(np.argmin(np.where(candidates, distance, np.inf)))
    return Choice(
        str(names[best]),
        "{0:.0f} m from the array's centre, {1:.1%} of its data flagged (median antenna "
        "{2:.1%})".format(distance[best], flagged[best], typical)
    )


def choose_combine(sample, nspws: int) -> Choice:
    """
    "spw" only if one window's phase solution over a whole scan can't reach twice minsnr, the
    target of solint="auto"; else each window has its own solutions
    """
    from .selfcalibration import preflight
    from .selfcalibration.steps import Phasecal

    target = preflight.AUTO_SNR_FACTOR * SolveOptions().minsnr
    per_window = preflight.predict(sample, Phasecal("inf"), SolveOptions()).snr
    if nspws > 1 and per_window < target:
        combined = preflight.predict(sample, Phasecal("inf"), SolveOptions(combine="spw")).snr
        return Choice(
            "spw", "one window's phase S/N over a scan is {0:.1f}, below {1:g}; the {2} windows "
            "combined reach {3:.1f}".format(per_window, target, nspws, combined)
        )
    return Choice(
        "", "one window's phase S/N over a scan is {0:.1f}, above {1:g}: each window gets its own "
        "solutions".format(per_window, target)
    )


def choose_amplitude_gaintype(sample, combine: str) -> Choice:
    """
    "G", a solution per polarization, if one polarization's amplitude solution over a scan reaches
    twice minsnr; else "T"
    """
    from .selfcalibration import preflight
    from .selfcalibration.steps import AmpPhasecal

    target = preflight.AUTO_SNR_FACTOR * SolveOptions().minsnr
    snr = preflight.predict(
        sample, AmpPhasecal("inf"), SolveOptions(combine=combine, gaintype="G")
    ).snr
    if snr >= target:
        return Choice(
            "G", "one polarization's amplitude S/N over a scan is {0:.1f}, above "
            "{1:g}".format(snr, target)
        )
    return Choice(
        "T", "one polarization's amplitude S/N over a scan is {0:.1f}, below {1:g}: "
        "both polarizations share a solution".format(snr, target)
    )


def choose_geometry(ms: str, selection: Selection, resolve: bool) -> Choice:
    """
    The grid: "auto", with what it resolves to now; or, with resolve, those values
    """
    from .imaging.contract import resolve_geometry

    resolved = resolve_geometry(ImageGeometry(imsize=AUTO, cell=AUTO), ms, selection)
    nx, ny = resolved.imsize
    now = "{0} x {1} pixels of {2}: the Nyquist cell / 2.5, and 2.5 primary-beam FWHMs".format(
        nx, ny, resolved.cell
    )
    if resolve:
        return Choice({"imsize": nx if nx == ny else [nx, ny], "cell": resolved.cell}, now)
    return Choice({"imsize": AUTO, "cell": AUTO}, "resolved when the run starts; now " + now)


def choose_steps(sample, combine: str, amplitude_gaintype: str, resolve: bool) -> List[Choice]:
    """
    Phases repeated at shorter "auto" solints until they reach their noise, then amplitudes and
    phases; or, with resolve, one round of each at the solint the data support now
    """
    from .selfcalibration import preflight
    from .selfcalibration.steps import AmpPhasecal, Phasecal

    amplitude_solve = {"gaintype": amplitude_gaintype}
    if combine:
        amplitude_solve["combine"] = ""
    if not resolve:
        per_scan = {"type": "phase", "solint": "inf"}
        shorter = {"type": "repeat", "step": {"type": "phase", "solint": AUTO}}
        amplitudes = {"type": "ampphase", "solint": AUTO, "solve": amplitude_solve}
        return [
            Choice(per_scan, "first, phases per scan: the initial model is the least reliable"),
            Choice(shorter, "then phases at solints shorter than a scan, until at their noise"),
            Choice(amplitudes, "then amplitudes, with gaintype \"{0}\"".format(amplitude_gaintype)),
        ]
    options = SolveOptions(combine=combine)
    phase, phase_why = preflight.shortest_solint(sample, Phasecal(AUTO), options)
    amplitude, amplitude_why = preflight.shortest_solint(
        sample, AmpPhasecal(AUTO, solve=amplitude_solve), options
    )
    if phase is None or amplitude is None:
        raise ValueError(
            "The data support no solint: {0}".format(phase_why if phase is None else amplitude_why)
        )
    return [
        Choice({
            "type": "phase",
            "solint": phase
        }, phase_why),
        Choice({
            "type": "ampphase",
            "solint": amplitude,
            "solve": amplitude_solve
        }, amplitude_why),
    ]


def render(
    ms: str, field: Choice, spws: Sequence[int], refant: Choice, combine: Choice,
    steps: Sequence[Choice], imager: str, geometry: Choice, sample
) -> str:
    """
    The recipe as TOML, a comment with each value's reason
    """
    from . import __version__

    lines = [
        "# Recipe suggested by wenuray suggest {0} on {1} from the data of {2}.".format(
            __version__,
            datetime.date.today().isoformat(), os.path.basename(ms)
        ),
        "# Each value comes from the data, with its reason. Review it, then: wenuray run <this file>",
        "",
        _line("ms", ms, "the calibrated target data in DATA"),
        "",
        "[selection]",
        _line("field", field.value, field.reason),
        _line("spw", "", "the windows with data: {0}".format(", ".join(str(s) for s in spws))),
        "",
        "[solve]",
        _line("refant", refant.value, refant.reason),
        _line("combine", combine.value, combine.reason),
        _line("gaintype", "T", "phases: the atmosphere delays both polarizations alike"),
        "",
        "[imager]",
        _line("type", imager),
        _line("geometry", geometry.value, geometry.reason),
        _line("weighting", {"weighting": "natural"}, "the most sensitive, for the models"),
    ]
    lines += _imager_lines(imager)
    lines += ["", _bandwidth_note(sample, imager)]
    for step in steps:
        lines += ["", "# " + step.reason, "[[steps]]"]
        lines += [_line(name, value) for name, value in step.value.items()]
    return "\n".join(lines) + "\n"


def _imager_lines(imager: str) -> List[str]:
    if imager == "tclean":
        return [
            _line("deconvolver", "hogbom"),
            _line("niter", 100000, "a ceiling: nsigma stops it"),
            _line("nsigma", 3.0, "clean the whole image down to 3 x the residual's noise"),
            "# A mask limits the model to the sources: usemask = \"auto-multithresh\" with "
            "thresholds suited to",
            "# the PSF's sidelobes (its defaults find no mask with high sidelobes), or a mask "
            "file",
        ]
    return [
        _line("terms", ["chi_squared"]),
        _line("optimizer", "lbfgsb", "keeps the model positive"),
        _line("max_iter", 100),
        "# Add a mask: mask_mode = \"file\" and mask_path, around the sources. Without one the "
        "model is",
        "# free in every pixel and can absorb the gain errors self-calibration should solve "
        "(Wenuray warns)",
    ]


def _bandwidth_note(sample, imager: str) -> str:
    """
    How much a spectral index changes the flux across the selected band (#117)
    """
    bandwidth = sample.fractional_bandwidth
    how = "spectral_index = <alpha>" if imager == "pyralysis" else \
        "deconvolver = \"mtmfs\" and nterms = 2"
    return (
        "# Fractional bandwidth {0:.1%}: a spectral index of 3 changes the flux by about {1:.0%} "
        "across the band.\n# If the residuals change with frequency, model it: {2} (#117)".format(
            bandwidth, 3 * bandwidth, how
        )
    )


def _line(name: str, value: Any, reason: str = "") -> str:
    text = "{0} = {1}".format(name, _toml(value))
    return text + "  # " + reason if reason else text


def _toml(value: Any) -> str:
    """
    A value as inline TOML
    """
    if isinstance(value, dict):
        return "{ " + ", ".join("{0} = {1}".format(k, _toml(v)) for k, v in value.items()) + " }"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml(item) for item in value) + "]"
    return tomli_w.dumps({"x": value}).split("=", 1)[1].strip()
