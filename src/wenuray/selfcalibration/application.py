"""
Applying a run's accepted calibration to other data (#68): the line data, other spectral windows,
or another measurement set of the same observation. CalStack builds the applycal lists, as in the
run, but each table's spwmap is derived again for the target: a split measurement set renumbers its
spectral windows, so they are matched to the tables' by frequency, never by id.
"""
from __future__ import annotations

import dataclasses
import logging
import os
from typing import Dict, Mapping, Tuple

import numpy as np

from ..config import ApplyOptions, Selection
from ..utils.selfcal_utils import channel_edges, get_spw_ids, ms_fingerprint, open_table
from .caltable import CalStack, CalTable
from .workspace import MSWorkspace

logger = logging.getLogger(__name__)

Band = Tuple[float, float]


@dataclasses.dataclass(frozen=True)
class Application:
    """
    A record of the accepted calibration applied to other data

    Parameters
    ----------
    ms :
        The measurement set calibrated
    selection :
        The data calibrated
    applymode :
        applycal's applymode
    stack :
        The tables as applied, each with the spwmap derived for the target
    fingerprint :
        ms_fingerprint of the target before the calibration was applied
    """
    ms: str
    selection: Selection
    applymode: str
    stack: CalStack
    fingerprint: str = ""


def map_spectral_windows(targets: Mapping[int, Band], solved: Mapping[int, Band],
                         combined: bool) -> Dict[int, int]:
    """
    The table spectral window each target spectral window takes its solutions from

    Parameters
    ----------
    targets :
        The target's spectral windows to calibrate, with their frequency ranges in Hz
    solved :
        The table's spectral windows that hold solutions, with their frequency ranges
    combined :
        Whether the table was solved combining spectral windows: one solution for the whole band,
        which every target spectral window takes

    Raises
    ------
    ValueError
        If a table solved per spectral window has no solutions overlapping a target spectral
        window in frequency: its gains at another frequency would be a guess
    """
    if not solved:
        raise ValueError("The table holds no solutions")
    if combined:
        # gaincal writes a spw-combined solution under one spectral window
        return {spw: min(solved) for spw in targets}
    mapping = {}
    for spw, (low, high) in targets.items():
        overlaps = {
            table_spw: min(high, table_high) - max(low, table_low)
            for table_spw, (table_low, table_high) in solved.items()
        }
        best = max(overlaps, key=overlaps.get)
        if overlaps[best] <= 0:
            centre = (low + high) / 2
            nearest = min(solved, key=lambda table_spw: abs(sum(solved[table_spw]) / 2 - centre))
            raise ValueError(
                "Spectral window {0} ({1:.6g}-{2:.6g} GHz) has no solutions: the table was solved "
                "per spectral window and none overlaps it (the nearest is {3}, {4:.6g}-{5:.6g} "
                "GHz). Solve with combine='spw' to calibrate other frequencies".format(
                    spw, low / 1e9, high / 1e9, nearest, solved[nearest][0] / 1e9,
                    solved[nearest][1] / 1e9
                )
            )
        mapping[spw] = best
    return mapping


def spwmap_list(mapping: Mapping[int, int]) -> Tuple[int, ...]:
    """
    applycal's spwmap: entry i is the table spectral window for spectral window i. Windows not
    calibrated map to themselves, as applycal does when the map is shorter
    """
    spwmap = list(range(max(mapping) + 1))
    for spw, table_spw in mapping.items():
        spwmap[spw] = table_spw
    return tuple(spwmap)


def apply_to(stack: CalStack, ms: str, selection: Selection, applymode: str) -> Application:
    """
    Applies the stack to the selected data of ms, writing CORRECTED_DATA, with each table's
    spwmap derived for ms

    Raises
    ------
    ValueError
        If ms doesn't hold the tables' antennas or times, or a spectral window can't be mapped
    """
    if not os.path.isdir(ms):
        raise ValueError("{0} does not exist".format(ms))
    target_spws = get_spw_ids(ms, selection.spw)
    with open_table(os.path.join(ms, "SPECTRAL_WINDOW")) as spw_table:
        bands = {spw: channel_edges(spw_table, spw) for spw in target_spws}
    antennas = _names(ms)
    times = _time_range(ms)
    tables = []
    for table in stack:
        _check_compatible(table, antennas, times, ms)
        mapping = map_spectral_windows(
            bands, _solved_bands(table.path), combined=len(set(table.spwmap)) == 1
        )
        tables.append(dataclasses.replace(table, spwmap=spwmap_list(mapping)))
    remapped = CalStack(tables)
    fingerprint = ms_fingerprint(ms)
    logger.info(
        "Applying %d accepted tables to %s (spw %s)", len(remapped), ms, selection.spw
        or "with data"
    )
    MSWorkspace(ms, selection, ApplyOptions(applymode)).apply_stack(remapped)
    return Application(ms, selection, applymode, remapped, fingerprint)


def _solved_bands(path: str) -> Dict[int, Band]:
    """
    The table's spectral windows that hold solutions, with their frequency ranges
    """
    with open_table(path) as table:
        solved = np.unique(table.getcol("SPECTRAL_WINDOW_ID"))
    with open_table(os.path.join(path, "SPECTRAL_WINDOW")) as spw_table:
        return {int(spw): channel_edges(spw_table, int(spw)) for spw in solved}


def _names(path: str) -> np.ndarray:
    with open_table(os.path.join(path, "ANTENNA")) as antenna_table:
        return np.asarray(antenna_table.getcol("NAME"))


def _time_range(ms: str) -> Tuple[float, float]:
    with open_table(ms) as ms_table:
        times = ms_table.getcol("TIME")
    return float(times.min()), float(times.max())


def _check_compatible(
    table: CalTable, antennas: np.ndarray, times: Tuple[float, float], ms: str
) -> None:
    """
    The table's solutions must be for the target's antennas, by the same ids, and its times
    """
    names = _names(table.path)
    with open_table(table.path) as cal_table:
        used = np.unique(cal_table.getcol("ANTENNA1"))
        solution_times = cal_table.getcol("TIME")
        half = float(np.max(cal_table.getcol("INTERVAL"))) / 2
    for antenna in used:
        if antenna >= len(antennas) or antennas[antenna] != names[antenna]:
            raise ValueError(
                "{0} has no antenna {1} with id {2}: the table was solved on other data".format(
                    ms, names[antenna], antenna
                )
            )
    if times[1] < solution_times.min() - half or times[0] > solution_times.max() + half:
        raise ValueError(
            "{0} doesn't overlap the times of {1}: apply tables to data of the same "
            "observation".format(ms, table.path)
        )
