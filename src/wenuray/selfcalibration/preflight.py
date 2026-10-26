"""
Pre-flight estimate of a step's solution signal-to-noise ratio (#28), before gaincal runs:

    S/N = (S / sigma) * sqrt(nantennas - 1) * sqrt(nintegrations * nchannels * npolarizations)

with everything taken from the data, so it holds for any array: S is the median amplitude of the
model visibilities (MODEL_DATA, the accepted model), sigma the noise of one visibility, from the
differences of consecutive integrations on each baseline (they cancel the source and slow gain
errors, which the residuals before the solve still hold), and the counts from the selection and
the step's solint and combine. The loop warns when the estimate is below minsnr and records it.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from casatools import ms as ms_tool_factory

from ..config import Selection, SolveOptions
from ..utils.selfcal_utils import open_table
from .acceptance import solution_snr
from .steps import CalibrationStep

# Rows read for the estimate: blocks spread through the selection, each holding this many
# consecutive integrations of every baseline and spectral window
BLOCKS = 8
INTEGRATIONS_PER_BLOCK = 4
# 1.4826 MAD is the standard deviation of a Gaussian
MAD_TO_SIGMA = 1.4826


@dataclass(frozen=True)
class VisibilitySample:
    """
    What the estimate needs from the data

    Parameters
    ----------
    signal :
        Median amplitude of the model visibilities, in Jy
    noise :
        Noise of the real or imaginary part of one visibility, in Jy
    nantennas :
        Antennas with unflagged cross-correlations
    interval :
        Integration time, in s
    scan_length :
        Median scan duration, in s
    span :
        Duration of the selected data, in s
    channels :
        Channels of each selected spectral window
    parallel_hands :
        Parallel-hand correlations (RR and LL, or XX and YY)
    """
    signal: float
    noise: float
    nantennas: int
    interval: float
    scan_length: float
    span: float
    channels: Tuple[int, ...]
    parallel_hands: int


@dataclass(frozen=True)
class SolutionSNR:
    """
    The predicted signal-to-noise ratio of one of a step's solutions

    Parameters
    ----------
    snr :
        The prediction for the step's solint and combine
    minsnr :
        The step's minsnr
    advice :
        How to reach minsnr when the prediction is below it
    """
    snr: float
    minsnr: float
    advice: str = ""

    @property
    def enough(self) -> bool:
        return self.snr >= self.minsnr


def predict(sample: VisibilitySample, step: CalibrationStep, options: SolveOptions) -> SolutionSNR:
    """
    The solution S/N of the step with the run's solve options, from a sample of the data
    """
    options = step.solve_options(options)
    combine = {axis.strip() for axis in options.combine.split(",")}
    longest = sample.span if "scan" in combine else sample.scan_length
    seconds = step.solint_seconds
    nintegrations = max(1, round(min(seconds, longest) / sample.interval)) if seconds else 1
    nchannels = sum(sample.channels) if "spw" in combine else min(sample.channels)
    npolarizations = sample.parallel_hands if options.gaintype == "T" else 1

    def snr(integrations: int, channels: int = nchannels) -> float:
        return solution_snr(
            sample.signal, sample.noise, sample.nantennas, integrations, channels, npolarizations
        )

    prediction = snr(nintegrations)
    advice = ""
    if prediction < options.minsnr:
        needed = math.ceil((options.minsnr / snr(1))**2)
        if needed * sample.interval <= longest:
            advice = "a solint of {0:g}s or longer would reach it".format(needed * sample.interval)
        else:
            advice = "even a solint of a whole {0} wouldn't; combine {1}".format(
                "observation" if "scan" in combine else "scan",
                "spectral windows" if "spw" not in combine and len(sample.channels) > 1 else
                "scans" if "scan" not in combine else "more data, or use a longer solint"
            )
    return SolutionSNR(prediction, options.minsnr, advice)


def read_sample(ms: str, selection: Selection) -> Optional[VisibilitySample]:
    """
    Reads what the estimate needs from the selected data (field and spectral windows; uvrange
    isn't applied). Returns None if the data can't support an estimate: no model, no unflagged
    cross-correlations, or no consecutive integrations
    """
    ms_tool = ms_tool_factory()
    try:
        indices = ms_tool.msseltoindex(vis=ms, field=selection.field, spw=selection.spw)
    finally:
        ms_tool.done()
    conditions = ["ANTENNA1 != ANTENNA2"] + [
        "{0} IN [{1}]".format(column, ",".join(str(int(i)) for i in ids))
        for column, ids in (("FIELD_ID", indices["field"]),
                            ("DATA_DESC_ID", indices["dd"])) if len(ids)
    ]
    with open_table(ms) as ms_table:
        if "MODEL_DATA" not in ms_table.colnames():
            return None
        selected = ms_table.query(query=" AND ".join(conditions))
        try:
            nrows = selected.nrows()
            if nrows == 0:
                return None
            times, scans = selected.getcol("TIME"), selected.getcol("SCAN_NUMBER")
            interval = float(np.median(selected.getcol("INTERVAL")))
            ddids = np.unique(selected.getcol("DATA_DESC_ID"))
            with open_table(os.path.join(ms, "ANTENNA")) as antenna_table:
                nbaselines = antenna_table.nrows() * (antenna_table.nrows() - 1) // 2
            block_rows = INTEGRATIONS_PER_BLOCK * nbaselines * len(ddids)
            blocks = [
                _read_block(selected, start, min(block_rows, nrows - start))
                for start in _block_starts(nrows, block_rows)
            ]
        finally:
            selected.close()
    with open_table(os.path.join(ms, "DATA_DESCRIPTION")) as dd_table:
        dd_spw = dd_table.getcol("SPECTRAL_WINDOW_ID")
    with open_table(os.path.join(ms, "SPECTRAL_WINDOW")) as spw_table:
        all_channels = spw_table.getcol("NUM_CHAN")
    channels = _selected_channels(indices, dd_spw, all_channels)

    model = np.concatenate([block["model"] for block in blocks])
    differences = np.concatenate([_differences(block, interval) for block in blocks])
    antennas = np.unique(np.concatenate([block["antennas"] for block in blocks]))
    if model.size == 0 or differences.size < 10 or not np.any(model > 0):
        return None
    components = np.concatenate([differences.real, differences.imag])
    noise = MAD_TO_SIGMA * np.median(np.abs(components - np.median(components))) / math.sqrt(2)
    if not noise > 0:
        return None
    scan_lengths = [
        times[scans == scan].max() - times[scans == scan].min() + interval
        for scan in np.unique(scans)
    ]
    return VisibilitySample(
        signal=float(np.median(model)),
        noise=float(noise),
        nantennas=len(antennas),
        interval=interval,
        scan_length=float(np.median(scan_lengths)),
        span=float(times.max() - times.min() + interval),
        channels=tuple(int(n) for n in channels),
        parallel_hands=blocks[0]["parallel_hands"],
    )


def _block_starts(nrows: int, block_rows: int) -> list:
    if nrows <= BLOCKS * block_rows:
        return list(range(0, nrows, block_rows))
    return [int(start) for start in np.linspace(0, nrows - block_rows, BLOCKS)]


def _read_block(selected, start: int, nrow: int) -> dict:
    """
    The parallel hands (the first and last correlations) of a block of rows
    """
    model = selected.getcol("MODEL_DATA", startrow=start, nrow=nrow)
    data = selected.getcol(
        "CORRECTED_DATA" if "CORRECTED_DATA" in selected.colnames() else "DATA",
        startrow=start,
        nrow=nrow
    )
    flags = selected.getcol("FLAG", startrow=start, nrow=nrow)
    hands = sorted({0, model.shape[0] - 1})
    antenna1 = selected.getcol("ANTENNA1", startrow=start, nrow=nrow)
    antenna2 = selected.getcol("ANTENNA2", startrow=start, nrow=nrow)
    good = ~flags[hands]
    unflagged_rows = good.any(axis=(0, 1))
    return {
        "model": np.abs(model[hands])[good],
        "data": data[hands],
        "flags": flags[hands],
        "baseline": antenna1 * 100000 + antenna2,
        "ddid": selected.getcol("DATA_DESC_ID", startrow=start, nrow=nrow),
        "time": selected.getcol("TIME", startrow=start, nrow=nrow),
        "antennas": np.concatenate([antenna1[unflagged_rows], antenna2[unflagged_rows]]),
        "parallel_hands": len(hands),
    }


def _differences(block: dict, interval: float) -> np.ndarray:
    """
    Differences of consecutive integrations of each baseline, correlation and channel
    """
    differences = []
    keys = block["baseline"] * 1000 + block["ddid"]
    for key in np.unique(keys):
        rows = np.flatnonzero(keys == key)
        rows = rows[np.argsort(block["time"][rows])]
        consecutive = np.diff(block["time"][rows]) <= 1.5 * interval
        first, second = rows[:-1][consecutive], rows[1:][consecutive]
        if not len(first):
            continue
        both = ~block["flags"][..., first] & ~block["flags"][..., second]
        delta = block["data"][..., second] - block["data"][..., first]
        differences.append(delta[both])
    return np.concatenate(differences) if differences else np.zeros(0, dtype=complex)


def _selected_channels(indices: dict, dd_spw: np.ndarray, all_channels: np.ndarray) -> list:
    """
    Channels of each selected spectral window, from the channel ranges of the spw selection
    """
    ranges = indices["channel"]
    if len(ranges):
        counts = {}
        for spw, start, end, step in ranges:
            counts[spw] = counts.get(spw, 0) + len(range(start, end + 1, max(step, 1)))
        return [counts[spw] for spw in sorted(counts)]
    spws = sorted({int(dd_spw[dd]) for dd in indices["dd"]}) if len(indices["dd"]) else \
        range(len(all_channels))
    return [int(all_channels[spw]) for spw in spws]
