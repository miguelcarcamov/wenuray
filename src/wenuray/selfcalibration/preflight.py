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
from ..utils.selfcal_utils import channel_edges, open_table
from .acceptance import solution_snr
from .steps import CalibrationStep

# Rows read for the estimate: blocks spread through the selection, each holding this many
# consecutive integrations of every baseline and spectral window
BLOCKS = 8
INTEGRATIONS_PER_BLOCK = 4
# 1.4826 MAD is the standard deviation of a Gaussian
MAD_TO_SIGMA = 1.4826
# Weight checks (#129): differences a baseline needs for its noise; how far measured noise over
# SIGMA may be from 1 (10% in noise is 20% in weight, and 10% in every noise-based number); and
# the robust z-score of a baseline that stands out
MIN_DIFFERENCES = 20
WEIGHTS_TOLERANCE = 0.1
OUTLIER_Z = 5.0


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
    band :
        Lowest and highest frequency of the selected spectral windows, channel edges included, in
        Hz
    weights :
        Sums of the data weights and of their squares over the selected, unflagged visibilities of
        the parallel hands (every channel counted), from the sample scaled to the whole selection
    noise_over_sigma :
        For each sampled baseline and spectral window, (antenna1, antenna2, spw, ratio): the
        noise of its consecutive-integration differences over the noise its SIGMA column states
        (#129). Near 1 when the weights are right
    """
    signal: float
    noise: float
    nantennas: int
    interval: float
    scan_length: float
    span: float
    channels: Tuple[int, ...]
    parallel_hands: int
    band: Tuple[float, float]
    weights: Tuple[float, float]
    noise_over_sigma: Tuple[Tuple[str, str, int, float], ...] = ()

    @property
    def fractional_bandwidth(self) -> float:
        low, high = self.band
        return 2 * (high - low) / (high + low)

    @property
    def thermal_noise(self) -> float:
        """
        The thermal noise of a Stokes I image of the selection, in Jy/beam (#119). A pixel of the
        dirty image is a weighted mean of the visibilities' real parts, so its noise is
        noise * sqrt(sum w^2) / sum w. The weights are the data weights: exact for natural
        weighting, and its floor (a lower bound) for briggs or uniform. The noise comes from the
        data rather than from the weights, which often misstate it
        """
        total, squares = self.weights
        return self.noise * math.sqrt(squares) / total


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


def absorbed_spectrum(
    sample: VisibilitySample, step: CalibrationStep, options: SolveOptions, snr: float,
    models_spectrum: bool
) -> str:
    """
    Why the step's amplitude gains would absorb the source's spectrum, or "" if they won't (#118).

    Solved per spectral window against a model without a spectrum, each window's amplitude gain
    takes up the source's flux at that window's frequency: the residual drops, but the calibrated
    data lose their spectral index. A spectral index alpha changes the amplitude by about
    alpha * (fractional bandwidth) across the band, and a solution with signal-to-noise ratio
    ``snr`` measures amplitudes to about 1 / snr; the warning is given when the gains would resolve
    the spectrum of a source with alpha = 1. One window, or windows combined, keep the spectrum.
    """
    options = step.solve_options(options)
    combine = {axis.strip() for axis in options.combine.split(",")}
    if not step.solves_amplitude or "spw" in combine or len(sample.channels) < 2:
        return ""
    if models_spectrum or sample.fractional_bandwidth <= 1 / snr:
        return ""
    return (
        "amplitudes are solved per spectral window across a {0:.1%} band against a model without "
        "a spectrum, with solutions precise to {1:.1%}: the gains can absorb the source's spectral "
        "index and bias the calibrated data's. Give the imager a spectral model (e.g. tclean's "
        "deconvolver='mtmfs', or Pyralysis's spectral_index), or combine='spw'".format(
            sample.fractional_bandwidth, 1 / snr
        )
    )


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
        edges = [channel_edges(spw_table, spw) for spw in _selected_spws(indices, dd_spw, ddids)]
    channels = _selected_channels(indices, dd_spw, all_channels, ddids)

    weight_sum = sum(block["weights"][0] for block in blocks)
    weight_squares = sum(block["weights"][1] for block in blocks)
    sampled = sum(block["rows"] for block in blocks)
    model = np.concatenate([block["model"] for block in blocks])
    per_baseline = {}
    for block in blocks:
        for key, (delta, sigma) in _differences(block, interval).items():
            per_baseline.setdefault(key, []).append((delta, sigma))
    differences = np.concatenate(
        [delta for pairs in per_baseline.values() for delta, _ in pairs]
        or [np.zeros(0, dtype=complex)]
    )
    antennas = np.unique(np.concatenate([block["antennas"] for block in blocks]))
    if model.size == 0 or differences.size < 10 or not np.any(model > 0):
        return None
    components = np.concatenate([differences.real, differences.imag])
    noise = MAD_TO_SIGMA * np.median(np.abs(components - np.median(components))) / math.sqrt(2)
    if not noise > 0 or not weight_sum > 0:
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
        band=(min(low for low, _ in edges), max(high for _, high in edges)),
        weights=(weight_sum * nrows / sampled, weight_squares * nrows / sampled),
        noise_over_sigma=_noise_over_sigma(per_baseline, _antenna_names(ms), dd_spw),
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
    sigma = selected.getcol("SIGMA", startrow=start, nrow=nrow)
    hands = sorted({0, model.shape[0] - 1})
    antenna1 = selected.getcol("ANTENNA1", startrow=start, nrow=nrow)
    antenna2 = selected.getcol("ANTENNA2", startrow=start, nrow=nrow)
    good = ~flags[hands]
    unflagged_rows = good.any(axis=(0, 1))
    # One weight per correlation and row, shared by its channels
    weights = selected.getcol("WEIGHT", startrow=start, nrow=nrow)[hands]
    unflagged_channels = good.sum(axis=1)
    return {
        "model": np.abs(model[hands])[good],
        "data": data[hands],
        "flags": flags[hands],
        "sigma": sigma[hands],
        "baseline": antenna1 * 100000 + antenna2,
        "ddid": selected.getcol("DATA_DESC_ID", startrow=start, nrow=nrow),
        "time": selected.getcol("TIME", startrow=start, nrow=nrow),
        "antennas": np.concatenate([antenna1[unflagged_rows], antenna2[unflagged_rows]]),
        "parallel_hands": len(hands),
        "weights": (
            float(np.sum(weights * unflagged_channels)),
            float(np.sum(weights**2 * unflagged_channels))
        ),
        "rows": nrow,
    }


def _differences(block: dict, interval: float) -> dict:
    """
    Differences of consecutive integrations of each baseline, correlation and channel, by
    (antenna1, antenna2, data description), with the SIGMA of the visibilities differenced
    """
    differences = {}
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
        sigma = np.broadcast_to(block["sigma"][:, None, first], delta.shape)
        baseline, ddid = divmod(int(key), 1000)
        differences[(*divmod(baseline, 100000), ddid)] = (delta[both], sigma[both])
    return differences


def _noise_over_sigma(per_baseline: dict, names: np.ndarray, dd_spw: np.ndarray) -> tuple:
    """
    Each baseline and spectral window's measured noise over the noise its SIGMA states. A
    difference of two visibilities has sqrt(2) times their noise in each of its real and
    imaginary parts; the visibilities are independent, so a plain standard deviation is right
    """
    ratios = []
    for (antenna1, antenna2, ddid), pairs in sorted(per_baseline.items()):
        delta = np.concatenate([delta for delta, _ in pairs])
        sigma = np.concatenate([sigma for _, sigma in pairs])
        if delta.size < MIN_DIFFERENCES or not np.median(sigma) > 0:
            continue
        measured = np.std(np.concatenate([delta.real, delta.imag])) / math.sqrt(2)
        ratios.append(
            (
                str(names[antenna1]), str(names[antenna2]), int(dd_spw[ddid]),
                float(measured / np.median(sigma))
            )
        )
    return tuple(ratios)


def _antenna_names(ms: str) -> np.ndarray:
    with open_table(os.path.join(ms, "ANTENNA")) as antenna_table:
        return np.asarray(antenna_table.getcol("NAME"))


def weight_anomalies(sample: VisibilitySample) -> str:
    """
    Why the data weights misstate the noise, or "" if they don't (#129): the median over baselines
    of measured noise over SIGMA more than WEIGHTS_TOLERANCE from 1, and the baselines whose ratio
    stands out (a robust z-score above OUTLIER_Z, and WEIGHTS_TOLERANCE from the median)
    """
    if not sample.noise_over_sigma:
        return ""
    ratios = np.array([ratio for *_, ratio in sample.noise_over_sigma])
    median = float(np.median(ratios))
    spread = MAD_TO_SIGMA * float(np.median(np.abs(ratios - median)))
    outliers = [
        "{0}-{1} spw {2} ({3:.2f})".format(*entry) for entry in sample.noise_over_sigma
        if abs(entry[3] - median) > max(OUTLIER_Z * spread, WEIGHTS_TOLERANCE * median)
    ]
    problems = []
    if abs(median - 1) > WEIGHTS_TOLERANCE:
        problems.append(
            "the data's noise is {0:.2f} times the noise their weights (SIGMA) state, on the "
            "median of {1} baselines: the weights {2} it, which skews the solve, the predicted "
            "S/N and imagers that use the weights".format(
                median, len(ratios), "understate" if median > 1 else "overstate"
            )
        )
    if outliers:
        problems.append(
            "{0} of {1} baselines' noise doesn't match their weights: {2}".format(
                len(outliers), len(ratios),
                ", ".join(outliers[:10]) + (", ..." if len(outliers) > 10 else "")
            )
        )
    if problems:
        problems.append("statwt, or Pyralysis's EmpiricalWeightEstimator, can re-estimate them")
    return "; ".join(problems)


def _selected_channels(
    indices: dict, dd_spw: np.ndarray, all_channels: np.ndarray, data_ddids: np.ndarray
) -> list:
    """
    Channels of each selected spectral window, from the channel ranges of the spw selection. An
    empty selection means the spectral windows that hold data (``data_ddids``, the data
    descriptions of the selected rows), not every SPECTRAL_WINDOW row: data split from a larger
    measurement set keep rows for windows they have no data in (#115)
    """
    ranges = indices["channel"]
    if len(ranges):
        counts = {}
        for spw, start, end, step in ranges:
            counts[spw] = counts.get(spw, 0) + len(range(start, end + 1, max(step, 1)))
        return [counts[spw] for spw in sorted(counts)]
    return [int(all_channels[spw]) for spw in _selected_spws(indices, dd_spw, data_ddids)]


def _selected_spws(indices: dict, dd_spw: np.ndarray, data_ddids: np.ndarray) -> list:
    """
    The selected spectral windows; for an empty selection, those that hold data (#115)
    """
    ddids = indices["dd"] if len(indices["dd"]) else data_ddids
    return sorted({int(dd_spw[dd]) for dd in ddids})
