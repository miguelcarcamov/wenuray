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
from .diagnostics import MAD_RELATIVE_ERROR, SIGNIFICANCE
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
# solint="auto" (#66) takes the shortest solint whose predicted S/N reaches this many times minsnr,
# so that the solutions' own scatter about the prediction leaves most of them above minsnr
AUTO_SNR_FACTOR = 2.0


@dataclass(frozen=True)
class VisibilitySample:
    """
    What the estimate needs from the data

    Parameters
    ----------
    signal :
        Median amplitude of the model visibilities, in Jy
    noise :
        Noise of the real or imaginary part of one visibility, in Jy, from the estimate named by
        noise_from (#153)
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
    noise_from :
        Which differences the noise comes from: "consecutive integrations" of a baseline, or "the
        parallel hands" (XX - YY or RR - LL) of a visibility (#153)
    other_noise :
        The other estimate, (where from, noise), when both could be made
    scan_lengths :
        The duration of every selected scan, in s, which decides where solution intervals end
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
    noise_from: str = "consecutive integrations"
    other_noise: Optional[Tuple[str, float]] = None
    scan_lengths: Tuple[float, ...] = ()

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


def _solution_snr(sample: VisibilitySample, options: SolveOptions, solves_amplitude: bool) -> tuple:
    """
    The longest solint the options allow (a scan, or the observation when scans are combined) and
    the predicted solution S/N as a function of the integrations a solution spans: the S/N CASA
    compares with minsnr. For a table that solves amplitudes too, that is the antenna's S/N over
    sqrt(2): on the same simulated data, gaincal reports SNR 75 for calmode "p" and 53 for "ap"
    (diagnostics.py, tests/integration)
    """
    casa = 1.0 / math.sqrt(2) if solves_amplitude else 1.0
    combine = {axis.strip() for axis in options.combine.split(",")}
    longest = sample.span if "scan" in combine else sample.scan_length
    nchannels = sum(sample.channels) if "spw" in combine else min(sample.channels)
    npolarizations = sample.parallel_hands if options.gaintype == "T" else 1

    def snr(integrations: int) -> float:
        return casa * solution_snr(
            sample.signal, sample.noise, sample.nantennas, integrations, nchannels, npolarizations
        )

    return combine, longest, snr


def predict(sample: VisibilitySample, step: CalibrationStep, options: SolveOptions) -> SolutionSNR:
    """
    The solution S/N of the step with the run's solve options, from a sample of the data
    """
    options = step.solve_options(options)
    combine, longest, snr = _solution_snr(sample, options, step.solves_amplitude)
    seconds = step.solint_seconds
    nintegrations = max(1, round(min(seconds, longest) / sample.interval)) if seconds else 1
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


def longest_solint(sample: VisibilitySample, step: CalibrationStep, options: SolveOptions) -> float:
    """
    The longest solint the step's options allow, in s: a scan, or the observation when scans are
    combined. What "inf" means for these data
    """
    return _solution_snr(sample, step.solve_options(options), step.solves_amplitude)[1]


def shortest_solint(
    sample: VisibilitySample,
    step: CalibrationStep,
    options: SolveOptions,
    at_least: float = 0.0,
    at_most: float = math.inf
) -> Tuple[Optional[str], str]:
    """
    What solint="auto" resolves to (#66): the shortest solint whose predicted solution S/N reaches
    AUTO_SNR_FACTOR times the step's minsnr. The S/N grows as the square root of the integrations a
    solution spans, so that is the first whole number of integrations past (target / S/N of one)
    squared. If not even the longest solint reaches the target but it reaches minsnr, it is "inf";
    if it doesn't reach minsnr either, there is none.

    gaincal cuts each scan into intervals from its start and leaves the rest as a last, shorter
    one, whose S/N, 2 x minsnr x sqrt(r / n) for r of n integrations, falls below minsnr when r < n
    / 4, and whose data calflag then flags. So of the lengths from n to 2 n integrations, the solint
    is the shortest that leaves the fewest integrations in such remainders across the scans
    (_even_length).

    A Repeat bounds it, so that it shortens step by step (Repeat.max_shortening): at_least, in s,
    keeps it from being shorter than a fraction of the last round's (at least the longest: "inf"),
    and at_most keeps the lengthening that avoids short remainders below the round's ceiling.

    Returns
    -------
    The solint ("int", "96s" or "inf"), or None, and why, for the log
    """
    options = step.solve_options(options)
    combine, longest, snr = _solution_snr(sample, options, step.solves_amplitude)
    target = AUTO_SNR_FACTOR * options.minsnr
    per_integration = snr(1)
    # Rounded before the ceiling, so that a ratio that is a whole number in exact arithmetic,
    # e.g. (6 / 0.6)**2 = 100.00000000000001, doesn't take one more integration
    needed = max(1, math.ceil(round((target / per_integration)**2, 9)))
    measure = "CASA's S/N, the antenna's over sqrt(2) for amplitude solutions, " \
        if step.solves_amplitude else ""
    if at_least >= longest:
        at_longest = snr(max(1, round(longest / sample.interval)))
        if at_longest >= options.minsnr:
            return "inf", "a repeat starts at the longest solint, a whole {0}: S/N {1:.1f}".format(
                "observation" if "scan" in combine else "scan", at_longest
            )
        return None, "S/N {0:.1f} over a whole {1} is below minsnr {2:g}".format(
            at_longest, "observation" if "scan" in combine else "scan", options.minsnr
        )
    bounded = max(needed, math.ceil(round(at_least / sample.interval, 9)))
    if bounded == 1:
        return "int", "{0}S/N {1:.1f} in one integration reaches {2:g} (2 x minsnr)".format(
            measure, per_integration, target
        )
    if bounded * sample.interval <= longest:
        scans = _scan_integrations(sample, combine)
        ceiling = math.floor(round(at_most /
                                   sample.interval, 9)) if math.isfinite(at_most) else None
        length, lost = _even_length(needed, scans, start=bounded, ceiling=ceiling)
        why = "the shortest solint whose {0}S/N reaches {1:g} (2 x minsnr)".format(measure, target)
        if bounded > needed:
            why = (
                "{0} integrations, the repeat's shortest step from its last round; the S/N would "
                "support {1}".format(bounded, needed)
            )
        if length > bounded:
            why += (
                ", {0} integrations rather than {1}, so that fewer scans end in a solution too short "
                "to reach minsnr".format(length, bounded)
            )
        if lost:
            why += "; {0:.1%} of the integrations still do".format(lost / sum(scans))
        return "{0:g}s".format(length * sample.interval), why
    at_longest = snr(max(1, round(longest / sample.interval)))
    if at_longest >= options.minsnr:
        return "inf", "S/N {0:.1f} over a whole {1} reaches minsnr {2:g}, not {3:g}".format(
            at_longest, "observation" if "scan" in combine else "scan", options.minsnr, target
        )
    return None, "S/N {0:.1f} over a whole {1} is below minsnr {2:g}".format(
        at_longest, "observation" if "scan" in combine else "scan", options.minsnr
    )


def _scan_integrations(sample: VisibilitySample, combine: set) -> list:
    """
    The integrations of each stretch of data gaincal cuts into solution intervals: each scan, or
    the whole observation when scans are combined
    """
    if "scan" in combine:
        lengths = [sample.span]
    else:
        lengths = list(sample.scan_lengths) or [sample.scan_length]
    return [max(1, round(length / sample.interval)) for length in lengths]


def _even_length(
    needed: int,
    scans: list,
    start: Optional[int] = None,
    ceiling: Optional[int] = None
) -> Tuple[int, int]:
    """
    The solint, in integrations from start (needed by default) to 2 x start, and no more than
    ceiling, that leaves the fewest integrations in solutions too short to reach minsnr, the
    shortest of those that tie, and how many it leaves. A remainder of r integrations reaches
    minsnr when 4 r >= needed (AUTO_SNR_FACTOR = 2, and the S/N grows as the square root of the
    integrations)
    """
    start = needed if start is None else start
    last = 2 * start if ceiling is None else max(start, min(2 * start, ceiling))

    def lost(length: int) -> int:
        return sum(
            remainder for remainder in (count % length for count in scans)
            if remainder and AUTO_SNR_FACTOR**2 * remainder < needed
        )

    best = min(range(start, last + 1), key=lambda length: (lost(length), length))
    return best, lost(best)


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
    estimates = {}
    for name, differenced in (
        ("consecutive integrations", lambda block: _differences(block, interval)),
        ("the parallel hands", _hand_differences),
    ):
        per_baseline = {}
        for block in blocks:
            for key, (delta, sigma) in differenced(block).items():
                per_baseline.setdefault(key, []).append((delta, sigma))
        noise, count = _difference_noise(per_baseline)
        if noise is not None:
            estimates[name] = (noise, count, per_baseline)
    antennas = np.unique(np.concatenate([block["antennas"] for block in blocks]))
    if model.size == 0 or not estimates or not np.any(model > 0) or not weight_sum > 0:
        return None
    noise_from = _least_raised(estimates)
    noise, _, per_baseline = estimates[noise_from]
    others = [(name, float(value[0])) for name, value in estimates.items() if name != noise_from]
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
        noise_from=noise_from,
        other_noise=others[0] if others else None,
        scan_lengths=tuple(float(length) for length in scan_lengths),
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


def _hand_differences(block: dict) -> dict:
    """
    Differences of the parallel hands (XX - YY or RR - LL) of each visibility, by (antenna1,
    antenna2, data description), with the noise the hands' SIGMA state together, the root mean
    square of the two. An unpolarized source cancels, and so do the antenna gains wherever they are
    the same in both hands, as the atmosphere's are, however fast they change (#153). None with
    one hand
    """
    if block["parallel_hands"] < 2:
        return {}
    differences = {}
    keys = block["baseline"] * 1000 + block["ddid"]
    both = ~block["flags"][0] & ~block["flags"][1]
    delta = block["data"][1] - block["data"][0]
    sigma = np.broadcast_to(
        np.sqrt((block["sigma"][0]**2 + block["sigma"][1]**2) / 2)[None, :], delta.shape
    )
    for key in np.unique(keys):
        rows = keys == key
        baseline, ddid = divmod(int(key), 1000)
        differences[(*divmod(baseline, 100000), ddid)
                    ] = (delta[:, rows][both[:, rows]], sigma[:, rows][both[:, rows]])
    return differences


def _difference_noise(per_baseline: dict) -> Tuple[Optional[float], int]:
    """
    The noise of the real or imaginary part of one visibility from differences of two
    visibilities, which have sqrt(2) times it, and how many components it was measured on: a robust
    (MAD) scatter. None with too few differences
    """
    differences = np.concatenate(
        [delta for pairs in per_baseline.values() for delta, _ in pairs]
        or [np.zeros(0, dtype=complex)]
    )
    if differences.size < 10:
        return None, 0
    components = np.concatenate([differences.real, differences.imag])
    noise = MAD_TO_SIGMA * np.median(np.abs(components - np.median(components))) / math.sqrt(2)
    return (float(noise), components.size) if noise > 0 else (None, 0)


def _least_raised(estimates: dict) -> str:
    """
    Which noise estimate to trust (#153). Each one measures the noise when the source cancels in
    its differences, and only a source that doesn't cancel raises it: gains that change within an
    integration raise the consecutive integrations', a polarized source or different gains in the
    two hands the parallel hands'. So the lower one is trusted, but only when it is lower by more
    than SIGNIFICANCE standard errors of a MAD estimate (MAD_RELATIVE_ERROR / sqrt(n) of it): the
    lower of two equal estimates is otherwise a little low, by chance
    """
    default = "consecutive integrations"
    if len(estimates) < 2 or default not in estimates:
        return next(iter(estimates))
    (time_noise, time_count, _), (hand_noise, hand_count,
                                  _) = (estimates[default], estimates["the parallel hands"])
    error = math.hypot(
        time_noise * MAD_RELATIVE_ERROR / math.sqrt(time_count),
        hand_noise * MAD_RELATIVE_ERROR / math.sqrt(hand_count)
    )
    return "the parallel hands" if time_noise - hand_noise > SIGNIFICANCE * error else default


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
