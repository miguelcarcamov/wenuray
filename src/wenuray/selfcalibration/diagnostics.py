"""
What a round's gain table says about the round (#126), at almost no cost next to imaging: whether
its corrections stand above the noise of the solutions, which antennas stand out, and how the two
polarizations differ. Each table is the increment over the accepted stack, so its corrections
falling to the noise is the round converging.

The noise of a solution comes from the table's SNR column. If S is an antenna's own solution S/N,
its fractional amplitude noise is 1/S; its phase is relative to the reference antenna, which adds
the reference's noise, so sqrt(2)/S rad; and an X - Y phase difference carries both polarizations'
referenced noise, 2/S. CASA's SNR column is S for a phase-only table and S/sqrt(2) for one that
solves amplitudes too: on simulated noise with the true model, the same data give SNR 75 (calmode
"p") and 53 ("ap"), and these noises then read 0.88 to 1.05 of the measured scatter in every case
(tests/integration). Scatters are robust (1.4826 times the median absolute value), measured about
zero phase and unit amplitude: a correction shared by every antenna is still a correction.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ..utils.selfcal_utils import open_table

# 1.4826 MAD is the standard deviation of a Gaussian
MAD_TO_SIGMA = 1.4826
# An antenna stands out when the modified z-score of its scatter, against what its distance from
# the array centre predicts, exceeds 3.5 (Iglewicz & Hoaglin 1993)
OUTLIER_Z = 3.5
# Fewer antennas than this with solutions and a distance give no fit of scatter against distance
MIN_ANTENNAS_FOR_FIT = 4
# The relative standard error of a MAD scatter of n Gaussian values is this over sqrt(n): the MAD
# is 37% efficient against the standard deviation, whose relative error is 1 / sqrt(2 n)
MAD_RELATIVE_ERROR = 1.167
# A scatter over its noise is a correction when it stands this many standard errors above 1
SIGNIFICANCE = 3.0


@dataclass(frozen=True)
class GainDiagnostics:
    """
    Diagnostics of one round's gain table

    Parameters
    ----------
    phase_over_noise :
        Robust phase scatter of the solutions over their phase noise: about 1 when the round
        corrected nothing beyond noise
    amplitude_over_noise :
        The same for the amplitudes, about 1; None for phase-only tables
    outlying_antennas :
        Antennas whose phase scatter stands out from what their distance from the array centre
        predicts (atmospheric phase grows with baseline length, so the outer antennas are
        expected to scatter more)
    xy_amplitude :
        Median of ``|amp_X / amp_Y - 1|`` for tables that solve each polarization and amplitudes;
        None otherwise
    xy_phase_over_noise :
        Robust X - Y phase scatter over its noise, for tables that solve each polarization: well
        above 1 when solving them separately (gaintype G) is justified
    solutions :
        Number of solutions the scatters are measured on, which decides how far above 1 a scatter
        must be to be a correction (above_noise); None in records written before it was kept
    """
    phase_over_noise: float
    amplitude_over_noise: Optional[float] = None
    outlying_antennas: Tuple[str, ...] = ()
    xy_amplitude: Optional[float] = None
    xy_phase_over_noise: Optional[float] = None
    solutions: Optional[int] = None

    def __post_init__(self):
        object.__setattr__(self, "outlying_antennas", tuple(self.outlying_antennas))

    def above_noise(self, ratio: float) -> Optional[bool]:
        """
        Whether a scatter over its noise, one of this table's, stands above what noise alone gives
        on this many solutions: more than SIGNIFICANCE standard errors of a MAD scatter above 1.
        None when the number of solutions is unknown
        """
        if not self.solutions:
            return None
        return ratio > 1.0 + SIGNIFICANCE * MAD_RELATIVE_ERROR / math.sqrt(self.solutions)


@dataclass(frozen=True)
class SolutionCoverage:
    """
    How completely a round's gains were solved: how many solutions failed, on which antennas, and
    how many solution intervals kept too few antennas for the gains to be determined

    Parameters
    ----------
    solutions :
        Antenna solutions in the table, one per antenna, solution interval and spectral window
    failed :
        Of those, the ones gaincal flagged: below minsnr, or without data. applycal (calflag)
        flags the data they would have calibrated
    intervals :
        Solution intervals, one per time and spectral window
    short_intervals :
        Of those, the ones with fewer than min_antennas solved antennas
    min_antennas :
        Antennas the step needs per interval: 3 for phases (closure phases), 4 for amplitudes
        (closure amplitudes)
    minsnr :
        The solve's minimum solution S/N
    failed_antennas :
        The antennas with failed solutions, most failures first
    spws :
        The spectral windows the table has solutions for
    uncalibrated_spws :
        Selected spectral windows with unflagged data but no solutions, in a table solved per
        spectral window: applycal leaves their data uncalibrated, without flagging them
    empty_spws :
        The measurement set's other spectral windows without solutions, which hold no unflagged
        data (e.g. a split that keeps its original ids, or a flagged window): CASA reports them
        ("no corresponding cal spws"), and nothing is lost
    """
    solutions: int
    failed: int
    intervals: int
    short_intervals: int
    min_antennas: int
    minsnr: float
    failed_antennas: Tuple[str, ...] = ()
    spws: Tuple[int, ...] = ()
    uncalibrated_spws: Tuple[int, ...] = ()
    empty_spws: Tuple[int, ...] = ()

    def __post_init__(self):
        for name in ("failed_antennas", "spws", "uncalibrated_spws", "empty_spws"):
            object.__setattr__(self, name, tuple(getattr(self, name)))


def solution_coverage(caltable: str, min_antennas: int,
                      minsnr: float) -> Optional[SolutionCoverage]:
    """
    The coverage of a gain table's solutions; None if it holds none
    """
    with open_table(caltable) as table:
        if table.nrows() == 0:
            return None
        flags = table.getcol("FLAG")  # (polarization, channel, row)
        times = table.getcol("TIME")
        spws = table.getcol("SPECTRAL_WINDOW_ID")
        antennas = table.getcol("ANTENNA1")
    with open_table(os.path.join(caltable, "ANTENNA")) as antenna_table:
        names = np.asarray(antenna_table.getcol("NAME"))
    return cover(flags, times, spws, antennas, names, min_antennas, minsnr)


def cover(
    flags: np.ndarray, times: np.ndarray, spws: np.ndarray, antennas: np.ndarray, names: np.ndarray,
    min_antennas: int, minsnr: float
) -> SolutionCoverage:
    """
    The coverage from a gain table's columns: flags are (polarization, channel, row). A solution
    is solved when any of its polarizations or channels is unflagged
    """
    solved = ~np.all(flags, axis=(0, 1))
    intervals = {}
    for time, spw, antenna, ok in zip(times, spws, antennas, solved):
        interval = intervals.setdefault((float(time), int(spw)), set())
        if ok:
            interval.add(int(antenna))
    failures = {}
    for antenna in antennas[~solved]:
        failures[int(antenna)] = failures.get(int(antenna), 0) + 1
    return SolutionCoverage(
        solutions=len(solved),
        failed=int(np.count_nonzero(~solved)),
        intervals=len(intervals),
        short_intervals=sum(
            len(solved_antennas) < min_antennas for solved_antennas in intervals.values()
        ),
        min_antennas=min_antennas,
        minsnr=float(minsnr),
        failed_antennas=tuple(
            str(names[antenna]) if 0 <= antenna < len(names) else str(antenna)
            for antenna, _ in sorted(failures.items(), key=lambda item: (-item[1], item[0]))
        ),
        spws=tuple(sorted({int(spw)
                           for spw in spws})),
    )


def gain_diagnostics(caltable: str, solves_amplitude: bool) -> Optional[GainDiagnostics]:
    """
    The diagnostics of a gain table; None if it holds no unflagged solutions with a signal-to-noise
    ratio
    """
    with open_table(caltable) as table:
        if table.nrows() == 0:
            return None
        gains = table.getcol("CPARAM")
        flags = table.getcol("FLAG")
        snr = table.getcol("SNR")
        antenna1 = table.getcol("ANTENNA1")
        antenna2 = table.getcol("ANTENNA2")
    with open_table(os.path.join(caltable, "ANTENNA")) as antenna_table:
        names = np.asarray(antenna_table.getcol("NAME"))
        positions = np.asarray(antenna_table.getcol("POSITION")).T  # (antenna, xyz)
    return diagnose(gains, flags, snr, antenna1, antenna2, names, positions, solves_amplitude)


def diagnose(
    gains: np.ndarray,
    flags: np.ndarray,
    snr: np.ndarray,
    antenna1: np.ndarray,
    antenna2: np.ndarray,
    names: np.ndarray,
    positions: np.ndarray,
    solves_amplitude: bool,
) -> Optional[GainDiagnostics]:
    """
    The diagnostics from a gain table's columns: gains, flags and snr are (polarization, channel,
    row); antenna2 is the reference antenna, whose own solutions are fixed and left out; positions
    are (antenna, xyz) in metres
    """
    usable = ~flags & (snr > 0) & (antenna1 != antenna2)[None, None, :]
    if not usable.any():
        return None
    # Each antenna's own solution S/N (see the module docstring)
    own = snr * (math.sqrt(2) if solves_amplitude else 1.0)
    phase_noise = float(np.median(math.sqrt(2) / own[usable]))
    amplitude_noise = float(np.median(1 / own[usable]))
    phases = np.angle(gains)
    amplitudes = np.abs(gains)
    xy_amplitude = xy_phase = None
    if gains.shape[0] == 2:
        both = usable[0] & usable[1]
        if both.any():
            difference = np.angle(gains[0] * np.conj(gains[1]))[both]
            difference_noise = np.hypot(math.sqrt(2) / own[0][both], math.sqrt(2) / own[1][both])
            xy_phase = _rms(difference) / float(np.median(difference_noise))
            if solves_amplitude:
                xy_amplitude = float(
                    np.median(np.abs(amplitudes[0][both] / amplitudes[1][both] - 1))
                )
    return GainDiagnostics(
        phase_over_noise=_rms(phases[usable]) / phase_noise,
        amplitude_over_noise=(
            _rms(amplitudes[usable] - 1) / amplitude_noise if solves_amplitude else None
        ),
        outlying_antennas=_outlying_antennas(phases, usable, antenna1, antenna2, names, positions),
        xy_amplitude=xy_amplitude,
        xy_phase_over_noise=xy_phase,
        solutions=int(np.count_nonzero(usable)),
    )


def _rms(values: np.ndarray) -> float:
    """
    Robust scatter about zero
    """
    return MAD_TO_SIGMA * float(np.median(np.abs(values)))


def _outlying_antennas(
    phases: np.ndarray, usable: np.ndarray, antenna1: np.ndarray, antenna2: np.ndarray,
    names: np.ndarray, positions: np.ndarray
) -> Tuple[str, ...]:
    """
    Antennas whose phase scatter is high for their distance from the array centre: a power law
    of scatter against distance is fitted in log-log, and the residuals are judged by their
    modified z-score
    """
    solved = sorted({int(a) for a in antenna1[usable.any(axis=(0, 1))]})
    # The centre of the array the table covers, the reference antenna included
    present = np.unique(np.concatenate([antenna1, antenna2]))
    centre = positions[present[(present >= 0) & (present < len(positions))]].mean(axis=0)
    rows = []
    for antenna in solved:
        own = usable & (antenna1 == antenna)[None, None, :]
        scatter = _rms(phases[own])
        distance = float(np.linalg.norm(positions[antenna] - centre))
        if scatter > 0 and distance > 0:
            rows.append((antenna, np.log(distance), np.log(scatter)))
    if len(rows) < MIN_ANTENNAS_FOR_FIT:
        return ()
    antennas, log_distance, log_scatter = (np.array(column) for column in zip(*rows))
    slope, intercept = np.polyfit(log_distance, log_scatter, 1)
    residuals = log_scatter - (slope * log_distance + intercept)
    deviation = float(np.median(np.abs(residuals - np.median(residuals))))
    if deviation == 0:
        return ()
    z = 0.6745 * (residuals - np.median(residuals)) / deviation
    return tuple(str(names[antenna]) for antenna, score in zip(antennas, z) if score > OUTLIER_Z)
