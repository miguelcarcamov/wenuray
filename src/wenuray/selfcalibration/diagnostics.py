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
    """
    phase_over_noise: float
    amplitude_over_noise: Optional[float] = None
    outlying_antennas: Tuple[str, ...] = ()
    xy_amplitude: Optional[float] = None
    xy_phase_over_noise: Optional[float] = None

    def __post_init__(self):
        object.__setattr__(self, "outlying_antennas", tuple(self.outlying_antennas))


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
