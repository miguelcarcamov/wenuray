"""
Observers of a self-calibration run (#22): logging, gain plots and anything else that watches a run
without changing it. Pass them to Selfcal.run(observers=...); each hook does nothing unless
overridden.
"""
from __future__ import annotations

import logging
import math
import time
from typing import TYPE_CHECKING, Optional

import numpy as np

from ..utils.selfcal_utils import open_table
from .advisories import spw_list
from .caltable import CalTable

if TYPE_CHECKING:
    from .loop import Selfcal
    from .result import RoundRecord, SelfcalResult
    from .steps import CalibrationStep, Finding

logger = logging.getLogger(__name__)


class SelfcalObserver:
    """
    Base class of the observers; override the hooks you need
    """

    def on_start(self, selfcal: Selfcal) -> None:
        """
        Before anything is written
        """

    def on_phase(self, phase: str, seconds: Optional[float]) -> None:
        """
        When a phase of the run (resetting the data, solving, applying, imaging) starts, with
        seconds None, and when it ends, with how long it took (#176)
        """

    def on_solved(self, step: CalibrationStep, table: CalTable) -> None:
        """
        After gaincal wrote a step's table, before it is applied
        """

    def on_round(self, record: RoundRecord) -> None:
        """
        After a round was imaged and judged (index 0 is the initial image)
        """

    def on_finding(self, finding: Finding) -> None:
        """
        When an advisory found something (#141); the loop logs it and records it with its round
        """

    def on_finished(self, result: SelfcalResult) -> None:
        """
        After the run, with the measurement set in its final state
        """


class LoggingObserver(SelfcalObserver):
    """
    Logs the run at INFO through the `wenuray` logger, as a trace a radio astronomer can follow
    (#179): each phase as it starts and ends, a block per round (its solutions, its gains against
    their noise, its image, the decision), and a summary of every round at the end
    """

    def __init__(self):
        self._steps = 0
        self._started = None
        self._accepted = None

    def on_phase(self, phase: str, seconds: Optional[float]) -> None:
        if seconds is None:
            logger.info("%s", phase)
        else:
            logger.info("%s: done in %s", phase, duration(seconds))

    def on_start(self, selfcal: Selfcal) -> None:
        self._steps, self._started, self._accepted = len(selfcal.steps), time.monotonic(), None
        logger.info(
            "Self-calibration of %s: %d steps imaged with %s%s, output in %s", selfcal.ms,
            len(selfcal.steps),
            type(selfcal.imager).__name__,
            ", after an initial stack of {0} tables".format(len(selfcal.initial_stack))
            if len(selfcal.initial_stack) else "", selfcal.output
        )

    def on_round(self, record: RoundRecord) -> None:
        logger.info("%s", round_block(record, self._steps, self._accepted))
        if record.accepted:
            self._accepted = record

    def on_finished(self, result: SelfcalResult) -> None:
        seconds = time.monotonic() - self._started if self._started is not None else None
        logger.info("%s", summary(result, seconds))


# The width of a round block's labels
LABEL = 10


def round_block(record: RoundRecord, steps: int, previous: Optional[RoundRecord]) -> str:
    """
    A round as a block of lines: what was solved and how completely, the gains against their
    noise, the image, and the decision with its reasons. previous is the last accepted round, the
    one this round is judged against; steps the number of steps in the run
    """
    if record.index == 0:
        lines = ["Round 0 initial image"]
    else:
        lines = [
            "Round {0}/{1} {2}: {3}".format(
                record.index, steps or "?", record.step,
                "accepted" if record.accepted else "rejected"
            )
        ]
    if record.solutions is not None:
        lines += _labelled("solutions", _solutions(record.solutions))
    if record.gains is not None:
        lines += _labelled("gains", [_gains(record.gains)])
    if not math.isnan(record.psnr):
        lines += _labelled("image", [_image(record, previous)])
    if record.index > 0:
        lines += _labelled(
            "decision", [
                "{0}: {1}".format(
                    "accepted" if record.accepted else "rejected", "; ".join(record.reasons)
                )
            ]
        )
        hint = _hint(record)
        if hint:
            lines += _labelled("hint", [hint])
    return "\n".join(lines)


def summary(result: SelfcalResult, seconds: Optional[float] = None) -> str:
    """
    Every round of a run in a table, and the final image
    """
    steps = len(result.rounds) - 1
    accepted = len(result.accepted_rounds) - 1
    rows = [("Round", "Step", "PSNR", "rms/thermal", "Model flux", "Gains/noise", "Decision")]
    for record in result.rounds:
        gains = "-"
        if record.gains is not None:
            gains = "{0:.1f}".format(record.gains.phase_over_noise)
            if record.gains.amplitude_over_noise is not None:
                gains += " / {0:.1f}".format(record.gains.amplitude_over_noise)
        rows.append(
            (
                str(record.index),
                record.step or "initial image",
                "-" if math.isnan(record.psnr) else "{0:.1f}".format(record.psnr),
                "{0:.2f}".format(record.rms / record.thermal_noise)
                if record.thermal_noise and not math.isnan(record.rms) else "-",
                "-" if math.isnan(record.model_flux) else "{0:.4g} Jy".format(record.model_flux),
                gains,
                "accepted" if record.accepted else "rejected",
            )
        )
    widths = [max(len(row[column]) for row in rows) for column in range(len(rows[0]))]
    final = result.final
    lines = [
        "Self-calibration finished{0}: {1} of {2} steps accepted".
        format("" if seconds is None else " in " + duration(seconds), accepted, steps)
    ] + [
        "  " + "  ".join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip()
        for row in rows
    ]
    noise = (
        ", rms {0:.2f} x thermal".format(final.rms /
                                         final.thermal_noise) if final.thermal_noise else ""
    )
    lines.append(
        "Final image: {0} (round {1}), PSNR {2:.1f}{3}".format(
            final.image, final.index, final.psnr, noise
        )
    )
    return "\n".join(lines)


def _labelled(label: str, texts) -> list:
    """
    Lines under a label, the label on the first only
    """
    return [
        "  {0} {1}".format((label if number == 0 else "").ljust(LABEL), text)
        for number, text in enumerate(texts)
    ]


def _solutions(coverage) -> list:
    counted = "{0} solutions over {1} interval{2}".format(
        coverage.solutions, coverage.intervals, "" if coverage.intervals == 1 else "s"
    )
    if coverage.failed == 0:
        texts = [counted + ", all with S/N above {0:g}".format(coverage.minsnr)]
    else:
        texts = [
            counted + "; {0} ({1:.1%}) failed (S/N below {2:g}, or no data), most on {3}: "
            "applycal flags the data they calibrate".format(
                coverage.failed, coverage.failed / coverage.solutions, coverage.minsnr,
                ", ".join(coverage.failed_antennas[:3])
            )
        ]
    if coverage.uncalibrated_spws:
        texts.append(
            "{0} unflagged data but no solutions: applycal leaves {1} uncalibrated".format(
                _spws_hold(coverage.uncalibrated_spws),
                "it" if len(coverage.uncalibrated_spws) == 1 else "them"
            )
        )
    if coverage.empty_spws:
        texts.append(
            "{0} no unflagged data: CASA's \"no corresponding cal spws\" about {1} is "
            "harmless".format(
                _spws_hold(coverage.empty_spws), "it" if len(coverage.empty_spws) == 1 else "them"
            )
        )
    # Enough antennas everywhere is the decision's to say; too few is what the solutions show
    if coverage.short_intervals:
        texts.append(
            "{0} of {1} intervals have fewer than {2} antennas: their gains are undetermined".
            format(coverage.short_intervals, coverage.intervals, coverage.min_antennas)
        )
    return texts


def _spws_hold(spws) -> str:
    return ("spw {0} holds" if len(spws) == 1 else "spws {0} hold").format(spw_list(spws))


def _gains(gains) -> str:
    """
    Each scatter over its noise, and what it means: a correction, or noise
    """

    def judged(name: str, ratio: float) -> str:
        text = "{0} scatter {1:.1f} x noise".format(name, ratio)
        above = gains.above_noise(ratio)
        if above is None:
            return text
        return text + (
            ": a correction"
            if above else ": consistent with noise on {0} solutions".format(gains.solutions)
        )

    parts = [judged("phase", gains.phase_over_noise)]
    if gains.amplitude_over_noise is not None:
        parts.append(judged("amplitude", gains.amplitude_over_noise))
    if gains.xy_phase_over_noise is not None:
        parts.append(judged("X-Y phase", gains.xy_phase_over_noise))
    if gains.xy_amplitude is not None:
        parts.append("X/Y amplitude {0:.1%} from 1".format(gains.xy_amplitude))
    text = "; ".join(parts)
    if gains.outlying_antennas:
        text += "; {0} scatter more than their distance from the centre predicts".format(
            ", ".join(gains.outlying_antennas)
        )
    return text


def _image(record: RoundRecord, previous: Optional[RoundRecord]) -> str:
    text = "PSNR {0:.1f}".format(record.psnr)
    if previous is not None and previous.psnr > 0:
        text += " ({0:+.1%} on round {1})".format(record.psnr / previous.psnr - 1, previous.index)
    text += ", rms {0:.3g} mJy/beam".format(record.rms * 1e3)
    if record.thermal_noise:
        text += " ({0:.2f} x thermal".format(record.rms / record.thermal_noise)
        if record.rms_in_mask is not None:
            text += ", {0:.2f} x inside the source mask".format(
                record.rms_in_mask / record.thermal_noise
            )
        text += ")"
    return text + ", model flux {0:.4g} Jy".format(record.model_flux)


def _hint(record: RoundRecord) -> Optional[str]:
    """
    What to try when a round was rejected for lack of S/N in its solutions; nothing otherwise
    """
    coverage = record.solutions
    starved = coverage is None or coverage.short_intervals > 0 or coverage.failed > 0
    if record.accepted or not starved:
        return None
    return 'a longer solint, or combining spectral windows (combine = "spw"), gives each solution more S/N'


class GainPlotObserver(SelfcalObserver):
    """
    Plots each solved table next to it (<table>.png): one line per antenna against time, the
    phase for phase steps and the amplitude for amplitude steps (the step's plot_yaxis). Needs
    matplotlib (pip install "wenuray[plots]").
    """

    def on_solved(self, step: CalibrationStep, table: CalTable) -> None:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt

        with open_table(table.path) as cal_table:
            gains = cal_table.getcol("CPARAM")  # (polarization, channel, row)
            flags = cal_table.getcol("FLAG")
            times = cal_table.getcol("TIME")
            antennas = cal_table.getcol("ANTENNA1")
        gains = np.where(flags, np.nan, gains)[:, 0, :]  # first channel
        values = np.degrees(np.angle(gains)) if step.plot_yaxis == "phase" else np.abs(gains)
        hours = (times - times.min()) / 3600.0
        figure, axes = plt.subplots(figsize=(8, 4))
        try:
            for antenna in np.unique(antennas):
                rows = antennas == antenna
                for polarization in range(values.shape[0]):
                    axes.plot(hours[rows], values[polarization, rows], ".", markersize=3)
            axes.set_xlabel("Time from start [h]")
            axes.set_ylabel("Gain phase [deg]" if step.plot_yaxis == "phase" else "Gain amplitude")
            axes.set_title("{0}({1!r})".format(type(step).__name__, step.solint))
            figure.savefig(table.path + ".png", dpi=100, bbox_inches="tight")
        finally:
            plt.close(figure)


def duration(seconds: float) -> str:
    """
    A duration as people read it: "8 s", "3 min 12 s", "1 h 05 min"
    """
    seconds = int(round(seconds))
    if seconds < 60:
        return "{0} s".format(seconds)
    if seconds < 3600:
        return "{0} min {1:02d} s".format(seconds // 60, seconds % 60)
    return "{0} h {1:02d} min".format(seconds // 3600, seconds % 3600 // 60)
