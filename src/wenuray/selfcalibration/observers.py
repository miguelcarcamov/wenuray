"""
Observers of a self-calibration run (#22): logging, gain plots and anything else that watches a run
without changing it. Pass them to Selfcal.run(observers=...); each hook does nothing unless
overridden.
"""
from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import numpy as np

from ..utils.selfcal_utils import open_table
from .caltable import CalTable

if TYPE_CHECKING:
    from .loop import Selfcal
    from .result import RoundRecord, SelfcalResult
    from .steps import CalibrationStep

logger = logging.getLogger(__name__)


class SelfcalObserver:
    """
    Base class of the observers; override the hooks you need
    """

    def on_start(self, selfcal: Selfcal) -> None:
        """
        Before anything is written
        """

    def on_solved(self, step: CalibrationStep, table: CalTable) -> None:
        """
        After gaincal wrote a step's table, before it is applied
        """

    def on_round(self, record: RoundRecord) -> None:
        """
        After a round was imaged and judged (index 0 is the initial image)
        """

    def on_finished(self, result: SelfcalResult) -> None:
        """
        After the run, with the measurement set in its final state
        """


class LoggingObserver(SelfcalObserver):
    """
    Logs the run at INFO through the `wenuray` logger: each round's metrics and decision
    """

    def on_start(self, selfcal: Selfcal) -> None:
        logger.info(
            "Self-calibration of %s: %d steps, initial stack of %d tables, output in %s",
            selfcal.ms, len(selfcal.steps), len(selfcal.initial_stack), selfcal.output
        )

    def on_round(self, record: RoundRecord) -> None:
        if math.isnan(record.psnr):
            logger.info(
                "Round %d %s - rejected: %s", record.index, record.step, "; ".join(record.reasons)
            )
            return
        logger.info(
            "Round %d %s - PSNR: %.3f, peak: %.3f mJy/beam, rms: %.3f mJy/beam, model flux: %.4g Jy, "
            "flagged: %.1f%% - %s: %s", record.index, record.step or "initial image", record.psnr,
            record.peak * 1e3, record.rms * 1e3, record.model_flux, 100 * record.flagged_fraction,
            "accepted" if record.accepted else "rejected", "; ".join(record.reasons)
        )

    def on_finished(self, result: SelfcalResult) -> None:
        logger.info(
            "Self-calibration finished: %d of %d steps accepted, final PSNR %.3f",
            len(result.accepted_rounds) - 1,
            len(result.rounds) - 1, result.final.psnr
        )


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
