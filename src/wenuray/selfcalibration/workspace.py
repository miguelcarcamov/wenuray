"""
The measurement-set side of a self-calibration run (#22): snapshots of the flags, and putting
CORRECTED_DATA back to DATA calibrated with an accepted stack. Rolling back a round therefore needs
neither a copy of the measurement set nor extra columns.
"""
from __future__ import annotations

import dataclasses
import logging

from casatasks import applycal, clearcal, flagmanager

from ..config import ApplyOptions, Selection
from .caltable import CalStack

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class MSWorkspace:
    """
    Parameters
    ----------
    ms :
        The measurement set being self-calibrated
    selection :
        The data selection shared by the solve, the apply and the imager
    apply :
        How tables are applied
    prefix :
        Prefix of the flag versions this run saves, so they don't clash with the user's
    """
    ms: str
    selection: Selection
    apply: ApplyOptions
    prefix: str = "wenuray_selfcal"

    def _version(self, name: str) -> str:
        return "{0}_{1}".format(self.prefix, name)

    def snapshot(self, name: str) -> None:
        """
        Saves the current flags under name
        """
        flagmanager(vis=self.ms, mode="save", versionname=self._version(name))

    def apply_stack(self, stack: CalStack) -> None:
        """
        Sets CORRECTED_DATA to DATA calibrated with stack: a copy of DATA when the stack is empty
        """
        if len(stack) == 0:
            clearcal(
                vis=self.ms, field=self.selection.field, spw=self.selection.spw, addmodel=False
            )
            return
        applycal(
            vis=self.ms,
            **dataclasses.asdict(self.selection),
            gainfield="",
            calwt=False,
            flagbackup=False,
            applymode=self.apply.applymode,
            **stack.gaincal_arguments()
        )

    def restore(self, name: str, stack: CalStack) -> None:
        """
        Puts the flags saved under name back and recalibrates CORRECTED_DATA with stack
        """
        logger.info("Restoring the flags of %s and the accepted calibration", name)
        flagmanager(vis=self.ms, mode="restore", versionname=self._version(name))
        self.apply_stack(stack)

    def discard(self, *names: str) -> None:
        """
        Deletes the flag versions saved by this run
        """
        for name in names:
            flagmanager(vis=self.ms, mode="delete", versionname=self._version(name))
