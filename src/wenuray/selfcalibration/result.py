"""
What a self-calibration run produced (#22): the accepted calibration stack and a record of every
round, written as TOML next to the run's outputs so the run can be reproduced and resumed.
"""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from typing import Mapping, Optional, Tuple

import tomli_w

from .caltable import CalStack


@dataclass(frozen=True)
class RoundRecord:
    """
    One round: the initial image (index 0) or a calibration step followed by imaging

    Parameters
    ----------
    index :
        0 for the initial image, then 1, 2, ... for the steps
    step :
        Description of the step, e.g. "Phasecal('60s')"; empty for the initial image
    caltable :
        Calibration table solved in the round; empty for the initial image
    image :
        Restored image of the round
    psnr, peak, rms, model_flux :
        Image statistics, as computed by ImageStatistics
    beam :
        Restoring beam (major, minor) in arcsec and position angle in degrees, if known
    flagged_fraction :
        Fraction of the round's gain solutions that were flagged
    accepted :
        Whether the round was kept
    reasons :
        Why the acceptance policy decided as it did
    data_flagged :
        Flagged fraction of the selected data before and after the step's residual flagging, if
        it flagged
    predicted_snr :
        The step's solution S/N predicted from the data before solving (#28), if it could be
    """
    index: int
    step: str
    caltable: str
    image: str
    psnr: float
    peak: float
    rms: float
    model_flux: float
    beam: Optional[Tuple[float, float, float]] = None
    flagged_fraction: float = 0.0
    accepted: bool = True
    reasons: Tuple[str, ...] = ()
    data_flagged: Optional[Tuple[float, float]] = None
    predicted_snr: Optional[float] = None

    def __post_init__(self):
        if self.beam is not None:
            object.__setattr__(self, "beam", tuple(self.beam))
        if self.data_flagged is not None:
            object.__setattr__(self, "data_flagged", tuple(self.data_flagged))
        object.__setattr__(self, "reasons", tuple(self.reasons))


@dataclass(frozen=True)
class SelfcalResult:
    """
    The outcome of Selfcal.run()

    Parameters
    ----------
    ms :
        The self-calibrated measurement set. CORRECTED_DATA holds DATA calibrated with the stack
        and MODEL_DATA the model of the last accepted round
    stack :
        The accepted calibration tables, in application order
    rounds :
        Every round, accepted or not, in the order they ran
    provenance :
        How the run can be reproduced (#36): when it started, the recipe written next to the
        result, a fingerprint of the input data (ms_fingerprint) and the software versions
    """
    ms: str
    stack: CalStack
    rounds: Tuple[RoundRecord, ...] = field(default=())
    provenance: Mapping = field(default_factory=dict, hash=False)

    def __post_init__(self):
        object.__setattr__(self, "rounds", tuple(self.rounds))

    @property
    def accepted_rounds(self) -> Tuple[RoundRecord, ...]:
        return tuple(record for record in self.rounds if record.accepted)

    @property
    def final(self) -> RoundRecord:
        """
        The last accepted round, whose image and model are the run's result
        """
        return self.accepted_rounds[-1]

    def to_toml(self, path: str) -> None:
        """
        Writes the result to a TOML file. Paths are written relative to the file's directory, so
        the run's directory can be moved as a whole
        """
        base = os.path.dirname(os.path.abspath(path))

        def relative(target: str) -> str:
            return os.path.relpath(os.path.abspath(target), base) if target else ""

        rounds = []
        for record in self.rounds:
            entry = {
                "index": record.index,
                "step": record.step,
                "caltable": relative(record.caltable),
                "image": relative(record.image),
                "psnr": record.psnr,
                "peak": record.peak,
                "rms": record.rms,
                "model_flux": record.model_flux,
                "flagged_fraction": record.flagged_fraction,
                "accepted": record.accepted,
                "reasons": list(record.reasons),
            }
            if record.beam is not None:
                entry["beam"] = list(record.beam)
            if record.data_flagged is not None:
                entry["data_flagged"] = list(record.data_flagged)
            if record.predicted_snr is not None:
                entry["predicted_snr"] = record.predicted_snr
            rounds.append(entry)
        data = {
            "ms": relative(self.ms),
            "stack": self.stack.to_dict(relative_to=base),
            "rounds": rounds
        }
        if self.provenance:
            data["provenance"] = dict(self.provenance)
        with open(path, "wb") as file:
            tomli_w.dump(data, file)

    @classmethod
    def from_toml(cls, path: str) -> SelfcalResult:
        """
        Reads a result written by to_toml
        """
        base = os.path.dirname(os.path.abspath(path))

        def absolute(target: str) -> str:
            return os.path.normpath(os.path.join(base, target)) if target else ""

        with open(path, "rb") as file:
            data = tomllib.load(file)
        rounds = [
            RoundRecord(
                **{
                    **entry, "caltable": absolute(entry["caltable"]),
                    "image": absolute(entry["image"])
                }
            ) for entry in data.get("rounds", [])
        ]
        return cls(
            absolute(data["ms"]),
            CalStack.from_dict(data["stack"], relative_to=base),
            rounds,
            data.get("provenance", {}),
        )
