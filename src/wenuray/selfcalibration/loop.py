"""
The self-calibration loop (#22): the only one in Wenuray. Steps provide data and hooks, the imager
follows the contract of wenuray.imaging.contract, the acceptance policies decide, and CalStack builds
every table list::

    result = Selfcal(
        ms="target.ms",
        imager=Tclean(ImageGeometry(imsize=1024, cell="0.05arcsec"), niter=1000, nsigma=3.0),
        steps=[Phasecal("inf"), Phasecal("60s"), AmpPhasecal("inf")],
        solve=SolveOptions(refant="DA51"),
    ).run()

The loop::

    validate the schedule                         (construction, before any I/O)
    CORRECTED_DATA = DATA, image round 0
    for each step:
        solve against DATA with the accepted stack pre-applied
        apply the stack plus the new table
        image CORRECTED_DATA (the imager writes MODEL_DATA)
        accept: push the table and snapshot the flags
        reject: restore flags and calibration, re-image the accepted model, stop
"""
from __future__ import annotations

import logging
import os
import platform
from datetime import datetime, timezone
from importlib import metadata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple, Union

import tomli_w
from casatasks import applycal, gaincal
from multimethod import multimethod

from .. import recipe
from ..config import ApplyOptions, Selection, SolveOptions
from ..imaging.contract import Imager, ImagingRequest, ImagingResult
from ..utils.selfcal_utils import (
    columns_match, count_spectral_windows, get_spw_ids, is_column_in_ms, ms_fingerprint, open_table,
    solution_summary
)
from . import preflight
from .acceptance import AcceptancePolicy, RoundMetrics
from .flagging import NoFlagging
from .caltable import CalStack, CalTable, derive_spwmap
from .observers import LoggingObserver, SelfcalObserver
from .result import RoundRecord, SelfcalResult
from .schedule import ScheduleError, validate_schedule
from .steps import CalibrationStep
from .workspace import MSWorkspace

logger = logging.getLogger(__name__)

RESULT_FILE = "selfcal.toml"
# The run as a resolved recipe: `wenuray run <output>/recipe.toml --output <new>` reruns it
RECIPE_FILE = "recipe.toml"


@multimethod
def as_result(result: SelfcalResult) -> SelfcalResult:
    """
    A result to resume from: a SelfcalResult, or the path to its selfcal.toml or its directory
    """
    return result


@as_result.register
def _(result: Union[str, os.PathLike]) -> SelfcalResult:
    path = Path(result)
    return SelfcalResult.from_toml(str(path / RESULT_FILE if path.is_dir() else path))


@dataclass(frozen=True)
class Selfcal:
    """
    A self-calibration run. Construction validates everything and touches no file; run() does the
    work.

    Parameters
    ----------
    ms :
        Measurement set with the calibrated data in DATA. gaincal solves against DATA, so
        CORRECTED_DATA is reset to DATA when the run starts
    imager :
        The imager; it must follow the self-calibration contract
    steps :
        The calibration steps, in the order they run. Invalid orders raise ScheduleError here
    selection :
        One data selection for the solve, the apply and the imager
    solve :
        gaincal options shared by every step; each step may override them
    apply :
        applycal options
    acceptance :
        Policy for every step that doesn't set its own; None uses each step type's default
    output :
        Directory for the calibration tables, the images and the result record; by default
        <ms name>_selfcal next to the measurement set
    """
    ms: str
    imager: Imager
    steps: Tuple[CalibrationStep, ...]
    selection: Selection = Selection()
    solve: SolveOptions = SolveOptions()
    apply: ApplyOptions = ApplyOptions()
    acceptance: Optional[AcceptancePolicy] = None
    output: str = ""
    # Tables accepted before this run, set by resume()
    initial_stack: CalStack = field(default=CalStack(), repr=False)

    @classmethod
    def resume(
        cls, result: Union[SelfcalResult, str, os.PathLike], imager: Imager,
        steps: Sequence[CalibrationStep], **options
    ) -> Selfcal:
        """
        Continues an earlier run: the same measurement set, with that run's accepted tables applied
        before every new solve. The schedule is validated against them, so e.g. an amplitude step
        may come first if the earlier run solved phases.

        Parameters
        ----------
        result :
            The earlier run's SelfcalResult, or the path to its selfcal.toml or its output
        imager, steps :
            As for Selfcal
        options :
            selection, solve, apply, acceptance or output, as for Selfcal. By default the output is
            a new directory next to the earlier one

        Raises
        ------
        TypeError
            If options set ms or initial_stack, which come from the result
        ScheduleError
            If a table of the result is missing or was solved on another measurement set
        """
        previous = as_result(result)
        fixed = sorted({"ms", "initial_stack"} & set(options))
        if fixed:
            raise TypeError("resume() takes {0} from the result".format(fixed))
        _check_solved_on(previous)
        if not options.get("output"):
            options["output"] = _next_free(str(Path(previous.ms).with_suffix("")) + "_selfcal")
        return cls(previous.ms, imager, steps, initial_stack=previous.stack, **options)

    @classmethod
    def from_recipe(cls, table: Mapping) -> Selfcal:
        """
        Builds a run from a recipe table (wenuray.recipe) whose paths are absolute or relative to the
        working directory
        """
        table = recipe.decode_configs(
            table, selection=Selection, solve=SolveOptions, apply=ApplyOptions
        )
        for name in ("imager", "steps"):
            if name not in table:
                raise recipe.RecipeError("A recipe needs {0}".format(name))
        table["imager"] = recipe.decode("imager", table["imager"])
        table["steps"] = [recipe.decode("step", step) for step in table["steps"]]
        if "acceptance" in table:
            table["acceptance"] = recipe.decode("acceptance", table["acceptance"])
        if "initial_stack" in table:
            table["initial_stack"] = CalStack.from_dict(table["initial_stack"])
        return cls(**table)

    @classmethod
    def from_toml(cls, path: Union[str, os.PathLike]) -> Selfcal:
        """
        Builds a run from a TOML recipe and its presets; see wenuray.recipe. Paths in the recipe are
        relative to its directory
        """
        return cls.from_recipe(recipe.read(os.fspath(path)))

    def to_recipe(self, relative_to: Optional[str] = None, defaults: bool = False) -> dict:
        """
        The recipe table of the run, leaving out the defaults unless asked for them

        Parameters
        ----------
        relative_to :
            Directory the paths are written relative to; absolute paths when None
        defaults :
            Also write the values that equal their defaults: the fully resolved run
        """
        if defaults and not recipe.writes_defaults():
            with recipe.with_defaults():
                return self.to_recipe(relative_to)

        def path(target: str) -> str:
            target = os.path.abspath(target)
            return target if relative_to is None else os.path.relpath(target, relative_to)

        fields = recipe.fields_table(self)
        table = {"ms": path(self.ms)}
        if recipe.writes_defaults() or self.output != _default_output(self.ms):
            table["output"] = path(self.output)
        # The order a reader expects: what data, how to solve, how to image, then the steps
        for name in ("selection", "solve", "apply", "acceptance", "imager", "steps"):
            if name in fields:
                table[name] = fields[name]
        if len(self.initial_stack):
            table["initial_stack"] = self.initial_stack.to_dict(relative_to=relative_to)
        return table

    def to_toml(self, path: Union[str, os.PathLike]) -> None:
        """
        Writes the run as a TOML recipe that from_toml() builds back, with paths relative to the
        recipe's directory
        """
        path = os.path.abspath(os.fspath(path))
        table = self.to_recipe(relative_to=os.path.dirname(path))
        with open(path, "wb") as file:
            tomli_w.dump(table, file)

    def __post_init__(self):
        if not isinstance(self.imager, Imager):
            raise TypeError("imager must be an Imager, not {0!r}".format(self.imager))
        object.__setattr__(self, "steps", tuple(self.steps))
        validate_schedule(self.steps, self.initial_stack)
        for step in self.steps:
            # Unknown imaging overrides raise here, before anything runs
            self.imager.with_overrides(**step.imaging)
        if not self.output:
            object.__setattr__(self, "output", _default_output(self.ms))

    def run(self, observers: Sequence[SelfcalObserver] = (LoggingObserver(), )) -> SelfcalResult:
        """
        Runs the self-calibration and writes the result record to <output>/selfcal.toml

        Parameters
        ----------
        observers :
            Watch the run, e.g. LoggingObserver() (the default) and GainPlotObserver()

        Raises
        ------
        FileExistsError
            If output already holds a run; resume() or a new output continues it
        ScheduleError
            If a table of the initial stack is missing
        """
        record_path = os.path.join(self.output, RESULT_FILE)
        if os.path.exists(record_path):
            raise FileExistsError(
                "{0} already holds a self-calibration run: continue it with Selfcal.resume() or "
                "choose another output".format(self.output)
            )
        self._check_initial_stack()
        for observer in observers:
            observer.on_start(self)
        os.makedirs(self.output, exist_ok=True)
        provenance = self._provenance()
        workspace = MSWorkspace(self.ms, self.selection, self.apply)
        self._reset_corrected(workspace)
        stack = self.initial_stack

        result = self._image(self.imager, "round0")
        accepted = RoundMetrics(result.statistics())
        rounds = [self._record(0, None, "", result, accepted, True, ("initial image", ))]
        _notify(observers, "on_round", rounds[-1])
        workspace.snapshot("round0")
        snapshots = ["round0"]
        accepted_imager, accepted_name = self.imager, "round0"

        for index, step in enumerate(self.steps, start=1):
            name = "round{0}".format(index)
            imager = self.imager.with_overrides(**step.imaging)
            predicted = self._predict_snr(step)
            table = self._solve(step, stack, name)
            if table is None:
                # Nothing was applied or imaged, so there is nothing to restore
                rounds.append(
                    self._failed_record(index, step, "gaincal found no solutions", predicted)
                )
                _notify(observers, "on_round", rounds[-1])
                break
            _notify(observers, "on_solved", step, table)
            self._apply(stack, table)
            # Residuals against the model the gains were solved with; restored if rejected
            data_flagged = step.flagging.flag(self.ms, self.selection)
            result = self._image(imager, name)
            flagged, antennas = solution_summary(table.path)
            current = RoundMetrics(result.statistics(), step, flagged, antennas)
            decision = step.acceptance_policy(self.acceptance).decide(accepted, current)
            rounds.append(
                self._record(
                    index, step, table.path, result, current, decision.accepted, decision.reasons,
                    data_flagged, predicted
                )
            )
            _notify(observers, "on_round", rounds[-1])
            if not decision.accepted:
                workspace.restore(snapshots[-1], stack)
                # MODEL_DATA must hold the accepted model again
                self._image(accepted_imager, accepted_name + "_model")
                break
            stack = stack.push(table)
            workspace.snapshot(name)
            snapshots.append(name)
            accepted, accepted_imager, accepted_name = current, imager, name

        workspace.discard(*snapshots)
        selfcal_result = SelfcalResult(self.ms, stack, rounds, provenance)
        selfcal_result.to_toml(record_path)
        _notify(observers, "on_finished", selfcal_result)
        return selfcal_result

    def _provenance(self) -> dict:
        """
        Writes the run as a resolved recipe, every value included, to <output>/recipe.toml before
        anything runs, and returns what the result records to reproduce it
        """
        recipe_path = os.path.join(self.output, RECIPE_FILE)
        try:
            table = self.to_recipe(relative_to=self.output, defaults=True)
        except recipe.RecipeError as error:
            logger.warning("The run can't be written as a recipe, so it isn't recorded: %s", error)
            recipe_path = ""
        else:
            with open(recipe_path, "wb") as file:
                tomli_w.dump(table, file)
        versions = {"python": platform.python_version()}
        for distribution in ("wenuray", "casatools", "casatasks", "numpy", "astropy"):
            try:
                versions[distribution] = metadata.version(distribution)
            except metadata.PackageNotFoundError:
                pass
        versions.update(self.imager.provenance())
        return {
            "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "recipe": os.path.basename(recipe_path),
            "ms_fingerprint": ms_fingerprint(self.ms),
            "versions": versions,
        }

    def _check_initial_stack(self) -> None:
        """
        The tables to continue from must exist. Tables of your own may come from another
        measurement set, e.g. the one this was split from; resume() checks that a result's tables
        belong to its measurement set
        """
        for table in self.initial_stack:
            if not os.path.isdir(table.path):
                raise ScheduleError(
                    "The table {0} to continue from does not exist".format(table.path)
                )

    def _reset_corrected(self, workspace: MSWorkspace) -> None:
        """
        gaincal solves against DATA, so CORRECTED_DATA must be DATA calibrated with the initial
        stack and nothing else
        """
        if len(self.initial_stack) == 0 and is_column_in_ms(self.ms, "CORRECTED_DATA") \
                and not columns_match(self.ms, "DATA", "CORRECTED_DATA"):
            logger.warning(
                "CORRECTED_DATA of %s differs from DATA and is reset. gaincal solves against "
                "DATA: split the calibrated data into DATA first if CORRECTED_DATA held them.",
                self.ms
            )
        workspace.apply_stack(self.initial_stack)

    def _image(self, imager: Imager, name: str) -> ImagingResult:
        return imager.image(
            ImagingRequest(self.ms, os.path.join(self.output, name), self.selection)
        )

    def _predict_snr(self, step: CalibrationStep) -> Optional[float]:
        """
        The step's solution S/N predicted from the data, against the accepted model in
        MODEL_DATA; warns when it is below minsnr. None if the data can't support an estimate
        """
        sample = preflight.read_sample(self.ms, self.selection)
        if sample is None:
            logger.debug("No solution S/N estimate for %s", _describe(step))
            return None
        prediction = preflight.predict(sample, step, self.solve)
        if not prediction.enough:
            logger.warning(
                "%s: predicted solution S/N %.1f is below minsnr %g, so gaincal may flag most "
                "solutions; %s", _describe(step), prediction.snr, prediction.minsnr,
                prediction.advice
            )
        return prediction.snr

    def _solve(self, step: CalibrationStep, stack: CalStack, name: str) -> Optional[CalTable]:
        """
        Solves the step's gains against DATA with the accepted stack pre-applied. Returns None if
        gaincal wrote no table (no solution reached minsnr)
        """
        path = os.path.join(self.output, "{0}_{1}.cal".format(name, step.calmode))
        gaincal(
            vis=self.ms,
            caltable=path,
            **step.gaincal_kwargs(self.solve, self.selection),
            **stack.gaincal_arguments()
        )
        if not os.path.exists(path):
            return None
        spwmap = derive_spwmap(
            step.solve_options(self.solve).combine, get_spw_ids(self.ms, self.selection.spw),
            count_spectral_windows(self.ms)
        )
        return CalTable(path, step.calmode, tuple(spwmap), self.apply.interp)

    def _apply(self, stack: CalStack, table: CalTable) -> None:
        applycal(
            vis=self.ms,
            field=self.selection.field,
            spw=self.selection.spw,
            uvrange=self.selection.uvrange,
            gainfield="",
            calwt=False,
            flagbackup=False,
            applymode=self.apply.applymode,
            **stack.applycal_arguments(table)
        )

    @staticmethod
    def _record(
        index: int,
        step: Optional[CalibrationStep],
        caltable: str,
        result: ImagingResult,
        metrics: RoundMetrics,
        accepted: bool,
        reasons: Sequence[str],
        data_flagged: Optional[Tuple[float, float]] = None,
        predicted_snr: Optional[float] = None
    ) -> RoundRecord:
        statistics = metrics.statistics
        return RoundRecord(
            index=index,
            step=_describe(step),
            caltable=caltable,
            image=result.restored,
            psnr=statistics.psnr,
            peak=statistics.peak,
            rms=statistics.rms,
            model_flux=statistics.model_flux,
            beam=statistics.beam,
            flagged_fraction=metrics.flagged_fraction,
            accepted=accepted,
            reasons=tuple(reasons),
            data_flagged=data_flagged,
            predicted_snr=predicted_snr,
        )

    @staticmethod
    def _failed_record(
        index: int,
        step: CalibrationStep,
        reason: str,
        predicted_snr: Optional[float] = None
    ) -> RoundRecord:
        nan = float("nan")
        return RoundRecord(
            index,
            _describe(step),
            "",
            "",
            nan,
            nan,
            nan,
            nan,
            None,
            1.0,
            False, (reason, ),
            predicted_snr=predicted_snr
        )


def _default_output(ms: str) -> str:
    return str(Path(ms).with_suffix("")) + "_selfcal"


def _describe(step: Optional[CalibrationStep]) -> str:
    if step is None:
        return ""
    details = [repr(step.solint)]
    details += ["solve={0}".format(dict(step.solve))] if step.solve else []
    details += ["imaging={0}".format(dict(step.imaging))] if step.imaging else []
    details += ["flagging={0!r}".format(step.flagging)] if step.flagging != NoFlagging() else []
    return "{0}({1})".format(type(step).__name__, ", ".join(details))


def _check_solved_on(result: SelfcalResult) -> None:
    """
    A result's tables must exist and have been solved on its measurement set
    """
    for table in result.stack:
        if not os.path.isdir(table.path):
            raise ScheduleError("The table {0} to continue from does not exist".format(table.path))
        with open_table(table.path) as cal_table:
            solved_on = cal_table.getkeywords().get("MSName", "")
        if solved_on and Path(solved_on).name != Path(result.ms).name:
            raise ScheduleError(
                "The table {0} was solved on {1}, not on {2}".format(
                    table.path, solved_on, result.ms
                )
            )


def _notify(observers: Sequence[SelfcalObserver], hook: str, *arguments) -> None:
    for observer in observers:
        getattr(observer, hook)(*arguments)


def _next_free(base: str) -> str:
    """
    The first of base_2, base_3, ... that doesn't exist yet
    """
    number = 2
    while os.path.exists("{0}_{1}".format(base, number)):
        number += 1
    return "{0}_{1}".format(base, number)
