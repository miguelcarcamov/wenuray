"""The single self-calibration loop (#22c), with CASA replaced by mocks."""
import dataclasses
import logging
import os
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from wenuray.config import Selection, SolveOptions
from wenuray.imaging import contract
from wenuray.imaging.contract import Imager, ImagingResult
from wenuray.imaging.statistics import ImageStatistics
from wenuray.selfcalibration import loop, observers, workspace
from wenuray.selfcalibration.acceptance import AllOf, PSNRImproves
from wenuray.selfcalibration.caltable import CalStack, CalTable
from wenuray.selfcalibration.flagging import Flagging
from wenuray.selfcalibration.loop import Selfcal
from wenuray.selfcalibration.observers import GainPlotObserver, SelfcalObserver
from wenuray.selfcalibration.result import SelfcalResult
from wenuray.selfcalibration.schedule import ScheduleError
from wenuray.selfcalibration.steps import AmpPhasecal, Ampcal, Phasecal


@dataclasses.dataclass(frozen=True)
class FakeResult(ImagingResult):
    peak: float = 1.0

    def statistics(self):
        return ImageStatistics(self.peak, 1e-3, 1.0, (1.0, 1.0, 0.0))


@dataclasses.dataclass(frozen=True)
class FakeImager(Imager):
    """Returns the next peak on every call and records the image names and its niter."""
    peaks: object = None
    calls: list = dataclasses.field(default_factory=list, hash=False)
    niter: int = 100

    def _reconstruct(self, request):
        self.calls.append((Path(request.name).name, self.niter))
        return FakeResult(request.name + ".image", "", "", peak=next(self.peaks))


@pytest.fixture
def casa(monkeypatch, tmp_path):
    """gaincal writes the table unless told not to; everything else is recorded."""
    mocks = SimpleNamespace(
        gaincal=mock.Mock(),
        applycal=mock.Mock(),
        workspace_applycal=mock.Mock(),
        clearcal=mock.Mock(),
        flagmanager=mock.Mock(),
        solves=True,
        corrected_matches=True,
        sample=None,
    )

    def gaincal(**kwargs):
        if mocks.solves:
            os.makedirs(kwargs["caltable"])

    mocks.gaincal.side_effect = gaincal
    monkeypatch.setattr(loop, "gaincal", mocks.gaincal)
    monkeypatch.setattr(loop, "applycal", mocks.applycal)
    monkeypatch.setattr(workspace, "applycal", mocks.workspace_applycal)
    monkeypatch.setattr(workspace, "clearcal", mocks.clearcal)
    monkeypatch.setattr(workspace, "flagmanager", mocks.flagmanager)
    monkeypatch.setattr(loop, "solution_summary", lambda table: (0.05, (8, 8, 8)))
    monkeypatch.setattr(loop.preflight, "read_sample", lambda ms, selection: mocks.sample)
    monkeypatch.setattr(loop, "get_spw_ids", lambda ms, spw: [0])
    monkeypatch.setattr(loop, "ms_fingerprint", lambda ms: "sha256:input")
    monkeypatch.setattr(loop, "count_spectral_windows", lambda ms: 1)
    monkeypatch.setattr(loop, "is_column_in_ms", lambda ms, column: True)
    monkeypatch.setattr(loop, "columns_match", lambda ms, a, b: mocks.corrected_matches)
    # The imager contract: CORRECTED_DATA exists and the model is always written
    monkeypatch.setattr(contract, "is_column_in_ms", lambda ms, column: True)
    monkeypatch.setattr(contract, "delmod", mock.Mock())
    monkeypatch.setattr(contract, "model_column_problem", lambda ms, field, spw: "")
    mocks.ms = str(tmp_path / "target.ms")
    return mocks


def imager(*peaks, **kwargs) -> FakeImager:
    return FakeImager(peaks=iter(peaks), **kwargs)


def table_names(call) -> list:
    return [Path(path).name for path in call.kwargs["gaintable"]]


class TestConstruction:

    def test_invalid_schedule_raises_before_any_io(self, casa, tmp_path):
        with pytest.raises(ScheduleError):
            Selfcal(casa.ms, imager(), [Ampcal("inf")])
        assert list(tmp_path.iterdir()) == []

    def test_unknown_imaging_override_raises_at_construction(self, casa):
        with pytest.raises(TypeError):
            Selfcal(casa.ms, imager(), [Phasecal("inf", imaging={"nitre": 5})])

    def test_imager_must_be_an_imager(self, casa):
        with pytest.raises(TypeError, match="Imager"):
            Selfcal(casa.ms, lambda request: None, [Phasecal("inf")])

    def test_output_defaults_next_to_the_measurement_set(self, casa, tmp_path):
        assert Selfcal(casa.ms, imager(),
                       [Phasecal("inf")]).output == str(tmp_path / "target_selfcal")


class TestAcceptedRounds:

    @pytest.fixture
    def result(self, casa):
        run = Selfcal(
            casa.ms,
            imager(1.0, 2.0, 3.0, 4.0),
            [Phasecal("inf"), Phasecal("60s"), AmpPhasecal("inf")],
            selection=Selection(field="target"),
            solve=SolveOptions(refant="A0"),
        )
        return run.run()

    def test_every_table_is_kept(self, result):
        assert [Path(t.path).name
                for t in result.stack] == ["round1_p.cal", "round2_p.cal", "round3_ap.cal"]
        assert all(record.accepted for record in result.rounds)
        assert [record.psnr for record in result.rounds] == [1000.0, 2000.0, 3000.0, 4000.0]

    def test_each_solve_preapplies_the_accepted_stack(self, casa, result):
        assert [table_names(call) for call in casa.gaincal.call_args_list
                ] == [[], ["round1_p.cal"], ["round1_p.cal", "round2_p.cal"]]

    def test_gaincal_gets_the_step_options_and_the_selection(self, casa, result):
        first = casa.gaincal.call_args_list[0].kwargs
        assert (first["calmode"], first["solint"], first["refant"],
                first["field"]) == ("p", "inf", "A0", "target")
        assert casa.gaincal.call_args_list[2].kwargs["solnorm"] is True

    def test_each_apply_is_the_stack_plus_the_new_table(self, casa, result):
        assert [table_names(call) for call in casa.applycal.call_args_list] == [
            ["round1_p.cal"], ["round1_p.cal", "round2_p.cal"],
            ["round1_p.cal", "round2_p.cal", "round3_ap.cal"]
        ]
        assert casa.applycal.call_args_list[0].kwargs["field"] == "target"

    def test_corrected_data_starts_as_data(self, casa, result):
        casa.clearcal.assert_called_once()

    def test_flag_snapshots_are_saved_and_discarded(self, casa, result):
        modes = [
            (call.kwargs["mode"], call.kwargs["versionname"])
            for call in casa.flagmanager.call_args_list
        ]
        saved = [name for mode, name in modes if mode == "save"]
        assert saved == [
            "wenuray_selfcal_round0", "wenuray_selfcal_round1", "wenuray_selfcal_round2",
            "wenuray_selfcal_round3"
        ]
        assert sorted(name for mode, name in modes if mode == "delete") == sorted(saved)

    def test_the_result_is_written_as_toml(self, casa, result):
        written = SelfcalResult.from_toml(
            str(Path(casa.ms).with_suffix("")) + "_selfcal/selfcal.toml"
        )

        assert written == result
        assert result.final.index == 3


class TestRejection:

    @pytest.fixture
    def run(self, casa):
        fake = imager(1.0, 2.0, 1.5, 2.0)
        result = Selfcal(casa.ms, fake, [Phasecal("inf"), Phasecal("60s"), Phasecal("int")]).run()
        return SimpleNamespace(result=result, imager=fake)

    def test_a_worse_round_stops_the_run(self, casa, run):
        assert [record.accepted for record in run.result.rounds] == [True, True, False]
        assert casa.gaincal.call_count == 2  # Phasecal("int") never ran
        assert [Path(t.path).name for t in run.result.stack] == ["round1_p.cal"]
        assert run.result.final.index == 1

    def test_flags_and_calibration_are_restored(self, casa, run):
        restored = [
            call.kwargs["versionname"] for call in casa.flagmanager.call_args_list
            if call.kwargs["mode"] == "restore"
        ]
        assert restored == ["wenuray_selfcal_round1"]
        assert table_names(casa.workspace_applycal.call_args) == ["round1_p.cal"]

    def test_the_accepted_model_is_imaged_again(self, run):
        assert [name
                for name, _ in run.imager.calls] == ["round0", "round1", "round2", "round1_model"]

    def test_the_reasons_are_recorded(self, run):
        assert "PSNR" in run.result.rounds[2].reasons[0]


def test_a_round_without_solutions_stops_the_run(casa):
    casa.solves = False

    result = Selfcal(casa.ms, imager(1.0), [Phasecal("inf"), Phasecal("60s")]).run()

    assert [record.accepted for record in result.rounds] == [True, False]
    assert result.rounds[1].reasons == ("gaincal found no solutions", )
    assert len(result.stack) == 0
    casa.applycal.assert_not_called()


@pytest.mark.parametrize("combine, spwmap", [("spw", (1, 1, 1, 1)), ("", ())])
def test_each_table_gets_its_own_spwmap(casa, monkeypatch, combine, spwmap):
    """Solutions combined over spws are stored under the lowest selected one (#26)."""
    monkeypatch.setattr(loop, "get_spw_ids", lambda ms, spw: [1, 2, 3])
    monkeypatch.setattr(loop, "count_spectral_windows", lambda ms: 4)

    result = Selfcal(
        casa.ms, imager(1.0, 2.0, 3.0),
        [Phasecal("inf", solve={"combine": combine}),
         AmpPhasecal("inf")]
    ).run(observers=())

    assert [table.spwmap for table in result.stack] == [spwmap, ()]
    assert casa.applycal.call_args.kwargs["spwmap"] == [list(spwmap), []]


def test_step_imaging_overrides_reach_the_imager(casa):
    fake = imager(1.0, 2.0, 3.0)

    Selfcal(casa.ms, fake, [Phasecal("inf"), Phasecal("60s", imaging={"niter": 500})]).run()

    assert fake.calls == [("round0", 100), ("round1", 100), ("round2", 500)]


def test_the_step_policy_wins_over_the_run_policy(casa):
    strict = AllOf(PSNRImproves(min_ratio=10.0))

    result = Selfcal(
        casa.ms,
        imager(1.0, 2.0, 1.0),
        [Phasecal("inf", acceptance=strict)],  # 1.0: re-imaged model
        acceptance=AllOf(PSNRImproves())
    ).run()

    assert not result.rounds[1].accepted


def test_corrected_data_that_differs_from_data_is_reset_with_a_warning(casa, caplog):
    casa.corrected_matches = False

    with caplog.at_level(logging.WARNING, logger="wenuray"):
        Selfcal(casa.ms, imager(1.0, 2.0), [Phasecal("inf")]).run()

    assert "differs from DATA" in caplog.text
    casa.clearcal.assert_called_once()


class Recorder(SelfcalObserver):
    """Records every hook call."""

    def __init__(self):
        self.events = []

    def on_start(self, selfcal):
        self.events.append(("start", len(selfcal.steps)))

    def on_solved(self, step, table):
        self.events.append(("solved", Path(table.path).name))

    def on_round(self, record):
        self.events.append(("round", record.index, record.accepted))

    def on_finished(self, result):
        self.events.append(("finished", len(result.stack)))


class TestObservers:

    def test_every_hook_is_called_in_order(self, casa):
        recorder = Recorder()

        Selfcal(casa.ms, imager(1.0, 2.0, 1.0, 2.0),
                [Phasecal("inf"), Phasecal("60s")]).run(observers=[recorder])

        assert recorder.events == [
            ("start", 2), ("round", 0, True), ("solved", "round1_p.cal"), ("round", 1, True),
            ("solved", "round2_p.cal"), ("round", 2, False), ("finished", 1)
        ]

    def test_logging_is_the_default(self, casa, caplog):
        with caplog.at_level(logging.INFO, logger="wenuray"):
            Selfcal(casa.ms, imager(1.0, 2.0), [Phasecal("inf")]).run()

        assert "Round 1 Phasecal('inf') - PSNR: 2000.000" in caplog.text
        assert "1 of 1 steps accepted" in caplog.text

    def test_no_observers_means_no_logging(self, casa, caplog):
        with caplog.at_level(logging.INFO, logger="wenuray.selfcalibration.observers"):
            Selfcal(casa.ms, imager(1.0, 2.0), [Phasecal("inf")]).run(observers=())

        assert [r for r in caplog.records if r.name == "wenuray.selfcalibration.observers"] == []


class TestResume:

    @pytest.fixture
    def first(self, casa):
        return Selfcal(casa.ms, imager(1.0, 2.0), [Phasecal("inf")]).run(observers=())

    @pytest.fixture(autouse=True)
    def tables_from(self, monkeypatch):
        """The result's tables were solved on the measurement set given (target.ms by default)."""

        def use(ms_name):
            keywords = mock.MagicMock()
            keywords.__enter__.return_value.getkeywords.return_value = {"MSName": ms_name}
            monkeypatch.setattr(loop, "open_table", lambda path: keywords)

        use("target.ms")
        return use

    def test_continues_with_the_accepted_stack(self, casa, first, tables_from):
        tables_from("target.ms")
        casa.gaincal.reset_mock()

        result = Selfcal.resume(first, imager(2.0, 3.0), [Ampcal("1h")]).run(observers=())

        assert table_names(casa.gaincal.call_args) == ["round1_p.cal"]
        assert table_names(casa.workspace_applycal.call_args) == ["round1_p.cal"]  # not clearcal
        assert [Path(t.path).name for t in result.stack] == ["round1_p.cal", "round1_a.cal"]

    def test_the_schedule_is_validated_against_the_earlier_stack(self, casa, first):
        """An amplitude step may come first only because the earlier run solved phases."""
        Selfcal.resume(first, imager(), [Ampcal("1h")])
        with pytest.raises(ScheduleError):
            Selfcal(casa.ms, imager(), [Ampcal("1h")])

    def test_writes_to_a_new_directory(self, casa, first, tmp_path):
        assert Selfcal.resume(first, imager(),
                              [Phasecal("60s")]).output == str(tmp_path / "target_selfcal_2")

    def test_from_the_result_file_or_its_directory(self, casa, first, tmp_path):
        directory = tmp_path / "target_selfcal"

        for source in (directory, directory / "selfcal.toml", str(directory)):
            assert Selfcal.resume(source, imager(), [Phasecal("60s")]).initial_stack == first.stack

    def test_measurement_set_and_stack_come_from_the_result(self, first):
        with pytest.raises(TypeError, match="from the result"):
            Selfcal.resume(first, imager(), [Phasecal("60s")], ms="other.ms")

    def test_tables_solved_on_another_measurement_set_raise(self, first, tables_from):
        tables_from("other.ms")

        with pytest.raises(ScheduleError, match="other.ms"):
            Selfcal.resume(first, imager(1.0), [Phasecal("60s")])

    def test_your_own_tables_may_come_from_another_measurement_set(
        self, casa, tables_from, tmp_path
    ):
        """E.g. a phase table solved on the measurement set this one was split from."""
        tables_from("parent.ms")
        (tmp_path / "pcal").mkdir()
        stack = CalStack([CalTable(str(tmp_path / "pcal"), "p")])

        result = Selfcal(casa.ms, imager(1.0, 2.0), [Ampcal("1h")],
                         initial_stack=stack).run(observers=())

        assert table_names(casa.gaincal.call_args) == ["pcal"]
        assert len(result.stack) == 2

    def test_missing_tables_raise(self, casa, first, tmp_path):
        for table in first.stack:
            os.rmdir(table.path)

        with pytest.raises(ScheduleError, match="does not exist"):
            Selfcal.resume(first, imager(1.0), [Phasecal("60s")])
        with pytest.raises(ScheduleError, match="does not exist"):
            Selfcal(
                casa.ms,
                imager(1.0), [Phasecal("60s")],
                output=str(tmp_path / "again"),
                initial_stack=first.stack
            ).run(observers=())


def test_an_existing_run_is_not_overwritten(casa):
    run = Selfcal(casa.ms, imager(1.0, 2.0, 1.0), [Phasecal("inf")])
    run.run(observers=())

    with pytest.raises(FileExistsError, match="resume"):
        run.run(observers=())


def test_gain_plots(monkeypatch, tmp_path):
    rows = 6
    tables = mock.MagicMock()
    tables.__enter__.return_value.getcol.side_effect = lambda column: {
        "CPARAM": np.exp(1j * np.linspace(0, 1, 2 * rows)).reshape(2, 1, rows),
        "FLAG": np.zeros((2, 1, rows), dtype=bool),
        "TIME": np.repeat([0.0, 60.0, 120.0], 2),
        "ANTENNA1": np.tile([0, 1], 3),
    }[column]
    monkeypatch.setattr(observers, "open_table", lambda path: tables)
    table = CalTable(str(tmp_path / "round1_p.cal"), "p")

    GainPlotObserver().on_solved(Phasecal("60s"), table)

    assert (tmp_path / "round1_p.cal.png").stat().st_size > 0


@dataclasses.dataclass(frozen=True)
class FakeFlagging(Flagging):
    """Records when it flags, between the apply and the imaging."""
    events: list = dataclasses.field(default_factory=list, hash=False)

    def flag(self, ms, selection):
        self.events.append(("flag", selection.field))
        return 0.1, 0.2


@dataclasses.dataclass(frozen=True)
class EventImager(FakeImager):
    """Records its imaging into a shared event list."""
    events: list = dataclasses.field(default=None, hash=False)

    def _reconstruct(self, request):
        self.events.append(("image", ))
        return super()._reconstruct(request)


class TestResidualFlagging:

    def test_flags_after_the_apply_and_before_the_imaging(self, casa):
        flagger = FakeFlagging()
        casa.applycal.side_effect = lambda **kwargs: flagger.events.append(("apply", ))
        fake = EventImager(peaks=iter([1.0, 2.0]), events=flagger.events)

        Selfcal(casa.ms, fake, [Phasecal("inf", flagging=flagger)],
                selection=Selection(field="t")).run(observers=())

        assert flagger.events == [("image", ), ("apply", ), ("flag", "t"), ("image", )]

    def test_the_flagged_fractions_are_recorded(self, casa):
        result = Selfcal(casa.ms, imager(1.0, 2.0),
                         [Phasecal("inf", flagging=FakeFlagging())]).run(observers=())

        assert result.rounds[0].data_flagged is None
        assert result.rounds[1].data_flagged == (0.1, 0.2)
        assert SelfcalResult.from_toml(
            str(Path(casa.ms).with_suffix("")) + "_selfcal/selfcal.toml"
        ) == result

    def test_steps_without_flagging_flag_nothing(self, casa):
        result = Selfcal(casa.ms, imager(1.0, 2.0), [Phasecal("inf")]).run(observers=())

        assert result.rounds[1].data_flagged is None


class TestPredictedSolutionSNR:
    """The pre-flight S/N estimate (#28): warned about when below minsnr, and recorded."""

    @staticmethod
    def sample(noise):
        from wenuray.selfcalibration.preflight import VisibilitySample
        return VisibilitySample(1.0, noise, 8, 60.0, 7200.0, 7200.0, (1, ), 2)

    def test_is_recorded_for_each_step(self, casa):
        casa.sample = self.sample(0.05)

        result = Selfcal(casa.ms, imager(1.0, 2.0), [Phasecal("int")]).run(observers=())

        assert result.rounds[0].predicted_snr is None
        assert result.rounds[1].predicted_snr == pytest.approx(20 * np.sqrt(7) * np.sqrt(2))
        assert SelfcalResult.from_toml(
            str(Path(casa.ms).with_suffix("")) + "_selfcal/selfcal.toml"
        ) == result

    def test_a_low_estimate_warns_and_the_step_still_runs(self, casa, caplog):
        casa.sample = self.sample(2.0)

        with caplog.at_level(logging.WARNING):
            result = Selfcal(casa.ms, imager(1.0, 2.0), [Phasecal("int")]).run(observers=())

        assert "predicted solution S/N 1.9 is below minsnr 3" in caplog.text
        assert "a solint of 180s or longer would reach it" in caplog.text
        assert result.rounds[1].accepted

    def test_no_estimate_without_a_sample(self, casa):
        result = Selfcal(casa.ms, imager(1.0, 2.0), [Phasecal("int")]).run(observers=())

        assert result.rounds[1].predicted_snr is None


@dataclasses.dataclass(frozen=True)
class RecordableImager(Imager):
    """A fake imager a recipe can hold: the n-th call returns the n-th peak"""
    peaks: tuple = (1.0, 2.0, 3.0)
    calls: list = dataclasses.field(default_factory=list, hash=False, compare=False)

    def __post_init__(self):
        object.__setattr__(self, "peaks", tuple(self.peaks))

    def _reconstruct(self, request):
        self.calls.append(request.name)
        return FakeResult(request.name + ".image", "", "", peak=self.peaks[len(self.calls) - 1])


class TestTheRunIsRecorded:
    """#36: the resolved recipe and the provenance are written for every run."""

    def test_the_resolved_recipe_rebuilds_the_run(self, casa):
        selfcal = Selfcal(
            casa.ms,
            RecordableImager(), [Phasecal("inf"), Phasecal("60s")],
            solve=SolveOptions(refant="A0")
        )

        result = selfcal.run(observers=())

        recipe_path = os.path.join(selfcal.output, "recipe.toml")
        assert result.provenance["recipe"] == "recipe.toml"
        assert Selfcal.from_toml(recipe_path) == selfcal
        with open(recipe_path) as file:
            assert "minblperant = 4" in file.read()  # defaults are written out

    def test_the_provenance_is_in_the_result(self, casa):
        result = Selfcal(casa.ms, RecordableImager(), [Phasecal("inf")]).run(observers=())

        provenance = result.provenance
        assert provenance["ms_fingerprint"] == "sha256:input"
        assert {"python", "numpy"} <= set(provenance["versions"])
        assert provenance["started"].endswith("+00:00")
        assert SelfcalResult.from_toml(
            str(Path(casa.ms).with_suffix("")) + "_selfcal/selfcal.toml"
        ) == result

    def test_a_run_a_recipe_cannot_hold_still_runs(self, casa, caplog):
        with caplog.at_level(logging.WARNING):
            result = Selfcal(casa.ms, imager(1.0, 2.0), [Phasecal("inf")]).run(observers=())

        assert "can't be written as a recipe" in caplog.text
        assert result.provenance["recipe"] == ""
        assert not os.path.exists(str(Path(casa.ms).with_suffix("")) + "_selfcal/recipe.toml")
