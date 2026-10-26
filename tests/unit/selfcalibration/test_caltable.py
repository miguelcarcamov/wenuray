import dataclasses

import pytest

from wenuray.selfcalibration.caltable import CalStack, CalTable, derive_spwmap


class TestDeriveSpwmap:

    @pytest.mark.parametrize("combine", ["", "scan", "field,scan"])
    def test_without_spw_combination_no_map_is_needed(self, combine):
        assert derive_spwmap(combine, [0, 1, 2, 3], 4) == []

    @pytest.mark.parametrize("combine", ["spw", "scan,spw", "scan, spw"])
    def test_combined_solutions_map_to_lowest_selected_spw(self, combine):
        assert derive_spwmap(combine, [3, 1, 2], 5) == [1, 1, 1, 1, 1]

    def test_combining_without_selected_spws_raises(self):
        with pytest.raises(ValueError, match="No spectral windows"):
            derive_spwmap("spw", [], 4)


class TestCalStack:

    def test_one_entry_per_table_in_application_order(self):
        stack = CalStack(
            [CalTable("pcal", "p", (0, 0, 0, 0)),
             CalTable("apcal", "ap", (), "nearest")]
        )

        assert stack.gaincal_arguments() == {
            "gaintable": ["pcal", "apcal"],
            "spwmap": [[0, 0, 0, 0], []],
            "interp": ["linear", "nearest"],
        }

    def test_empty_stack_gives_empty_lists_not_empty_names(self):
        assert CalStack().gaincal_arguments() == {"gaintable": [], "spwmap": [], "interp": []}

    def test_applycal_applies_the_stack_plus_the_new_table(self):
        stack = CalStack([CalTable("pcal", "p")])

        arguments = stack.applycal_arguments(CalTable("apcal", "ap"))

        assert arguments["gaintable"] == ["pcal", "apcal"]
        assert len(stack) == 1  # the stack itself is unchanged

    def test_push_returns_a_new_stack(self):
        stack = CalStack()
        pushed = stack.push(CalTable("pcal", "p"))

        assert len(stack) == 0 and [t.path for t in pushed] == ["pcal"]
        with pytest.raises(dataclasses.FrozenInstanceError):
            pushed.tables = ()

    def test_reports_which_solutions_it_holds(self):
        phase = CalStack([CalTable("pcal", "p")])

        assert phase.solves_phase and not phase.solves_amplitude
        assert phase.push(CalTable("acal", "a")).solves_amplitude
        assert not CalStack().solves_phase


class TestCalStackToml:

    def test_round_trip_keeps_every_table_field(self, tmp_path):
        stack = CalStack(
            [
                CalTable(str(tmp_path / "out_pcal0"), "p", (1, 1, 1, 1)),
                CalTable(str(tmp_path / "out_apcal0"), "ap", (), "nearest"),
            ]
        )
        stack.to_toml(str(tmp_path / "stack.toml"))

        assert CalStack.from_toml(str(tmp_path / "stack.toml")) == stack

    def test_paths_are_relative_to_the_file_so_the_run_can_move(self, tmp_path):
        run = tmp_path / "run"
        run.mkdir()
        CalStack([CalTable(str(run / "tables" / "pcal0"), "p")]).to_toml(str(run / "stack.toml"))
        assert 'path = "tables/pcal0"' in (run / "stack.toml").read_text()

        moved = tmp_path / "moved"
        run.rename(moved)

        loaded = CalStack.from_toml(str(moved / "stack.toml"))
        assert [t.path for t in loaded] == [str(moved / "tables" / "pcal0")]


def test_invalid_calmode_raises():
    with pytest.raises(ValueError, match="calmode"):
        CalTable("t", "k")


def test_spwmap_is_stored_as_a_tuple():
    assert CalTable("t", "p", [0, 0]).spwmap == (0, 0)


def test_calmode_properties():
    assert CalTable("t", "p").solves_phase and not CalTable("t", "p").solves_amplitude
    assert CalTable("t", "a").solves_amplitude and not CalTable("t", "a").solves_phase
    assert CalTable("t", "ap").solves_phase and CalTable("t", "ap").solves_amplitude
