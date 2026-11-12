"""
#190: wenuray suggest writes a recipe with values derived from the data, each with its reason. The
choices are made from a DataSummary, so they are tested here without CASA.
"""
import dataclasses
import math
import tomllib

import numpy as np
import pytest

from wenuray import cli, suggest
from wenuray.config import ImageGeometry, SolveOptions
from wenuray.imaging.backends import Pyralysis, Tclean
from wenuray.selfcalibration import AmpPhasecal, Phasecal, Repeat, Selfcal
from wenuray.selfcalibration.preflight import VisibilitySample
from wenuray.suggest import Antenna, DataSummary

# 0.05 Jy per visibility over 2500 equal weights: 1 mJy/beam
SAMPLE = VisibilitySample(
    signal=math.nan,
    noise=0.05,
    nantennas=6,
    interval=60.0,
    scan_length=600.0,
    span=7200.0,
    channels=(1, ),
    parallel_hands=2,
    band=(5.0e9, 5.1e9),
    weights=(2500.0, 2500.0),
)


def antenna(name, east, flagged=0.0):
    return Antenna(name, (east, 0.0, 6.4e6), flagged)


# Six antennas on a line: A2 and A3 nearest the centre (the median, at 250 m)
ANTENNAS = tuple(
    antenna("A{0}".format(i), east) for i, east in enumerate((-2000, -500, 0, 500, 1000, 3000))
)
SUMMARY = DataSummary(
    ms="/data/target.ms",
    field="target",
    spws=(0, ),
    spws_flagged=(),
    antennas=ANTENNAS,
    sample=SAMPLE,
    geometry=ImageGeometry(imsize=(1280, 1280), cell="0.312arcsec"),
)


def recipe(summary=SUMMARY, imager="tclean"):
    return suggest.recipe_text(summary, imager)


def loaded(tmp_path, text):
    path = tmp_path / "recipe.toml"
    path.write_text(text)
    return Selfcal.from_toml(str(path))


class TestReferenceAntenna:

    def test_the_most_central_with_unflagged_data(self):
        choice = suggest.reference_antenna(ANTENNAS)

        assert choice.value == "A2"
        assert "250 m from the array centre, 0.0% of its sampled data flagged" in choice.why

    def test_flagging_moves_it_to_a_less_flagged_neighbour(self):
        flagged = tuple(
            dataclasses.replace(a, flagged=0.6) if a.name in ("A2", "A3") else a for a in ANTENNAS
        )

        assert suggest.reference_antenna(flagged).value == "A1"

    def test_unknown_flagging_leaves_the_distance(self):
        unknown = tuple(dataclasses.replace(a, flagged=math.nan) for a in ANTENNAS)

        choice = suggest.reference_antenna(unknown)

        assert choice.value == "A2" and "flagging is unknown" in choice.why

    def test_no_antennas(self):
        assert suggest.reference_antenna(()).value == ""


class TestBaselinesPerAntenna:

    @pytest.mark.parametrize("amplitudes, value", [(False, 2), (True, 3)])
    def test_one_fewer_than_the_closure_antennas(self, amplitudes, value):
        choice = suggest.baselines_per_antenna(10, amplitudes)

        assert choice.value == value
        assert "closure {0}".format("amplitudes" if amplitudes else "phases") in choice.why

    def test_at_most_the_other_antennas(self):
        choice = suggest.baselines_per_antenna(3, amplitudes=True)

        assert choice.value == 2
        assert "only 3 antennas have data" in choice.why


class TestRecipe:

    def test_loads_as_a_run(self, tmp_path):
        run = loaded(tmp_path, recipe())

        assert run.ms == "/data/target.ms"
        assert run.selection.field == "target" and run.selection.spw == ""
        assert run.solve.refant == "A2" and run.solve.minblperant == 3
        assert isinstance(run.imager, Tclean)
        assert run.imager.geometry.cell == "0.312arcsec"
        assert run.imager.geometry.imsize == (1280, 1280)
        assert run.steps == (Phasecal("inf"), Repeat(Phasecal("auto")), AmpPhasecal("auto"))

    def test_every_derived_value_says_why(self):
        lines = recipe().splitlines()

        for key in ("field", "spw", "refant", "minblperant", "nsigma", "geometry", "weighting"):
            line = next(index for index, text in enumerate(lines) if text.startswith(key + " ="))
            assert "#" in lines[line] or lines[line - 1].startswith("#"), key

    def test_lines_fit_the_width(self):
        assert all(len(line) <= suggest.WIDTH for line in recipe().splitlines())

    def test_is_valid_toml_with_the_comments(self):
        assert tomllib.loads(recipe())["solve"]["refant"] == "A2"

    def test_states_the_thermal_noise_with_nsigma(self):
        text = recipe()

        assert "nsigma = 3.0" in text
        assert "thermal noise from the data" in text and "1 mJy/beam" in text

    def test_spws_flagged_throughout_are_left_out(self, tmp_path):
        summary = dataclasses.replace(SUMMARY, spws=(0, 2), spws_flagged=(1, ))

        run = loaded(tmp_path, recipe(summary))

        assert run.selection.spw == "0,2"
        assert "1 is all flagged" in recipe(summary)

    def test_several_spws_mention_combine_without_setting_it(self, tmp_path):
        summary = dataclasses.replace(SUMMARY, spws=(0, 1, 2, 3))

        run = loaded(tmp_path, recipe(summary))

        assert run.solve.combine == ""
        assert "combine = \"spw\" isn't set, so the 4 spectral windows" in recipe(summary)

    def test_three_antennas_get_no_amplitude_step(self, tmp_path):
        summary = dataclasses.replace(SUMMARY, antennas=ANTENNAS[:3])

        run = loaded(tmp_path, recipe(summary))

        assert run.steps == (Phasecal("inf"), Repeat(Phasecal("auto")))
        assert run.solve.minblperant == 2
        assert "No amplitude step: closure amplitudes need 4 antennas, and 3 have data" \
            in recipe(summary)

    def test_fewer_than_three_antennas_cant_be_self_calibrated(self):
        with pytest.raises(ValueError, match="needs at least 3"):
            recipe(dataclasses.replace(SUMMARY, antennas=ANTENNAS[:2]))

    def test_weights_that_misstate_the_noise_are_a_warning(self):
        ratios = tuple(("A{0}".format(i), "A{0}".format(i + 1), 0, 1.41) for i in range(5))
        summary = dataclasses.replace(
            SUMMARY, sample=dataclasses.replace(SAMPLE, noise_over_sigma=ratios)
        )

        assert "# Warning: " in recipe(summary) and "statwt" in recipe(summary)

    def test_without_a_noise_estimate(self, tmp_path):
        summary = dataclasses.replace(SUMMARY, sample=None)

        assert "the thermal noise is unknown" in recipe(summary)
        assert loaded(tmp_path, recipe(summary)).imager.options["nsigma"] == 3.0

    def test_a_grid_the_data_cant_give_is_left_auto(self, tmp_path):
        summary = dataclasses.replace(
            SUMMARY, geometry=None, geometry_problem="the antennas have no dish diameter"
        )

        run = loaded(tmp_path, recipe(summary))

        assert run.imager.geometry.cell == "auto"
        assert "no dish diameter" in recipe(summary)

    def test_pyralysis(self, tmp_path, monkeypatch):
        # Building the imager checks Pyralysis's version in its worker; the recipe is what's tested
        from wenuray.imaging.backends import pyralysis as backend
        monkeypatch.setattr(backend, "_check", lambda options: None)

        run = loaded(tmp_path, recipe(imager="pyralysis"))

        assert isinstance(run.imager, Pyralysis)
        assert run.imager.options["mask_mode"] == "auto"
        assert "1 mJy/beam" in recipe(imager="pyralysis")

    def test_an_unknown_imager(self):
        with pytest.raises(ValueError, match="imager must be one of"):
            recipe(imager="wsclean")

    def test_names_no_instrument(self):
        text = recipe().lower()

        assert not any(name in text for name in ("alma", "vla", "meerkat", "ska", "atca"))


class TestCommand:

    @pytest.fixture
    def read(self, monkeypatch, tmp_path):
        (tmp_path / "target.ms").mkdir()
        monkeypatch.setattr(
            suggest,
            "read_summary",
            lambda ms, field, image=False: dataclasses.replace(SUMMARY, ms=ms)
        )
        return str(tmp_path / "target.ms")

    def test_to_a_file_with_the_ms_relative_to_it(self, read, tmp_path, capsys):
        output = tmp_path / "recipes" / "selfcal.toml"
        output.parent.mkdir()

        assert cli.main(["suggest", read, "-o", str(output)]) == 0
        assert 'ms = "../target.ms"' in output.read_text()
        assert cli.main(["validate", str(output)]) == 0

    def test_to_the_standard_output(self, read, capsys):
        assert cli.main(["suggest", read, "--imager", "pyralysis"]) == 0

        output = capsys.readouterr().out
        assert 'type = "pyralysis"' in output and read in output

    def test_doesnt_overwrite(self, read, tmp_path, capsys):
        output = tmp_path / "selfcal.toml"
        output.write_text("")

        assert cli.main(["suggest", read, "-o", str(output)]) == 1
        assert "already exists" in capsys.readouterr().err

    def test_image_asks_for_a_model(self, monkeypatch, tmp_path, capsys):
        (tmp_path / "target.ms").mkdir()
        asked = []
        monkeypatch.setattr(
            suggest,
            "read_summary",
            lambda ms, field, image=False: asked.append(image) or dataclasses.
            replace(SUMMARY, ms=ms)
        )

        assert cli.main(["suggest", str(tmp_path / "target.ms"), "--image"]) == 0
        assert asked == [True]

    def test_needs_a_measurement_set(self, tmp_path, capsys):
        assert cli.main(["suggest", str(tmp_path / "missing.ms")]) == 1
        assert "is not a measurement set" in capsys.readouterr().err


# The S/N of SAMPLE's solutions at solint "inf" (10 integrations of a 600 s scan, 2 hands, 6
# antennas) for a model signal S: (S / 0.05) sqrt(5) sqrt(10 x 2 x channels) = 200 S per channel,
# over sqrt(2) for amplitudes; 2 x minsnr is 6
def modelled(signal, spws=(0, )):
    model = suggest.ModelEstimate(signal, 1.0, 12, 4.8, 1e5)
    sample = dataclasses.replace(SAMPLE, channels=(1, ) * len(spws))
    return dataclasses.replace(SUMMARY, spws=spws, sample=sample, model=model)


class TestQuickModel:
    """--image (#190): a quick model's S/N decides the amplitude step and combine"""

    @pytest.mark.parametrize("beams, nsigma", [(1e5, math.sqrt(2 * math.log(1e5))), (50, 3.0)])
    def test_cleans_to_the_noise_maximum(self, beams, nsigma):
        assert suggest.noise_maximum(beams) == pytest.approx(nsigma)
        assert suggest.noise_maximum(1e5) == pytest.approx(4.80, abs=0.01)

    def test_counts_the_beams_from_the_grid(self):
        geometry = ImageGeometry(imsize=(1000, 1000), cell="0.1arcsec", oversampling=2.5)

        assert suggest.independent_beams(geometry) == pytest.approx(1e6 / (1.133 * 25))

    def test_a_point_at_the_centre_has_its_flux_everywhere(self):
        u, v = np.linspace(-1e5, 1e5, 50), np.linspace(3e4, -2e4, 50)

        amplitudes = suggest.model_amplitudes(
            np.array([0.7]), np.array([0.0]), np.array([0.0]), u, v
        )

        assert amplitudes == pytest.approx(np.full(50, 0.7))

    def test_two_points_beat_in_batches(self, monkeypatch):
        # Two 1 Jy points 1e-5 rad apart in l: |V| = 2 |cos(pi u 1e-5)|, the same in any batches
        monkeypatch.setattr(suggest, "COMPONENT_BATCH", 1)
        u, v = np.linspace(0, 1e5, 40), np.zeros(40)

        amplitudes = suggest.model_amplitudes(
            np.ones(2), np.array([-5e-6, 5e-6]), np.zeros(2), u, v
        )

        assert amplitudes == pytest.approx(2 * np.abs(np.cos(np.pi * u * 1e-5)))

    @pytest.mark.parametrize(
        "signal, included, words", [(0.1, True, "14.1 (CASA's"), (0.02, False, "2.8 (CASA's")]
    )
    def test_the_amplitude_step_needs_twice_minsnr(self, signal, included, words):
        sample = dataclasses.replace(SAMPLE, signal=signal)

        choice = suggest.amplitude_step(sample, SolveOptions())

        assert choice.value is included and words in choice.why

    @pytest.mark.parametrize(
        "signal, spws, combine",
        [
            (0.1, 4, ""),  # 20 per window
            (0.02, 4, "spw"),  # 4 per window, 8 combined
            (0.005, 4, "spw"),  # 1 per window, 2 combined: still the best there is
            (0.02, 1, ""),  # one window: nothing to combine
        ]
    )
    def test_windows_are_combined_when_one_is_short(self, signal, spws, combine):
        sample = dataclasses.replace(SAMPLE, signal=signal, channels=(1, ) * spws)

        assert suggest.spw_combination(sample, SolveOptions()).value == combine

    def test_a_faint_source_gets_combined_windows_and_no_amplitudes(self, tmp_path):
        summary = modelled(0.02, spws=(0, 1, 2, 3))

        run = loaded(tmp_path, recipe(summary))
        text = recipe(summary)

        assert run.solve.combine == "spw"
        assert run.steps == (Phasecal("inf"), Repeat(Phasecal("auto")))
        assert "the 4 windows combined give 8.0" in text
        assert "No amplitude step: the amplitude solutions' predicted S/N" in text
        assert "5.7 (CASA's" in text  # 400 x 0.02 / sqrt(2)

    def test_a_bright_source_keeps_its_amplitudes(self, tmp_path):
        summary = modelled(0.1, spws=(0, 1, 2, 3))

        run = loaded(tmp_path, recipe(summary))

        assert run.solve.combine == ""
        assert run.steps[-1] == AmpPhasecal("auto")
        assert "reaches 6 (2 x minsnr)" in recipe(summary)
        assert "--image: a quick tclean model of DATA" in recipe(summary)

    def test_no_emission_is_a_warning(self, tmp_path):
        summary = dataclasses.replace(SUMMARY, model=suggest.ModelEstimate(0.0, 0.0, 0, 4.8, 1e5))

        assert "found no emission above the noise's expected maximum" in recipe(summary)
        assert loaded(tmp_path, recipe(summary)).steps[-1] == AmpPhasecal("auto")

    def test_without_a_model_nothing_is_decided(self, tmp_path):
        summary = dataclasses.replace(
            SUMMARY, model_problem="it needs the grid, which couldn't be derived"
        )

        assert "no quick model was made: it needs the grid" in recipe(summary)
        assert "the S/N isn't known without a model" in recipe(summary)


class TestReadSummary:
    """read_summary's CASA reads are integration-tested; here, what it does with a failed image"""

    def test_a_failed_quick_image_is_a_problem_not_an_error(self, monkeypatch):
        monkeypatch.setattr(suggest, "_field", lambda ms, field: "target")
        monkeypatch.setattr(suggest, "get_spw_ids", lambda ms, spw: [0])
        monkeypatch.setattr(suggest, "spws_with_unflagged_data", lambda ms, spws: [0])
        monkeypatch.setattr(suggest.preflight, "read_sample", lambda ms, sel, model: SAMPLE)
        monkeypatch.setattr(suggest, "_antennas", lambda ms, sample: ANTENNAS)
        monkeypatch.setattr(
            "wenuray.imaging.contract.resolve_geometry", lambda g, ms, sel: SUMMARY.geometry
        )

        def fail(ms, selection, geometry):
            raise RuntimeError("tclean: no data selected")

        monkeypatch.setattr(suggest, "estimate_model", fail)

        summary = suggest.read_summary("target.ms", image=True)

        assert summary.model is None
        assert "tclean couldn't image the data: tclean: no data selected" in summary.model_problem
