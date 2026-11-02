"""Advisories (#141): checks that warn, never decide, each returning Findings at its hook."""
import dataclasses

import pytest

from wenuray.config import SolveOptions
from wenuray.selfcalibration.advisories import (
    ADVISORIES, Advisory, ResidualBelowThermal, RoundContext, SolutionSNRBelowMinimum, SolveContext,
    SpectrumAbsorbed, StartContext, WeightsMisstateNoise
)
from wenuray.selfcalibration.preflight import SolutionSNR, VisibilitySample
from wenuray.selfcalibration.result import RoundRecord
from wenuray.selfcalibration.steps import AmpPhasecal, Phasecal, Severity

SAMPLE = VisibilitySample(
    1.0, 0.05, 8, 60.0, 7200.0, 7200.0, (1, 1), 2, (4.75e9, 5.25e9), (2500.0, 2500.0)
)


class FlatImager:
    models_spectrum = False


def solve_context(step, snr, minsnr=3.0):
    return SolveContext(
        step, "Step", SolveOptions(), SAMPLE, SolutionSNR(snr, minsnr, "advice"), FlatImager()
    )


def record(rms_in_mask, thermal):
    return RoundRecord(
        1,
        "Phasecal('inf')",
        "t.cal",
        "i.image",
        10.0,
        1.0,
        0.1,
        1.0,
        rms_in_mask=rms_in_mask,
        thermal_noise=thermal
    )


def test_an_advisory_finds_nothing_by_default():

    class Quiet(Advisory):
        pass

    assert Quiet().at_start(StartContext("t.ms", None)) == []
    assert Quiet().before_solve(solve_context(Phasecal("inf"), 50.0)) == []
    assert Quiet().after_round(RoundContext(record(None, None))) == []


def test_a_low_predicted_snr_is_a_warning():
    (finding, ) = SolutionSNRBelowMinimum().before_solve(solve_context(Phasecal("int"), 1.9))

    assert finding.severity is Severity.WARNING
    assert finding.message == (
        "Step: predicted solution S/N 1.9 is below minsnr 3, so gaincal "
        "may flag most solutions; advice"
    )
    assert SolutionSNRBelowMinimum().before_solve(solve_context(Phasecal("int"), 3.0)) == []


def test_an_absorbed_spectrum_is_a_warning():
    # Amplitudes per spw over a 10% band, against a flat model
    (finding, ) = SpectrumAbsorbed().before_solve(solve_context(AmpPhasecal("inf"), 500.0))

    assert "absorb the source's spectral index" in finding.message
    assert SpectrumAbsorbed().before_solve(solve_context(Phasecal("inf"), 500.0)) == []


def test_misstated_weights_are_a_warning():
    ratios = tuple(("A{0}".format(i), "B", 0, 1.41) for i in range(6))
    sample = dataclasses.replace(SAMPLE, noise_over_sigma=ratios)

    (finding, ) = WeightsMisstateNoise().at_start(StartContext("t.ms", sample))

    assert finding.message.startswith("t.ms: the data's noise is 1.41 times")
    assert WeightsMisstateNoise().at_start(StartContext("t.ms", None)) == []


@pytest.mark.parametrize(
    "rms_in_mask, thermal, warns",
    [(0.6e-3, 1e-3, True), (0.95e-3, 1e-3, False), (None, 1e-3, False)]
)
def test_a_residual_below_thermal_in_the_mask_is_a_warning(rms_in_mask, thermal, warns):
    findings = ResidualBelowThermal().after_round(RoundContext(record(rms_in_mask, thermal)))

    assert bool(findings) is warns


def test_every_run_makes_the_four_advisories():
    assert {type(advisory)
            for advisory in ADVISORIES} == {
                WeightsMisstateNoise, SolutionSNRBelowMinimum, SpectrumAbsorbed,
                ResidualBelowThermal
            }
