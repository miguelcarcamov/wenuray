"""Acceptance policies (#28): composable checks, and PSNR never decides alone."""
import math

import pytest

from wenuray.imaging.statistics import ImageStatistics
from wenuray.selfcalibration.acceptance import (
    AllOf, AnyOf, BeamStable, Check, EnoughAntennas, FlaggedFractionBelow, FluxNotLost,
    PSNRImproves, RoundMetrics, solution_snr
)
from wenuray.selfcalibration.steps import AmpPhasecal, Phasecal


def metrics(
    peak=1.0,
    rms=1e-3,
    flux=1.0,
    beam=(1.0, 0.8, 30.0),
    flagged=0.0,
    step=None,
    antennas=(),
    extra=None
) -> RoundMetrics:
    return RoundMetrics(
        ImageStatistics(peak, rms, flux, beam), step, flagged, antennas, extra or {}
    )


BEFORE = metrics()


class TestPSNRImproves:

    def test_higher_psnr_is_accepted(self):
        assert PSNRImproves().decide(BEFORE, metrics(rms=0.5e-3)).accepted

    def test_equal_psnr_is_rejected(self):
        assert not PSNRImproves().decide(BEFORE, metrics()).accepted

    def test_margin(self):
        policy = PSNRImproves(min_ratio=1.1)

        assert not policy.decide(BEFORE, metrics(peak=1.05)).accepted
        assert policy.decide(BEFORE, metrics(peak=1.2)).accepted


class TestFluxNotLost:

    def test_more_flux_after_phase_calibration_is_fine(self):
        assert FluxNotLost().decide(BEFORE, metrics(flux=1.3)).accepted

    def test_small_loss_within_tolerance_is_fine(self):
        assert FluxNotLost(max_loss=0.1).decide(BEFORE, metrics(flux=0.95)).accepted

    def test_flux_absorbed_by_the_gains_is_rejected(self):
        decision = FluxNotLost(max_loss=0.1).decide(BEFORE, metrics(flux=0.8))

        assert not decision.accepted and "model flux" in decision.reasons[0]


class TestBeamStable:

    def test_small_change_is_fine(self):
        assert BeamStable(0.05).decide(BEFORE, metrics(beam=(1.03, 0.8, 35.0))).accepted

    def test_large_change_is_rejected(self):
        assert not BeamStable(0.05).decide(BEFORE, metrics(beam=(1.0, 0.9, 30.0))).accepted

    def test_rounds_without_a_beam_pass(self):
        assert BeamStable(0.05).decide(BEFORE, metrics(beam=None)).accepted


class TestFlaggedFractionBelow:

    def test_few_flagged_solutions_are_fine(self):
        assert FlaggedFractionBelow(0.25).decide(BEFORE, metrics(flagged=0.1)).accepted

    def test_many_flagged_solutions_are_rejected(self):
        assert not FlaggedFractionBelow(0.25).decide(BEFORE, metrics(flagged=0.4)).accepted


class TestCombinators:

    def test_all_of_needs_every_policy_and_lists_every_reason(self):
        decision = AllOf(PSNRImproves(),
                         FluxNotLost()).decide(BEFORE, metrics(rms=0.5e-3, flux=0.5))

        assert not decision.accepted
        assert len(decision.reasons) == 2

    def test_any_of_needs_one_policy(self):
        policy = AnyOf(PSNRImproves(), FluxNotLost())

        assert policy.decide(BEFORE, metrics(flux=1.2)).accepted
        assert not policy.decide(BEFORE, metrics(rms=2e-3, flux=0.5)).accepted

    def test_combinators_nest_and_are_hashable(self):
        policy = AllOf(PSNRImproves(), AnyOf(FluxNotLost(), BeamStable(0.05)))

        assert hash(policy) == hash(AllOf(PSNRImproves(), AnyOf(FluxNotLost(), BeamStable(0.05))))

    @pytest.mark.parametrize("combinator", [AllOf, AnyOf])
    def test_combinators_need_a_policy(self, combinator):
        with pytest.raises(ValueError):
            combinator()


class TestEnoughAntennas:

    def test_limit_comes_from_the_step(self):
        """Phase needs 3 antennas per interval, amplitude 4, whatever the array."""
        counts = (3, 3, 5)

        assert EnoughAntennas().decide(
            BEFORE, metrics(step=Phasecal("inf"), antennas=counts)
        ).accepted
        assert not EnoughAntennas(
        ).decide(BEFORE, metrics(step=AmpPhasecal("inf"), antennas=counts)).accepted

    def test_explicit_limit_and_allowed_fraction(self):
        counts = (10, 10, 10, 2)

        assert not EnoughAntennas(min_antennas=4).decide(BEFORE, metrics(antennas=counts)).accepted
        assert EnoughAntennas(min_antennas=4, max_short_fraction=0.25
                              ).decide(BEFORE, metrics(antennas=counts)).accepted

    def test_rounds_without_solutions_pass(self):
        assert EnoughAntennas().decide(BEFORE, metrics(step=Phasecal("inf"))).accepted


class TestCheck:

    def test_custom_metric(self):
        closure = Check(
            "closure phases improve",
            lambda previous, current: current.extra["closure_rms"] < previous.extra["closure_rms"],
        )
        before = metrics(extra={"closure_rms": 5.0})

        assert closure.decide(before, metrics(extra={"closure_rms": 3.0})).accepted
        decision = closure.decide(before, metrics(extra={"closure_rms": 6.0}))
        assert not decision.accepted and decision.reasons == ("closure phases improve: failed", )

    def test_composes_with_the_other_policies(self):
        policy = AllOf(
            PSNRImproves(), Check("peak above 1 Jy", lambda p, c: c.statistics.peak > 1.0)
        )

        assert policy.decide(BEFORE, metrics(peak=1.5)).accepted
        assert not policy.decide(BEFORE, metrics(peak=0.99, rms=0.1e-3)).accepted

    def test_extra_metrics_are_read_only(self):
        with pytest.raises(TypeError):
            metrics(extra={"a": 1.0}).extra["a"] = 2.0


class TestStepDefaults:
    """Default policies hold for any interferometer: their limits come from theory, not an instrument."""

    def test_phase_rounds_check_psnr_and_antennas(self):
        policy = Phasecal("inf").acceptance_policy()

        assert policy == AllOf(PSNRImproves(), EnoughAntennas())

    def test_amplitude_rounds_also_check_the_model_flux(self):
        policy = AmpPhasecal("inf").acceptance_policy()

        assert policy == AllOf(PSNRImproves(), EnoughAntennas(), FluxNotLost())

    def test_higher_psnr_with_flux_absorbed_is_rejected_for_amplitude(self):
        """The failure PSNR-only acceptance misses: amplitude gains absorbing flux and noise."""
        step = AmpPhasecal("inf")
        current = metrics(peak=0.9, rms=0.4e-3, flux=0.7, step=step, antennas=(8, 8))
        assert current.statistics.psnr > BEFORE.statistics.psnr

        assert not step.acceptance_policy().decide(BEFORE, current).accepted

    def test_higher_psnr_with_too_few_antennas_is_rejected(self):
        step = Phasecal("int")
        current = metrics(rms=0.5e-3, step=step, antennas=(2, 2, 8))

        assert not step.acceptance_policy().decide(BEFORE, current).accepted

    def test_step_policy_wins_over_the_run_policy(self):
        own = AllOf(PSNRImproves(min_ratio=1.2))
        run = AllOf(PSNRImproves())

        assert Phasecal("inf", acceptance=own).acceptance_policy(run) is own
        assert Phasecal("inf").acceptance_policy(run) is run

    def test_acceptance_must_be_a_policy(self):
        with pytest.raises(TypeError, match="AcceptancePolicy"):
            Phasecal("inf", acceptance=lambda p, c: True)


@pytest.mark.parametrize("policy", [BeamStable, FlaggedFractionBelow])
def test_instrument_dependent_checks_have_no_default(policy):
    with pytest.raises(TypeError):
        policy()


@pytest.mark.parametrize(
    "build", [
        lambda: PSNRImproves(min_ratio=0),
        lambda: FluxNotLost(max_loss=1.5),
        lambda: BeamStable(tolerance=-0.1),
        lambda: FlaggedFractionBelow(max_fraction=2),
        lambda: EnoughAntennas(min_antennas=1),
        lambda: EnoughAntennas(max_short_fraction=1.5),
        lambda: RoundMetrics(BEFORE.statistics, flagged_fraction=1.5),
    ]
)
def test_invalid_parameters_raise(build):
    with pytest.raises(ValueError):
        build()


class TestSolutionSNR:

    def test_formula(self):
        snr = solution_snr(
            flux=1.0,
            sigma_visibility=0.5,
            nantennas=10,
            nintegrations=6,
            nchannels=4,
            npolarizations=2
        )

        assert snr == pytest.approx(2.0 * 3.0 * math.sqrt(48))

    def test_longer_solint_raises_the_snr(self):
        assert solution_snr(0.01, 1.0, 27, 60) > solution_snr(0.01, 1.0, 27, 1)

    @pytest.mark.parametrize("arguments", [(1.0, 0.0, 10, 1), (1.0, 1.0, 1, 1), (1.0, 1.0, 10, 0)])
    def test_invalid_inputs_raise(self, arguments):
        with pytest.raises(ValueError):
            solution_snr(*arguments)
