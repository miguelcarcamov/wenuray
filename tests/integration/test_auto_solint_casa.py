"""
#66 on simulated data: Repeat(Phasecal("auto")) shortens the solint step by step, from a scan down,
as long as the gains correct more than their noise. At high S/N, coming down step by step calibrates
far better than jumping to the shortest solint the data support at once (max_shortening=1).
"""
import math

import pytest

from wenuray.config import ImageGeometry, SolveOptions
from wenuray.imaging.backends import Tclean
from wenuray.selfcalibration import AmpPhasecal, Phasecal, Repeat, Selfcal
from wenuray.selfcalibration.steps import solint_seconds
from wenuray.testing import simulate_point_source_ms

IMAGER = Tclean(ImageGeometry(imsize=128, cell="1arcsec"), niter=200, nsigma=3.0)
SOLVE = SolveOptions(refant="A0", minblperant=3)
# The simulation is one scan of two hours
SCAN = 7200.0


def resolved(result) -> list:
    """The solints "auto" resolved to, round by round"""
    return [record.solint for record in result.rounds[1:]]


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """Each schedule on each simulation, run once for the module"""
    cache = {}

    def run(noise, max_shortening=0.5):
        key = (noise, max_shortening)
        if key not in cache:
            directory = tmp_path_factory.mktemp("auto")
            ms = simulate_point_source_ms(str(directory / "target.ms"), noise=noise)
            steps = [Repeat(Phasecal("auto"), max_shortening=max_shortening), AmpPhasecal("auto")]
            cache[key] = Selfcal(ms, IMAGER, steps, solve=SOLVE,
                                 output=str(directory / "selfcal")).run(observers=())
        return cache[key]

    return run


def phase_rounds(result) -> list:
    return [record for record in result.rounds[1:] if record.step.startswith("Phasecal")]


def test_at_high_snr_it_comes_down_from_a_scan_step_by_step(runs):
    # 1 Jy over 0.05 Jy: about 75 per integration, so S/N alone would allow one at once
    result = runs(0.05)

    lengths = [min(solint_seconds(record.solint), SCAN) for record in phase_rounds(result)]
    assert phase_rounds(result)[0].solint == "inf"
    assert len(lengths) > 2
    # Each round at least half the last and at most 3/4 of it
    assert all(0.5 * last <= this <= 0.75 * last for last, this in zip(lengths, lengths[1:]))
    assert all(record.accepted for record in result.rounds)


def test_coming_down_step_by_step_calibrates_better_than_jumping(runs):
    gradual, jump = runs(0.05), runs(0.05, max_shortening=1.0)

    assert phase_rounds(jump)[0].solint == "int"
    # On this simulation: PSNR 849 against 249
    assert gradual.final.psnr > 2 * jump.final.psnr


def test_with_more_noise_it_stops_at_a_longer_solint(runs):
    # 1 Jy over 1 Jy: about 3.7 per 60 s integration, so 2 x minsnr needs a few of them
    result = runs(1.0)

    solints = resolved(result)
    assert solints and all(solint_seconds(solint) > 60.0 for solint in solints)
    assert result.rounds[1].accepted
    lengths = [solint_seconds(record.solint) for record in phase_rounds(result)]
    assert lengths == sorted(lengths, reverse=True) and len(set(lengths)) == len(lengths)


@pytest.mark.parametrize("noise", [0.05, 1.0])
def test_the_resolved_solints_are_recorded(runs, noise):
    result = runs(noise)

    assert all("'auto' -> " in record.step for record in result.rounds[1:])
    assert all(record.solint for record in result.rounds[1:])
    assert not math.isnan(result.final.psnr)
