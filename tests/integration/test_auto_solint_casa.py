"""
#66 on simulated data: Repeat(Phasecal("auto")) takes the shortest solint the data support. At
high S/N that is one integration; with the noise raised, a longer solint.
"""
import pytest

from wenuray.config import ImageGeometry, SolveOptions
from wenuray.imaging.backends import Tclean
from wenuray.selfcalibration import Phasecal, Repeat, Selfcal
from wenuray.selfcalibration.steps import solint_seconds
from wenuray.testing import simulate_point_source_ms

IMAGER = Tclean(ImageGeometry(imsize=128, cell="1arcsec"), niter=200, nsigma=3.0)
SOLVE = SolveOptions(refant="A0", minblperant=3)


def resolved(result) -> list:
    """The solints "auto" resolved to, round by round"""
    return [record.solint for record in result.rounds[1:]]


def run(tmp_path, noise):
    ms = simulate_point_source_ms(str(tmp_path / "target.ms"), noise=noise)
    return Selfcal(ms, IMAGER, [Repeat(Phasecal("auto"))], solve=SOLVE).run(observers=())


def test_at_high_snr_the_ladder_reaches_one_integration(tmp_path):
    # 1 Jy over 0.05 Jy: about 75 per integration, far above 2 x minsnr
    result = run(tmp_path, noise=0.05)

    assert resolved(result)[0] == "int"
    assert result.rounds[1].accepted


def test_with_more_noise_it_stops_at_a_longer_solint(tmp_path):
    # 1 Jy over 1 Jy: about 3.7 per 60 s integration, so 2 x minsnr needs a few of them
    result = run(tmp_path, noise=1.0)

    solints = resolved(result)
    assert solints and all(solint_seconds(solint) > 60.0 for solint in solints)
    assert result.rounds[1].accepted
    # Each round of the repeat is shorter than the last
    lengths = [solint_seconds(solint) for solint in solints]
    assert lengths == sorted(lengths, reverse=True) and len(set(lengths)) == len(lengths)


@pytest.mark.parametrize("noise", [0.05, 1.0])
def test_the_resolved_solints_are_recorded(tmp_path, noise):
    result = run(tmp_path, noise)

    assert all("'auto' -> " in record.step for record in result.rounds[1:])
    assert all(record.solint for record in result.rounds[1:])
