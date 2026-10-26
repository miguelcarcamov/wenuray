"""ImageStatistics (#28): one noise estimator for every imager, with the source masked."""
import numpy as np
import pytest

from wenuray.imaging.statistics import ImageStatistics


@pytest.fixture
def images():
    """A 1 Jy/beam point source on Gaussian noise of 1 mJy/beam, with a strong source residual."""
    rng = np.random.default_rng(0)
    residual = rng.normal(0.0, 1e-3, (128, 128))
    residual[60:68, 60:68] += 0.05  # unmodelled emission at the source position
    restored = residual.copy()
    restored[64, 64] += 1.0
    model = np.zeros((128, 128))
    model[64, 64] = 1.0
    mask = np.zeros((128, 128), dtype=bool)
    mask[56:72, 56:72] = True
    return restored, residual, model, mask


def test_noise_excludes_the_source_mask(images):
    restored, residual, model, mask = images

    statistics = ImageStatistics.from_arrays(restored, residual, model, source_mask=mask)

    assert statistics.rms == pytest.approx(1e-3, rel=0.05)
    assert statistics.peak == pytest.approx(1.05, rel=0.01)
    assert statistics.psnr == pytest.approx(statistics.peak / statistics.rms)
    assert statistics.model_flux == pytest.approx(1.0)


def test_sigma_clipping_removes_source_residuals_without_a_mask(images):
    restored, residual, model, _ = images

    statistics = ImageStatistics.from_arrays(restored, residual, model)

    assert statistics.rms == pytest.approx(1e-3, rel=0.1)


def test_nan_pixels_are_ignored(images):
    restored, residual, model, mask = images
    residual = residual.copy()
    residual[:, :10] = np.nan
    restored = restored.copy()
    restored[:, :10] = np.nan

    statistics = ImageStatistics.from_arrays(restored, residual, model, source_mask=mask)

    assert np.isfinite(statistics.rms) and np.isfinite(statistics.peak)


def test_a_two_dimensional_mask_applies_to_every_plane(images):
    restored, residual, model, mask = images

    statistics = ImageStatistics.from_arrays(
        restored[None, None], residual[None, None], model[None, None], source_mask=mask
    )

    assert statistics.rms == pytest.approx(1e-3, rel=0.05)


def test_a_residual_with_no_noise_left_raises(images):
    restored, residual, model, _ = images

    with pytest.raises(ValueError, match="no finite noise"):
        ImageStatistics.from_arrays(
            restored, residual, model, source_mask=np.ones_like(residual, dtype=bool)
        )


@pytest.mark.parametrize("beam", [(1.0, 0.5), (0.0, 0.5, 30.0), (1.0, -0.5, 30.0)])
def test_invalid_beam_raises(beam):
    with pytest.raises(ValueError, match="beam"):
        ImageStatistics(peak=1.0, rms=1e-3, model_flux=1.0, beam=beam)


def test_rms_must_be_positive():
    with pytest.raises(ValueError, match="rms"):
        ImageStatistics(peak=1.0, rms=0.0, model_flux=1.0)
