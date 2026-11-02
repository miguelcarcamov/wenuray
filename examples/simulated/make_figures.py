"""
Makes the tutorial's figures (docs/tutorial.md) from a run of recipe.toml on the data simulate.py
made:

    python examples/simulated/make_figures.py tutorial.ms tutorial_selfcal docs/_static/tutorial

- images.png: the initial image, the self-calibrated one, and the data imaged without gain errors
  (the ideal a perfect calibration reaches), with their residuals below;
- convergence.png: each round's PSNR, its noise against the ideal image's (the thermal floor), and
  its gain corrections against their noise;
- gains.png: the antenna phases self-calibration found, against those simulate.py injected.

Needs matplotlib (`pip install "wenuray[plots]"`) and CASA.
"""
import argparse
import os

import numpy as np

from wenuray.config import ImageGeometry
from wenuray.imaging.backends import Tclean
from wenuray.imaging.contract import ImagingRequest
from wenuray.selfcalibration import SelfcalResult
from wenuray.utils.image_utils import read_image
from wenuray.utils.selfcal_utils import open_table


def image_ideal(ms: str, result: SelfcalResult, output: str):
    """
    The data without gain errors (simulate.py's <ms>.ideal), imaged with the run's grid and
    settings
    """
    grid = result.provenance["geometry"]
    imager = Tclean(
        ImageGeometry(imsize=tuple(grid["imsize"]), cell=grid["cell"]),
        niter=2000,
        nsigma=3.0,
        usemask="auto-multithresh"
    )
    # The ideal copy has no CORRECTED_DATA yet: tclean images CORRECTED_DATA, which equals DATA here
    from casatasks import clearcal

    clearcal(vis=ms + ".ideal", addmodel=False)
    return imager.image(ImagingRequest(ms + ".ideal", os.path.join(output, "ideal")))


def plot_images(result: SelfcalResult, ideal, path: str, extent: float = 1.0) -> None:
    """
    The initial, self-calibrated and ideal images (top) and their residuals (bottom), each row on
    one colour scale, within extent arcsec of the centre
    """
    import matplotlib.pyplot as plt

    def residual(image: str) -> str:
        return image[:-len(".image")] + ".residual"

    columns = [
        ("Initial image", result.rounds[0].image, "PSNR {0:.0f}".format(result.rounds[0].psnr)),
        ("Self-calibrated", result.final.image, "PSNR {0:.0f}".format(result.final.psnr)),
        ("No gain errors (ideal)", ideal.restored, "same noise, true gains"),
    ]
    cell = float(result.provenance["geometry"]["cell"].replace("arcsec", ""))
    rows = []
    for paths in ([image for _, image, _ in columns], [residual(image) for _, image, _ in columns]):
        images = [np.squeeze(read_image(image)[0]).T * 1e3 for image in paths]
        centre = images[0].shape[0] // 2
        cut = int(round(extent / cell))
        rows.append(
            [image[centre - cut:centre + cut, centre - cut:centre + cut] for image in images]
        )
    figure, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    peak = max(np.nanmax(image) for image in rows[0])
    spread = max(np.nanmax(np.abs(image)) for image in rows[1])
    for row, (images, scale, label) in enumerate(
        (
            (rows[0], (-0.05 * peak, peak), "mJy/beam"),
            (rows[1], (-spread, spread), "residual, mJy/beam")
        )
    ):
        for axis, image, (title, _, subtitle) in zip(axes[row], images, columns):
            shown = axis.imshow(
                image,
                origin="lower",
                cmap="inferno" if row == 0 else "RdBu_r",
                vmin=scale[0],
                vmax=scale[1],
                extent=(extent, -extent, -extent, extent)
            )
            if row == 0:
                axis.set_title("{0}\n{1}".format(title, subtitle))
            axis.set_xlabel("ΔRA (arcsec)")
        axes[row][0].set_ylabel("ΔDec (arcsec)")
        figure.colorbar(shown, ax=axes[row], label=label, shrink=0.9)
    figure.savefig(path, dpi=110)
    plt.close(figure)


def plot_convergence(result: SelfcalResult, floor: float, path: str) -> None:
    """
    floor is the ideal image's noise: the same thermal noise, no gain errors
    """
    import matplotlib.pyplot as plt

    rounds = result.rounds
    labels = ["initial"] + [
        record.step.split("(")[1].rstrip(")").strip("'") + "\n" +
        record.step.split("(")[0].replace("cal", "") for record in rounds[1:]
    ]
    x = np.arange(len(rounds))
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    axes[0].plot(x, [record.psnr for record in rounds], "o-")
    axes[0].set_yscale("log")
    axes[0].set_title("Peak S/N of the image")
    axes[1].plot(x, [record.rms / floor for record in rounds], "o-")
    axes[1].axhline(1, color="grey", linestyle=":")
    axes[1].set_yscale("log")
    axes[1].set_title("Image noise / ideal image's")
    solved = [(i, record.gains.phase_over_noise) for i, record in enumerate(rounds) if record.gains]
    axes[2].plot(*zip(*solved), "o-", label="phase")
    amplitude = [
        (i, record.gains.amplitude_over_noise) for i, record in enumerate(rounds)
        if record.gains and record.gains.amplitude_over_noise is not None
    ]
    if amplitude:
        axes[2].plot(*zip(*amplitude), "s", label="amplitude")
    axes[2].axhline(1, color="grey", linestyle=":")
    axes[2].set_yscale("log")
    axes[2].set_title("Gain corrections / their noise")
    axes[2].legend()
    for axis in axes:
        axis.set_xticks(x, labels, fontsize=8)
    figure.savefig(path, dpi=110)
    plt.close(figure)


def solved_gains(result: SelfcalResult, times: np.ndarray, nantennas: int) -> np.ndarray:
    """
    Each antenna's gain from all accepted tables together, at the given times: the product of
    the tables' gains, each taken from its solution nearest in time (one per integration here)
    """
    total = np.ones((nantennas, len(times)), dtype=complex)
    for table in result.stack:
        with open_table(table.path) as cal:
            gains, flags = cal.getcol("CPARAM")[0, 0], cal.getcol("FLAG")[0, 0]
            antennas, epochs = cal.getcol("ANTENNA1"), cal.getcol("TIME")
        for antenna in range(nantennas):
            rows = (antennas == antenna) & ~flags
            if not rows.any():
                continue
            nearest = np.abs(times[:, None] - epochs[rows][None, :]).argmin(axis=1)
            total[antenna] *= gains[rows][nearest]
    return total


def plot_gains(ms: str, result: SelfcalResult, path: str) -> None:
    """
    The phases self-calibration found against those simulate.py injected, both relative to the
    reference antenna: one antenna over time, and every antenna at every time
    """
    import matplotlib.pyplot as plt

    injected = np.load(ms + ".gains.npz")
    times, truth, names = injected["times"], injected["gains"], injected["names"]
    reference = int(injected["reference"])
    found = solved_gains(result, times, len(names))
    found = found * np.conj(found[reference]) / np.abs(found[reference])
    true_phase = np.degrees(np.angle(truth))
    found_phase = np.degrees(np.angle(found))
    error = (found_phase - true_phase + 180) % 360 - 180
    others = [antenna for antenna in range(len(names)) if antenna != reference]
    shown = others[len(others) // 2]
    minutes = (times - times[0]) / 60
    figure, axes = plt.subplots(
        1, 2, figsize=(12, 4), constrained_layout=True, gridspec_kw={"width_ratios": [2, 1]}
    )
    axes[0].plot(minutes, true_phase[shown], color="grey", label="injected")
    axes[0].plot(minutes, found_phase[shown], ".", color="C1", label="self-calibrated")
    axes[0].set_xlabel("Time (minutes)")
    axes[0].set_ylabel("Phase of {0} relative to {1} (deg)".format(names[shown], names[reference]))
    axes[0].legend()
    axes[1].plot(true_phase[others].ravel(), found_phase[others].ravel(), ",", alpha=0.5)
    axes[1].plot([-180, 180], [-180, 180], color="grey", linestyle=":")
    axes[1].set_xlabel("Injected phase (deg)")
    axes[1].set_ylabel("Self-calibrated phase (deg)")
    axes[1].set_title(
        "All antennas: rms difference {0:.1f} deg".format(np.sqrt(np.mean(error[others]**2)))
    )
    figure.savefig(path, dpi=110)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ms", help="the measurement set simulate.py made")
    parser.add_argument("run", help="the run's output directory")
    parser.add_argument("figures", help="where to write the figures")
    args = parser.parse_args()
    os.makedirs(args.figures, exist_ok=True)
    result = SelfcalResult.from_toml(os.path.join(args.run, "selfcal.toml"))
    ideal = image_ideal(args.ms, result, args.run)
    plot_images(result, ideal, os.path.join(args.figures, "images.png"))
    plot_convergence(result, ideal.statistics().rms, os.path.join(args.figures, "convergence.png"))
    plot_gains(args.ms, result, os.path.join(args.figures, "gains.png"))


if __name__ == "__main__":
    main()
