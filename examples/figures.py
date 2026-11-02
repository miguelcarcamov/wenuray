"""
Figures for Wenuray's examples (the tutorial), in one style: images in magma with a
square-root stretch, so faint emission shows beside a bright core; residuals in red and blue on one
symmetric scale; a beam ellipse and a scale bar on every image; the same colours for "before",
"after" and "ideal" in every chart.

Needs matplotlib (pip install "wenuray[plots]") and CASA, and the files pyralysis_simulation.py
writes next to each observation (<ms>.ideal, <ms>.gains.npz).
"""
import math
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from wenuray.imaging.contract import Imager, ImagingRequest, ImagingResult
from wenuray.selfcalibration import SelfcalResult
from wenuray.utils.image_utils import read_image
from wenuray.utils.selfcal_utils import open_table

BEFORE, AFTER, IDEAL = "#d1495b", "#00798c", "#30343f"


def style() -> None:
    """
    The figures' look: clean axes, readable sizes, white backgrounds
    """
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.dpi": 120,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "legend.frameon": False,
        }
    )


@dataclass(frozen=True)
class Panel:
    """
    A square cut of an image around its centre, in mJy/beam, with its beam and pixel size
    """
    data: np.ndarray
    cell: float  # arcsec
    beam: Optional[Tuple[float, float, float]]  # major, minor (arcsec), position angle (deg)

    @classmethod
    def read(cls, path: str, cell: float, extent: float) -> "Panel":
        data, beam = read_image(path)
        # CASA images are (x, y), FITS files (y, x); panels are (y, x)
        data = np.squeeze(data) * 1e3
        if os.path.isdir(path):
            data = data.T
        centre, half = data.shape[0] // 2, max(1, int(round(extent / cell)))
        return cls(data[centre - half:centre + half, centre - half:centre + half], cell, beam)

    @property
    def extent(self) -> float:
        return self.data.shape[0] * self.cell / 2


def draw(axis, panel: Panel, norm, cmap: str, label: str = "", corner: str = "") -> object:
    """
    Draws a panel with its beam (bottom left), a scale bar (bottom right) and labels
    """
    from matplotlib.patches import Ellipse, Rectangle

    half = panel.extent
    shown = axis.imshow(
        panel.data,
        origin="lower",
        cmap=cmap,
        norm=norm,
        extent=(half, -half, -half, half),
        interpolation="bicubic"
    )
    axis.set_xticks([])
    axis.set_yticks([])
    axis.grid(False)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color("#888888")
    colour = "white" if cmap != "RdBu_r" else "#30343f"
    if panel.beam is not None:
        major, minor, angle = panel.beam
        axis.add_patch(
            Ellipse(
                (half - 0.12 * 2 * half, -half + 0.12 * 2 * half),
                minor,
                major,
                angle=-angle,
                facecolor="none",
                edgecolor=colour,
                linewidth=1.2
            )
        )
    length = scale_length(0.25 * 2 * half)
    right = -half + 0.06 * 2 * half
    axis.add_patch(
        Rectangle((right, -half + 0.07 * 2 * half), length, 0.012 * 2 * half, color=colour)
    )
    axis.text(
        right + length / 2,
        -half + 0.1 * 2 * half,
        arcsec_label(length),
        color=colour,
        ha="center",
        va="bottom",
        fontsize=8
    )
    # Labels on a translucent box, readable on bright and saturated pixels alike
    box = {
        "boxstyle": "round,pad=0.25",
        "linewidth": 0,
        "alpha": 0.55,
        "facecolor": "black" if colour == "white" else "white"
    }
    if label:
        axis.text(
            0.04,
            0.96,
            label,
            transform=axis.transAxes,
            color=colour,
            ha="left",
            va="top",
            fontsize=9,
            fontweight="bold",
            bbox=box
        )
    if corner:
        axis.text(
            0.96,
            0.96,
            corner,
            transform=axis.transAxes,
            color=colour,
            ha="right",
            va="top",
            fontsize=8,
            bbox=box
        )
    return shown


def nice_length(length: float) -> float:
    """
    The round number (1, 2 or 5 times a power of ten) nearest below length
    """
    power = 10**math.floor(math.log10(length))
    return max(step * power for step in (1, 2, 5) if step * power <= length)


def scale_length(length: float) -> float:
    """
    A scale bar's length, in arcsec, at most length: round in the unit it is shown in
    """
    unit = 60.0 if length >= 60 else 1.0 if length >= 1 else 1e-3
    return nice_length(length / unit) * unit


def arcsec_label(length: float) -> str:
    if length >= 60:
        return "{0:g}′".format(round(length / 60, 6))
    if length >= 1:
        return "{0:g}″".format(round(length, 6))
    return "{0:g} mas".format(round(length * 1e3, 6))


def colorbar(figure, shown, axes, label: str) -> None:
    from matplotlib.ticker import MaxNLocator

    bar = figure.colorbar(shown, ax=axes, label=label, shrink=0.85, pad=0.02)
    low, high = shown.norm.vmin, shown.norm.vmax
    bar.set_ticks([tick for tick in MaxNLocator(5).tick_values(low, high) if low <= tick <= high])


def plain_log(axis, which: str = "x") -> None:
    """
    A log axis labelled with plain numbers (0.5, 1, 2, 100, 1000), not powers of ten
    """
    from matplotlib.ticker import FuncFormatter, LogLocator

    scale = axis.xaxis if which == "x" else axis.yaxis
    scale.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
    scale.set_major_formatter(FuncFormatter(lambda value, _: "{0:g}".format(value)))
    scale.set_minor_formatter(FuncFormatter(lambda value, _: ""))


def image_norm(panels: Sequence[Panel]):
    """
    A square-root stretch from a little below zero to the brightest panel's peak
    """
    from matplotlib.colors import PowerNorm

    peak = max(np.nanmax(panel.data) for panel in panels)
    return PowerNorm(0.5, vmin=0.0, vmax=peak, clip=True)


def residual_norm(reference: Panel, sigmas: float = 6.0):
    """
    A symmetric scale of sigmas times the reference residual's noise (MAD standard deviation)
    """
    from astropy.stats import mad_std
    from matplotlib.colors import Normalize

    limit = sigmas * float(mad_std(reference.data, ignore_nan=True))
    return Normalize(-limit, limit, clip=True)


def image_ideal(ms: str, imager: Imager, result: SelfcalResult, output: str) -> ImagingResult:
    """
    The data without gain errors (<ms>.ideal), imaged with the run's grid and imager settings
    """
    from casatasks import clearcal

    from wenuray.config import ImageGeometry

    grid = result.provenance.get("geometry")
    if grid:
        imager = imager.with_overrides(
            geometry=ImageGeometry(imsize=tuple(grid["imsize"]), cell=grid["cell"])
        )
    # The ideal copy has no CORRECTED_DATA yet; equal to DATA, it is what the imager images
    clearcal(vis=ms + ".ideal", addmodel=False)
    return imager.image(ImagingRequest(ms + ".ideal", output + "/ideal"))


def residual_path(image: str) -> str:
    """
    The residual next to a restored image: tclean's <name>.residual, Pyralysis's
    <name>.residual.fits
    """
    if image.endswith(".image.fits"):
        return image[:-len(".image.fits")] + ".residual.fits"
    return image[:-len(".image")] + ".residual"


def cell_of(result: SelfcalResult) -> float:
    """
    The run's cell in arcsec: the grid an "auto" geometry was resolved to, or the first image's
    """
    grid = result.provenance.get("geometry")
    if grid:
        return float(grid["cell"].replace("arcsec", ""))
    from casatools import image

    reader = image()
    reader.open(result.rounds[0].image)
    try:
        increment = reader.coordsys().increment(type="direction")["numeric"]
    finally:
        reader.close()
    return float(abs(np.degrees(increment[0])) * 3600)


def before_after(
    result: SelfcalResult, ideal: ImagingResult, path: str, extent: float, title: str = ""
) -> None:
    """
    The initial, self-calibrated and ideal images (top) and their residuals (bottom)
    """
    import matplotlib.pyplot as plt

    cell = cell_of(result)
    images = [
        Panel.read(p, cell, extent)
        for p in (result.rounds[0].image, result.final.image, ideal.restored)
    ]
    residuals = [
        Panel.read(residual_path(p), cell, extent)
        for p in (result.rounds[0].image, result.final.image, ideal.restored)
    ]
    floor = ideal.statistics().rms
    labels = ["Before self-calibration", "Self-calibrated", "No gain errors (ideal)"]
    corners = [
        "PSNR {0:.0f}".format(result.rounds[0].psnr),
        "PSNR {0:.0f}".format(result.final.psnr),
        "PSNR {0:.0f}".format(ideal.statistics().psnr),
    ]
    noise = [
        "noise {0:.1f}× ideal".format(record.rms / floor)
        for record in (result.rounds[0], result.final)
    ] + ["thermal floor"]
    figure, axes = plt.subplots(2, 3, figsize=(11, 7.4), constrained_layout=True)
    norm = image_norm(images)
    for axis, panel, label, corner in zip(axes[0], images, labels, corners):
        shown = draw(axis, panel, norm, "magma", label, corner)
    colorbar(figure, shown, axes[0], "mJy/beam")
    scale = residual_norm(residuals[2])
    for axis, panel, corner in zip(axes[1], residuals, noise):
        shown = draw(axis, panel, scale, "RdBu_r", "Residual", corner)
    colorbar(figure, shown, axes[1], "mJy/beam (±6σ of the ideal)")
    if title:
        figure.suptitle(title, fontsize=12, fontweight="bold")
    figure.savefig(path)
    plt.close(figure)


def round_labels(result: SelfcalResult) -> List[str]:
    labels = ["initial"]
    for record in result.rounds[1:]:
        kind, solint = record.step.split("(", 1)
        labels.append(
            "{0}\n{1}".format(
                kind.replace("cal", "").replace("AmpPhase", "amp+phase").lower(),
                solint.split(",")[0].strip("')")
            )
        )
    return labels


def convergence(result: SelfcalResult, floor: float, path: str) -> None:
    """
    Each round's peak S/N, its noise against the ideal image's (the thermal floor), and its gain
    corrections against their noise
    """
    import matplotlib.pyplot as plt

    rounds = result.rounds
    x = np.arange(len(rounds))
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    axes[0].plot(x, [r.psnr for r in rounds], "o-", color=AFTER)
    axes[0].set_yscale("log")
    axes[0].set_title("Peak S/N of the image")
    axes[1].plot(x, [r.rms / floor for r in rounds], "o-", color=AFTER)
    axes[1].axhline(1, color=IDEAL, linestyle=":", label="ideal (thermal floor)")
    axes[1].set_yscale("log")
    axes[1].set_title("Image noise / ideal image's")
    axes[1].legend()
    solved = [(i, r.gains.phase_over_noise) for i, r in enumerate(rounds) if r.gains]
    axes[2].plot(*zip(*solved), "o-", color=AFTER, label="phase")
    amplitude = [
        (i, r.gains.amplitude_over_noise) for i, r in enumerate(rounds)
        if r.gains and r.gains.amplitude_over_noise is not None
    ]
    if amplitude:
        axes[2].plot(*zip(*amplitude), "s", color=BEFORE, label="amplitude")
    axes[2].axhline(1, color=IDEAL, linestyle=":", label="at their noise")
    axes[2].set_yscale("log")
    axes[2].set_title("Gain corrections / their noise")
    axes[2].legend()
    for axis in axes:
        axis.set_xticks(x, round_labels(result), fontsize=8)
        plain_log(axis, "y")
    figure.savefig(path)
    plt.close(figure)


def solved_gains(result: SelfcalResult, times: np.ndarray, nantennas: int) -> np.ndarray:
    """
    Each antenna's gain from all accepted tables together, at the given times: the product of
    the tables' gains, each taken from its solution nearest in time
    """
    total = np.ones((nantennas, len(times)), dtype=complex)
    for table in result.stack:
        with open_table(table.path) as cal:
            gains, flags = cal.getcol("CPARAM")[0, 0], cal.getcol("FLAG")[0, 0]
            antennas, epochs = cal.getcol("ANTENNA1"), cal.getcol("TIME")
        for antenna in range(nantennas):
            rows = (antennas == antenna) & ~flags
            if rows.any():
                nearest = np.abs(times[:, None] - epochs[rows][None, :]).argmin(axis=1)
                total[antenna] *= gains[rows][nearest]
    return total


def phase_errors(ms: str, result: SelfcalResult) -> Tuple[np.ndarray, np.ndarray, list, int]:
    """
    The injected and the self-calibrated phases (degrees, antenna by time), both relative to the
    reference antenna, the antenna names and the reference's index
    """
    injected = np.load(ms + ".gains.npz")
    times, truth, names = injected["times"], injected["gains"], list(injected["names"])
    reference = int(injected["reference"])
    found = solved_gains(result, times, len(names))
    found = found * np.conj(found[reference]) / np.abs(found[reference])
    return np.degrees(np.angle(truth)), np.degrees(np.angle(found)), names, reference


def wrapped(difference: np.ndarray) -> np.ndarray:
    return (difference + 180) % 360 - 180


def gains(ms: str, result: SelfcalResult, path: str) -> None:
    """
    The phases self-calibration found against those injected: one antenna over time, and every
    antenna at every time
    """
    import matplotlib.pyplot as plt

    truth, found, names, reference = phase_errors(ms, result)
    times = np.load(ms + ".gains.npz")["times"]
    others = [antenna for antenna in range(len(names)) if antenna != reference]
    shown = others[len(others) // 2]
    minutes = (times - times[0]) / 60
    error = wrapped(found[others] - truth[others])
    figure, axes = plt.subplots(
        1, 2, figsize=(12, 4), constrained_layout=True, gridspec_kw={"width_ratios": [2, 1]}
    )
    axes[0].plot(minutes, truth[shown], color=IDEAL, linewidth=1, label="injected")
    axes[0].plot(minutes, found[shown], "o", color=AFTER, markersize=3.5, label="self-calibrated")
    axes[0].set_xlabel("Time (minutes)")
    axes[0].set_ylabel("Phase of {0} − {1} (deg)".format(names[shown], names[reference]))
    axes[0].legend(loc="upper right")
    axes[1].plot(
        truth[others].ravel(), found[others].ravel(), ".", color=AFTER, markersize=1, alpha=0.3
    )
    axes[1].plot([-180, 180], [-180, 180], color=IDEAL, linestyle=":")
    axes[1].set_xlim(-180, 180)
    axes[1].set_ylim(-180, 180)
    axes[1].set_xlabel("Injected phase (deg)")
    axes[1].set_ylabel("Self-calibrated phase (deg)")
    axes[1].set_title("All antennas: {0:.1f}° rms difference".format(np.sqrt(np.mean(error**2))))
    figure.savefig(path)
    plt.close(figure)
