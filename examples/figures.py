"""
Figures for Wenuray's examples (the tutorial and the gallery), in one style: images in magma with a
square-root stretch, so faint emission shows beside a bright core; residuals in red and blue on one
symmetric scale; a beam ellipse and a scale bar on every image; the same colours for "before",
"after" and "ideal" in every chart.

Needs matplotlib (pip install "wenuray[plots]") and CASA, and the files pyralysis_simulation.py
writes next to each observation (<ms>.ideal, <ms>.gains.npz).
"""
import dataclasses
import json
import math
import os
from dataclasses import dataclass
from typing import List, Mapping, Optional, Sequence, Tuple

import numpy as np

from wenuray.imaging.contract import Imager, ImagingRequest, ImagingResult
from wenuray.selfcalibration import SelfcalResult
from wenuray.utils.image_utils import read_image
from wenuray.utils.selfcal_utils import open_table

BEFORE, AFTER, IDEAL = "#d1495b", "#00798c", "#7b8496"
# Text and axes: mid grey, legible on white and on a dark page
GREY = "#808080"


def style() -> None:
    """
    The figures' look: clean axes, readable sizes, transparent backgrounds, and text and axes in a
    grey that reads on the documentation's light and dark themes alike
    """
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.facecolor": "none",
            "axes.facecolor": "none",
            "savefig.facecolor": "none",
            "savefig.transparent": True,
            "text.color": GREY,
            "axes.labelcolor": GREY,
            "axes.edgecolor": GREY,
            "xtick.color": GREY,
            "ytick.color": GREY,
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


def save(figure, path: str) -> None:
    """
    Saves a figure: as WebP at quality 90 for a .webp path, a seventh of a PNG's size for images
    with no visible difference; otherwise in the format the extension names
    """
    options = {"pil_kwargs": {"quality": 90, "method": 6}} if path.endswith(".webp") else {}
    figure.savefig(path, **options)


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
    """
    A colour bar right of a row of image panels, exactly as tall as the images: its axes are fixed
    to the last panel's, which an image's equal aspect sizes
    """
    from matplotlib.ticker import MaxNLocator

    bar = figure.colorbar(shown, cax=axes[-1].inset_axes([1.04, 0.0, 0.045, 1.0]), label=label)
    bar.outline.set_edgecolor(GREY)
    bar.ax.grid(False)
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


def residual_norm(noise: float, sigmas: float = 6.0):
    """
    A symmetric scale of sigmas times noise, in Jy/beam: the ideal image's, measured over the whole
    image with the source masked. The noise of a panel would follow what the model fits there: a
    model free inside its mask, as Pyralysis's, also fits the noise around the source
    """
    from matplotlib.colors import Normalize

    limit = sigmas * noise * 1e3
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
    # Imaged once: redrawing the figures reuses it
    record = output + "/ideal.json"
    if os.path.exists(record):
        with open(record) as file:
            return ImagingResult(**json.load(file))
    # The ideal copy has no CORRECTED_DATA yet; equal to DATA, it is what the imager images
    clearcal(vis=ms + ".ideal", addmodel=False)
    ideal = imager.image(ImagingRequest(ms + ".ideal", output + "/ideal"))
    with open(record, "w") as file:
        json.dump(dataclasses.asdict(ideal), file)
    return ideal


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
    scale = residual_norm(floor)
    for axis, panel, corner in zip(axes[1], residuals, noise):
        shown = draw(axis, panel, scale, "RdBu_r", "Residual", corner)
    colorbar(figure, shown, axes[1], "mJy/beam (±6σ of the ideal)")
    if title:
        figure.suptitle(title, fontsize=12, fontweight="bold")
    save(figure, path)
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
    save(figure, path)
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
    save(figure, path)
    plt.close(figure)


@dataclass(frozen=True)
class Scenario:
    """
    What the gallery shows of one scenario's run
    """
    title: str
    subtitle: str
    about: str
    result: SelfcalResult
    ideal: ImagingResult
    extent: float
    phase_error: float  # injected, rms degrees
    phase_residual: float  # after self-calibration, rms degrees
    imager: str = "tclean"


def gallery(scenarios: Sequence[Scenario], path: str, columns: int = 3) -> None:
    """
    One tile per scenario, before (left) and after (right), each on its own scale
    """
    import matplotlib.pyplot as plt

    rows = math.ceil(len(scenarios) / columns)
    figure, axes = plt.subplots(
        rows,
        2 * columns,
        figsize=(4.2 * columns, 2.35 * rows + 0.4),
        gridspec_kw={
            "wspace": 0.04,
            "hspace": 0.32
        }
    )
    axes = np.atleast_2d(axes)
    for index, scenario in enumerate(scenarios):
        row, column = divmod(index, columns)
        cell = cell_of(scenario.result)
        panels = [
            Panel.read(p, cell, scenario.extent)
            for p in (scenario.result.rounds[0].image, scenario.result.final.image)
        ]
        norm = image_norm(panels)
        for side, (panel, record, label) in enumerate(
            zip(panels, (scenario.result.rounds[0], scenario.result.final), ("before", "after"))
        ):
            axis = axes[row, 2 * column + side]
            draw(axis, panel, norm, "magma", label, "PSNR {0:.0f}".format(record.psnr))
        axes[row, 2 * column].set_title(
            "{0} · {1}\n{2}".format(scenario.title, scenario.subtitle, scenario.about),
            loc="left",
            fontsize=9,
            x=0.0
        )
    for index in range(len(scenarios), rows * columns):
        row, column = divmod(index, columns)
        axes[row, 2 * column].axis("off")
        axes[row, 2 * column + 1].axis("off")
    save(figure, path)
    plt.close(figure)


# How each imager's results are marked in the charts that compare them
MARKERS = {"tclean": "o", "Pyralysis": "s"}


def summary(groups: Mapping[str, Sequence[Scenario]], path: str) -> None:
    """
    For each scenario and imager: the peak S/N before, after and with no gain errors, and the phase
    errors injected against those left after self-calibration. groups maps each imager's name to
    its runs of the same scenarios
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    first = next(iter(groups.values()))
    names = ["{0}\n{1}".format(s.title, s.subtitle.split("·")[0].strip()) for s in first]
    rows = np.arange(len(first))[::-1]
    shifts = np.linspace(0.18, -0.18, len(groups)) if len(groups) > 1 else [0.0]
    figure, axes = plt.subplots(
        1, 2, figsize=(12, 0.7 * len(first) + 1.6), constrained_layout=True, sharey=True
    )
    for (imager, scenarios), shift in zip(groups.items(), shifts):
        y = rows + shift
        marker = MARKERS.get(imager, "o")
        before = [s.result.rounds[0].psnr for s in scenarios]
        after = [s.result.final.psnr for s in scenarios]
        ideal = [s.ideal.statistics().psnr for s in scenarios]
        axes[0].hlines(y, before, after, color="#bbbbbb", linewidth=2, zorder=1)
        axes[0].scatter(before, y, color=BEFORE, marker=marker, zorder=2)
        axes[0].scatter(after, y, color=AFTER, marker=marker, zorder=3)
        axes[0].scatter(ideal, y, marker="|", s=150, color=IDEAL, zorder=4)
        injected = [s.phase_error for s in scenarios]
        left = [s.phase_residual for s in scenarios]
        axes[1].hlines(y, left, injected, color="#bbbbbb", linewidth=2, zorder=1)
        axes[1].scatter(injected, y, color=BEFORE, marker=marker, zorder=2)
        axes[1].scatter(left, y, color=AFTER, marker=marker, zorder=3)
    colours = [
        Line2D([], [], color=BEFORE, marker="o", linestyle="", label="before"),
        Line2D([], [], color=AFTER, marker="o", linestyle="", label="self-calibrated"),
        Line2D(
            [], [], color=IDEAL, marker="|", markersize=12, linestyle="", label="no gain errors"
        ),
    ]
    shapes = [
        Line2D(
            [], [], color="#888888", marker=MARKERS.get(imager, "o"), linestyle="", label=imager
        ) for imager in groups
    ]
    for axis in axes:
        axis.set_xscale("log")
        plain_log(axis)
    axes[0].set_xlabel("Peak S/N of the image")
    axes[0].set_yticks(rows, names, fontsize=8)
    axes[0].legend(
        handles=colours + shapes, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=5, fontsize=8
    )
    axes[1].set_xlabel("Antenna phase errors relative to the reference, rms (deg)")
    axes[1].legend(
        handles=[
            Line2D([], [], color=BEFORE, marker="o", linestyle="", label="injected"),
            Line2D(
                [], [], color=AFTER, marker="o", linestyle="", label="left after self-calibration"
            ),
        ] + shapes,
        loc="lower left",
        bbox_to_anchor=(0, 1.0),
        ncol=4,
        fontsize=8
    )
    save(figure, path)
    plt.close(figure)


def convergence_all(groups: Mapping[str, Sequence[Scenario]], path: str) -> None:
    """
    Each scenario's image noise against its ideal image's, round by round, one panel per imager
    """
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        1, len(groups), figsize=(6.5 * len(groups), 4.4), constrained_layout=True, sharey=True
    )
    axes = np.atleast_1d(axes)
    for axis, (imager, scenarios) in zip(axes, groups.items()):
        colours = plt.get_cmap("viridis")(np.linspace(0, 0.9, len(scenarios)))
        for scenario, colour in zip(scenarios, colours):
            floor = scenario.ideal.statistics().rms
            rounds = scenario.result.rounds
            axis.plot(
                range(len(rounds)), [r.rms / floor for r in rounds],
                "o-",
                color=colour,
                label=scenario.title + " · " + scenario.subtitle.split("·")[0].strip()
            )
        axis.axhline(1, color=IDEAL, linestyle=":")
        axis.set_yscale("log")
        plain_log(axis, "y")
        longest = max((s.result for s in scenarios), key=lambda result: len(result.rounds))
        axis.set_xticks(range(len(longest.rounds)), round_labels(longest), fontsize=8)
        axis.set_title(imager, loc="left")
    axes[0].set_ylabel("Image noise / ideal image's")
    axes[-1].legend(fontsize=7, ncol=2, loc="upper right")
    save(figure, path)
    plt.close(figure)
