"""
Makes the gallery (docs/gallery.md): for each scenario of scenarios.toml, simulates the observation
with Pyralysis, self-calibrates it with Wenuray, and draws it; then the figures that compare them.
Reproducible: every scenario has its seed and a fixed date. In Wenuray's image:

    python examples/gallery/make_gallery.py gallery docs/_static/gallery
    python examples/gallery/make_gallery.py gallery docs/_static/gallery --pyralysis-backend cupy

Work already done in the output directory (simulations, runs) is reused, so a run can be resumed.
The simulations run with WENURAY_PYRALYSIS_PYTHON if it is set, as Pyralysis's worker does; by
default with this Python.
"""
import argparse
import json
import logging
import os
import subprocess
import sys
import tomllib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import figures  # noqa: E402

from wenuray.config import ImageGeometry, SolveOptions, Weighting  # noqa: E402
from wenuray.imaging import Pyralysis, Tclean  # noqa: E402
from wenuray.selfcalibration import (  # noqa: E402
    AmpPhasecal, Phasecal, Selfcal, SelfcalResult
)

HERE = os.path.dirname(os.path.abspath(__file__))
SCENARIOS = os.path.join(HERE, "scenarios.toml")
# Briggs weighting, between natural's sensitivity and uniform's resolution, for the targets' detail
WEIGHTING = Weighting("briggs", robust=0.5)
# The targets are extended: multiscale CLEAN on point sources, one and three beams (the "auto" cell
# puts five pixels across the beam), and an automatic mask whose sidelobe threshold is low enough
# to take in the faint emission next to the bright parts (tclean's default, 3, leaves it uncleaned)
TCLEAN = Tclean(
    ImageGeometry(imsize=512, cell="auto"),
    WEIGHTING,
    niter=20000,
    nsigma=3.0,
    deconvolver="multiscale",
    scales=[0, 5, 15],
    usemask="auto-multithresh",
    sidelobethreshold=1.25,
)
# The same self-calibration in every scenario, with either imager. The gains drift over minutes:
# phases over the whole observation, then over two minutes, then every integration; then
# amplitudes, which drift less and need more signal, over five minutes
STEPS = [Phasecal("inf"), Phasecal("120s"), Phasecal("int"), AmpPhasecal("300s")]
# Panels show the target with this much sky around it
MARGIN = 1.15


def pyralysis_imager(tclean: SelfcalResult, directory: str, backend: str) -> Pyralysis:
    """
    Pyralysis on tclean's grid, with a mask around the sources: the mask of tclean's last round,
    grown by a beam (five pixels), as one draws CLEAN boxes after a first look at the field. A model
    free in every pixel could absorb the gain errors self-calibration should solve for. backend is
    Pyralysis's array backend: numpy, or cupy on a GPU
    """
    import numpy as np
    from astropy.io import fits
    from casatools import image
    from scipy.ndimage import binary_dilation

    path = os.path.join(directory, "pyralysis_mask.fits")
    if not os.path.exists(path):
        reader = image()
        reader.open(tclean.final.image[:-len(".image")] + ".mask")
        try:
            mask = reader.getchunk().squeeze() > 0
        finally:
            reader.close()
        # CASA stores (x, y); FITS (y, x)
        grown = binary_dilation(mask.T, iterations=5)
        fits.PrimaryHDU(grown.astype(np.float32)).writeto(path)
    grid = tclean.provenance["geometry"]
    return Pyralysis(
        ImageGeometry(imsize=tuple(grid["imsize"]), cell=grid["cell"]),
        WEIGHTING,
        terms=PYRALYSIS_TERMS,
        optimizer="lbfgsb",
        max_iter=PYRALYSIS_ITERATIONS,
        mask_mode="file",
        mask_path=path,
        array_backend=backend,
    )


# Pyralysis's reconstruction: chi-squared with a small total squared variation, which keeps the
# extended emission smooth
PYRALYSIS_TERMS = ["chi_squared", "tsv:1e-5"]
PYRALYSIS_ITERATIONS = 100
IMAGERS = ("tclean", "Pyralysis")


def simulate(name: str, ms: str) -> None:
    if os.path.exists(ms + ".gains.npz"):
        return
    python = os.environ.get("WENURAY_PYRALYSIS_PYTHON", sys.executable)
    subprocess.run(
        [python, os.path.join(HERE, "simulate_scenario.py"), SCENARIOS, name, ms], check=True
    )


def self_calibrate(ms: str, imager, output: str) -> SelfcalResult:
    import numpy as np

    record = os.path.join(output, "selfcal.toml")
    if os.path.exists(record):
        return SelfcalResult.from_toml(record)
    injected = np.load(ms + ".gains.npz")
    refant = str(injected["names"][int(injected["reference"])])
    return Selfcal(ms, imager, STEPS, solve=SolveOptions(refant=refant), output=output).run()


def show(
    table: dict, name: str, imager: str, ms: str, directory: str, result: SelfcalResult, imaging,
    extent: float, figures_directory: str
) -> "figures.Scenario":
    """
    Draws one run of a scenario and returns what the comparisons need
    """
    import numpy as np

    ideal = figures.image_ideal(ms, imaging, result, directory)
    suffix = "" if imager == "tclean" else "-" + imager.lower()
    title = "{0} · {1}: {2} ({3})".format(table["title"], table["subtitle"], table["about"], imager)
    figures.before_after(
        result,
        ideal,
        os.path.join(figures_directory, name + suffix + ".webp"),
        extent,
        title=title
    )
    truth, found, names, reference = figures.phase_errors(ms, result)
    others = [antenna for antenna in range(len(names)) if antenna != reference]
    residual = figures.wrapped(found[others] - truth[others])
    scenario = figures.Scenario(
        title=table["title"],
        subtitle=table["subtitle"],
        about=table["about"],
        result=result,
        ideal=ideal,
        extent=extent,
        phase_error=float(np.sqrt(np.mean(figures.wrapped(truth[others])**2))),
        phase_residual=float(np.sqrt(np.mean(residual**2))),
        imager=imager,
    )
    print(
        "{0} ({1}): PSNR {2:.0f} -> {3:.0f} (ideal {4:.0f}); phases {5:.1f} -> {6:.1f} deg rms".
        format(
            name, imager, result.rounds[0].psnr, result.final.psnr,
            ideal.statistics().psnr, scenario.phase_error, scenario.phase_residual
        ),
        flush=True
    )
    return scenario


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", help="where the simulations and the runs go")
    parser.add_argument("figures", help="where the figures go")
    parser.add_argument("--only", nargs="*", help="these scenarios only")
    parser.add_argument("--imagers", nargs="*", default=IMAGERS, help="these imagers only")
    parser.add_argument(
        "--pyralysis-backend",
        default="numpy",
        choices=("numpy", "cupy"),
        help="Pyralysis's array backend: cupy runs it on a GPU (Wenuray's :gpu image)"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    os.makedirs(args.figures, exist_ok=True)
    figures.style()
    with open(SCENARIOS, "rb") as file:
        tables = tomllib.load(file)["scenario"]
    groups = {imager: [] for imager in args.imagers}
    for table in tables:
        name = table["name"]
        if args.only and name not in args.only:
            continue
        directory = os.path.join(os.path.abspath(args.work), name)
        os.makedirs(directory, exist_ok=True)
        ms = os.path.join(directory, "data.ms")
        simulate(name, ms)
        with open(ms + ".sky.json") as file:
            extent = MARGIN * json.load(file)["extent"]
        # tclean first: Pyralysis takes its grid and its mask
        tclean = self_calibrate(ms, TCLEAN, os.path.join(directory, "tclean"))
        runs = {"tclean": (tclean, TCLEAN)}
        if "Pyralysis" in groups:
            imaging = pyralysis_imager(tclean, directory, args.pyralysis_backend)
            runs["Pyralysis"] = (
                self_calibrate(ms, imaging, os.path.join(directory, "pyralysis")), imaging
            )
        for imager in groups:
            result, imaging = runs[imager]
            run = os.path.join(directory, imager.lower())
            groups[imager].append(
                show(table, name, imager, ms, run, result, imaging, extent, args.figures)
            )
    if len(next(iter(groups.values()))) > 1:
        for imager, scenarios in groups.items():
            suffix = "" if imager == "tclean" else "-" + imager.lower()
            figures.gallery(scenarios, os.path.join(args.figures, "gallery" + suffix + ".webp"))
        figures.summary(groups, os.path.join(args.figures, "summary.png"))
        figures.convergence_all(groups, os.path.join(args.figures, "convergence.png"))


if __name__ == "__main__":
    main()
