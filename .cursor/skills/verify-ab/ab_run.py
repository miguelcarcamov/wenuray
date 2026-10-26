"""
Runs one seeded self-calibration with the Wenuray on PYTHONPATH and records it for ab_compare.py:

    PYTHONPATH=<tree>/src python ab_run.py <out> [--recipe recipe.toml] [--imager tclean|pyralysis]

<out>/selfcal.toml is the run's record; <out>/ab.json adds the time and peak memory.
"""
import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np

STEPS = '''
[[steps]]
type = "phase"
solint = "inf"

[[steps]]
type = "phase"
solint = "int"

[[steps]]
type = "ampphase"
solint = "inf"
'''
IMAGERS = {
    "tclean": '''
[imager]
type = "tclean"
niter = 200
nsigma = 3.0
geometry = { imsize = 128, cell = "1arcsec" }
''',
    "pyralysis": '''
[imager]
type = "pyralysis"
terms = ["chi_squared"]
optimizer = "lbfgsb"
max_iter = 50
mask_mode = "file"
mask_path = "{mask}"
geometry = { imsize = 128, cell = "1arcsec" }
''',
}


def box_mask(path: Path) -> str:
    from astropy.io import fits

    mask = np.zeros((128, 128), dtype=np.float32)
    mask[60:69, 60:69] = 1.0
    fits.PrimaryHDU(mask).writeto(path, overwrite=True)
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--recipe", type=Path, help="a recipe of your own; its ms is replaced")
    parser.add_argument("--imager", choices=sorted(IMAGERS), default="tclean")
    args = parser.parse_args()

    import wenuray
    from wenuray import recipe
    from wenuray.selfcalibration import Selfcal
    from wenuray.testing import simulate_point_source_ms

    work = args.out.with_name(args.out.name + "_work")
    work.mkdir(parents=True, exist_ok=False)
    ms = simulate_point_source_ms(str(work / "target.ms"))
    if args.recipe:
        table = recipe.read(str(args.recipe))
    else:
        text = '[solve]\nrefant = "A0"\nminblperant = 3\n' + IMAGERS[args.imager] + STEPS
        text = text.replace("{mask}", box_mask(work / "mask.fits"))
        (work / "recipe.toml").write_text(text)
        table = recipe.read(str(work / "recipe.toml"))
    table.update(ms=ms, output=str(args.out))

    start = time.perf_counter()
    result = Selfcal.from_recipe(table).run(observers=())
    seconds = time.perf_counter() - start

    record = {
        "wenuray": wenuray.__version__,
        "seconds": seconds,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "children_peak_rss_mb": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024,
        "accepted_rounds": len(result.accepted_rounds),
    }
    (args.out / "ab.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
