"""
Compares two runs recorded by ab_run.py (or any two Wenuray output directories):

    python ab_compare.py <branch_out> <dev_out> [--rtol 1e-6]

Round by round: step, decision, reasons and metrics; then the gain solutions of the accepted
tables (needs casatools); then time and memory from ab.json when present. Exits 1 on a difference
beyond --rtol.
"""
import argparse
import json
import math
import sys
import tomllib
from pathlib import Path

METRICS = ("psnr", "peak", "rms", "model_flux", "flagged_fraction", "predicted_snr")


def load(directory: Path) -> dict:
    with open(directory / "selfcal.toml", "rb") as file:
        return tomllib.load(file)


def close(a, b, rtol: float) -> bool:
    if a is None or b is None:
        return a is b
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return math.isclose(a, b, rel_tol=rtol, abs_tol=0.0)


def compare_rounds(branch: dict, dev: dict, rtol: float) -> list:
    problems = []
    rounds = list(zip(branch["rounds"], dev["rounds"]))
    if len(branch["rounds"]) != len(dev["rounds"]):
        problems.append("rounds: {0} vs {1}".format(len(branch["rounds"]), len(dev["rounds"])))
    print(
        "{0:>5} {1:<28} {2:>8} {3:>12} {4:>12} {5:>10}".format(
            "round", "step", "accepted", "psnr", "dev psnr", "rel diff"
        )
    )
    for a, b in rounds:
        diff = abs(a["psnr"] - b["psnr"]) / abs(b["psnr"]) if b["psnr"] else 0.0
        print(
            "{0:>5} {1:<28} {2:>8} {3:>12.6g} {4:>12.6g} {5:>10.2e}".format(
                a["index"], a["step"][:28] or "initial image", str(a["accepted"]), a["psnr"],
                b["psnr"], diff
            )
        )
        # The reasons quote the metrics, which are compared below with a tolerance
        for key in ("step", "accepted"):
            if a.get(key) != b.get(key):
                problems.append(
                    "round {0} {1}: {2!r} vs {3!r}".format(a["index"], key, a.get(key), b.get(key))
                )
        for key in METRICS:
            if not close(a.get(key), b.get(key), rtol):
                problems.append(
                    "round {0} {1}: {2} vs {3}".format(a["index"], key, a.get(key), b.get(key))
                )
    return problems


def compare_gains(branch_dir: Path, dev_dir: Path, branch: dict, dev: dict, rtol: float) -> list:
    try:
        import numpy as np
        from casatools import table
    except ImportError:
        print("casatools not available: gain solutions not compared")
        return []
    problems = []
    pairs = zip(branch["stack"]["tables"], dev["stack"]["tables"])
    for a, b in pairs:
        gains = []
        for directory, entry in ((branch_dir, a), (dev_dir, b)):
            tool = table()
            tool.open(str((directory / entry["path"]).resolve()))
            gains.append(np.where(tool.getcol("FLAG"), np.nan, tool.getcol("CPARAM")))
            tool.close()
        if gains[0].shape != gains[1].shape:
            problems.append(
                "{0}: shape {1} vs {2}".format(a["path"], gains[0].shape, gains[1].shape)
            )
            continue
        worst = np.nanmax(np.abs(gains[0] - gains[1]) / np.maximum(np.abs(gains[1]), 1e-30))
        print("{0}: worst relative gain difference {1:.2e}".format(a["path"], worst))
        if worst > rtol:
            problems.append("{0}: gains differ by {1:.2e}".format(a["path"], worst))
    return problems


def compare_cost(branch_dir: Path, dev_dir: Path) -> None:
    files = [directory / "ab.json" for directory in (branch_dir, dev_dir)]
    if not all(path.exists() for path in files):
        return
    branch, dev = (json.loads(path.read_text()) for path in files)
    for key in ("seconds", "peak_rss_mb", "children_peak_rss_mb"):
        print(
            "{0:>22}: branch {1:10.2f}  dev {2:10.2f}  ratio {3:6.3f}".format(
                key, branch[key], dev[key], branch[key] / dev[key] if dev[key] else math.nan
            )
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("branch", type=Path)
    parser.add_argument("dev", type=Path)
    parser.add_argument("--rtol", type=float, default=1e-6)
    args = parser.parse_args()
    branch, dev = load(args.branch), load(args.dev)
    problems = compare_rounds(branch, dev, args.rtol)
    problems += compare_gains(args.branch, args.dev, branch, dev, args.rtol)
    compare_cost(args.branch, args.dev)
    for problem in problems:
        print("DIFFERENT:", problem)
    print("same results" if not problems else "{0} differences".format(len(problems)))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
