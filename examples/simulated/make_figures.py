"""
Makes the tutorial's figures (docs/tutorial.md) from a run of recipe.toml on the data simulate.py
made:

    python examples/simulated/make_figures.py tutorial.ms tutorial_selfcal docs/_static/tutorial

- images.png: the initial image, the self-calibrated one, and the data imaged without gain errors
  (the ideal a perfect calibration reaches), with their residuals below;
- convergence.png: each round's peak S/N, its noise against the ideal image's, and its gain
  corrections against their noise;
- gains.png: the antenna phases self-calibration found, against those simulate.py injected.

Needs matplotlib (`pip install "wenuray[plots]"`) and CASA; see ../figures.py for the style.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import figures  # noqa: E402

from wenuray.selfcalibration import Selfcal, SelfcalResult  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ms", help="the measurement set simulate.py made")
    parser.add_argument("run", help="the run's output directory")
    parser.add_argument("figures", help="where to write the figures")
    args = parser.parse_args()
    os.makedirs(args.figures, exist_ok=True)
    figures.style()
    result = SelfcalResult.from_toml(os.path.join(args.run, "selfcal.toml"))
    # The ideal data imaged as the run imaged its own
    imager = Selfcal.from_toml(os.path.join(HERE, "recipe.toml")).imager
    ideal = figures.image_ideal(args.ms, imager, result, args.run)
    figures.before_after(result, ideal, os.path.join(args.figures, "images.png"), extent=0.8)
    figures.convergence(
        result,
        ideal.statistics().rms, os.path.join(args.figures, "convergence.png")
    )
    figures.gains(args.ms, result, os.path.join(args.figures, "gains.png"))


if __name__ == "__main__":
    main()
