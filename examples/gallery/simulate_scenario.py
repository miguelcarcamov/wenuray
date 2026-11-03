"""
Simulates one of the gallery's scenarios (scenarios.toml) with Pyralysis: its target (skies.py),
sized to its resolution, with its gain errors and thermal noise. Run with Pyralysis's Python
(make_gallery.py does):

    python examples/gallery/simulate_scenario.py scenarios.toml vla-l-a vla-l-a.ms
"""
import argparse
import json
import os
import sys
import tomllib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from pyralysis_simulation import Observation, simulate, target  # noqa: E402
from skies import Scale, build  # noqa: E402

DESCRIBING = ("name", "title", "subtitle", "sky", "about")


def scenario(path: str, name: str) -> tuple:
    """
    The scenario's observation and its target's name
    """
    with open(path, "rb") as file:
        scenarios = tomllib.load(file)["scenario"]
    (table, ) = [table for table in scenarios if table["name"] == name]
    observation = Observation(
        **{
            key: value
            for key, value in table.items() if key not in DESCRIBING
        }
    )
    return observation, table["sky"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenarios", help="the gallery's scenarios.toml")
    parser.add_argument("name", help="the scenario to simulate")
    parser.add_argument("ms", help="the measurement set to write")
    args = parser.parse_args()
    observation, sky = scenario(args.scenarios, args.name)
    scale = Scale(observation.resolution, observation.largest_scale, observation.image_noise)
    sources, extent = build(sky, target(observation), scale)
    print(
        "{0}: {1}, resolution {2:.4g} arcsec, largest scale {3:.4g} arcsec, image noise {4:.3g} "
        "Jy, reference antenna {5}, {6} components".format(
            args.name, sky, observation.resolution, observation.largest_scale,
            observation.image_noise, observation.reference_antenna, len(sources)
        )
    )
    simulate(observation, sources, args.ms)
    # How large the target is, for the figures
    with open(args.ms + ".sky.json", "w") as file:
        json.dump({"sky": sky, "extent": extent, "resolution": observation.resolution}, file)


if __name__ == "__main__":
    main()
