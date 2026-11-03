"""
Simulates the tutorial's observation with Pyralysis (docs/tutorial.md): an ALMA Band 7 continuum
observation of a protoplanetary disk, a ring with a compact source at its centre, corrupted with
Pyralysis's noise injectors: antenna gain errors, then thermal noise. See ../pyralysis_simulation.py
for what it writes (the observation, its ideal copy and the injected gains).

Run it in an environment with Pyralysis (Wenuray's conda files and images have it):

    python examples/simulated/simulate.py tutorial.ms

The values (array, frequency, source, noise, gains) are this example's, chosen to show
self-calibration at work, not Wenuray's.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyralysis_simulation import Observation, disk_with_core, simulate, target  # noqa: E402

# ALMA in configuration C-10.6 at 345 GHz (Band 7): one 2 GHz channel, 30 s integrations over 2 h.
# The gains: 40 degrees of phase and 5% of amplitude per integration; the noise: a 150 K system.
OBSERVATION = Observation(
    configuration="alma.cycle10.6.cfg",
    frequency=345e9,
    bandwidth=2e9,
    system_temperature=150.0,
    phase_error=40.0,
    amplitude_error=0.05,
    declination="-37d09m16s",
    seed=2026,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ms", help="the measurement set to write")
    path = parser.parse_args().ms
    # A ring 0.45 arcsec in radius with a 0.3 arcsec hole (0.15 Jy), and a 0.03 Jy core
    sources = disk_with_core(0.45, 0.30, 0.15, 0.03, 0.0, 0.0, target(OBSERVATION))
    simulate(OBSERVATION, sources, path)


if __name__ == "__main__":
    main()
