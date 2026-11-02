"""
Simulates the tutorial's observation with Pyralysis (docs/tutorial.md): an ALMA Band 7 continuum
observation of a protoplanetary disk, a ring with a compact source at its centre, corrupted with
Pyralysis's noise injectors as its documentation recommends: antenna gain errors, then thermal
noise (multiplicative before additive, as in a receiver chain).

Writes three things next to each other:

- <ms>: the observation, with gain errors and thermal noise, in DATA;
- <ms>.ideal: the same observation and the same thermal noise without gain errors, what a perfect
  calibration would give;
- <ms>.gains.npz: each antenna's injected gain relative to the reference antenna, per integration.

Pyralysis lives in its own environment in Wenuray's images, so run this with that Python:

    /opt/pyralysis/bin/python examples/simulated/simulate.py tutorial.ms

It imports neither wenuray nor casatools. The values (array, frequency, source, noise, gains) are
this example's, chosen to show self-calibration at work, not Wenuray's.
"""
import argparse
from importlib import resources

import astropy.units as u
import numpy as np

SEED = 2026
# The reference antenna of the tutorial's recipe: the one nearest the array's centre
REFANT = "A042"
# Continuum: one 2 GHz channel per spectral window at 345 GHz, 30 s integrations over 2 h
FREQUENCY = 345 * u.GHz
BANDWIDTH = 2 * u.GHz
INTEGRATION = 30 * u.s


def gain_errors():
    """
    Antenna-based gains, as the atmosphere and the electronics make them: 40 degrees of phase and
    5% of amplitude, drawn anew for every 30 s integration. Pyralysis draws them independently for
    each integration (and channel; there is one), so only solutions at every integration can
    follow them
    """
    from pyralysis.injectors import AntennaGainInjector

    return AntennaGainInjector(amp_sigma=0.05, phase_sigma=np.radians(40.0), seed=SEED)


def thermal_noise():
    """
    Radiometer-equation noise for a 150 K system; Pyralysis also writes SIGMA and WEIGHT for it
    """
    from pyralysis.injectors import ThermalNoiseInjector

    return ThermalNoiseInjector(
        system_temperature=150.0,
        integration_time=INTEGRATION.to_value(u.s),
        channel_bandwidth=BANDWIDTH.to_value(u.Hz),
        seed=SEED + 1,
    )


def observe():
    """
    The noiseless observation: Pyralysis's simulation pipeline, an ALMA cycle 10 configuration 6
    observing a disk 0.45 arcsec in radius, with an inner hole of 0.3 arcsec and a compact source
    """
    from astropy.coordinates import Angle, SkyCoord
    from astropy.time import Time
    from pyralysis.models.sky.disk_source import DiskSource
    from pyralysis.models.sky.point_source import PointSource
    from pyralysis.pipelines.simulation import SimulationContext, SimulationPipeline

    centre = SkyCoord("15h15m48.4s", "-37d09m16s")  # an arbitrary southern field
    configs = resources.files("pyralysis.simulation") / "antenna_configs"
    ring = DiskSource(
        reference_intensity=0.15,
        sky_position=centre,
        profile_type="uniform",
        major_radius=Angle(0.45 * u.arcsec),
        inner_radius=Angle(0.30 * u.arcsec),
        inclination=Angle(30 * u.deg),
        position_angle=Angle(60 * u.deg),
    )
    star = PointSource(reference_intensity=0.03, sky_position=centre)
    context = SimulationContext(
        config={
            "antenna_config_path": str(configs / "alma.cycle10.6.cfg"),
            "sources": [ring, star],
            "observation_config": {
                "center_frequency": FREQUENCY,
                "channel_width": BANDWIDTH,
                "n_channels": 1,
                "pointing_position": centre,
                "hour_angle": "-1h",
                "integration_time": INTEGRATION,
                "observation_time": "2h",
                # A fixed date: by default the hour angle refers to the current time, so every
                # run would observe at other times
                "reference_time": Time("2026-06-21T00:00:00", scale="utc"),
            },
        }
    )
    SimulationPipeline().run(context)
    return context.dataset


def write(dataset, path: str) -> None:
    """
    Writes the dataset as a measurement set CASA reads, patching what Pyralysis 3.3's simulation
    leaves out (see docs/tutorial.md, "What Pyralysis's simulation needs for CASA")
    """
    from casacore.tables import table
    from pyralysis.io import DaskMS

    # Subtables are written only when asked for
    DaskMS(output_name=path).write(dataset, tables="ALL")
    # REST (0) frequencies can't be converted to the frame CASA images in; observatories record
    # TOPO (5)
    with table(path + "/SPECTRAL_WINDOW", readonly=False, ack=False) as windows:
        windows.putcol("MEAS_FREQ_REF", np.full(windows.nrows(), 5, dtype=np.int32))
    # The FEED table is left empty: each antenna gets ALMA's linear feeds, X and Y
    with table(path + "/ANTENNA", ack=False) as antennas:
        nantennas = antennas.nrows()
    with table(path + "/FEED", readonly=False, ack=False) as feeds:
        feeds.addrows(nantennas)
        feeds.putcol("ANTENNA_ID", np.arange(nantennas, dtype=np.int32))
        feeds.putcol("FEED_ID", np.zeros(nantennas, dtype=np.int32))
        feeds.putcol("SPECTRAL_WINDOW_ID", np.full(nantennas, -1, dtype=np.int32))
        feeds.putcol("BEAM_ID", np.full(nantennas, -1, dtype=np.int32))
        feeds.putcol("NUM_RECEPTORS", np.full(nantennas, 2, dtype=np.int32))
        feeds.putcol("TIME", np.zeros(nantennas))
        feeds.putcol("INTERVAL", np.zeros(nantennas))
        feeds.putcol("POSITION", np.zeros((nantennas, 3)))
        feeds.putcol("BEAM_OFFSET", np.zeros((nantennas, 2, 2)))
        feeds.putcol("POLARIZATION_TYPE", np.tile(["X", "Y"], (nantennas, 1)))
        feeds.putcol("POL_RESPONSE", np.tile(np.eye(2, dtype=np.complex64), (nantennas, 1, 1)))
        feeds.putcol("RECEPTOR_ANGLE", np.tile([0.0, np.pi / 2], (nantennas, 1)))


def write_ideal(dataset, path: str) -> None:
    """
    <ms>.ideal: a copy of the observation whose DATA are the dataset's. The writer may store rows
    in another order than the dataset holds them, so rows are matched by time and baseline, which
    one field and one spectral window make unique
    """
    import shutil

    from casacore.tables import table

    (ms, ) = dataset.ms_list
    visibilities = ms.visibilities.dataset
    keys = [visibilities[column].data.compute() for column in ("TIME", "ANTENNA1", "ANTENNA2")]
    data = visibilities["DATA"].data.compute()
    shutil.copytree(path, path + ".ideal")
    with table(path + ".ideal", readonly=False, ack=False) as ideal:
        written = [ideal.getcol(column) for column in ("TIME", "ANTENNA1", "ANTENNA2")]
        # The dataset's rows in key order, then placed where each key sits in the measurement set
        order = np.lexsort(keys[::-1])
        place = np.lexsort(written[::-1])
        if not all(np.array_equal(key[order], column[place]) for key, column in zip(keys, written)):
            raise RuntimeError("The ideal observation doesn't hold the observation's rows")
        rows = np.empty_like(order)
        rows[place] = order
        ideal.putcol("DATA", data[rows])


def injected_gains(path: str) -> None:
    """
    Each antenna's injected gain relative to the reference antenna: the same seeded gain errors
    applied to unit visibilities give g_i conj(g_j) on each baseline, and the baselines to the
    reference antenna give g_i conj(g_ref)
    """
    import dask.array as da
    from casacore.tables import table

    dataset = observe()
    for ms in dataset.ms_list:
        visibilities = ms.visibilities.dataset
        visibilities["DATA"] = visibilities["DATA"].copy(
            data=da.ones_like(visibilities["DATA"].data)
        )
    gain_errors().apply(dataset)
    (ms, ) = dataset.ms_list
    visibilities = ms.visibilities.dataset
    product = visibilities["DATA"].data[:, 0, 0].compute()
    antenna1 = visibilities["ANTENNA1"].data.compute()
    antenna2 = visibilities["ANTENNA2"].data.compute()
    times = visibilities["TIME"].data.compute()
    with table(path + "/ANTENNA", ack=False) as antennas:
        names = list(antennas.getcol("NAME"))
    reference = names.index(REFANT)
    epochs = np.unique(times)
    gains = np.ones((len(names), len(epochs)), dtype=complex)
    for rows, antenna, conjugate in (
        (antenna2 == reference, antenna1, False), (antenna1 == reference, antenna2, True)
    ):
        index = np.searchsorted(epochs, times[rows])
        values = product[rows]
        gains[antenna[rows], index] = np.conj(values) if conjugate else values
    np.savez(path + ".gains.npz", times=epochs, gains=gains, names=names, reference=reference)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ms", help="the measurement set to write")
    path = parser.parse_args().ms
    # A fresh injector for each use: a seeded injector continues its random stream once used
    observation = observe()
    (gain_errors() + thermal_noise()).apply(observation)
    write(observation, path)
    ideal = observe()
    thermal_noise().apply(ideal)
    write_ideal(ideal, path)
    injected_gains(path)


if __name__ == "__main__":
    main()
