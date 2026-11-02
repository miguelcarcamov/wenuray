"""
Simulated observations for Wenuray's examples (the tutorial), made with Pyralysis's
simulation framework and its noise injectors, and written as measurement sets CASA reads.

For each observation it writes, next to each other:

- <ms>: the observation, with antenna gain errors then thermal noise (multiplicative before
  additive, as in a receiver chain) in DATA;
- <ms>.ideal: the same observation and the same thermal noise without gain errors, what a perfect
  calibration would give;
- <ms>.gains.npz: each antenna's injected gain relative to the reference antenna, per integration.

It runs with Pyralysis's Python (in Wenuray's images, /opt/pyralysis/bin/python) and imports
neither wenuray nor casatools. Pyralysis 3.3's measurement sets need three patches for CASA, made in
write(): pyralysis#405 (subtables), #406 (frequency frame) and #407 (FEED).
"""
from dataclasses import dataclass
from importlib import resources
from typing import List, Tuple

import astropy.units as u
import numpy as np

BOLTZMANN = 1.380649e-23  # J/K
JANSKY = 1e-26  # W m^-2 Hz^-1
SPEED_OF_LIGHT = 299792458.0
APERTURE_EFFICIENCY = 0.7  # ThermalNoiseInjector's default
# A fixed date: by default an hour angle refers to the current time, so every run would observe at
# other times
REFERENCE_TIME = "2026-06-21T00:00:00"


@dataclass(frozen=True)
class Observation:
    """
    One simulated observation

    Parameters
    ----------
    configuration :
        A Pyralysis antenna configuration, e.g. "vla.a.cfg" or "alma.cycle10.6.cfg"
    frequency, bandwidth :
        The centre frequency and the width of its single channel, in Hz
    system_temperature :
        In K, for the radiometer equation
    phase_error, amplitude_error :
        The antenna gain errors: degrees of phase and fractional amplitude, per integration
    declination :
        Of the target, e.g. "-37d09m16s"; the right ascension is fixed
    seed :
        Of the gain errors (the thermal noise uses seed + 1)
    integration, duration :
        In seconds
    """
    configuration: str
    frequency: float
    bandwidth: float
    system_temperature: float
    phase_error: float
    amplitude_error: float
    declination: str
    seed: int
    integration: float = 30.0
    duration: float = 7200.0

    @property
    def path(self) -> str:
        return str(resources.files("pyralysis.simulation") / "antenna_configs" / self.configuration)

    def antennas(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        The antennas' positions (m), dish diameters (m) and names, from the configuration file
        """
        rows = [
            line.split() for line in open(self.path)
            if line.strip() and not line.lstrip().startswith("#")
        ]
        positions = np.array([[float(value) for value in row[:3]] for row in rows])
        return positions, np.array([float(row[3]) for row in rows]), [row[4] for row in rows]

    def baselines(self) -> np.ndarray:
        positions, _, _ = self.antennas()
        first, second = np.triu_indices(len(positions), 1)
        return np.linalg.norm(positions[first] - positions[second], axis=1)

    @property
    def longest_baseline(self) -> float:
        return float(self.baselines().max())

    @property
    def shortest_baseline(self) -> float:
        return float(self.baselines().min())

    @property
    def resolution(self) -> float:
        """
        lambda / the longest baseline, in arcsec: about the synthesized beam's width
        """
        return float(np.degrees(SPEED_OF_LIGHT / self.frequency / self.longest_baseline) * 3600)

    @property
    def largest_scale(self) -> float:
        """
        About the largest angular scale the array images, 0.6 lambda / the shortest baseline, in
        arcsec: emission much larger is resolved out
        """
        return float(
            np.degrees(0.6 * SPEED_OF_LIGHT / self.frequency / self.shortest_baseline) * 3600
        )

    @property
    def reference_antenna(self) -> str:
        """
        The antenna nearest the array's centre: it sees every other on short baselines
        """
        positions, _, names = self.antennas()
        return names[int(np.argmin(np.linalg.norm(positions - positions.mean(axis=0), axis=1)))]

    @property
    def image_noise(self) -> float:
        """
        The thermal noise of a naturally weighted Stokes I image, in Jy/beam, from the radiometer
        equation: SEFD / sqrt(npol N (N - 1) bandwidth duration)
        """
        _, diameters, _ = self.antennas()
        area = APERTURE_EFFICIENCY * np.pi * (np.median(diameters) / 2)**2
        sefd = 2 * BOLTZMANN * self.system_temperature / area / JANSKY
        nantennas = len(diameters)
        return float(
            sefd / np.sqrt(2 * nantennas * (nantennas - 1) * self.bandwidth * self.duration)
        )


def disk_with_core(
    radius: float, inner_radius: float, ring_flux: float, core_flux: float, companion_flux: float,
    companion_offset: float, centre
) -> list:
    """
    The sky: a ring with a gap (radii in arcsec), inclined by 30 degrees, a compact core at its
    centre, and a faint compact companion companion_offset arcsec away
    """
    from astropy.coordinates import Angle
    from pyralysis.models.sky.disk_source import DiskSource
    from pyralysis.models.sky.point_source import PointSource

    ring = DiskSource(
        reference_intensity=ring_flux,
        sky_position=centre,
        profile_type="ring",  # inner_radius applies to "ring" (and "power_law") only
        major_radius=Angle(radius * u.arcsec),
        inner_radius=Angle(inner_radius * u.arcsec),
        inclination=Angle(30 * u.deg),
        position_angle=Angle(60 * u.deg),
    )
    sources = [ring, PointSource(reference_intensity=core_flux, sky_position=centre)]
    if companion_flux:
        companion = centre.spherical_offsets_by(
            companion_offset * 0.6 * u.arcsec, companion_offset * 0.8 * u.arcsec
        )
        sources.append(PointSource(reference_intensity=companion_flux, sky_position=companion))
    return sources


def target(observation: Observation):
    from astropy.coordinates import SkyCoord

    return SkyCoord("15h15m48.4s", observation.declination)


def observe(observation: Observation, sources: list):
    """
    The noiseless observation: Pyralysis's simulation pipeline, from an hour before transit for
    the observation's duration
    """
    from astropy.time import Time
    from pyralysis.pipelines.simulation import SimulationContext, SimulationPipeline

    context = SimulationContext(
        config={
            "antenna_config_path": observation.path,
            "sources": sources,
            "observation_config": {
                "center_frequency": observation.frequency * u.Hz,
                "channel_width": observation.bandwidth * u.Hz,
                "n_channels": 1,
                "pointing_position": target(observation),
                "hour_angle": "-{0:g}h".format(observation.duration / 7200),
                "integration_time": observation.integration * u.s,
                "observation_time": "{0:g}s".format(observation.duration),
                "reference_time": Time(REFERENCE_TIME, scale="utc"),
            },
        }
    )
    SimulationPipeline().run(context)
    return context.dataset


def gain_errors(observation: Observation):
    """
    Antenna-based gains, drawn anew for every integration (and channel; there is one): Pyralysis's
    AntennaGainInjector draws them independently (pyralysis#409), so only solutions at every
    integration can follow them
    """
    from pyralysis.injectors import AntennaGainInjector

    return AntennaGainInjector(
        amp_sigma=observation.amplitude_error,
        phase_sigma=np.radians(observation.phase_error),
        seed=observation.seed
    )


def thermal_noise(observation: Observation):
    """
    Radiometer-equation noise; Pyralysis also writes SIGMA and WEIGHT for it
    """
    from pyralysis.injectors import ThermalNoiseInjector

    return ThermalNoiseInjector(
        system_temperature=observation.system_temperature,
        integration_time=observation.integration,
        channel_bandwidth=observation.bandwidth,
        aperture_efficiency=APERTURE_EFFICIENCY,
        seed=observation.seed + 1,
    )


def write(dataset, path: str) -> None:
    """
    Writes the dataset as a measurement set CASA reads, patching what Pyralysis 3.3's simulation
    leaves out
    """
    from casacore.tables import table
    from pyralysis.io import DaskMS

    # pyralysis#405: subtables are written only when asked for
    DaskMS(output_name=path).write(dataset, tables="ALL")
    # pyralysis#406: REST (0) frequencies can't be converted to the frame CASA images in;
    # observatories record TOPO (5)
    with table(path + "/SPECTRAL_WINDOW", readonly=False, ack=False) as windows:
        windows.putcol("MEAS_FREQ_REF", np.full(windows.nrows(), 5, dtype=np.int32))
    # pyralysis#407: the FEED table is left empty; each antenna gets linear feeds, X and Y (the
    # simulation's correlations are XX and YY)
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


def write_injected_gains(observation: Observation, sources: list, path: str) -> None:
    """
    <ms>.gains.npz: each antenna's injected gain relative to the reference antenna. The same seeded
    gain errors applied to unit visibilities give g_i conj(g_j) on each baseline, and the baselines
    to the reference antenna give g_i conj(g_ref)
    """
    import dask.array as da
    from casacore.tables import table

    dataset = observe(observation, sources)
    for ms in dataset.ms_list:
        visibilities = ms.visibilities.dataset
        visibilities["DATA"] = visibilities["DATA"].copy(
            data=da.ones_like(visibilities["DATA"].data)
        )
    gain_errors(observation).apply(dataset)
    (ms, ) = dataset.ms_list
    visibilities = ms.visibilities.dataset
    product = visibilities["DATA"].data[:, 0, 0].compute()
    antenna1 = visibilities["ANTENNA1"].data.compute()
    antenna2 = visibilities["ANTENNA2"].data.compute()
    times = visibilities["TIME"].data.compute()
    with table(path + "/ANTENNA", ack=False) as antennas:
        names = list(antennas.getcol("NAME"))
    reference = names.index(observation.reference_antenna)
    epochs = np.unique(times)
    gains = np.ones((len(names), len(epochs)), dtype=complex)
    for rows, antenna, conjugate in (
        (antenna2 == reference, antenna1, False), (antenna1 == reference, antenna2, True)
    ):
        index = np.searchsorted(epochs, times[rows])
        values = product[rows]
        gains[antenna[rows], index] = np.conj(values) if conjugate else values
    np.savez(path + ".gains.npz", times=epochs, gains=gains, names=names, reference=reference)


def simulate(observation: Observation, sources: list, path: str) -> None:
    """
    Writes the observation, its ideal copy and its injected gains. A fresh injector for each use:
    a seeded injector continues its random stream once used
    """
    corrupted = observe(observation, sources)
    (gain_errors(observation) + thermal_noise(observation)).apply(corrupted)
    write(corrupted, path)
    ideal = observe(observation, sources)
    thermal_noise(observation).apply(ideal)
    write_ideal(ideal, path)
    write_injected_gains(observation, sources, path)
