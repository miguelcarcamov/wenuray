"""
The gallery's targets: for each scenario, a source typical for its array and band, built from
Pyralysis's analytic components (points, elliptical Gaussians, disks and rings), so visibilities are
exact.

Sizes are in units of a few resolutions (lambda / the longest baseline), shrunk when the target
would be larger than the array can image (Scale.fit). Fluxes are in units of the image's
thermal noise per beam (Scale.flux), so every scenario has a comparable signal-to-noise ratio. Each
target has a compact bright part, as real ones do, which the longest baselines solve on.

Runs with Pyralysis's Python (simulate_scenario.py imports it).
"""
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import astropy.units as u
import numpy as np

# Resolutions per unit of a target's size, where the array's largest scale allows it: enough for its
# finest parts (a ring, a jet) to span a couple of beams
ZOOM = 2.5


@dataclass(frozen=True)
class Scale:
    """
    A scenario's units: its resolution and the largest scale it images (arcsec), and its image
    noise (Jy/beam)
    """
    resolution: float
    largest_scale: float
    noise: float

    def fit(self, extent: float) -> float:
        """
        The target's unit of size, for a target extent units in radius: ZOOM resolutions, so it
        spans tens of beams, or less, so the whole target stays within the largest scale. Its
        structure (clumps, filaments) is much smaller, and what the array resolves out of the
        smoothest parts it resolves out of the ideal image too
        """
        return self.resolution * min(ZOOM, self.largest_scale / (2 * extent * self.resolution))

    def flux(self, snr: float, size: float = 0.0) -> float:
        """
        The flux (Jy) of a component of FWHM size (arcsec) whose peak is snr times the noise in a
        beam
        """
        return snr * self.noise * (1 + (size / self.resolution)**2)


class Sky:
    """
    Collects components around a centre: offsets east and north in arcsec, sizes as FWHM in arcsec,
    position angles east of north in degrees
    """

    def __init__(self, centre):
        self.centre = centre
        self.sources: List = []

    def position(self, east: float, north: float):
        return self.centre.spherical_offsets_by(east * u.arcsec, north * u.arcsec)

    def point(self, flux: float, east: float = 0.0, north: float = 0.0) -> None:
        from pyralysis.models.sky.point_source import PointSource

        self.sources.append(
            PointSource(reference_intensity=flux, sky_position=self.position(east, north))
        )

    def gaussian(
        self,
        flux: float,
        east: float,
        north: float,
        major: float,
        minor: float,
        angle: float = 0.0
    ) -> None:
        from astropy.coordinates import Angle
        from pyralysis.models.sky.gaussian_source import GaussianSource

        self.sources.append(
            GaussianSource(
                reference_intensity=flux,
                sky_position=self.position(east, north),
                major_fwhm=Angle(major * u.arcsec),
                minor_fwhm=Angle(minor * u.arcsec),
                theta_angle=Angle(angle * u.deg),
            )
        )

    def disk(
        self,
        flux: float,
        profile: str,
        outer: float,
        inner: float = 0.0,
        inclination: float = 0.0,
        angle: float = 0.0,
        power: float = 0.0,
        east: float = 0.0,
        north: float = 0.0
    ) -> None:
        from astropy.coordinates import Angle
        from pyralysis.models.sky.disk_source import DiskSource

        self.sources.append(
            DiskSource(
                reference_intensity=flux,
                sky_position=self.position(east, north),
                profile_type=profile,
                major_radius=Angle(outer * u.arcsec),
                inner_radius=Angle(inner * u.arcsec),
                power_index=power,
                inclination=Angle(inclination * u.deg),
                position_angle=Angle(angle * u.deg),
            )
        )

    def filament(
        self, scale: Scale, path: np.ndarray, width: float, snr: Tuple[float, float],
        random: np.random.Generator
    ) -> None:
        """
        A filament along path (east, north offsets in arcsec, one row per point): overlapping
        Gaussians width wide, three per width, elongated along it, each snr[0] to snr[1] times the
        noise, so its brightness varies as real ones do
        """
        steps = np.cumsum(np.r_[0, np.linalg.norm(np.diff(path, axis=0), axis=1)])
        for along in np.arange(0, steps[-1], width / 3):
            east, north = (np.interp(along, steps, path[:, i]) for i in (0, 1))
            k = min(np.searchsorted(steps, along, side="right"), len(path) - 1)
            de, dn = path[k] - path[k - 1]
            self.gaussian(
                scale.flux(random.uniform(*snr), width), east, north, 1.6 * width, width,
                np.degrees(np.arctan2(de, dn))
            )

    def clumps(
        self, scale: Scale, count: int, centre: Tuple[float, float], size: Tuple[float, float],
        angle: float, width: Tuple[float, float], snr: Tuple[float,
                                                             float], random: np.random.Generator
    ) -> None:
        """
        count Gaussian clumps scattered in an ellipse of semi-axes size (arcsec) at position angle
        angle around centre, each width[0] to width[1] wide and snr[0] to snr[1] times the noise
        """
        theta = np.radians(angle)
        for _ in range(count):
            radius, phase = np.sqrt(random.uniform()), random.uniform(0, 2 * np.pi)
            a, b = size[0] * radius * np.cos(phase), size[1] * radius * np.sin(phase)
            east = centre[0] + a * np.sin(theta) + b * np.cos(theta)
            north = centre[1] + a * np.cos(theta) - b * np.sin(theta)
            major = random.uniform(*width)
            self.gaussian(
                scale.flux(random.uniform(*snr), major), east, north, major,
                major * random.uniform(0.4, 1.0), random.uniform(0, 180)
            )

    def background(self, scale: Scale, count: int, radius: float, seed: int) -> None:
        """
        Unrelated sources scattered out to radius arcsec, 15 to 150 times the noise: most compact,
        some small doubles
        """
        rng = np.random.default_rng(seed)
        for _ in range(count):
            distance, direction = radius * np.sqrt(rng.uniform(0.3, 1.0)), rng.uniform(0, 2 * np.pi)
            east, north = distance * np.sin(direction), distance * np.cos(direction)
            if rng.uniform() < 0.7:
                self.point(scale.flux(rng.uniform(15, 150)), east, north)
                continue
            split = rng.uniform(1.5, 4) * scale.resolution
            turn = rng.uniform(0, np.pi)
            for side in (1, -1):
                self.gaussian(
                    scale.flux(rng.uniform(15, 80),
                               scale.resolution), east + side * split * np.sin(turn),
                    north + side * split * np.cos(turn), scale.resolution, 0.6 * scale.resolution,
                    np.degrees(turn)
                )


def ring_flux(scale: Scale, snr: float, outer: float, inner: float, inclination: float) -> float:
    """
    The flux of a ring whose surface brightness is snr times the noise per beam
    """
    beams = np.pi * (outer**2 - inner**2) * np.cos(np.radians(inclination)
                                                   ) / (1.133 * scale.resolution**2)
    return snr * scale.noise * max(1.0, beams)


def along(
    axis_angle: float,
    distances: np.ndarray,
    wiggle: float = 0.0,
    turns: float = 2.0
) -> np.ndarray:
    """
    A path from the centre outwards at a position angle (deg), bending sideways by wiggle (arcsec)
    over turns half-periods
    """
    theta = np.radians(axis_angle)
    axis, side = np.array([np.sin(theta), np.cos(theta)]), np.array([np.cos(theta), -np.sin(theta)])
    bend = wiggle * np.sin(np.pi * turns * distances / max(distances.max(), 1e-12))
    return distances[:, None] * axis + bend[:, None] * side


def radio_galaxy(
    sky: Sky, scale: Scale, r: float, angle: float, length: float, lobe: Tuple[float, float],
    seed: int
) -> None:
    """
    A double radio galaxy: a core, two jets that bend a little, turbulent lobes of many clumps on a
    diffuse base, and a bright hotspot with a secondary one where each jet ends. Lengths are in r
    """
    random = np.random.default_rng(seed)
    sky.point(scale.flux(150))
    for side in (1, -1):
        heading = angle if side > 0 else angle + 180
        jet = along(heading, np.linspace(0.8, length - lobe[0], 40) * r, 0.25 * r, 1.5)
        sky.filament(scale, jet, 0.35 * r, (8, 20), random)
        theta = np.radians(heading)
        axis = np.array([np.sin(theta), np.cos(theta)])
        centre = (length - 0.6 * lobe[0]) * r * axis
        sky.gaussian(
            scale.flux(12, 2 * lobe[1] * r), *centre, 2 * lobe[0] * r, 2 * lobe[1] * r, heading
        )
        sky.clumps(
            scale, 45, tuple(centre), (lobe[0] * r, lobe[1] * r), heading, (0.4 * r, 1.6 * r),
            (6, 30), random
        )
        tip = length * r * axis
        sky.point(scale.flux(700), *tip)
        sky.gaussian(
            scale.flux(120, 0.5 * r),
            *(tip - 0.7 * r * axis + 0.6 * r * axis[::-1] * np.array([1, -1])), 0.5 * r, 0.35 * r,
            heading
        )


def giant_radio_galaxy(sky: Sky, scale: Scale) -> float:
    """
    MeerKAT at UHF: a giant radio galaxy, a core with two-sided jets feeding turbulent lobes that
    end in hotspots, in a field of unrelated background sources
    """
    r = scale.fit(13)
    radio_galaxy(sky, scale, r, 35, 10.5, (3.0, 2.0), seed=11)
    sky.background(scale, 12, 12.5 * r, seed=1)
    return 12.5 * r


def cygnus_a(sky: Sky, scale: Scale) -> float:
    """
    The VLA at 1.5 GHz in A: a powerful FR II radio galaxy, a faint core, narrow jets, and lobes
    full of filaments with bright hotspots at their ends
    """
    r = scale.fit(10)
    radio_galaxy(sky, scale, r, 110, 9.0, (3.2, 1.8), seed=13)
    return 10 * r


def spiral_galaxy(sky: Sky, scale: Scale) -> float:
    """
    The VLA at 1.5 GHz in D: a nearby spiral galaxy, a synchrotron disk, two arms of star-forming
    clumps, a nucleus, and background sources
    """
    r = scale.fit(9)
    random = np.random.default_rng(12)
    inclination, cosine = 35.0, np.cos(np.radians(35.0))
    sky.disk(
        ring_flux(scale, 30, 5 * r, 0, inclination),
        "exponential",
        2.2 * r,
        inclination=inclination,
        angle=20
    )
    sky.point(scale.flux(300))
    for arm in (0, np.pi):
        t = np.linspace(0.3, 3.0, 60)
        radius, phase = r * 1.2 * np.exp(0.55 * t), arm + t
        path = np.c_[radius * np.cos(phase), radius * np.sin(phase) * cosine]
        sky.filament(scale, path, 0.7 * r, (10, 25), random)
        for k in random.choice(len(t), 14, replace=False):
            sky.gaussian(
                scale.flux(random.uniform(30, 110), 0.5 * r),
                *(path[k] + random.normal(0, 0.3 * r, 2)), 0.5 * r, 0.4 * r, random.uniform(0, 180)
            )
    sky.background(scale, 6, 9 * r, seed=2)
    return 9.5 * r


def star_forming_region(sky: Sky, scale: Scale) -> float:
    """
    The VLA at 45 GHz in D: a massive star-forming region, an ionised shell around a cluster,
    compact and ultracompact HII regions of several sizes, and diffuse gas
    """
    r = scale.fit(6)
    random = np.random.default_rng(14)
    sky.gaussian(scale.flux(15, 7 * r), 0, 0, 7 * r, 5 * r, 40)
    sky.disk(
        ring_flux(scale, 40, 2.6 * r, 1.8 * r, 0), "ring", 2.6 * r, 1.8 * r, 0, 0, 0, -1 * r, 1 * r
    )
    sky.clumps(
        scale, 25, (-1 * r, 1 * r), (2.6 * r, 2.4 * r), 0, (0.3 * r, 0.9 * r), (15, 60), random
    )
    for east, north, size, snr in (
        (1.6, -1.4, 0.4, 900), (2.6, -2.2, 0.8, 250), (-3.4, -2.0, 1.2, 150), (3.6, 2.6, 0.6, 200),
        (-4.2, 3.0, 0.9, 100), (0.6, -3.6, 0.3, 300)
    ):
        sky.gaussian(scale.flux(snr, size * r), east * r, north * r, size * r, 0.8 * size * r, 30)
    return 6 * r


def young_disk(sky: Sky, scale: Scale) -> float:
    """
    The VLA at 45 GHz in A: a young star's disk at 7 mm, an inner disk and two rings, a crescent
    where dust is trapped, and knots of free-free emission along a jet
    """
    r = scale.fit(5)
    random = np.random.default_rng(15)
    inclination, angle = 40.0, 140.0
    sky.disk(
        ring_flux(scale, 60, 1.8 * r, 0, inclination),
        "exponential",
        0.9 * r,
        inclination=inclination,
        angle=angle
    )
    for inner, outer, snr in ((2.2, 2.9, 90), (3.6, 4.4, 60)):
        sky.disk(
            ring_flux(scale, snr, outer * r, inner * r, inclination), "ring", outer * r, inner * r,
            inclination, angle
        )
    theta = np.radians(angle)
    crescent = 4.0 * r * np.array(
        [np.sin(theta + 1.2), np.cos(theta + 1.2) * np.cos(np.radians(40))]
    )
    sky.gaussian(scale.flux(110, 1.1 * r), *crescent, 1.3 * r, 0.5 * r, angle + 60)
    sky.point(scale.flux(800))
    jet = along(angle + 90, np.linspace(1.2, 4.8, 4) * r)
    for knot in jet:
        for side in (1, -1):
            sky.gaussian(
                scale.flux(random.uniform(30, 70), 0.4 * r), *(side * knot), 0.4 * r, 0.3 * r
            )
    return 5 * r


def protocluster(sky: Sky, scale: Scale) -> float:
    """
    ALMA at 100 GHz in C-3: a hub of dense gas where filaments meet, cores along each, the
    birthplace of a star cluster
    """
    r = scale.fit(5)
    random = np.random.default_rng(16)
    sky.gaussian(scale.flux(30, 2 * r), 0, 0, 2.2 * r, 1.6 * r, 60)
    sky.point(scale.flux(500))
    for heading, length in ((60, 4.6), (150, 3.6), (250, 4.2), (320, 3.2)):
        path = along(heading, np.linspace(0.5, length, 30) * r, 0.4 * r, 1.0)
        sky.filament(scale, path, 0.6 * r, (10, 25), random)
        for k in random.choice(len(path), 3, replace=False):
            sky.gaussian(
                scale.flux(random.uniform(80, 250), 0.4 * r), *path[k], 0.45 * r, 0.35 * r,
                random.uniform(0, 180)
            )
    return 5 * r


def ringed_disk(sky: Sky, scale: Scale, spiral: bool = False) -> float:
    """
    ALMA at long baselines: a protoplanetary disk with bright rings and dark gaps around an inner
    disk, a crescent where dust is trapped, and, with spiral, two spiral arms in its outer part
    """
    r = scale.fit(8)
    random = np.random.default_rng(17)
    inclination, angle = 35.0, 130.0
    cosine = np.cos(np.radians(inclination))
    sky.disk(
        ring_flux(scale, 250, 1.4 * r, 0, inclination),
        "power_law",
        1.4 * r,
        0.1 * r,
        inclination,
        angle,
        power=0.6
    )
    for inner, outer, snr in ((2.0, 2.9, 220), (3.7, 4.4, 160), (5.3, 6.0, 110), (6.9, 7.4, 70)):
        sky.disk(
            ring_flux(scale, snr, outer * r, inner * r, inclination), "ring", outer * r, inner * r,
            inclination, angle
        )
    sky.point(scale.flux(1000))
    theta = np.radians(angle)
    crescent = 6.4 * r * np.array([np.sin(theta + 2.0), np.cos(theta + 2.0) * cosine])
    sky.gaussian(scale.flux(160, 0.8 * r), *crescent, 1.8 * r, 0.5 * r, angle + 110)
    if spiral:
        for arm in (0, np.pi):
            t = np.linspace(0, 1.7, 50)
            radius, phase = r * (4.4 + 1.7 * t), arm + 2.2 * t
            path = np.c_[radius * np.cos(phase), radius * np.sin(phase) * cosine]
            sky.filament(scale, path, 0.6 * r, (25, 50), random)
    return 8 * r


def lensed_galaxy(sky: Sky, scale: Scale) -> float:
    """
    ALMA at 345 GHz in C-3: a dusty galaxy behind a massive one, lensed into an Einstein ring with
    two bright, clumpy arcs and a faint counter-image
    """
    r = scale.fit(2.5)
    random = np.random.default_rng(18)
    sky.disk(ring_flux(scale, 50, 1.8 * r, 1.4 * r, 0), "ring", 1.8 * r, 1.4 * r)
    for start, stop, snr in ((0.2, 1.4, (60, 160)), (3.2, 4.2, (50, 130))):
        phase = np.linspace(start, stop, 25)
        arc = 1.6 * r * np.c_[np.cos(phase), np.sin(phase)]
        sky.filament(scale, arc, 0.35 * r, snr, random)
    sky.gaussian(scale.flux(70, 0.4 * r), -0.4 * r, -1.1 * r, 0.4 * r, 0.3 * r, 20)
    sky.point(scale.flux(40))
    return 2.5 * r


TARGETS: Dict[str, Callable[[Sky, Scale], float]] = {
    "giant_radio_galaxy": giant_radio_galaxy,
    "spiral_galaxy": spiral_galaxy,
    "cygnus_a": cygnus_a,
    "star_forming_region": star_forming_region,
    "young_disk": young_disk,
    "protocluster": protocluster,
    "ringed_disk": ringed_disk,
    "spiral_disk": lambda sky, scale: ringed_disk(sky, scale, spiral=True),
    "lensed_galaxy": lensed_galaxy,
}


def build(target: str, centre, scale: Scale) -> Tuple[list, float]:
    """
    The target's components and its extent (radius, arcsec) for the figures
    """
    sky = Sky(centre)
    extent = TARGETS[target](sky, scale)
    return sky.sources, extent
