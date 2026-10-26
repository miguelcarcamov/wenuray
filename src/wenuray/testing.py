"""
Test helpers for imagers and self-calibration with real CASA (#23, #32). They need pytest and CASA.

ImagerContractSuite checks that an imager, also a user's own, follows the self-calibration
contract:

    from wenuray.testing import ImagerContractSuite

    class TestMyImager(ImagerContractSuite):
        imager = MyImager(...)
"""
from __future__ import annotations

import fnmatch
import os
from types import SimpleNamespace
from typing import Optional, Sequence, Tuple

import numpy as np
import pytest
from casatasks import clearcal
from casatools import componentlist, measures, simulator, table

from .imaging.contract import Imager, ImagingRequest
from .utils.selfcal_utils import open_table, selected_rows

DIRECTION = ("J2000", "10h00m00s", "+30d00m00s")
NANTENNAS = 8


def simulate_point_source_ms(
    path: str,
    seed: int = 1,
    phase_errors_in: Sequence[str] = ("DATA", "CORRECTED_DATA"),
    noise: float = 0.05
) -> str:
    """
    Simulates a small VLA observation of a 1 Jy point source at the phase center: eight antennas,
    one 128 MHz channel at 5 GHz, two hours of 60 s integrations. Everything is seeded, so the
    data are the same on every run.

    Parameters
    ----------
    path :
        Path of the new measurement set
    seed :
        Seed of the antenna positions, the phase errors and the noise
    phase_errors_in :
        Columns, of DATA and CORRECTED_DATA, that get antenna-based phase errors; the others hold
        the true visibilities. Both get the same noise
    noise :
        Standard deviation of the noise of the real and imaginary parts of each visibility, in Jy

    Returns
    -------
    The path of the measurement set
    """
    rng = np.random.default_rng(seed)
    east, north = rng.uniform(-2000, 2000, NANTENNAS), rng.uniform(-2000, 2000, NANTENNAS)
    me, sm, cl = measures(), simulator(), componentlist()
    cl.addcomponent(dir=" ".join(DIRECTION), flux=1.0, fluxunit="Jy", freq="5GHz", shape="point")
    cl.rename(path + ".cl")
    cl.close()
    sm.open(path)
    sm.setconfig(
        telescopename="VLA",
        x=east.tolist(),
        y=north.tolist(),
        z=[0.0] * NANTENNAS,
        dishdiameter=[25.0] * NANTENNAS,
        mount=["alt-az"] * NANTENNAS,
        antname=["A{0}".format(i) for i in range(NANTENNAS)],
        padname=["P{0}".format(i) for i in range(NANTENNAS)],
        coordsystem="local",
        referencelocation=me.observatory("VLA")
    )
    sm.setspwindow(
        spwname="C",
        freq="5GHz",
        deltafreq="128MHz",
        freqresolution="128MHz",
        nchannels=1,
        stokes="RR LL"
    )
    sm.setfeed(mode="perfect R L")
    sm.setfield(sourcename="target", sourcedirection=me.direction(*DIRECTION))
    sm.setlimits(shadowlimit=0.001, elevationlimit="8deg")
    sm.setauto(autocorrwt=0.0)
    sm.settimes(
        integrationtime="60s",
        usehourangle=True,
        referencetime=me.epoch("utc", "2024/01/01/00:00:00")
    )
    sm.observe(sourcename="target", spwname="C", starttime="-1h", stoptime="1h")
    sm.predict(complist=path + ".cl")
    sm.close()
    me.done()
    clearcal(vis=path, addmodel=False)  # CORRECTED_DATA starts as a copy of DATA
    _corrupt(path, rng, phase_errors_in, noise)
    return path


def injected_phases(path: str, seed: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    The antenna phase errors that simulate_point_source_ms(path, seed) injected, to compare
    self-calibration solutions with. Antenna A0, the reference, has none.

    Returns
    -------
    The integration times (the measurement set's TIME, in s), and the phase of each antenna at
    those times in radians, shaped (antenna, time)
    """
    rng = np.random.default_rng(seed)
    rng.uniform(-2000, 2000, NANTENNAS), rng.uniform(-2000, 2000, NANTENNAS)  # the positions
    with open_table(path) as ms_table:
        times = np.unique(ms_table.getcol("TIME"))
    return times, _phases(_phase_model(rng), times - times.min())


def _phase_model(rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Each antenna's phase error: an offset, plus a drift of some amplitude and period in hours
    """
    offset = rng.uniform(-np.pi / 3, np.pi / 3, NANTENNAS)
    drift, period = rng.uniform(-np.pi / 9, np.pi / 9, NANTENNAS), rng.uniform(1.0, 3.0, NANTENNAS)
    return offset, drift, period


def _phases(model: Tuple[np.ndarray, np.ndarray, np.ndarray], seconds: np.ndarray) -> np.ndarray:
    offset, drift, period = model
    hours = seconds / 3600.0
    phase = offset[:, None] + drift[:, None] * np.sin(2 * np.pi * hours / period[:, None])
    phase[0] = 0.0  # the reference antenna
    return phase


def _corrupt(
    path: str, rng: np.random.Generator, phase_errors_in: Sequence[str], noise: float
) -> None:
    """
    Antenna-based phase errors (a constant offset plus a slow drift) and thermal noise, from
    seeded numpy rather than the simulator's corruption, which is not reproducible from run to run
    """
    tb = table()
    tb.open(path, nomodify=False)
    try:
        true = tb.getcol("DATA")  # (correlation, channel, row)
        antenna1, antenna2 = tb.getcol("ANTENNA1"), tb.getcol("ANTENNA2")
        times = tb.getcol("TIME")
        phase = _phases(_phase_model(rng), times - times.min())
        rows = np.arange(len(times))
        gains = np.exp(1j * (phase[antenna1, rows] - phase[antenna2, rows]))
        thermal = noise * (rng.standard_normal(true.shape) + 1j * rng.standard_normal(true.shape))
        for column in ("DATA", "CORRECTED_DATA"):
            tb.putcol(column, (true * gains if column in phase_errors_in else true) + thermal)
    finally:
        tb.close()


class ImagerContractSuite:
    """
    Conformance tests of the self-calibration imager contract. Subclass it in a test module and set
    imager; pytest then runs every test below for that imager.

    The imager images a simulated 1 Jy point source whose DATA has phase errors and whose
    CORRECTED_DATA is clean. The tests check what the contract says, in the visibility domain, so
    they don't depend on how well an imager deconvolves:

    - MODEL_DATA holds the true visibilities. An imager that reads DATA instead of CORRECTED_DATA
      fits a decorrelated source and fails;
    - the result images can be measured by ImageStatistics;
    - nothing is left in the working directory.

    Override the tolerances if your imager needs it.
    """
    imager: Optional[Imager] = None
    # Largest allowed error of the median MODEL_DATA amplitude, in Jy, of the 1 Jy source
    flux_tolerance: float = 0.15
    # Largest allowed median phase of MODEL_DATA, in degrees (the source is at the phase center)
    max_model_phase: float = 5.0
    # Files an imager may leave in the working directory, e.g. CASA's log
    allowed_leftovers: tuple = ("casa-*.log", )

    @pytest.fixture(scope="class")
    @classmethod
    def contract_run(cls, tmp_path_factory) -> SimpleNamespace:
        if not isinstance(cls.imager, Imager):
            pytest.fail("Set imager to the Imager under test")
        outputs = tmp_path_factory.mktemp("outputs")
        workdir = tmp_path_factory.mktemp("workdir")
        ms = simulate_point_source_ms(str(outputs / "contract.ms"), phase_errors_in=("DATA", ))
        previous_cwd = os.getcwd()
        os.chdir(workdir)
        try:
            result = cls.imager.image(ImagingRequest(ms, str(outputs / "contract")))
        finally:
            os.chdir(previous_cwd)
        return SimpleNamespace(ms=ms, result=result, workdir=workdir)

    def test_model_data_fits_the_corrected_data(self, contract_run):
        rows = selected_rows(contract_run.ms)
        with open_table(contract_run.ms) as ms_table:
            model = np.concatenate([ms_table.getcell("MODEL_DATA", row).ravel() for row in rows])

        assert np.median(np.abs(model)) == pytest.approx(1.0, abs=self.flux_tolerance), (
            "MODEL_DATA should hold the 1 Jy source of CORRECTED_DATA; DATA has phase errors "
            "that decorrelate it"
        )
        assert np.degrees(np.median(np.abs(np.angle(model)))) < self.max_model_phase

    def test_result_images_can_be_measured(self, contract_run):
        statistics = contract_run.result.statistics()

        assert np.isfinite(statistics.psnr) and statistics.peak > 0

    def test_nothing_is_left_in_the_working_directory(self, contract_run):
        left = [
            name for name in os.listdir(contract_run.workdir)
            if not any(fnmatch.fnmatch(name, pattern) for pattern in self.allowed_leftovers)
        ]

        assert left == [], "The imager wrote {0} to the working directory".format(left)
