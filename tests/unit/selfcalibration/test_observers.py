"""The observers' own behaviour (#176)."""
import logging

import pytest

from wenuray.selfcalibration.observers import LoggingObserver, duration


@pytest.mark.parametrize(
    "seconds, text", [(0.4, "0 s"), (8.2, "8 s"), (192, "3 min 12 s"), (3900, "1 h 05 min")]
)
def test_durations_read_as_people_say_them(seconds, text):
    assert duration(seconds) == text


def test_a_phase_is_logged_when_it_starts_and_when_it_ends(caplog):
    with caplog.at_level(logging.INFO, logger="wenuray"):
        LoggingObserver().on_phase("Round 2/4 Phasecal('120s'): imaging with Pyralysis", None)
        LoggingObserver().on_phase("Round 2/4 Phasecal('120s'): imaging with Pyralysis", 192.0)

    assert [record.getMessage() for record in caplog.records] == [
        "Round 2/4 Phasecal('120s'): imaging with Pyralysis",
        "Round 2/4 Phasecal('120s'): imaging with Pyralysis: done in 3 min 12 s",
    ]


def record(index=1, **fields):
    from wenuray.selfcalibration.result import RoundRecord
    values = dict(
        index=index,
        step="Phasecal('inf')" if index else "",
        caltable="round1_p.cal" if index else "",
        image="round{0}.image".format(index),
        psnr=331.0,
        peak=0.69,
        rms=2.08e-3,
        model_flux=2.22,
        thermal_noise=0.612e-3,
        reasons=("PSNR 301.9 -> 331.0", )
    )
    values.update(fields)
    return RoundRecord(**values)


COMPLETE = dict(solutions=8, failed=0, intervals=1, short_intervals=0, min_antennas=3, minsnr=3.0)


class TestRoundBlock:
    """#179: a round reads as what was solved, how well, the image and the decision."""

    def test_an_accepted_round(self):
        from wenuray.selfcalibration.diagnostics import GainDiagnostics
        from wenuray.selfcalibration.observers import round_block

        text = round_block(
            record(gains=GainDiagnostics(3.2, solutions=7), solutions=COMPLETE), 3,
            record(0, psnr=301.9)
        )

        assert text.splitlines() == [
            "Round 1/3 Phasecal('inf'): accepted",
            "  solutions  8 solutions over 1 interval, all with S/N above 3",
            "  gains      phase scatter 3.2 x noise: a correction",
            "  image      PSNR 331.0 (+9.6% on round 0), rms 2.08 mJy/beam (3.40 x thermal), "
            "model flux 2.22 Jy",
            "  decision   accepted: PSNR 301.9 -> 331.0",
        ]

    def test_an_amplitude_correction_at_the_noise_says_so(self):
        from wenuray.selfcalibration.diagnostics import GainDiagnostics
        from wenuray.selfcalibration.observers import round_block

        text = round_block(record(gains=GainDiagnostics(4.3, 1.1, solutions=7)), 3, None)

        assert "phase scatter 4.3 x noise: a correction; amplitude scatter 1.1 x noise: " \
            "consistent with noise on 7 solutions" in text

    def test_a_round_short_of_antennas_says_why_and_what_to_try(self):
        from wenuray.selfcalibration.observers import round_block

        coverage = dict(
            COMPLETE, solutions=480, failed=58, intervals=60, short_intervals=12, min_antennas=4
        )
        text = round_block(
            record(
                step="AmpPhasecal('120s')",
                accepted=False,
                solutions=coverage,
                reasons=("12 of 60 intervals < 4 antennas (max 0%)", )
            ), 3, None
        )

        assert "58 (12.1%) failed (S/N below 3, or no data)" in text
        assert "12 of 60 intervals have fewer than 4 antennas: their gains are undetermined" in text
        assert "rejected: 12 of 60 intervals < 4 antennas" in text
        assert 'hint       a longer solint, or combining spectral windows (combine = "spw")' in text

    def test_a_round_rejected_on_its_image_gives_no_hint(self):
        from wenuray.selfcalibration.observers import round_block

        text = round_block(record(accepted=False, solutions=COMPLETE), 3, None)

        assert "hint" not in text

    def test_a_round_without_solutions(self):
        from wenuray.selfcalibration.observers import round_block

        nan = float("nan")
        text = round_block(
            record(
                psnr=nan,
                peak=nan,
                rms=nan,
                model_flux=nan,
                accepted=False,
                reasons=("gaincal found no solutions", )
            ), 3, None
        )

        assert text.splitlines()[:2] == [
            "Round 1/3 Phasecal('inf'): rejected",
            "  decision   rejected: gaincal found no solutions",
        ]
        assert "hint" in text

    def test_the_initial_image(self):
        from wenuray.selfcalibration.observers import round_block

        assert round_block(record(0, psnr=301.9), 3, None).splitlines() == [
            "Round 0 initial image",
            "  image      PSNR 301.9, rms 2.08 mJy/beam (3.40 x thermal), model flux 2.22 Jy",
        ]


def test_the_summary_tabulates_every_round():
    from wenuray.selfcalibration.caltable import CalStack
    from wenuray.selfcalibration.diagnostics import GainDiagnostics
    from wenuray.selfcalibration.observers import summary
    from wenuray.selfcalibration.result import SelfcalResult

    rounds = (
        record(0, psnr=301.9),
        record(gains=GainDiagnostics(3.2)),
        record(
            2,
            step="AmpPhasecal('inf')",
            psnr=310.0,
            accepted=False,
            gains=GainDiagnostics(4.3, 1.1)
        ),
    )

    lines = summary(SelfcalResult("point.ms", CalStack(), rounds), 202.0).splitlines()

    assert lines[0] == "Self-calibration finished in 3 min 22 s: 1 of 2 steps accepted"
    assert lines[1].split() == [
        "Round", "Step", "PSNR", "rms/thermal", "Model", "flux", "Gains/noise", "Decision"
    ]
    assert lines[4].split() == [
        "2", "AmpPhasecal('inf')", "310.0", "3.40", "2.22", "Jy", "4.3", "/", "1.1", "rejected"
    ]
    assert lines[-1] == "Final image: round1.image (round 1), PSNR 331.0, rms 3.40 x thermal"


def test_spectral_windows_without_solutions_are_explained():
    from wenuray.selfcalibration.observers import round_block

    coverage = dict(COMPLETE, spws=(13, 15), uncalibrated_spws=(17, ), empty_spws=(0, 1, 2, 14))
    text = round_block(record(solutions=coverage), 3, None)

    assert "spw 17 holds unflagged data but no solutions: applycal leaves it uncalibrated" in text
    assert 'spws 0-2, 14 hold no unflagged data: CASA\'s "no corresponding cal spws" about them ' \
        "is harmless" in text
