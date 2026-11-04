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
