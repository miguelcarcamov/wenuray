"""CASA tables must always be closed, also when a call fails (#29)."""
import numpy as np
import pytest

from wenuray.utils import selfcal_utils


class FakeTable:
    """Records how a casatools table tool is used."""

    def __init__(self, columns=(), nrows=0, fail_on=None):
        self.columns, self.rows, self.fail_on = list(columns), nrows, fail_on
        self.calls = []
        self.queries = []

    def __record(self, name):
        self.calls.append(name)
        if name == self.fail_on:
            raise RuntimeError(name + " failed")

    def open(self, tablename):
        self.calls.append(("open", tablename))

    def colnames(self):
        self.__record("colnames")
        return self.columns

    def nrows(self):
        self.__record("nrows")
        return self.rows

    def query(self, query):
        self.calls.append(("query", query))
        result = FakeTable(nrows=self.rows - 1)
        self.queries.append(result)
        return result

    def close(self):
        self.calls.append("close")

    def done(self):
        self.calls.append("done")

    @property
    def closed(self) -> bool:
        return "close" in self.calls


@pytest.fixture
def ms(tmp_path) -> str:
    path = tmp_path / "target.ms"
    path.mkdir()
    return str(path)


def use_table(monkeypatch, fake: FakeTable) -> FakeTable:
    monkeypatch.setattr(selfcal_utils, "table", lambda: fake)
    return fake


class TestOpenTable:

    def test_closes_and_releases_the_tool(self, monkeypatch, ms):
        fake = use_table(monkeypatch, FakeTable())

        with selfcal_utils.open_table(ms):
            pass

        assert fake.calls == [("open", ms), "close", "done"]

    def test_closes_when_the_body_raises(self, monkeypatch, ms):
        fake = use_table(monkeypatch, FakeTable(fail_on="colnames"))

        with pytest.raises(RuntimeError):
            with selfcal_utils.open_table(ms) as table_tool:
                table_tool.colnames()

        assert fake.closed


class TestIsColumnInMs:

    @pytest.mark.parametrize("column, expected", [("CORRECTED_DATA", True), ("MODEL_DATA", False)])
    def test_reports_the_column_and_closes_the_table(self, monkeypatch, ms, column, expected):
        fake = use_table(monkeypatch, FakeTable(columns=["DATA", "CORRECTED_DATA"]))

        assert selfcal_utils.is_column_in_ms(ms, column) is expected
        assert fake.closed

    def test_missing_measurement_set_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            selfcal_utils.is_column_in_ms(str(tmp_path / "missing.ms"), "DATA")


def test_get_table_rows_closes_the_table(monkeypatch, ms):
    fake = use_table(monkeypatch, FakeTable(nrows=8))

    assert selfcal_utils.get_table_rows(ms + "/SPECTRAL_WINDOW") == 8
    assert fake.closed


def test_calculate_number_antennas_closes_table_and_query(monkeypatch, ms):
    fake = use_table(monkeypatch, FakeTable(nrows=7))

    assert selfcal_utils.calculate_number_antennas(ms) == 6
    assert ("open", ms + "/ANTENNA") in fake.calls
    assert ("query", "!FLAG_ROW") in fake.calls
    assert fake.closed and fake.queries[0].closed


class TestColumnsMatch:

    @pytest.fixture
    def table_with(self, monkeypatch):

        def make(corrected_scale, nrows=1000):
            fake = FakeTable(nrows=nrows)
            fake.rows_read = []

            def getcell(column, row):
                fake.rows_read.append(row)
                return np.ones(3) * (corrected_scale if column == "CORRECTED_DATA" else 1.0)

            fake.getcell = getcell
            return use_table(monkeypatch, fake)

        return make

    @pytest.mark.parametrize("scale, expected", [(1.0, True), (2.0, False)])
    def test_compares_the_columns(self, table_with, ms, scale, expected):
        fake = table_with(scale)

        assert selfcal_utils.columns_match(ms, "DATA", "CORRECTED_DATA") is expected
        assert fake.closed

    def test_reads_only_a_sample_of_rows_spread_over_the_table(self, table_with, ms):
        fake = table_with(1.0, nrows=1000)

        selfcal_utils.columns_match(ms, "DATA", "CORRECTED_DATA", samples=10)

        rows = sorted(set(fake.rows_read))
        assert len(rows) == 10 and rows[0] == 0 and rows[-1] == 999
