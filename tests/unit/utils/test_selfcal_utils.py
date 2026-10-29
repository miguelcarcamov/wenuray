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


class TestGetSpwIdsWithoutSelection:
    """
    An empty selection means the spectral windows that hold data, not every row of the
    SPECTRAL_WINDOW table: data split from a larger MS keep their original ids
    """

    class Tables:
        """A main table whose rows use some data descriptions, and the DATA_DESCRIPTION table."""

        def __init__(self, used_data_descriptions, spw_of_data_description):
            self.used = np.array(used_data_descriptions)
            self.spws = np.array(spw_of_data_description)
            self.opened = []

        def __call__(self):
            tables = self

            class Tool(FakeTable):

                def open(self, tablename):
                    super().open(tablename)
                    tables.opened.append(self)
                    self.name = tablename

                def query(self, query="", columns=""):
                    result = FakeTable()
                    result.getcol = lambda column: tables.used
                    self.calls.append(("query", query, columns))
                    self.queries.append(result)
                    return result

                def getcol(self, column):
                    assert self.name.endswith("/DATA_DESCRIPTION")
                    assert column == "SPECTRAL_WINDOW_ID"
                    return tables.spws

            return Tool(nrows=len(tables.spws))

    @pytest.mark.parametrize("spw", ["", "*"])
    def test_returns_the_spectral_windows_of_the_data(self, monkeypatch, ms, spw):
        # 21 spectral windows, of which the data use 13, 15, 17 and 19 (an ALMA split)
        tables = self.Tables([19, 13, 17, 15], list(range(21)))
        monkeypatch.setattr(selfcal_utils, "table", tables)

        assert selfcal_utils.get_spw_ids(ms, spw) == [13, 15, 17, 19]
        assert all(tool.closed for tool in tables.opened)
        assert all(query.closed for tool in tables.opened for query in tool.queries)

    def test_maps_data_descriptions_to_their_spectral_windows(self, monkeypatch, ms):
        tables = self.Tables([0, 1], [7, 3])
        monkeypatch.setattr(selfcal_utils, "table", tables)

        assert selfcal_utils.get_spw_ids(ms, "") == [3, 7]
