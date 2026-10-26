import hashlib
import os
from contextlib import contextmanager
from typing import Tuple

import numpy as np
from casatools import ms, table

# m/s
SPEED_OF_LIGHT = 299792458.0


@contextmanager
def open_table(table_name: str):
    """
    Context manager that opens a CASA table read-only and always closes it, also on errors, so
    no table lock is left behind

    Parameters
    ----------
    table_name :
        Path to the CASA table, e.g. a measurement set or one of its subtables

    Yields
    ------
    The opened casatools table tool
    """
    table_tool = table()
    table_tool.open(tablename=table_name)
    try:
        yield table_tool
    finally:
        table_tool.close()
        table_tool.done()


def is_column_in_ms(ms_name: str = "", column_name: str = "") -> bool:
    """
        Function that returns True if column_name is present in Measurement Set file

        Parameters
        ----------
        ms_name :
            Measurement set name
        column_name:
            Column name

        Returns
        -------
        True if column_name exists False otherwise
    """
    if ms_name != "":
        if os.path.exists(ms_name):
            with open_table(ms_name) as ms_table:
                return column_name in ms_table.colnames()
        else:
            raise FileNotFoundError("The Measurement Set File does not exist")
    else:
        raise ValueError("Measurement Set File cannot be empty")


def columns_match(ms_name: str, first: str, second: str, samples: int = 100) -> bool:
    """
        Function that returns True if two data columns of a Measurement Set hold the same values,
        compared on rows sampled evenly through the table so that large datasets are not read

        Parameters
        ----------
        ms_name :
            Measurement set name
        first :
            First column name, e.g. "DATA"
        second :
            Second column name, e.g. "CORRECTED_DATA"
        samples :
            Number of rows compared

        Returns
        -------
        True if the sampled rows match, False otherwise
    """
    with open_table(ms_name) as ms_table:
        nrows = ms_table.nrows()
        # Row by row, since rows of different spectral windows can have different shapes
        for row in np.unique(np.linspace(0, nrows - 1, min(nrows, samples), dtype=int)):
            row = int(row)
            if not np.allclose(ms_table.getcell(first, row), ms_table.getcell(second, row)):
                return False
    return True


def get_table_rows(ms_table: str = "") -> int:
    """
    Function that returns the number of rows of a measurement set table

    Parameters
    ----------
    ms_table :
        Measurement set table name

    Returns
    -------
    rows :
        The number of rows of the measurement set table

    """
    with open_table(ms_table) as table_tool:
        return table_tool.nrows()


def calculate_number_antennas(ms_name: str = "") -> int:
    """
    Function that counts the antennas of a measurement set that are not flagged in its ANTENNA
    table (FLAG_ROW). Antennas whose data are all flagged in the main table are still counted.

    Parameters
    ----------
    ms_name :
        Absolute file name to the measurement set file

    Returns
    -------
    nrows :
        Number of antennas not flagged in the ANTENNA table
    """
    if ms_name != "":
        if os.path.exists(ms_name):
            with open_table(ms_name + "/ANTENNA") as antenna_table:
                unflagged = antenna_table.query(query="!FLAG_ROW")
                try:
                    return unflagged.nrows()
                finally:
                    unflagged.close()
        else:
            raise FileNotFoundError("The Measurement Set File does not exist")
    else:
        raise ValueError("Measurement Set File cannot be empty")


def count_spectral_windows(ms_name: str = "") -> int:
    """
    Function that returns the number of spectral windows of a measurement set

    Parameters
    ----------
    ms_name :
        Measurement set name

    Returns
    -------
    The number of rows of the SPECTRAL_WINDOW table
    """
    return get_table_rows(ms_name + "/SPECTRAL_WINDOW")


def get_spw_ids(ms_name: str = "", spw: str = "") -> list:
    """
    Function that returns the ids of the spectral windows selected by a CASA spw selection

    Parameters
    ----------
    ms_name :
        Measurement set name
    spw :
        CASA spectral window selection, e.g. "0~3" or "1,3:10~20". Empty selects every spectral
        window

    Returns
    -------
    Sorted list of selected spectral window ids
    """
    if spw in ("", "*"):
        return list(range(count_spectral_windows(ms_name)))
    ms_tool = ms()
    try:
        selection = ms_tool.msseltoindex(vis=ms_name, spw=spw)
    finally:
        ms_tool.done()
    return sorted(int(spw_id) for spw_id in selection["spw"])


def selected_rows(ms_name: str, field: str = "", spw: str = "", samples: int = 100) -> list:
    """
    Function that returns row numbers of a measurement set in a field and spectral window
    selection, sampled evenly so that large datasets are not read

    Parameters
    ----------
    ms_name :
        Measurement set name
    field :
        CASA field selection; empty selects every field
    spw :
        CASA spectral window selection; empty selects every spectral window
    samples :
        Largest number of rows returned

    Returns
    -------
    Sorted list of row numbers
    """
    ms_tool = ms()
    try:
        selection = ms_tool.msseltoindex(vis=ms_name, field=field, spw=spw)
    finally:
        ms_tool.done()
    conditions = [
        "{0} IN [{1}]".format(column, ",".join(str(int(i)) for i in ids))
        for column, ids in (("FIELD_ID", selection["field"]), ("DATA_DESC_ID", selection["dd"]))
        if len(ids)
    ]
    with open_table(ms_name) as ms_table:
        if not conditions:
            rows = np.arange(ms_table.nrows())
        else:
            subtable = ms_table.query(query=" AND ".join(conditions))
            try:
                rows = np.asarray(subtable.rownumbers())
            finally:
                subtable.close()
    if len(rows) == 0:
        return []
    return [
        int(rows[i])
        for i in np.unique(np.linspace(0, len(rows) - 1, min(len(rows), samples), dtype=int))
    ]


def uv_extent(ms_name: str,
              field: str = "",
              spw: str = "",
              uvrange: str = "") -> Tuple[np.ndarray, np.ndarray]:
    """
    Function that returns, for each unflagged row of a selection, the larger of ``|u|`` and ``|v|`` in
    metres and the highest selected frequency of the row's spectral window. Their product over the
    speed of light is how far out the uv plane, in wavelengths, an image of the selection reaches

    Parameters
    ----------
    ms_name :
        Measurement set name
    field :
        CASA field selection; empty selects every field
    spw :
        CASA spectral window selection, channels included; empty selects every spectral window
    uvrange :
        CASA uv-range selection; empty selects every baseline

    Returns
    -------
    The extents in metres and the frequencies in Hz, one per selected row; empty if the selection
    has no rows
    """
    with open_table(os.path.join(ms_name, "SPECTRAL_WINDOW")) as spw_table:
        frequencies = [
            np.abs(spw_table.getcell("CHAN_FREQ", spw_id)) for spw_id in range(spw_table.nrows())
        ]
    with open_table(os.path.join(ms_name, "DATA_DESCRIPTION")) as dd_table:
        dd_spw = np.asarray(dd_table.getcol("SPECTRAL_WINDOW_ID"))
    ms_tool = ms()
    try:
        if spw in ("", "*"):
            highest = np.array([channels.max() for channels in frequencies])
        else:
            highest = np.zeros(len(frequencies))
            for spw_id, start, end, step in ms_tool.msseltoindex(vis=ms_name, spw=spw)["channel"]:
                highest[spw_id] = max(
                    highest[spw_id], frequencies[spw_id][start:end + 1:step].max()
                )
        ms_tool.open(ms_name)
        items = {"field": field, "spw": spw, "uvdist": uvrange}
        items = {name: value for name, value in items.items() if value}
        # msselect returns False for an empty selection, and also when there is nothing to select
        if items and not ms_tool.msselect(items):
            return np.zeros(0), np.zeros(0)
        data = ms_tool.getdata(["u", "v", "flag_row", "data_desc_id"])
    finally:
        ms_tool.done()
    keep = ~data["flag_row"]
    metres = np.maximum(np.abs(data["u"]), np.abs(data["v"]))[keep]
    return metres, highest[dd_spw[data["data_desc_id"][keep]]]


def ms_fingerprint(ms_name: str, samples: int = 100) -> str:
    """
    Function that returns a fingerprint of a measurement set's data, to tell later whether a run
    used the same data: a hash of its size, time range, antennas, fields and frequencies, and of
    DATA and FLAG in rows sampled evenly through the table, so large datasets are not read

    Parameters
    ----------
    ms_name :
        Measurement set name
    samples :
        Number of rows whose DATA and FLAG are hashed

    Returns
    -------
    "sha256:" followed by the hex digest
    """
    digest = hashlib.sha256()

    def add(value) -> None:
        digest.update(np.ascontiguousarray(value).tobytes())

    with open_table(ms_name) as ms_table:
        nrows = ms_table.nrows()
        add(np.array([nrows], dtype=np.int64))
        if nrows:
            times = ms_table.getcol("TIME")
            add(np.array([times.min(), times.max()]))
            for row in np.unique(np.linspace(0, nrows - 1, min(nrows, samples), dtype=int)):
                add(ms_table.getcell("DATA", int(row)))
                add(ms_table.getcell("FLAG", int(row)))
    for subtable, column in (("ANTENNA", "NAME"), ("FIELD", "NAME")):
        with open_table(os.path.join(ms_name, subtable)) as table_tool:
            digest.update("\n".join(table_tool.getcol(column)).encode())
    with open_table(os.path.join(ms_name, "SPECTRAL_WINDOW")) as spw_table:
        for spw_id in range(spw_table.nrows()):
            add(spw_table.getcell("CHAN_FREQ", spw_id))
    return "sha256:" + digest.hexdigest()


def model_column_problem(ms_name: str, field: str = "", spw: str = "") -> str:
    """
    Function that checks that MODEL_DATA holds a model for the selected data: the column exists,
    its cells have the shape of DATA, and it is not all zeros

    Parameters
    ----------
    ms_name :
        Measurement set name
    field :
        CASA field selection
    spw :
        CASA spectral window selection

    Returns
    -------
    A description of what is wrong, or an empty string if the model was written
    """
    if not is_column_in_ms(ms_name, "MODEL_DATA"):
        return "MODEL_DATA does not exist"
    rows = selected_rows(ms_name, field, spw)
    if not rows:
        return "the selection (field={0!r}, spw={1!r}) has no rows".format(field, spw)
    with open_table(ms_name) as ms_table:
        written = False
        for row in rows:
            model = ms_table.getcell("MODEL_DATA", row)
            if model.shape != ms_table.getcell("DATA", row).shape:
                return "MODEL_DATA has shape {0} but DATA has {1} in row {2}".format(
                    model.shape,
                    ms_table.getcell("DATA", row).shape, row
                )
            written = written or bool(np.any(model != 0))
    if not written:
        return "MODEL_DATA is all zeros in the selected rows"
    return ""


def solution_summary(caltable: str) -> tuple:
    """
    Function that summarises the gain solutions of a calibration table: the fraction flagged, and
    how many antennas have unflagged solutions in each solution interval (one per time and
    spectral window)

    Parameters
    ----------
    caltable :
        Path to the calibration table

    Returns
    -------
    A tuple with the flagged fraction and a tuple of antenna counts, one per solution interval
    """
    with open_table(caltable) as cal_table:
        if cal_table.nrows() == 0:
            return 1.0, ()
        flags = cal_table.getcol("FLAG")  # (polarization, channel, row)
        times = cal_table.getcol("TIME")
        spws = cal_table.getcol("SPECTRAL_WINDOW_ID")
        antennas = cal_table.getcol("ANTENNA1")
    # An antenna has a solution in an interval if any of its polarizations or channels is unflagged
    solved = ~np.all(flags, axis=(0, 1))
    counts = {}
    for time, spw, antenna, ok in zip(times, spws, antennas, solved):
        interval = counts.setdefault((float(time), int(spw)), set())
        if ok:
            interval.add(int(antenna))
    return float(np.mean(flags)), tuple(len(counts[key]) for key in sorted(counts))
