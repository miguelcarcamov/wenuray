from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from typing import Iterator, Sequence

import tomli_w


@dataclass(frozen=True)
class CalTable:
    """
    Calibration table accepted by a self-calibration iteration

    Parameters
    ----------
    path :
        Path to the CASA calibration table
    calmode :
        Calibration mode the table was solved with: "p", "a" or "ap"
    spwmap :
        Spectral window map used to apply the table. Empty when every spectral window has its own
        solutions
    interp :
        Temporal interpolation used to apply the table, e.g. "linear" or "nearest"
    """
    path: str
    calmode: str
    spwmap: tuple = ()
    interp: str = "linear"

    def __post_init__(self):
        if self.calmode not in ("p", "a", "ap"):
            raise ValueError("calmode must be 'p', 'a' or 'ap', not {0!r}".format(self.calmode))
        # Accept any sequence (e.g. a list read from a file); store a hashable tuple
        object.__setattr__(self, "spwmap", tuple(int(spw) for spw in self.spwmap))

    @property
    def solves_phase(self) -> bool:
        return "p" in self.calmode

    @property
    def solves_amplitude(self) -> bool:
        return "a" in self.calmode


def derive_spwmap(combine: str, spw_ids: Sequence[int], nspw: int) -> list:
    """
    Function that derives the spectral window map needed to apply a gain solution. When the
    solution combines spectral windows, CASA stores it under the lowest selected spectral window,
    so every spectral window maps to it. Otherwise each spectral window has its own solution and
    no map is needed.

    Parameters
    ----------
    combine :
        Data axes combined when solving, e.g. "spw" or "scan,spw"
    spw_ids :
        Spectral window ids selected when solving
    nspw :
        Total number of spectral windows in the measurement set

    Returns
    -------
    The spectral window map, or an empty list if the solution does not combine spectral windows
    """
    if "spw" not in [axis.strip() for axis in combine.split(",")]:
        return []
    if not spw_ids:
        raise ValueError("No spectral windows selected to solve for")
    return [min(spw_ids)] * nspw


@dataclass(frozen=True)
class CalStack:
    """
    Immutable, ordered stack of accepted calibration tables. It is the only place that builds the
    gaintable, spwmap and interp arguments of gaincal and applycal, one entry per table, so a table
    can't be dropped from one list but kept in another (#25, #26).

    Parameters
    ----------
    tables :
        Calibration tables in the order they are applied
    """
    tables: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, "tables", tuple(self.tables))

    def __iter__(self) -> Iterator[CalTable]:
        return iter(self.tables)

    def __len__(self) -> int:
        return len(self.tables)

    def push(self, table: CalTable) -> CalStack:
        """
        Returns a new stack with table applied after the current ones
        """
        return CalStack(self.tables + (table, ))

    @property
    def solves_phase(self) -> bool:
        return any(table.solves_phase for table in self.tables)

    @property
    def solves_amplitude(self) -> bool:
        return any(table.solves_amplitude for table in self.tables)

    def gaincal_arguments(self) -> dict:
        """
        The gaintable, spwmap and interp arguments of gaincal: the tables pre-applied while solving

        Returns
        -------
        Dictionary with the gaintable, spwmap and interp keyword arguments
        """
        return {
            "gaintable": [table.path for table in self.tables],
            "spwmap": [list(table.spwmap) for table in self.tables],
            "interp": [table.interp for table in self.tables],
        }

    def applycal_arguments(self, new: CalTable) -> dict:
        """
        The gaintable, spwmap and interp arguments of applycal: the stack plus the newly solved table

        Parameters
        ----------
        new :
            The newly solved calibration table

        Returns
        -------
        Dictionary with the gaintable, spwmap and interp keyword arguments
        """
        return self.push(new).gaincal_arguments()

    def to_dict(self, relative_to: str = None) -> dict:
        """
        Returns the stack as a dictionary that TOML can hold

        Parameters
        ----------
        relative_to :
            Directory the table paths are written relative to, so a run directory can be moved as a
            whole. Absolute paths are kept when None
        """
        tables = []
        for table in self.tables:
            path = os.path.abspath(table.path)
            if relative_to is not None:
                path = os.path.relpath(path, os.path.abspath(relative_to))
            tables.append(
                {
                    "path": path,
                    "calmode": table.calmode,
                    "spwmap": list(table.spwmap),
                    "interp": table.interp
                }
            )
        return {"tables": tables}

    @classmethod
    def from_dict(cls, data: dict, relative_to: str = None) -> CalStack:
        """
        Builds a stack from a dictionary written by to_dict

        Parameters
        ----------
        data :
            The dictionary
        relative_to :
            Directory that relative table paths are resolved against
        """
        tables = []
        for table in data.get("tables", []):
            table = dict(table)
            if relative_to is not None and not os.path.isabs(table["path"]):
                table["path"] = os.path.normpath(os.path.join(relative_to, table["path"]))
            tables.append(CalTable(**table))
        return cls(tables)

    def to_toml(self, path: str) -> None:
        """
        Writes the stack to a TOML file, with table paths relative to the file's directory
        """
        with open(path, "wb") as file:
            tomli_w.dump(self.to_dict(relative_to=os.path.dirname(os.path.abspath(path))), file)

    @classmethod
    def from_toml(cls, path: str) -> CalStack:
        """
        Reads a stack written by to_toml
        """
        with open(path, "rb") as file:
            data = tomllib.load(file)
        return cls.from_dict(data, relative_to=os.path.dirname(os.path.abspath(path)))
