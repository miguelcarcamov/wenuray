"""
TOML recipes (#36). A recipe is a serialisation of the objects, not a second API: every table maps
to a constructor and its keys are that class's arguments. A recipe and a Python script therefore
build equal objects, and they raise the same errors, because validation lives only in the classes.

    ms = "target.ms"

    [selection]
    spw = "0~3"

    [solve]
    refant = "DA51"

    [imager]
    type = "tclean"
    niter = 1000
    geometry = { imsize = [1024, 1024], cell = "0.05arcsec" }

    [[steps]]
    type = "phase"
    solint = "inf"

    [[steps]]
    type = "ampphase"
    solint = "inf"
    imaging = { threshold = "0.025mJy" }

is Selfcal(ms="target.ms", selection=Selection(spw="0~3"), solve=SolveOptions(refant="DA51"),
imager=Tclean(ImageGeometry(imsize=1024, cell="0.05arcsec"), niter=1000),
steps=[Phasecal("inf"), AmpPhasecal("inf", imaging={"threshold": "0.025mJy"})]).

The tables of the imager, the steps, the acceptance policies and the flagging name their class with
type: a built-in name (see FAMILIES), an imager registered under the "wenuray.imagers" entry point, or
"package.module:Class" for a class of your own.

preset = "setup.toml" lays the recipe over another TOML file, e.g. the solve options and imager
settings a group shares; tables merge key by key and the recipe wins. Wenuray's own paths in a recipe
(ms, output, preset, the initial_stack tables) are relative to the recipe's directory; the
imager's options are passed as written. Wenuray ships no presets: its defaults come from theory
or the data, not from an instrument.

Classes serialise their dataclass fields, leaving out the values that equal the defaults. A class
whose constructor differs from its fields (e.g. one taking ``**options``) provides to_recipe() and the
classmethod from_recipe(table) instead.
"""
from __future__ import annotations

import contextlib
import contextvars
import dataclasses
import importlib
import os
import tomllib
from importlib.metadata import entry_points
from typing import Any, Iterator, Mapping, Optional

# Built-in names of each family's classes, imported when needed (the classes import this module)
FAMILIES = {
    "step": {
        "phase": "wenuray.selfcalibration.steps:Phasecal",
        "amp": "wenuray.selfcalibration.steps:Ampcal",
        "ampphase": "wenuray.selfcalibration.steps:AmpPhasecal",
    },
    "acceptance": {
        "psnr_improves": "wenuray.selfcalibration.acceptance:PSNRImproves",
        "flux_not_lost": "wenuray.selfcalibration.acceptance:FluxNotLost",
        "enough_antennas": "wenuray.selfcalibration.acceptance:EnoughAntennas",
        "beam_stable": "wenuray.selfcalibration.acceptance:BeamStable",
        "flagged_fraction_below": "wenuray.selfcalibration.acceptance:FlaggedFractionBelow",
        "all_of": "wenuray.selfcalibration.acceptance:AllOf",
        "any_of": "wenuray.selfcalibration.acceptance:AnyOf",
    },
    "flagging": {
        "none": "wenuray.selfcalibration.flagging:NoFlagging",
        "residual": "wenuray.selfcalibration.flagging:ResidualFlagging",
    },
    # Imagers come from the "wenuray.imagers" entry points (wenuray.imaging.registry)
    "imager": {},
}
BASES = {
    "step": "wenuray.selfcalibration.steps:CalibrationStep",
    "acceptance": "wenuray.selfcalibration.acceptance:AcceptancePolicy",
    "flagging": "wenuray.selfcalibration.flagging:Flagging",
    "imager": "wenuray.imaging.contract:Imager",
}

# Whether encode() writes the values that equal their defaults too
_WITH_DEFAULTS = contextvars.ContextVar("with_defaults", default=False)


@contextlib.contextmanager
def with_defaults() -> Iterator[None]:
    """
    Within it, encode() also writes the values that equal their defaults: the fully resolved
    recipe, e.g. for `wenuray show --defaults` or a run record
    """
    token = _WITH_DEFAULTS.set(True)
    try:
        yield
    finally:
        _WITH_DEFAULTS.reset(token)


def writes_defaults() -> bool:
    """
    Whether values equal to their defaults are written; for to_recipe() hooks
    """
    return _WITH_DEFAULTS.get()


class RecipeError(ValueError):
    """
    A recipe can't be read, or an object can't be written to one
    """


def _load(reference: str) -> type:
    module, _, name = reference.partition(":")
    if not module or not name:
        raise RecipeError("{0!r} is not of the form 'package.module:Class'".format(reference))
    try:
        value = importlib.import_module(module)
    except ImportError as error:
        raise RecipeError("Can't import {0!r}: {1}".format(module, error)) from None
    for part in name.split("."):
        value = getattr(value, part, None)
    if not isinstance(value, type):
        raise RecipeError("{0!r} is not a class".format(reference))
    return value


def _names(family: str) -> dict:
    """
    The family's names and the classes they refer to, as "module:Class"
    """
    if family == "imager":
        from .imaging.registry import GROUP
        return {entry.name: entry.value for entry in entry_points(group=GROUP)}
    return FAMILIES[family]


def _family(value: Any) -> Optional[str]:
    for family, base in BASES.items():
        if isinstance(value, _load(base)):
            return family
    return None


def class_of(family: str, name: str) -> type:
    """
    The class a recipe's type refers to

    Raises
    ------
    RecipeError
        If the name is unknown, or the class isn't of the family
    """
    names = _names(family)
    if name not in names and ":" not in name:
        raise RecipeError(
            "Unknown {0} type {1!r}; the built-in ones are {2}, or name a class of your own as "
            "'package.module:Class'".format(family, name, sorted(names))
        )
    cls = _load(names.get(name, name))
    if not issubclass(cls, _load(BASES[family])):
        raise RecipeError("{0!r} is not a {1} class".format(name, family))
    return cls


def type_name(family: str, cls: type) -> str:
    """
    The name a recipe gives to a class: its built-in name, else "module:Class"
    """
    reference = "{0}:{1}".format(cls.__module__, cls.__qualname__)
    for name, target in _names(family).items():
        if target == reference:
            return name
    if cls.__module__ == "__main__" or "<locals>" in cls.__qualname__:
        raise RecipeError(
            "{0} is defined in a script, so a recipe can't import it; define it in a module to "
            "write it to a recipe".format(cls.__qualname__)
        )
    return reference


def encode(value: Any) -> Any:
    """
    A value as TOML holds it: config objects and mappings become tables, the imager, steps,
    policies and flagging tables with their type, tuples arrays. None values are left out.
    """
    if isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, (list, tuple)):
        return [encode(item) for item in value]
    if isinstance(value, Mapping):
        return {str(name): encode(item) for name, item in value.items() if item is not None}
    family = _family(value)
    if family is not None:
        to_recipe = getattr(value, "to_recipe", None)
        table = to_recipe() if to_recipe is not None else fields_table(value)
        return {"type": type_name(family, type(value)), **table}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return fields_table(value)
    raise RecipeError("A recipe can't hold {0!r}".format(value))


def fields_table(config: Any) -> dict:
    """
    The dataclass fields of an object that differ from their defaults (all of them within
    with_defaults()), encoded
    """
    table = {}
    for item in dataclasses.fields(config):
        value = getattr(config, item.name)
        if not item.init or value is None or (not writes_defaults() and value == _default(item)):
            continue
        table[item.name] = encode(value)
    return table


def _default(item: dataclasses.Field) -> Any:
    if item.default is not dataclasses.MISSING:
        return item.default
    if item.default_factory is not dataclasses.MISSING:
        return item.default_factory()
    return dataclasses.MISSING


def decode(family: str, table: Mapping) -> Any:
    """
    Builds the object a typed table describes, e.g. decode("step", {"type": "phase", "solint": "inf"})
    """
    if not isinstance(table, Mapping):
        raise RecipeError("A {0} must be a table, not {1!r}".format(family, table))
    table = dict(table)
    if "type" not in table:
        raise RecipeError(
            "A {0} table needs a type, one of {1}".format(family, sorted(_names(family)))
        )
    cls = class_of(family, table.pop("type"))
    from_recipe = getattr(cls, "from_recipe", None)
    return from_recipe(table) if from_recipe is not None else cls(**table)


def decode_configs(table: Mapping, **configs: type) -> dict:
    """
    A copy of the table with the named sub-tables built as config objects, e.g.
    decode_configs(table, geometry=ImageGeometry)
    """
    table = dict(table)
    for name, config in configs.items():
        if isinstance(table.get(name), Mapping):
            table[name] = config(**table[name])
    return table


def grid_tables(imager: Any, **fields: Any) -> dict:
    """
    The recipe table of an imager made of a geometry, a weighting and backend options, such as
    Tclean and Pyralysis, with any further constructor arguments it has
    """
    from .config import Weighting

    table = {"geometry": encode(imager.geometry), **encode(fields)}
    if writes_defaults() or imager.weighting != Weighting():
        table["weighting"] = encode(imager.weighting)
    return {**table, **encode(dict(imager.options))}


def merge(base: Mapping, override: Mapping) -> dict:
    """
    The override laid over the base: tables merge key by key, any other value is replaced
    """
    merged = dict(base)
    for name, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(name), Mapping):
            merged[name] = merge(merged[name], value)
        else:
            merged[name] = value
    return merged


def read(path: str, _seen: tuple = ()) -> dict:
    """
    Reads a recipe with its presets laid underneath, and makes its paths absolute

    Raises
    ------
    RecipeError
        If a file isn't valid TOML, or presets include each other
    """
    path = os.path.abspath(path)
    if path in _seen:
        raise RecipeError("Presets include each other: {0}".format(" -> ".join(_seen + (path, ))))
    try:
        with open(path, "rb") as file:
            recipe = tomllib.load(file)
    except tomllib.TOMLDecodeError as error:
        raise RecipeError("{0} is not valid TOML: {1}".format(path, error)) from None
    directory = os.path.dirname(path)
    for name in ("ms", "output", "preset"):
        if isinstance(recipe.get(name), str) and recipe[name]:
            recipe[name] = os.path.normpath(os.path.join(directory, recipe[name]))
    for table in recipe.get("initial_stack", {}).get("tables", []):
        table["path"] = os.path.normpath(os.path.join(directory, table["path"]))
    preset = recipe.pop("preset", None)
    if preset is not None:
        recipe = merge(read(preset, _seen + (path, )), recipe)
    return recipe
