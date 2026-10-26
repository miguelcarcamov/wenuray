"""
Imagers by name, for configuration files (#23, #36): "tclean", or a third-party imager that a
package registers under the "wenuray.imagers" entry point group:

    [project.entry-points."wenuray.imagers"]
    myimager = "mypackage.imaging:MyImager"
"""
from __future__ import annotations

from importlib.metadata import entry_points

from .contract import Imager

GROUP = "wenuray.imagers"


def available_imagers() -> list:
    """
    Names of the imagers that can be built by name
    """
    return sorted(entry_point.name for entry_point in entry_points(group=GROUP))


def imager_from_name(name: str, *args, **kwargs) -> Imager:
    """
    Builds the imager registered under name, passing it the arguments

    Raises
    ------
    KeyError
        If no imager is registered under name
    """
    matches = entry_points(group=GROUP, name=name)
    if not matches:
        raise KeyError(
            "No imager named {0!r}; available: {1}. Imagers register under the {2!r} entry "
            "point group".format(name, available_imagers(), GROUP)
        )
    imager_class = next(iter(matches)).load()
    if not (isinstance(imager_class, type) and issubclass(imager_class, Imager)):
        raise TypeError("The imager registered as {0!r} is not an Imager subclass".format(name))
    return imager_class(*args, **kwargs)
