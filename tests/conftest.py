"""Shared test configuration.

CASA is not installable everywhere (wheels exist only for some platforms and Python versions), so
CASA-free tests stub ``casatasks``/``casatools`` when they are missing; unit tests replace the CASA
tasks a module uses with mocks of their own. Tests that need real CASA live under
``tests/integration``, which marks them ``casa``.
"""
import importlib.util
import sys
from unittest import mock

for _module in ("casatasks", "casatools"):
    if importlib.util.find_spec(_module) is None:
        sys.modules[_module] = mock.MagicMock(name=_module)
