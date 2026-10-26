"""Integration tests run real CASA tasks; they are skipped where CASA is not installed."""
import sys
from unittest import mock

import pytest

if isinstance(sys.modules.get("casatools"), mock.MagicMock):  # stubbed by tests/conftest.py
    collect_ignore_glob = ["test_*.py"]


def pytest_collection_modifyitems(items):
    for item in items:
        item.add_marker(pytest.mark.casa)
