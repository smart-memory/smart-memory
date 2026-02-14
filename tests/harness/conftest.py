"""Auto-mark all tests in this directory as harness tests."""

import pathlib

import pytest

_THIS_DIR = pathlib.Path(__file__).parent


def pytest_collection_modifyitems(config, items):
    for item in items:
        if pathlib.Path(item.fspath).is_relative_to(_THIS_DIR):
            item.add_marker(pytest.mark.harness)
