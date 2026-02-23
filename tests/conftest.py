"""Pytest configuration for smartmemory-cc tests."""

import pytest


def pytest_collection_modifyitems(items):
    """Auto-mark tests in tests/integration/ with the integration marker."""
    for item in items:
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
