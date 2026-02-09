"""Unit tests for stores.registry — generic backend registry."""

from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.unit

from smartmemory.stores.registry import (
    register_store,
    create_store,
    list_store_backends,
    _STORE_REGISTRY,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    """Snapshot and restore the global registry around each test."""
    snapshot = dict(_STORE_REGISTRY)
    yield
    _STORE_REGISTRY.clear()
    _STORE_REGISTRY.update(snapshot)


class TestRegisterStore:
    def test_register_and_create(self):
        factory = MagicMock(return_value="instance")
        register_store("test_backend", factory)
        result = create_store("test_backend")
        assert result == "instance"
        factory.assert_called_once()

    def test_register_overwrites(self):
        register_store("dup", lambda **kw: "first")
        register_store("dup", lambda **kw: "second")
        assert create_store("dup") == "second"

    def test_register_empty_key_raises(self):
        with pytest.raises(ValueError, match="Invalid"):
            register_store("", lambda **kw: None)

    def test_register_non_callable_raises(self):
        with pytest.raises(ValueError, match="Invalid"):
            register_store("bad", "not_callable")  # type: ignore


class TestCreateStore:
    def test_unknown_key_raises(self):
        with pytest.raises(ValueError, match="Unknown store backend"):
            create_store("nonexistent_xyz")

    def test_kwargs_forwarded(self):
        factory = MagicMock(return_value="ok")
        register_store("kw_test", factory)
        create_store("kw_test", data_dir="/tmp", optimize_for="semantic")
        factory.assert_called_once_with(data_dir="/tmp", optimize_for="semantic")


class TestListStoreBackends:
    def test_returns_copy(self):
        backends = list_store_backends()
        assert isinstance(backends, dict)
        # Mutating the copy should not affect the global registry
        backends["fake"] = lambda: None
        assert "fake" not in _STORE_REGISTRY
