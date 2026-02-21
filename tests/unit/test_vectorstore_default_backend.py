"""Tests for VectorStore class-level default backend (P0-4 prerequisite for DIST-LITE-1)."""
from unittest.mock import MagicMock
import pytest

from smartmemory.stores.vector.vector_store import VectorStore


@pytest.fixture(autouse=True)
def reset_default_backend():
    """Always reset to None before and after each test."""
    VectorStore.set_default_backend(None)
    yield
    VectorStore.set_default_backend(None)


def test_set_default_backend_injects_into_new_instances():
    """After set_default_backend(mock), new VectorStore()._backend is mock."""
    mock_backend = MagicMock()
    VectorStore.set_default_backend(mock_backend)
    vs = VectorStore()
    assert vs._backend is mock_backend


def test_set_default_backend_none_restores_normal_behavior():
    """After set_default_backend(None), VectorStore() tries normal backend resolution."""
    mock_backend = MagicMock()
    VectorStore.set_default_backend(mock_backend)
    VectorStore.set_default_backend(None)
    # With None, VectorStore() will try to resolve from config (may fail without FalkorDB)
    # We just verify the flag is cleared
    import smartmemory.stores.vector.vector_store as vs_mod
    assert vs_mod._DEFAULT_BACKEND is None


def test_set_default_backend_covers_all_instances():
    """Multiple VectorStore() instances all get the injected backend."""
    mock_backend = MagicMock()
    VectorStore.set_default_backend(mock_backend)
    vs1 = VectorStore()
    vs2 = VectorStore()
    assert vs1._backend is mock_backend
    assert vs2._backend is mock_backend
