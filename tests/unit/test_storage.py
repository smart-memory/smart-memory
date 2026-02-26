"""Tests for smartmemory_pkg.storage singleton and operations."""

import pytest
from unittest.mock import MagicMock, patch


def _reset_singleton():
    """Reset the storage module singleton between tests."""
    import smartmemory_pkg.storage as storage

    storage._memory = None


@pytest.fixture(autouse=True)
def reset_storage():
    _reset_singleton()
    yield
    _reset_singleton()


def test_get_memory_singleton(tmp_path):
    """get_memory() returns the same instance on repeated calls."""
    import smartmemory_pkg.storage as storage

    mock_mem = MagicMock()
    mock_pm = MagicMock()
    mock_pm.get_patterns.return_value = {}

    with (
        patch("smartmemory_pkg.storage._resolve_data_dir", return_value=tmp_path),
        patch("smartmemory_pkg.patterns.LitePatternManager", return_value=mock_pm),
        patch("smartmemory.tools.factory.create_lite_memory", return_value=mock_mem),
    ):
        m1 = storage.get_memory()
        m2 = storage.get_memory()
    assert m1 is m2, "get_memory() must return the same singleton"


def test_get_memory_registers_atexit(tmp_path):
    """get_memory() registers _shutdown with atexit on first init."""
    import smartmemory_pkg.storage as storage

    mock_mem = MagicMock()
    mock_pm = MagicMock()
    mock_pm.get_patterns.return_value = {}

    with (
        patch("smartmemory_pkg.storage._resolve_data_dir", return_value=tmp_path),
        patch("smartmemory_pkg.patterns.LitePatternManager", return_value=mock_pm),
        patch("smartmemory.tools.factory.create_lite_memory", return_value=mock_mem),
        patch("atexit.register") as mock_register,
    ):
        storage.get_memory()
        mock_register.assert_called_once_with(storage._shutdown)


def test_shutdown_calls_save_and_close():
    """_shutdown() calls _save() and backend.close() on the memory instance."""
    import smartmemory_pkg.storage as storage

    mock_vector = MagicMock()
    mock_graph = MagicMock()
    mock_mem = MagicMock()
    mock_mem._vector_backend = mock_vector
    mock_mem._graph = mock_graph
    storage._memory = mock_mem

    storage._shutdown()

    mock_vector._save.assert_called_once()
    mock_graph.backend.close.assert_called_once()


def test_shutdown_clears_singleton():
    """_shutdown() sets _memory to None after running."""
    import smartmemory_pkg.storage as storage

    storage._memory = MagicMock()
    storage._shutdown()
    assert storage._memory is None


def test_shutdown_noop_if_memory_none():
    """_shutdown() does nothing and does not raise if _memory is None."""
    import smartmemory_pkg.storage as storage

    storage._memory = None
    storage._shutdown()  # must not raise


def test_normalize_ingest_result_str():
    """_normalize_ingest_result returns str directly."""
    from smartmemory_pkg.storage import _normalize_ingest_result

    assert _normalize_ingest_result("abc-123") == "abc-123"


def test_normalize_ingest_result_dict():
    """_normalize_ingest_result extracts item_id from dict."""
    from smartmemory_pkg.storage import _normalize_ingest_result

    assert (
        _normalize_ingest_result({"item_id": "abc-123", "queued": False}) == "abc-123"
    )


def test_ingest_acquires_lock(tmp_path):
    """ingest() acquires FileLock before calling mem.ingest()."""
    import smartmemory_pkg.storage as storage

    mock_mem = MagicMock()
    mock_mem.ingest.return_value = "item-123"
    storage._memory = mock_mem

    mock_lock_instance = MagicMock()
    mock_lock_instance.__enter__ = MagicMock(return_value=None)
    mock_lock_instance.__exit__ = MagicMock(return_value=False)

    with (
        patch("smartmemory_pkg.storage._resolve_data_dir", return_value=tmp_path),
        patch("smartmemory_pkg.storage.FileLock", return_value=mock_lock_instance),
    ):
        result = storage.ingest("test content")

    mock_lock_instance.__enter__.assert_called_once()
    mock_mem.ingest.assert_called_once_with("test content", memory_type="episodic")
    assert result == "item-123"
