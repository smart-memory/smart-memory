"""Tests for smartmemory_app.storage singleton and operations."""

import pytest
from unittest.mock import MagicMock, patch


def _reset_singleton():
    """Reset the storage module singletons between tests."""
    import smartmemory_app.storage as storage

    storage._memory = None
    storage._remote_memory = None


@pytest.fixture(autouse=True)
def reset_storage():
    _reset_singleton()
    yield
    _reset_singleton()


def test_get_memory_singleton(tmp_path):
    """get_memory() returns the same instance on repeated calls."""
    import smartmemory_app.storage as storage

    mock_mem = MagicMock()
    mock_pm = MagicMock()
    mock_pm.get_patterns.return_value = {}
    mock_pm.add_patterns.return_value = 0
    mock_store = MagicMock()

    with (
        patch("smartmemory_app.storage._resolve_data_dir", return_value=tmp_path),
        patch("smartmemory_app.patterns.JSONLPatternStore", return_value=mock_store),
        patch("smartmemory.ontology.pattern_manager.PatternManager", return_value=mock_pm),
        patch("smartmemory.tools.factory.create_lite_memory", return_value=mock_mem),
    ):
        m1 = storage.get_memory()
        m2 = storage.get_memory()
    assert m1 is m2, "get_memory() must return the same singleton"


def test_get_memory_registers_atexit(tmp_path):
    """get_memory() registers _shutdown with atexit on first init."""
    import smartmemory_app.storage as storage

    mock_mem = MagicMock()
    mock_pm = MagicMock()
    mock_pm.get_patterns.return_value = {}
    mock_pm.add_patterns.return_value = 0
    mock_store = MagicMock()

    with (
        patch("smartmemory_app.storage._resolve_data_dir", return_value=tmp_path),
        patch("smartmemory_app.patterns.JSONLPatternStore", return_value=mock_store),
        patch("smartmemory.ontology.pattern_manager.PatternManager", return_value=mock_pm),
        patch("smartmemory.tools.factory.create_lite_memory", return_value=mock_mem),
        patch("atexit.register") as mock_register,
    ):
        storage.get_memory()
        mock_register.assert_called_once_with(storage._shutdown)


def test_shutdown_calls_save_and_close():
    """_shutdown() calls _save() and SmartMemory.close() on the memory instance.

    _shutdown() prefers SmartMemory.close() (which orchestrates evolution worker,
    ontology store, and graph backend shutdown). Only falls back to
    _graph.backend.close() when close() is unavailable.
    """
    import smartmemory_app.storage as storage

    mock_vector = MagicMock()
    mock_mem = MagicMock()
    mock_mem._vector_backend = mock_vector
    storage._memory = mock_mem

    storage._shutdown()

    mock_vector._save.assert_called_once()
    mock_mem.close.assert_called_once()


def test_shutdown_clears_singleton():
    """_shutdown() sets _memory to None after running."""
    import smartmemory_app.storage as storage

    storage._memory = MagicMock()
    storage._shutdown()
    assert storage._memory is None


def test_shutdown_noop_if_memory_none():
    """_shutdown() does nothing and does not raise if _memory is None."""
    import smartmemory_app.storage as storage

    storage._memory = None
    storage._shutdown()  # must not raise


def test_normalize_ingest_result_str():
    """_normalize_ingest_result returns str directly."""
    from smartmemory_app.storage import _normalize_ingest_result

    assert _normalize_ingest_result("abc-123") == "abc-123"


def test_normalize_ingest_result_dict():
    """_normalize_ingest_result extracts item_id from dict."""
    from smartmemory_app.storage import _normalize_ingest_result

    assert (
        _normalize_ingest_result({"item_id": "abc-123", "queued": False}) == "abc-123"
    )


def test_get_memory_raises_unconfigured_error():
    """get_memory() raises UnconfiguredError when not configured and migration fails."""
    import smartmemory_app.storage as storage
    from smartmemory_app.config import UnconfiguredError

    with (
        patch("smartmemory_app.storage.is_configured", return_value=False),
        patch("smartmemory_app.storage._detect_and_migrate", return_value=False),
    ):
        with pytest.raises(UnconfiguredError, match="smartmemory setup"):
            storage.get_memory()


def test_ingest_remote_branch_skips_lock(tmp_path):
    """ingest() in remote mode delegates to RemoteMemory without acquiring a lock."""
    import smartmemory_app.storage as storage
    from smartmemory_app.remote_backend import RemoteMemory

    mock_mem = MagicMock(spec=RemoteMemory)
    mock_mem.ingest.return_value = "remote-item-id"

    with (
        patch("smartmemory_app.storage.get_memory", return_value=mock_mem),
        patch("smartmemory_app.storage._get_lock_file") as mock_lock,
    ):
        result = storage.ingest("remote content")

    mock_mem.ingest.assert_called_once_with("remote content", "episodic")
    mock_lock.assert_not_called()
    assert result == "remote-item-id"


def test_get_remote_memory_singleton(tmp_path):
    """_get_remote_memory() returns the same RemoteMemory instance on repeated calls."""
    import smartmemory_app.storage as storage
    from smartmemory_app.config import SmartMemoryConfig

    cfg = SmartMemoryConfig(mode="remote", api_url="https://api.example.com", team_id="t1")
    # RemoteMemory is lazily imported inside _get_remote_memory — patch at source module
    with patch("smartmemory_app.remote_backend.RemoteMemory") as MockRemote:
        m1 = storage._get_remote_memory(cfg)
        m2 = storage._get_remote_memory(cfg)
    assert m1 is m2
    MockRemote.assert_called_once()  # constructor called only once


def test_ingest_acquires_lock(tmp_path):
    """ingest() acquires FileLock before calling mem.ingest() in local mode."""
    import smartmemory_app.storage as storage

    mock_mem = MagicMock()
    mock_mem.ingest.return_value = "item-123"

    mock_lock_instance = MagicMock()
    mock_lock_instance.__enter__ = MagicMock(return_value=None)
    mock_lock_instance.__exit__ = MagicMock(return_value=False)

    with (
        # get_memory() now checks is_configured() — patch it to return mock directly
        patch("smartmemory_app.storage.get_memory", return_value=mock_mem),
        patch("smartmemory_app.storage._resolve_data_dir", return_value=tmp_path),
        # _get_lock_file() does the lazy filelock import — patch at that boundary
        patch("smartmemory_app.storage._get_lock_file", return_value=mock_lock_instance),
    ):
        result = storage.ingest("test content")

    mock_lock_instance.__enter__.assert_called_once()
    mock_mem.ingest.assert_called_once_with("test content", context={"memory_type": "episodic"}, sync=True)
    assert result == "item-123"
