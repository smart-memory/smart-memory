"""Unit tests for stores.mixins — graph operation mixins."""

from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.unit

from smartmemory.models.memory_item import MemoryItem
from smartmemory.stores.mixins import (
    GraphErrorHandlingMixin,
    GraphLoggingMixin,
    GraphValidationMixin,
    GraphPerformanceMixin,
    GraphCachingMixin,
)


# ---------------------------------------------------------------------------
# GraphErrorHandlingMixin
# ---------------------------------------------------------------------------
class TestGraphErrorHandlingMixin:
    def _make(self):
        obj = GraphErrorHandlingMixin()
        obj._last_error = None
        return obj

    def test_handle_operation_error_logs(self):
        obj = self._make()
        obj._handle_operation_error("add", "key1", ValueError("bad"))
        assert obj._last_error is not None
        assert obj._last_error["operation"] == "add"
        assert obj._last_error["key"] == "key1"

    def test_handle_error_returns_none(self):
        obj = self._make()
        result = obj.handle_error("something broke")
        assert result is None

    def test_validate_key_valid(self):
        obj = self._make()
        assert obj._validate_key("abc", "get") is True

    def test_validate_key_empty(self):
        obj = self._make()
        assert obj._validate_key("", "get") is False

    def test_validate_key_none(self):
        obj = self._make()
        assert obj._validate_key(None, "get") is False

    def test_validate_item_valid(self):
        obj = self._make()
        item = MemoryItem(content="test", memory_type="semantic")
        assert obj._validate_item(item, "add") is True

    def test_validate_item_none(self):
        obj = self._make()
        assert obj._validate_item(None, "add") is False

    def test_validate_item_no_id(self):
        obj = self._make()
        item = MagicMock(spec=[])
        assert obj._validate_item(item, "add") is False


# ---------------------------------------------------------------------------
# GraphLoggingMixin
# ---------------------------------------------------------------------------
class TestGraphLoggingMixin:
    def test_log_operation(self):
        obj = GraphLoggingMixin()
        # Should not raise
        obj.log_operation("add", "Added item", key="k1")

    def test_log_operation_internal(self):
        obj = GraphLoggingMixin()
        obj._log_operation("get", "key1", success=True)
        obj._log_operation("get", "key2", success=False)

    def test_log_stats(self):
        obj = GraphLoggingMixin()
        obj._log_stats({"items": 10, "edges": 5})


# ---------------------------------------------------------------------------
# GraphValidationMixin
# ---------------------------------------------------------------------------
class TestGraphValidationMixin:
    def test_validate_memory_item(self):
        obj = GraphValidationMixin()
        item = MemoryItem(content="test", memory_type="semantic")
        assert obj.validate_item(item) is True

    def test_validate_dict_with_id(self):
        obj = GraphValidationMixin()
        assert obj.validate_item({"item_id": "abc"}) is True

    def test_validate_dict_without_id(self):
        obj = GraphValidationMixin()
        assert obj.validate_item({}) is False

    def test_validate_none(self):
        obj = GraphValidationMixin()
        assert obj.validate_item(None) is False

    def test_validate_metadata_valid(self):
        obj = GraphValidationMixin()
        assert obj._validate_metadata({"key": "val"}) is True

    def test_validate_metadata_not_dict(self):
        obj = GraphValidationMixin()
        assert obj._validate_metadata("not a dict") is False

    def test_validate_metadata_reserved_key(self):
        obj = GraphValidationMixin()
        assert obj._validate_metadata({"_internal": "bad"}) is False

    def test_validate_metadata_underscore_non_reserved_ok(self):
        obj = GraphValidationMixin()
        assert obj._validate_metadata({"_custom": "ok"}) is True


# ---------------------------------------------------------------------------
# GraphPerformanceMixin
# ---------------------------------------------------------------------------
class TestGraphPerformanceMixin:
    def _make(self):
        # Need to call __init__ properly
        obj = object.__new__(GraphPerformanceMixin)
        GraphPerformanceMixin.__init__(obj)
        return obj

    def test_initial_stats(self):
        obj = self._make()
        stats = obj.get_performance_stats()
        assert stats["total_operations"] == 0
        assert stats["add_count"] == 0

    def test_track_operation(self):
        obj = self._make()
        obj._track_operation("add")
        obj._track_operation("add")
        obj._track_operation("get")
        stats = obj.get_performance_stats()
        assert stats["add_count"] == 2
        assert stats["get_count"] == 1
        assert stats["total_operations"] == 3

    def test_reset_stats(self):
        obj = self._make()
        obj._track_operation("add")
        obj.reset_performance_stats()
        stats = obj.get_performance_stats()
        assert stats["total_operations"] == 0

    def test_performance_context(self):
        obj = self._make()
        with obj.performance_context("search"):
            pass
        assert obj.get_performance_stats()["search_count"] == 1


# ---------------------------------------------------------------------------
# GraphCachingMixin
# ---------------------------------------------------------------------------
class TestGraphCachingMixin:
    def _make(self, enable_cache=True, cache_max_size=1000):
        obj = GraphCachingMixin.__new__(GraphCachingMixin)
        obj._cache = {}
        obj._cache_enabled = enable_cache
        obj._cache_max_size = cache_max_size
        return obj

    def test_cache_put_and_get(self):
        obj = self._make()
        item = MemoryItem(content="cached", memory_type="semantic")
        obj._cache_put("k1", item)
        assert obj._cache_get("k1") is item

    def test_cache_get_miss(self):
        obj = self._make()
        assert obj._cache_get("missing") is None

    def test_cache_remove(self):
        obj = self._make()
        item = MemoryItem(content="temp", memory_type="semantic")
        obj._cache_put("k1", item)
        obj._cache_remove("k1")
        assert obj._cache_get("k1") is None

    def test_cache_clear(self):
        obj = self._make()
        obj._cache_put("a", "x")
        obj._cache_put("b", "y")
        obj._cache_clear()
        assert obj._cache_get("a") is None
        assert obj._cache_get("b") is None

    def test_cache_disabled(self):
        obj = self._make(enable_cache=False)
        obj._cache_put("k1", "val")
        assert obj._cache_get("k1") is None

    def test_cache_eviction(self):
        obj = self._make(cache_max_size=2)
        obj._cache_put("a", 1)
        obj._cache_put("b", 2)
        obj._cache_put("c", 3)  # should evict "a"
        assert obj._cache_get("a") is None
        assert obj._cache_get("b") == 2
        assert obj._cache_get("c") == 3
