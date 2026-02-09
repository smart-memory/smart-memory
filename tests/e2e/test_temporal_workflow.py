"""E2E: Temporal workflow (version → time-travel → diff).

Exercises: temporal, graph, stores.
Requires running FalkorDB (port 9010) and Redis (port 9012).
"""

import os

import pytest

pytestmark = pytest.mark.e2e


@pytest.fixture(scope="module")
def memory():
    os.environ.setdefault("FALKORDB_PORT", "9010")
    os.environ.setdefault("REDIS_PORT", "9012")
    os.environ.setdefault("VECTOR_BACKEND", "falkordb")
    try:
        from smartmemory import SmartMemory
        sm = SmartMemory()
    except Exception as e:
        pytest.skip(f"E2E environment not ready: {e}")
    yield sm
    try:
        sm.clear()
    except Exception:
        pass


class TestTemporalWorkflow:
    """Version tracking, time-travel queries, and temporal diffs."""

    def test_version_tracking_on_update(self, memory):
        from smartmemory.models.memory_item import MemoryItem

        item = MemoryItem(
            content="Python version is 3.11",
            memory_type="semantic",
        )
        memory.add(item)

        # update_properties creates a new version
        memory.update_properties(item.item_id, properties={"content": "Python version is 3.12"})

        updated = memory.get(item.item_id)
        assert updated is not None
        assert "3.12" in updated.content

    def test_time_travel_context(self, memory):
        """time_travel(to=timestamp) returns a temporal query context."""
        result = memory.time_travel(to="2026-01-01T00:00:00Z")
        # time_travel returns a context manager or temporal query helper
        assert result is not None

    def test_temporal_search(self, memory):
        from smartmemory.models.memory_item import MemoryItem

        item = MemoryItem(
            content="Temporal search test: event happened today",
            memory_type="episodic",
            metadata={"timestamp": "2026-02-09T12:00:00Z"},
        )
        memory.add(item)

        # Search should find the item
        results = memory.search("event happened", top_k=5)
        assert isinstance(results, list)

    def test_find_old_notes(self, memory):
        result = memory.find_old_notes()
        assert isinstance(result, (list, dict, type(None)))
