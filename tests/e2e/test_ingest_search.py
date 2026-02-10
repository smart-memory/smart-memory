"""E2E: Ingest → Search → Retrieve workflow.

Exercises: smart_memory, memory, pipeline, stores, graph.
Requires running FalkorDB (port 9010) and Redis (port 9012).
"""

import os

import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.golden]


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


class TestIngestSearchRetrieve:
    """Full lifecycle: add items, search, get by ID, delete."""

    def test_add_and_get(self, memory):
        from smartmemory.models.memory_item import MemoryItem

        item = MemoryItem(
            content="Python was created by Guido van Rossum in 1991",
            memory_type="semantic",
            metadata={"topic": "programming"},
        )
        memory.add(item)
        retrieved = memory.get(item.item_id)
        assert retrieved is not None
        assert "Guido" in retrieved.content

    def test_search_returns_relevant(self, memory):
        from smartmemory.models.memory_item import MemoryItem

        memory.add(MemoryItem(content="Rust is a systems programming language", memory_type="semantic"))
        memory.add(MemoryItem(content="Go was designed at Google", memory_type="semantic"))

        results = memory.search("systems programming", top_k=5)
        assert len(results) >= 1
        contents = " ".join(r.content for r in results)
        assert "Rust" in contents or "systems" in contents

    def test_remove_succeeds(self, memory):
        from smartmemory.models.memory_item import MemoryItem

        item = MemoryItem(content="Temporary item for deletion test", memory_type="working")
        memory.add(item)
        assert memory.get(item.item_id) is not None

        result = memory.remove(item.item_id)
        assert result is True or result is None  # remove returns bool or None

    def test_update_properties_modifies_content(self, memory):
        from smartmemory.models.memory_item import MemoryItem

        item = MemoryItem(content="Original content", memory_type="semantic")
        memory.add(item)

        memory.update_properties(item.item_id, properties={"content": "Updated content"})
        updated = memory.get(item.item_id)
        assert updated is not None
        assert "Updated" in updated.content

    def test_summary_returns_stats(self, memory):
        result = memory.summary()
        assert isinstance(result, dict)

    def test_multiple_memory_types(self, memory):
        from smartmemory.models.memory_item import MemoryItem

        types = ["semantic", "episodic", "procedural", "working"]
        for mt in types:
            memory.add(MemoryItem(content=f"Test item of type {mt}", memory_type=mt))

        summary = memory.summary()
        assert isinstance(summary, dict)
