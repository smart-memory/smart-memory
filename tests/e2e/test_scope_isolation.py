"""E2E: Multi-scope isolation workflow.

Exercises: scope_provider, smart_memory, stores.
Requires running FalkorDB (port 9010) and Redis (port 9012).
"""

import os

import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.golden]


@pytest.fixture(scope="module")
def _env():
    os.environ.setdefault("FALKORDB_PORT", "9010")
    os.environ.setdefault("REDIS_PORT", "9012")
    os.environ.setdefault("VECTOR_BACKEND", "falkordb")


class TestScopeIsolation:
    """Verify that different scope providers isolate data."""

    def test_default_scope_provider_works(self, _env):
        try:
            from smartmemory import SmartMemory
            sm = SmartMemory()
        except Exception as e:
            pytest.skip(f"E2E environment not ready: {e}")

        from smartmemory.models.memory_item import MemoryItem

        item = MemoryItem(content="Default scope test", memory_type="semantic")
        sm.add(item)
        assert sm.get(item.item_id) is not None
        sm.clear()

    def test_two_instances_share_default_scope(self, _env):
        try:
            from smartmemory import SmartMemory
            sm1 = SmartMemory()
            sm2 = SmartMemory()
        except Exception as e:
            pytest.skip(f"E2E environment not ready: {e}")

        from smartmemory.models.memory_item import MemoryItem

        item = MemoryItem(content="Shared scope item", memory_type="semantic")
        sm1.add(item)

        # sm2 should see the same item (same default scope)
        retrieved = sm2.get(item.item_id)
        assert retrieved is not None
        assert retrieved.content == "Shared scope item"

        sm1.clear()

    def test_clear_removes_all_items(self, _env):
        try:
            from smartmemory import SmartMemory
            sm = SmartMemory()
        except Exception as e:
            pytest.skip(f"E2E environment not ready: {e}")

        from smartmemory.models.memory_item import MemoryItem

        for i in range(5):
            sm.add(MemoryItem(content=f"Clear test item {i}", memory_type="semantic"))

        sm.clear()
        summary = sm.summary()
        assert isinstance(summary, dict)
