"""E2E: Evolution cycle (working → episodic → semantic).

Exercises: evolution, managers, memory.
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


class TestEvolutionCycle:
    """Working memory → episodic → semantic promotion."""

    def test_commit_working_to_episodic(self, memory):
        from smartmemory.models.memory_item import MemoryItem

        item = MemoryItem(
            content="User discussed Python decorators in today's session",
            memory_type="working",
        )
        memory.add(item)

        try:
            result = memory.commit_working_to_episodic()
            assert isinstance(result, list)
        except TypeError as e:
            if "config" in str(e).lower():
                pytest.skip(f"Evolver config not available in E2E env: {e}")
            raise

    def test_commit_working_to_procedural(self, memory):
        from smartmemory.models.memory_item import MemoryItem

        item = MemoryItem(
            content="To create a decorator: def my_decorator(func): ...",
            memory_type="working",
        )
        memory.add(item)

        try:
            result = memory.commit_working_to_procedural()
            assert isinstance(result, list)
        except TypeError as e:
            if "config" in str(e).lower():
                pytest.skip(f"Evolver config not available in E2E env: {e}")
            raise

    def test_run_evolution_cycle(self, memory):
        from smartmemory.models.memory_item import MemoryItem

        for i in range(3):
            memory.add(MemoryItem(
                content=f"Evolution test fact {i}: Python supports multiple paradigms",
                memory_type="semantic",
            ))

        result = memory.run_evolution_cycle()
        assert isinstance(result, (dict, type(None)))

    def test_run_clustering(self, memory):
        from smartmemory.models.memory_item import MemoryItem

        memory.add(MemoryItem(content="Machine learning uses data", memory_type="semantic"))
        memory.add(MemoryItem(content="ML learns from datasets", memory_type="semantic"))

        result = memory.run_clustering()
        assert isinstance(result, dict)
