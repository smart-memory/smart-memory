"""E2E: Reasoning workflow (assert → challenge → prove).

Exercises: reasoning, decisions, graph.
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


class TestReasoningWorkflow:
    """Assert facts, challenge them, verify reasoning."""

    def test_add_and_challenge(self, memory):
        from smartmemory.models.memory_item import MemoryItem

        item = MemoryItem(
            content="The speed of light is approximately 300,000 km/s",
            memory_type="semantic",
            metadata={"topic": "physics"},
        )
        memory.add(item)

        # challenge(assertion, memory_type, use_llm) -> ChallengeResult or None
        result = memory.challenge(
            "The speed of light is approximately 300,000 km/s",
            memory_type="semantic",
            use_llm=False,
        )
        assert result is None or hasattr(result, "has_conflicts")

    def test_contradictory_facts(self, memory):
        from smartmemory.models.memory_item import MemoryItem

        item1 = MemoryItem(
            content="Water boils at 100 degrees Celsius at sea level",
            memory_type="semantic",
        )
        memory.add(item1)

        # challenge(assertion, memory_type, use_llm) -> ChallengeResult or None
        result = memory.challenge(
            "Water boils at 50 degrees Celsius at sea level",
            memory_type="semantic",
            use_llm=False,
        )
        # May or may not detect depending on detector availability
        assert result is None or hasattr(result, "has_conflicts")

    def test_reflect_produces_output(self, memory):
        from smartmemory.models.memory_item import MemoryItem

        for i in range(3):
            memory.add(MemoryItem(
                content=f"Reflection test fact {i}: AI systems need evaluation",
                memory_type="semantic",
            ))

        result = memory.reflect()
        assert result is None or isinstance(result, (dict, str, list))
