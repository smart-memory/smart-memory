"""Integration tests for InferenceEngine against real FalkorDB graph.

Requires running FalkorDB (port 9010) and Redis (port 9012).
Tests are skipped gracefully if backends are unavailable.
"""

import os

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def memory():
    """Create a real SmartMemory instance for inference integration tests."""
    os.environ.setdefault("FALKORDB_PORT", "9010")
    os.environ.setdefault("REDIS_PORT", "9012")
    os.environ.setdefault("VECTOR_BACKEND", "falkordb")

    try:
        from smartmemory.smart_memory import SmartMemory
        sm = SmartMemory()
    except Exception as e:
        pytest.skip(f"Integration environment not ready: {e}")

    yield sm

    try:
        sm.clear()
    except Exception:
        pass
    for key in ("FALKORDB_PORT", "REDIS_PORT", "VECTOR_BACKEND"):
        os.environ.pop(key, None)


class TestInferenceEngineIntegration:
    """Test InferenceEngine running real Cypher queries against FalkorDB."""

    def test_run_with_no_data(self, memory):
        from smartmemory.inference.engine import InferenceEngine
        from smartmemory.inference.rules import get_default_rules

        engine = InferenceEngine(memory, graph=memory._graph, rules=get_default_rules())
        result = engine.run()
        assert result.edges_created == 0
        assert result.errors == []

    def test_run_dry_run(self, memory):
        from smartmemory.inference.engine import InferenceEngine
        from smartmemory.inference.rules import get_default_rules

        engine = InferenceEngine(memory, graph=memory._graph, rules=get_default_rules())
        result = engine.run(dry_run=True)
        assert result.edges_created == 0
        assert result.errors == []

    def test_run_with_seeded_data(self, memory):
        from smartmemory.models.memory_item import MemoryItem
        from smartmemory.inference.engine import InferenceEngine
        from smartmemory.inference.rules import get_default_rules

        # Seed some data so the graph has nodes
        memory.add(MemoryItem(content="Python is a programming language", memory_type="semantic"))
        memory.add(MemoryItem(content="Python is used for machine learning", memory_type="semantic"))

        engine = InferenceEngine(memory, graph=memory._graph, rules=get_default_rules())
        result = engine.run()
        # May or may not create edges depending on graph structure
        assert isinstance(result.edges_created, int)
        assert result.errors == []

    def test_custom_rule_execution(self, memory):
        from smartmemory.inference.engine import InferenceEngine
        from smartmemory.inference.rules import InferenceRule

        custom_rule = InferenceRule(
            name="test_custom",
            description="Test rule that matches nothing",
            pattern_cypher="MATCH (a:Memory)-[:NONEXISTENT]->(b:Memory) RETURN a.item_id, b.item_id",
            edge_type="TEST_EDGE",
            confidence=0.5,
        )
        engine = InferenceEngine(memory, graph=memory._graph, rules=[custom_rule])
        result = engine.run()
        assert result.edges_created == 0
        assert result.errors == []
