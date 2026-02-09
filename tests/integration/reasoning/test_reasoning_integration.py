"""Integration tests for reasoning module against real SmartMemory backends.

Requires running FalkorDB (port 9010) and Redis (port 9012).
Tests are skipped gracefully if backends are unavailable.
"""

import os
import time

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def memory():
    """Create a real SmartMemory instance for reasoning integration tests."""
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


class TestProofTreeIntegration:
    """Test ProofTreeBuilder against a real graph."""

    def test_build_proof_for_existing_node(self, memory):
        from smartmemory.models.memory_item import MemoryItem
        from smartmemory.reasoning.proof_tree import ProofTreeBuilder

        item = MemoryItem(content="Python is a programming language", memory_type="semantic")
        item_id = memory.add(item)
        assert item_id is not None

        builder = ProofTreeBuilder(memory._graph)
        tree = builder.build_proof(item_id, max_depth=2)
        assert tree is not None
        assert tree.root is not None
        assert tree.root.node_id == item_id

    def test_build_proof_nonexistent_returns_none(self, memory):
        from smartmemory.reasoning.proof_tree import ProofTreeBuilder

        builder = ProofTreeBuilder(memory._graph)
        tree = builder.build_proof("nonexistent_id_xyz")
        assert tree is None

    def test_render_text(self, memory):
        from smartmemory.models.memory_item import MemoryItem
        from smartmemory.reasoning.proof_tree import ProofTreeBuilder

        item = MemoryItem(content="Proof render test", memory_type="semantic")
        item_id = memory.add(item)

        builder = ProofTreeBuilder(memory._graph)
        tree = builder.build_proof(item_id)
        text = tree.render_text()
        assert "Proof render test" in text


class TestChallengerIntegration:
    """Test AssertionChallenger against real search + graph."""

    def test_challenge_novel_assertion(self, memory):
        from smartmemory.reasoning.challenger import AssertionChallenger

        challenger = AssertionChallenger(
            memory,
            use_llm=False,
            use_embedding=False,
        )
        result = challenger.challenge("Quantum computing uses qubits for computation")
        assert result is not None
        assert hasattr(result, "has_conflicts")
        assert hasattr(result, "overall_confidence")

    def test_challenge_with_related_facts(self, memory):
        from smartmemory.models.memory_item import MemoryItem
        from smartmemory.reasoning.challenger import AssertionChallenger

        memory.add(MemoryItem(
            content="The capital of France is Paris",
            memory_type="semantic",
        ))
        time.sleep(0.3)

        challenger = AssertionChallenger(
            memory,
            use_llm=False,
            use_embedding=False,
        )
        result = challenger.challenge("The capital of France is Berlin")
        assert result is not None
        # May or may not detect conflict depending on search results
        assert isinstance(result.overall_confidence, float)


class TestConfidenceManagerIntegration:
    """Test ConfidenceManager with real storage."""

    def test_apply_decay_and_get_history(self, memory):
        from smartmemory.models.memory_item import MemoryItem
        from smartmemory.reasoning.confidence import ConfidenceManager

        item = MemoryItem(
            content="Confidence decay integration test",
            memory_type="semantic",
            metadata={"confidence": 0.9},
        )
        item_id = memory.add(item)
        assert item_id is not None

        cm = ConfidenceManager(memory)
        success = cm.apply_decay(item_id, decay_factor=0.2, reason="test challenge")
        assert success is True

        history = cm.get_history(item_id)
        # Graph backend may serialize the list as a JSON string
        if isinstance(history, str):
            import json
            history = json.loads(history)
        assert len(history) >= 1
        entry = history[-1]
        if isinstance(entry, str):
            import json
            entry = json.loads(entry)
        assert entry["old_confidence"] == 0.9
        assert entry["new_confidence"] == pytest.approx(0.7)

    def test_apply_decay_nonexistent_item(self, memory):
        from smartmemory.reasoning.confidence import ConfidenceManager

        cm = ConfidenceManager(memory)
        result = cm.apply_decay("nonexistent_xyz", decay_factor=0.1)
        assert result is False


class TestQueryRouterIntegration:
    """Test QueryRouter with real backends."""

    def test_route_query(self, memory):
        from smartmemory.models.memory_item import MemoryItem
        from smartmemory.reasoning.query_router import QueryRouter

        memory.add(MemoryItem(content="Machine learning uses neural networks", memory_type="semantic"))
        time.sleep(0.3)

        router = QueryRouter(memory, graph=memory._graph)
        result = router.route("neural networks", top_k=5)
        # route() returns a dict with 'results' key
        assert isinstance(result, dict)
        assert "results" in result
        assert isinstance(result["results"], list)
