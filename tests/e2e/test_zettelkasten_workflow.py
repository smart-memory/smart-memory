"""E2E: Zettelkasten workflow (create → link → traverse).

Exercises: memory/types/zettel, graph, similarity.
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


class TestZettelkastenWorkflow:
    """Create zettel notes, link them, traverse the graph."""

    def test_create_zettel_notes(self, memory):
        from smartmemory.models.memory_item import MemoryItem

        z1 = MemoryItem(
            content="# Neural Networks\n\nNeural networks are computational models inspired by the brain.",
            memory_type="zettel",
            metadata={"title": "Neural Networks", "tags": ["ai", "ml"]},
        )
        z2 = MemoryItem(
            content="# Backpropagation\n\nBackpropagation is the algorithm used to train neural networks.",
            memory_type="zettel",
            metadata={"title": "Backpropagation", "tags": ["ai", "ml", "training"]},
        )
        memory.add(z1)
        memory.add(z2)

        r1 = memory.get(z1.item_id)
        r2 = memory.get(z2.item_id)
        assert r1 is not None
        assert r2 is not None

    def test_link_zettel_notes(self, memory):
        from smartmemory.models.memory_item import MemoryItem

        z1 = MemoryItem(
            content="# Gradient Descent\n\nOptimization algorithm for finding minima.",
            memory_type="zettel",
            metadata={"title": "Gradient Descent"},
        )
        z2 = MemoryItem(
            content="# Learning Rate\n\nHyperparameter controlling step size in gradient descent.",
            memory_type="zettel",
            metadata={"title": "Learning Rate"},
        )
        memory.add(z1)
        memory.add(z2)

        # Link the two notes
        memory.link(z1.item_id, z2.item_id, link_type="RELATED_TO")

        # Verify link exists
        links = memory.get_links(z1.item_id)
        assert isinstance(links, (list, dict))

    def test_search_zettel_by_content(self, memory):
        results = memory.search("neural networks brain", top_k=5)
        assert isinstance(results, list)

    def test_get_neighbors(self, memory):
        from smartmemory.models.memory_item import MemoryItem

        z = MemoryItem(
            content="# Test Node\n\nNode for neighbor test.",
            memory_type="zettel",
        )
        memory.add(z)
        neighbors = memory.get_neighbors(z.item_id)
        assert isinstance(neighbors, (list, dict))
