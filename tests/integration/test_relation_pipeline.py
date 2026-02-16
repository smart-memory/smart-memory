"""
Integration test: verify LLM-extracted semantic relations survive the full
ingest pipeline and appear as edges in the knowledge graph.

This test validates the fix for CORE-REL-1: 3 cascading bugs that caused 100%
of LLM-extracted semantic relations to be silently dropped during ingestion.

Run:
    PYTHONPATH=. pytest tests/integration/test_relation_pipeline.py -v -s
"""
import pytest
from unittest.mock import patch

from smartmemory import SmartMemory
from smartmemory.models.memory_item import MemoryItem


# Deterministic mock extraction result — avoids LLM API calls.
# These item_ids mimic the SHA256 hashes that LLMSingleExtractor produces.
MOCK_ENTITIES = [
    MemoryItem(
        content="Django",
        item_id="a1b2c3d4e5f67890",
        memory_type="concept",
        metadata={"name": "Django", "entity_type": "technology", "confidence": 0.95},
    ),
    MemoryItem(
        content="Python",
        item_id="b2c3d4e5f6789012",
        memory_type="concept",
        metadata={"name": "Python", "entity_type": "technology", "confidence": 0.98},
    ),
    MemoryItem(
        content="Adrian Holovaty",
        item_id="c3d4e5f678901234",
        memory_type="concept",
        metadata={"name": "Adrian Holovaty", "entity_type": "person", "confidence": 0.97},
    ),
]

MOCK_RELATIONS = [
    {"source_id": "a1b2c3d4e5f67890", "target_id": "b2c3d4e5f6789012", "relation_type": "framework_for"},
    {"source_id": "c3d4e5f678901234", "target_id": "a1b2c3d4e5f67890", "relation_type": "created"},
]

MOCK_EXTRACTION_RESULT = {
    "entities": MOCK_ENTITIES,
    "relations": MOCK_RELATIONS,
}


@pytest.fixture
def memory():
    """Create a SmartMemory instance with real graph infrastructure."""
    sm = SmartMemory()
    yield sm


def _patch_llm_extractor(memory):
    """Patch the LLM extractor in the pipeline v2 to return mock results.

    The pipeline v2 uses LLMExtractStage which creates an LLMSingleExtractor.
    We patch the extract() on the underlying LLMSingleExtractor to avoid real
    LLM API calls while letting the stage logic (entity/relation processing)
    run as normal.
    """
    return patch(
        "smartmemory.plugins.extractors.llm_single.LLMSingleExtractor.extract",
        return_value=MOCK_EXTRACTION_RESULT,
    )


class TestRelationPipeline:
    """Verify semantic relation edges survive the full ingest pipeline."""

    def test_ingest_creates_semantic_edges(self, memory):
        """Core test: ingest text with relations, verify semantic edges exist in graph.

        This is the primary regression test for CORE-REL-1. Before the fix,
        ALL semantic edges (framework_for, created, etc.) were silently dropped
        and only structural edges (MENTIONED_IN, CONTAINS_ENTITY) survived.
        """
        with _patch_llm_extractor(memory):
            item_id = memory.ingest("Django is a Python web framework created by Adrian Holovaty.")

        assert item_id is not None

        # Look up entity nodes that were created
        # The StoreStage creates entities via add_dual_node, so they should exist
        # Find Django entity by searching graph
        try:
            neighbors = memory._graph.get_neighbors(item_id, max_depth=1)
        except Exception:
            neighbors = []

        # Get all edges for the memory node
        edges = memory._graph.edges.get_edges_for_node(item_id)

        # There should be structural edges (CONTAINS_ENTITY, MENTIONED_IN)
        structural_types = {"MENTIONED_IN", "CONTAINS_ENTITY"}
        structural_edges = [e for e in edges if e.get("type") in structural_types]
        assert len(structural_edges) > 0, f"Missing structural edges. All edges: {edges}"

        # Now check entity-to-entity semantic edges
        # Find an entity node ID (Django)
        entity_ids = []
        for edge in structural_edges:
            if edge.get("type") == "CONTAINS_ENTITY":
                target = edge.get("target")
                if target and target != item_id:
                    entity_ids.append(target)

        assert len(entity_ids) > 0, f"No entity nodes found via CONTAINS_ENTITY edges. Edges: {edges}"

        # Check edges between entity nodes for semantic relations
        all_semantic_edges = []
        for eid in entity_ids:
            entity_edges = memory._graph.edges.get_edges_for_node(eid)
            for e in entity_edges:
                if e.get("type") not in structural_types:
                    all_semantic_edges.append(e)

        assert len(all_semantic_edges) > 0, (
            f"No semantic edges found between entities. "
            f"Entity IDs: {entity_ids}. "
            f"This means relations are still being dropped (CORE-REL-1 regression)."
        )

        # Verify meaningful relation types (not all collapsed to RELATED)
        edge_types = {e.get("type") for e in all_semantic_edges}
        assert "framework_for" in edge_types or "FRAMEWORK_FOR" in edge_types, (
            f"Expected 'framework_for' edge, found: {edge_types}"
        )

    def test_edges_created_for_all_relations(self, memory):
        """Verify both relations from the extraction produce edges."""
        with _patch_llm_extractor(memory):
            item_id = memory.ingest("Django is a Python framework by Adrian Holovaty.")

        # Collect all edges across all entity nodes
        memory_edges = memory._graph.edges.get_edges_for_node(item_id)
        entity_node_ids = [
            e.get("target") for e in memory_edges
            if e.get("type") == "CONTAINS_ENTITY" and e.get("target") != item_id
        ]

        semantic_edge_types = set()
        for eid in entity_node_ids:
            for e in memory._graph.edges.get_edges_for_node(eid):
                if e.get("type") not in ("MENTIONED_IN", "CONTAINS_ENTITY"):
                    semantic_edge_types.add(e.get("type"))

        # We expect at least framework_for and created
        assert len(semantic_edge_types) >= 2, (
            f"Expected >= 2 distinct semantic edge types, got {len(semantic_edge_types)}: {semantic_edge_types}"
        )

    def test_no_crash_with_empty_entities(self, memory):
        """Relations with no entities should be silently dropped, not crash."""
        mock_result = {
            "entities": [],
            "relations": [{"source_id": "a", "target_id": "b", "relation_type": "RELATED"}],
        }
        with patch(
            "smartmemory.plugins.extractors.llm_single.LLMSingleExtractor.extract",
            return_value=mock_result,
        ):
            item_id = memory.ingest("Some text with no extractable entities.")

        assert item_id is not None
        edges = memory._graph.edges.get_edges_for_node(item_id)
        structural_types = {"MENTIONED_IN", "CONTAINS_ENTITY", "HAS_VERSION", "NEXT_VERSION"}
        semantic_edges = [e for e in edges if e.get("type") not in structural_types]
        assert len(semantic_edges) == 0, (
            f"Expected no semantic edges when entities list is empty, got: {semantic_edges}"
        )

    def test_unresolvable_relation_target_dropped(self, memory):
        """Relation referencing a non-existent entity ID should be dropped."""
        mock_result = {
            "entities": MOCK_ENTITIES[:1],  # Only Django
            "relations": [
                {
                    "source_id": "a1b2c3d4e5f67890",  # Django — exists
                    "target_id": "nonexistent_id_999",  # Does not exist
                    "relation_type": "RELATED",
                }
            ],
        }
        with patch(
            "smartmemory.plugins.extractors.llm_single.LLMSingleExtractor.extract",
            return_value=mock_result,
        ):
            item_id = memory.ingest("Django is a framework.")

        assert item_id is not None
        edges = memory._graph.edges.get_edges_for_node(item_id)
        entity_nodes = [
            e.get("target") for e in edges
            if e.get("type") == "CONTAINS_ENTITY" and e.get("target") != item_id
        ]
        assert len(entity_nodes) == 1, "Should have 1 entity node"
        # No semantic edges — the relation target doesn't exist
        for eid in entity_nodes:
            entity_edges = memory._graph.edges.get_edges_for_node(eid)
            semantic = [e for e in entity_edges if e.get("type") not in ("MENTIONED_IN", "CONTAINS_ENTITY")]
            assert len(semantic) == 0, f"Expected no semantic edges, got: {semantic}"
