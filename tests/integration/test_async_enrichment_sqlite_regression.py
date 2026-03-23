"""Regression coverage for lite-mode async enrichment node preservation."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from smartmemory.models.memory_item import MemoryItem
from smartmemory.background.extraction_worker import process_extract_job


@pytest.fixture()
def lite_memory(tmp_path):
    from smartmemory.pipeline.config import PipelineConfig
    from smartmemory.tools.factory import create_lite_memory

    mem = create_lite_memory(data_dir=str(tmp_path), pipeline_profile=PipelineConfig.lite(llm_enabled=False))
    yield mem
    try:
        mem._graph.backend.close()
    except Exception:
        pass


def test_process_extract_job_does_not_overwrite_tier1_node_on_sqlite(lite_memory):
    result = lite_memory.ingest("Alice leads Project Atlas.", sync=False)
    item_id = result["item_id"]
    backend = lite_memory._graph.backend
    original = backend.get_node(item_id)
    assert original is not None

    colliding_entity = MemoryItem(
        content="Injected collision",
        item_id=item_id,
        memory_type="concept",
        metadata={"name": "Injected collision", "entity_type": "concept", "confidence": 0.95},
    )

    with patch(
        "smartmemory.plugins.extractors.llm_single.LLMSingleExtractor.extract",
        return_value={"entities": [colliding_entity], "relations": []},
    ):
        enrich_result = process_extract_job(
            lite_memory,
            {"item_id": item_id, "workspace_id": "", "entity_ids": {}, "enable_ontology": False},
        )

    assert enrich_result["status"] == "ok"
    assert enrich_result["new_entities"] == 1

    preserved = backend.get_node(item_id)
    assert preserved is not None
    assert preserved.get("content") == original.get("content")
    assert preserved.get("memory_type") == original.get("memory_type")
    assert preserved.get("node_category") == original.get("node_category")


def test_reingest_same_content_preserves_both_memory_nodes(lite_memory):
    """Re-ingesting the same content must preserve both memory nodes.

    When the same content is ingested twice, Tier 2 enrichment runs on both.
    The LLM extracts the same entities for both. The critical assertion:
    neither memory node is overwritten by entity writes.

    When Tier 1 entity_ids include the LLM entities (simulating learned
    EntityRuler patterns), the second enrichment produces 0 net-new entities.
    """
    content = "Alice leads Project Atlas at Acme Corp."

    alice_entity = MemoryItem(
        content="Alice",
        item_id="sha_alice_00000001",
        memory_type="concept",
        metadata={"name": "Alice", "entity_type": "person", "confidence": 0.95},
    )
    atlas_entity = MemoryItem(
        content="Project Atlas",
        item_id="sha_atlas_00000001",
        memory_type="concept",
        metadata={"name": "Project Atlas", "entity_type": "project", "confidence": 0.90},
    )
    llm_result = {"entities": [alice_entity, atlas_entity], "relations": []}

    # First ingest + enrichment
    result1 = lite_memory.ingest(content, sync=False)
    item_id_1 = result1["item_id"]
    entity_ids_1 = result1.get("entity_ids", {})

    with patch(
        "smartmemory.plugins.extractors.llm_single.LLMSingleExtractor.extract",
        return_value=llm_result,
    ):
        enrich1 = process_extract_job(
            lite_memory,
            {"item_id": item_id_1, "workspace_id": "", "entity_ids": entity_ids_1, "enable_ontology": False},
        )
    first_new = enrich1["new_entities"]

    # Second ingest — same content, new memory node
    result2 = lite_memory.ingest(content, sync=False)
    item_id_2 = result2["item_id"]
    assert item_id_2 != item_id_1

    # Simulate learned EntityRuler: Tier 1 now recognizes entities from round 1.
    # In production, add_entity_pattern writes patterns that the next Tier 1 picks up.
    # Here we pass them directly as entity_ids so the dedup gate sees them.
    entity_ids_2 = result2.get("entity_ids", {})
    entity_ids_2["alice"] = "known_alice_id"
    entity_ids_2["project atlas"] = "known_atlas_id"

    with patch(
        "smartmemory.plugins.extractors.llm_single.LLMSingleExtractor.extract",
        return_value=llm_result,
    ):
        enrich2 = process_extract_job(
            lite_memory,
            {"item_id": item_id_2, "workspace_id": "", "entity_ids": entity_ids_2, "enable_ontology": False},
        )

    # Both memory nodes must survive (no UPSERT overwrite)
    backend = lite_memory._graph.backend
    assert backend.get_node(item_id_1) is not None, "First memory node disappeared"
    assert backend.get_node(item_id_2) is not None, "Second memory node disappeared"

    # With learned patterns, second enrichment finds 0 net-new entities
    assert enrich2["new_entities"] == 0, (
        f"Re-ingest with known entities produced {enrich2['new_entities']} new entities, expected 0"
    )
