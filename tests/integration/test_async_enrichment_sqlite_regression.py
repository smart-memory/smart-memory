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
