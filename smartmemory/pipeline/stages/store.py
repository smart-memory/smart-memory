"""Store stage — wraps CRUD.add() + StoragePipeline."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING, List

from smartmemory.pipeline.state import PipelineState

if TYPE_CHECKING:
    from smartmemory.pipeline.config import PipelineConfig

logger = logging.getLogger(__name__)


class StoreStage:
    """Persist the memory item and extracted entities to the graph."""

    def __init__(self, memory):
        """Args: memory — a SmartMemory instance."""
        self._memory = memory

    @property
    def name(self) -> str:
        return "store"

    def execute(self, state: PipelineState, config: PipelineConfig) -> PipelineState:
        from smartmemory.models.memory_item import MemoryItem

        if config.mode == "preview":
            return replace(state, item_id="preview_item")

        content = state.resolved_text or state.text
        item = MemoryItem(
            content=content,
            memory_type=state.memory_type or "semantic",
            metadata=dict(state.raw_metadata),
        )

        # Build ontology_extraction payload
        entities = state.entities
        relations = state.relations
        ontology_extraction = None
        if entities or relations:
            ontology_extraction = {"entities": entities, "relations": relations}

        # Store via CRUD
        add_result = self._memory._crud.add(item, ontology_extraction=ontology_extraction)

        # Process result
        entity_ids = {}
        if isinstance(add_result, dict):
            item_id = add_result.get("memory_node_id")
            created_ids = add_result.get("entity_node_ids", []) or []
            entity_ids = self._map_entity_ids(entities, created_ids, item_id)
        else:
            item_id = add_result
            entity_ids = self._map_entity_ids(entities, [], item_id)

        item.item_id = item_id
        item.update_status("created", notes="Item ingested")

        # Process external relations via StoragePipeline
        if relations:
            self._process_relations(state, item_id, entities, relations)

        # Save to vector and graph
        context = {
            "item": item,
            "entity_ids": entity_ids,
            "entities": entities,
        }
        try:
            from smartmemory.memory.ingestion.storage import StoragePipeline
            from smartmemory.memory.ingestion.observer import IngestionObserver

            storage = StoragePipeline(self._memory, IngestionObserver())
            storage.save_to_vector_and_graph(context)
        except Exception as e:
            logger.warning("Vector/graph save failed (non-fatal): %s", e)

        return replace(state, item_id=item_id, entity_ids=entity_ids)

    def undo(self, state: PipelineState) -> PipelineState:
        return replace(state, item_id=None, entity_ids={})

    @staticmethod
    def _map_entity_ids(entities: List, created_ids: List, item_id) -> dict:
        entity_ids = {}
        for i, entity in enumerate(entities):
            name = _entity_name(entity, i)
            real_id = created_ids[i] if i < len(created_ids) else f"{item_id}_entity_{i}"
            entity_ids[name] = real_id
        return entity_ids

    def _process_relations(self, state, item_id, entities, relations):
        """Filter and process external relations."""
        internal_ids = set()
        for e in entities:
            if hasattr(e, "item_id") and e.item_id:
                internal_ids.add(e.item_id)
            elif isinstance(e, dict):
                eid = e.get("item_id") or e.get("id")
                if eid:
                    internal_ids.add(eid)

        external = []
        for r in relations:
            src = r.get("source_id") or r.get("subject") or r.get("source")
            tgt = r.get("target_id") or r.get("object") or r.get("target")
            if src and tgt and src in internal_ids and tgt in internal_ids:
                continue
            external.append(r)

        if external:
            try:
                from smartmemory.memory.ingestion.storage import StoragePipeline
                from smartmemory.memory.ingestion.observer import IngestionObserver

                context = dict(state._context)
                storage = StoragePipeline(self._memory, IngestionObserver())
                storage.process_extracted_relations(context, item_id, external)
            except Exception as e:
                logger.warning("Failed to process relations: %s", e)


def _entity_name(entity, index: int) -> str:
    """Extract display name from an entity (MemoryItem or dict)."""
    if hasattr(entity, "metadata") and isinstance(getattr(entity, "metadata", None), dict):
        name = entity.metadata.get("name")
        if name:
            return name
    if isinstance(entity, dict):
        meta = entity.get("metadata", {})
        return meta.get("name") or entity.get("name", f"entity_{index}")
    return f"entity_{index}"
