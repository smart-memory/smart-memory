"""Ground stage — wraps WikipediaGrounder.ground()."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING

from smartmemory.pipeline.state import PipelineState

if TYPE_CHECKING:
    from smartmemory.pipeline.config import PipelineConfig

logger = logging.getLogger(__name__)


class GroundStage:
    """Ground entities to Wikipedia for provenance."""

    def __init__(self, memory):
        """Args: memory — a SmartMemory instance (needs _graph, _grounding)."""
        self._memory = memory

    @property
    def name(self) -> str:
        return "ground"

    def execute(self, state: PipelineState, config: PipelineConfig) -> PipelineState:
        if config.mode == "preview":
            return state

        if not config.enrich.wikidata.enabled:
            return state

        entities = state.entities
        if not entities:
            return state

        try:
            from smartmemory.plugins.grounders import WikipediaGrounder
            from smartmemory.models.memory_item import MemoryItem

            item = MemoryItem(
                content=state.resolved_text or state.text,
                memory_type=state.memory_type or "semantic",
                metadata=dict(state.raw_metadata),
                item_id=state.item_id,
            )

            # Update entity item_ids from stored mapping
            entity_ids = state.entity_ids
            for entity in entities:
                if hasattr(entity, "metadata") and entity.metadata:
                    ename = entity.metadata.get("name")
                    if ename and ename in entity_ids:
                        entity.item_id = entity_ids[ename]

            grounder = WikipediaGrounder()
            provenance = grounder.ground(item, entities, self._memory._graph)

            # Create GROUNDED_IN edges
            if provenance:
                context = {
                    "provenance_candidates": provenance,
                    "entity_ids": entity_ids,
                    "item": item,
                }
                self._memory._grounding.ground(context)

            ctx = dict(state._context)
            ctx["provenance_candidates"] = provenance or []

            return replace(state, _context=ctx)

        except Exception as e:
            logger.warning("Wikipedia grounding failed: %s", e)
            return state

    def undo(self, state: PipelineState) -> PipelineState:
        ctx = dict(state._context)
        ctx.pop("provenance_candidates", None)
        return replace(state, _context=ctx)
