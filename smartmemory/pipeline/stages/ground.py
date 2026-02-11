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

    # Estimated tokens for a typical Wikipedia API + LLM grounding call
    _AVG_GROUND_TOKENS: int = 200

    def execute(self, state: PipelineState, config: PipelineConfig) -> PipelineState:
        if config.mode == "preview":
            return state

        if not config.enrich.wikidata.enabled:
            if state.token_tracker:
                state.token_tracker.record_avoided(
                    "ground",
                    self._AVG_GROUND_TOKENS,
                    reason="stage_disabled",
                )
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

            # Count entities before grounding to detect graph-gated skips
            entity_count = len(
                [e for e in entities if hasattr(e, "metadata") and e.metadata and e.metadata.get("name")]
            )

            provenance = grounder.ground(item, entities, self._memory._graph)

            # Record avoided tokens for entities that were graph-gated (not sent to Wikipedia API)
            # The grounder returns provenance for ALL entities (both cached and fresh).
            # Graph-gated entities are those where the grounder reused an existing node.
            if state.token_tracker and entity_count > 0:
                # provenance_count == entities that got provenance.
                # If all entities got provenance without any API calls, that's a full graph hit.
                # We estimate ~200 tokens per API call avoided per entity graph-gated.
                api_calls_avoided = getattr(grounder, "_graph_hits", 0)
                if api_calls_avoided > 0:
                    state.token_tracker.record_avoided(
                        "ground",
                        self._AVG_GROUND_TOKENS * api_calls_avoided,
                        reason="graph_lookup",
                    )

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
