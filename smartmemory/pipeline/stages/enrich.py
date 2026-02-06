"""Enrich stage — wraps EnrichmentPipeline.run_enrichment()."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING

from smartmemory.pipeline.state import PipelineState

if TYPE_CHECKING:
    from smartmemory.memory.ingestion.enrichment import EnrichmentPipeline
    from smartmemory.pipeline.config import PipelineConfig

logger = logging.getLogger(__name__)


class EnrichStage:
    """Run enrichment plugins on the stored item."""

    def __init__(self, enrichment_pipeline: EnrichmentPipeline):
        self._pipeline = enrichment_pipeline

    @property
    def name(self) -> str:
        return "enrich"

    def execute(self, state: PipelineState, config: PipelineConfig) -> PipelineState:
        if config.mode == "preview":
            return state

        from smartmemory.models.memory_item import MemoryItem

        item = MemoryItem(
            content=state.resolved_text or state.text,
            memory_type=state.memory_type or "semantic",
            metadata=dict(state.raw_metadata),
            item_id=state.item_id,
        )

        context = {
            "item": item,
            "node_ids": dict(state.entity_ids),
            "entity_ids": dict(state.entity_ids),
            "entities": list(state.entities),
        }

        # Pass enricher names from config if specified
        if config.enrich.enricher_names:
            context["enricher_names"] = config.enrich.enricher_names

        try:
            result = self._pipeline.run_enrichment(context)
        except Exception as e:
            logger.warning("Enrichment failed (non-fatal): %s", e)
            result = {}

        return replace(state, enrichments=result or {})

    def undo(self, state: PipelineState) -> PipelineState:
        return replace(state, enrichments={})
