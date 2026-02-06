"""Extract stage — wraps ExtractionPipeline.extract_semantics()."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING

from smartmemory.pipeline.state import PipelineState

if TYPE_CHECKING:
    from smartmemory.memory.ingestion.extraction import ExtractionPipeline
    from smartmemory.pipeline.config import PipelineConfig

logger = logging.getLogger(__name__)


class ExtractStage:
    """Run entity/relation extraction on the (resolved) text."""

    def __init__(self, extraction_pipeline: ExtractionPipeline):
        self._pipeline = extraction_pipeline

    @property
    def name(self) -> str:
        return "extract"

    def execute(self, state: PipelineState, config: PipelineConfig) -> PipelineState:
        from smartmemory.models.memory_item import MemoryItem
        from smartmemory.memory.pipeline.config import ExtractionConfig as LegacyExtractionConfig

        # Build a MemoryItem from state for the extractor
        content = state.resolved_text or state.text
        item = MemoryItem(
            content=content,
            memory_type=state.memory_type or "semantic",
            metadata=dict(state.raw_metadata),
        )

        # Map v2 config → legacy ExtractionConfig
        ext_cfg = config.extraction
        legacy_conf = LegacyExtractionConfig(
            extractor_name=ext_cfg.extractor_name,
            max_entities=ext_cfg.llm_extract.max_entities,
            enable_relations=ext_cfg.llm_extract.enable_relations,
            max_extraction_attempts=ext_cfg.max_extraction_attempts,
            model=ext_cfg.llm_extract.model,
            temperature=ext_cfg.llm_extract.temperature,
            max_tokens=ext_cfg.llm_extract.max_tokens,
        )

        # Build conversation context from coreference chains if available
        conversation_context = None
        coref_result = state._context.get("coreference_result", {})
        coref_chains = coref_result.get("chains", [])
        if coref_chains:
            conversation_context = {"coreference_chains": coref_chains}

        extraction = self._pipeline.extract_semantics(
            item,
            ext_cfg.extractor_name,
            legacy_conf,
            conversation_context=conversation_context,
        )

        entities = extraction.get("entities") or extraction.get("nodes") or []
        relations = extraction.get("relations", [])

        return replace(
            state,
            entities=entities,
            relations=relations,
        )

    def undo(self, state: PipelineState) -> PipelineState:
        return replace(state, entities=[], relations=[], rejected=[], promotion_candidates=[])
