"""LLMExtract stage — extract entities and relations via LLMSingleExtractor."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING

from smartmemory.pipeline.state import PipelineState

if TYPE_CHECKING:
    from smartmemory.pipeline.config import PipelineConfig
    from smartmemory.plugins.extractors.llm_single import LLMSingleExtractor

logger = logging.getLogger(__name__)


class LLMExtractStage:
    """Extract entities and relations using an LLM in a single call."""

    def __init__(self, extractor: LLMSingleExtractor | None = None):
        """Args: extractor — optional pre-built LLMSingleExtractor (for testing/DI)."""
        self._extractor = extractor

    @property
    def name(self) -> str:
        return "llm_extract"

    def execute(self, state: PipelineState, config: PipelineConfig) -> PipelineState:
        llm_cfg = config.extraction.llm_extract
        if not llm_cfg.enabled:
            return state

        # Build input text from simplified sentences or resolved/raw text
        if state.simplified_sentences:
            text = " ".join(state.simplified_sentences)
        else:
            text = state.resolved_text or state.text

        if not text or not text.strip():
            return state

        try:
            extractor = self._extractor or self._create_extractor(llm_cfg)
            result = extractor.extract(text)

            entities = result.get("entities", [])
            relations = result.get("relations", [])

            # Truncate to configured limits
            entities = entities[: llm_cfg.max_entities]
            relations = relations[: llm_cfg.max_relations]

            return replace(state, llm_entities=entities, llm_relations=relations)
        except Exception as e:
            logger.warning("LLM extraction failed: %s", e)
            return state

    def _create_extractor(self, llm_cfg):
        """Build an LLMSingleExtractor from config."""
        from smartmemory.plugins.extractors.llm_single import LLMSingleExtractor, LLMSingleExtractorConfig

        ext_config = LLMSingleExtractorConfig()
        if llm_cfg.model:
            ext_config.model_name = llm_cfg.model
        if llm_cfg.temperature is not None:
            ext_config.temperature = llm_cfg.temperature
        if llm_cfg.max_tokens is not None:
            ext_config.max_tokens = llm_cfg.max_tokens

        return LLMSingleExtractor(config=ext_config)

    def undo(self, state: PipelineState) -> PipelineState:
        return replace(state, llm_entities=[], llm_relations=[])
