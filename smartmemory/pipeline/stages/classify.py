"""Classify stage — wraps MemoryIngestionFlow.classify_item()."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from smartmemory.pipeline.state import PipelineState

if TYPE_CHECKING:
    from smartmemory.pipeline.config import PipelineConfig


class ClassifyStage:
    """Determine memory types for the incoming text."""

    def __init__(self, ingestion_flow):
        """Args: ingestion_flow — a MemoryIngestionFlow instance (has classify_item)."""
        self._flow = ingestion_flow

    @property
    def name(self) -> str:
        return "classify"

    def execute(self, state: PipelineState, config: PipelineConfig) -> PipelineState:
        from smartmemory.models.memory_item import MemoryItem
        from smartmemory.memory.pipeline.config import ClassificationConfig

        # Build a MemoryItem so classify_item can inspect metadata
        item = MemoryItem(
            content=state.text,
            memory_type=state.memory_type or "semantic",
            metadata=dict(state.raw_metadata),
        )

        # Map v2 config → legacy ClassificationConfig
        legacy_conf = ClassificationConfig(
            content_analysis_enabled=config.classify.content_analysis_enabled,
            default_confidence=config.classify.default_confidence,
            inferred_confidence=config.classify.inferred_confidence,
        )

        types = self._flow.classify_item(item, legacy_conf)

        return replace(
            state,
            classified_types=types,
            memory_type=state.memory_type or (types[0] if types else "semantic"),
        )

    def undo(self, state: PipelineState) -> PipelineState:
        return replace(state, classified_types=[], memory_type=None)
