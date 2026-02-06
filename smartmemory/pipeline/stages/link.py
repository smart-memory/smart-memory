"""Link stage — wraps Linking.link_new_item()."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from smartmemory.pipeline.state import PipelineState

if TYPE_CHECKING:
    from smartmemory.memory.pipeline.stages.linking import Linking
    from smartmemory.pipeline.config import PipelineConfig


class LinkStage:
    """Create semantic links between the new item and existing memory."""

    def __init__(self, linking: Linking):
        self._linking = linking

    @property
    def name(self) -> str:
        return "link"

    def execute(self, state: PipelineState, config: PipelineConfig) -> PipelineState:
        if config.mode == "preview":
            return state

        # Build an IngestionContext-compatible dict for the legacy Linking API
        context = {
            "item": _build_item_stub(state),
            "entity_ids": dict(state.entity_ids),
            "entities": list(state.entities),
            "relations": list(state.relations),
        }

        self._linking.link_new_item(context)

        return replace(state, links=context.get("links") or {})

    def undo(self, state: PipelineState) -> PipelineState:
        return replace(state, links={})


def _build_item_stub(state: PipelineState):
    """Build a minimal MemoryItem-like object for legacy APIs."""
    from smartmemory.models.memory_item import MemoryItem

    return MemoryItem(
        content=state.resolved_text or state.text,
        memory_type=state.memory_type or "semantic",
        metadata=dict(state.raw_metadata),
        item_id=state.item_id,
    )
