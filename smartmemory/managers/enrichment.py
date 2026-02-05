"""Enrichment and grounding operations extracted from SmartMemory."""

import logging
from typing import List, Optional

from smartmemory.models.memory_item import MemoryItem

logger = logging.getLogger(__name__)


class EnrichmentManager:
    """Manages enrichment, grounding, and external resolution."""

    def __init__(self, enrichment, grounding, external_resolver):
        self._enrichment = enrichment
        self._grounding = grounding
        self._external_resolver = external_resolver

    def enrich(self, item_id: str, routines: Optional[List[str]] = None) -> None:
        """Enrich a memory item using registered enrichment routines."""
        return self._enrichment.enrich(item_id, routines)

    def ground(self, item_id: str, source_url: str, validation: Optional[dict] = None) -> None:
        """Ground a memory item to an external source."""
        context = {
            "item_id": item_id,
            "source_url": source_url,
            "validation": validation,
        }
        return self._grounding.ground(context)

    def ground_context(self, context: dict):
        """Ground using a pre-built context dict."""
        return self._grounding.ground(context)

    def resolve_external(self, node: MemoryItem) -> Optional[list]:
        """Delegate resolve_external to ExternalResolver submodule."""
        return self._external_resolver.resolve_external(node)
