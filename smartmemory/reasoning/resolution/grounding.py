"""Resolve conflicts by checking existing grounding/provenance."""

import logging
from typing import Any, Dict, Optional

from smartmemory.reasoning.challenger import Conflict, ResolutionStrategy

from .base import ConflictResolver

logger = logging.getLogger(__name__)


class GroundingResolver(ConflictResolver):
    """Check if either fact has existing grounding/provenance."""

    name = "grounding"

    def resolve(self, conflict: Conflict) -> Optional[Dict[str, Any]]:
        try:
            existing_metadata = conflict.existing_item.metadata or {}

            grounded = existing_metadata.get("grounded_to")
            provenance = existing_metadata.get("provenance", {})

            if grounded or provenance.get("wikipedia_id"):
                return {
                    "auto_resolved": True,
                    "resolution": ResolutionStrategy.KEEP_EXISTING,
                    "confidence": 0.75,
                    "method": "grounding",
                    "evidence": f"Existing fact has provenance: {grounded or provenance}",
                    "actions_taken": ["Existing fact has verified grounding"],
                }

            source = provenance.get("source", "")
            trusted_sources = {"wikipedia", "wikidata", "official", "verified", "authoritative"}
            if any(ts in source.lower() for ts in trusted_sources):
                return {
                    "auto_resolved": True,
                    "resolution": ResolutionStrategy.KEEP_EXISTING,
                    "confidence": 0.7,
                    "method": "grounding",
                    "evidence": f"Existing fact from trusted source: {source}",
                    "actions_taken": [f"Trusted source: {source}"],
                }

            return None

        except Exception as e:
            logger.debug(f"Grounding resolution failed: {e}")
            return None
