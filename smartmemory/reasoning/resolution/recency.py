"""Resolve temporal conflicts by preferring more recent information."""

import logging
from typing import Any, Dict, Optional

from smartmemory.reasoning.models import Conflict, ResolutionStrategy

from .base import ConflictResolver

logger = logging.getLogger(__name__)


class RecencyResolver(ConflictResolver):
    """For temporal conflicts, prefer more recent information."""

    name = "recency"

    def resolve(self, conflict: Conflict) -> Optional[Dict[str, Any]]:
        try:
            existing_metadata = conflict.existing_item.metadata or {}
            existing_time = existing_metadata.get("valid_start_time") or existing_metadata.get("timestamp")

            if existing_time:
                new_lower = conflict.new_fact.lower()
                recency_indicators = ["now", "currently", "as of", "today", "recent"]

                if any(ind in new_lower for ind in recency_indicators):
                    return {
                        "auto_resolved": True,
                        "resolution": ResolutionStrategy.ACCEPT_NEW,
                        "confidence": 0.65,
                        "method": "recency",
                        "evidence": "New fact appears to be more recent update",
                        "actions_taken": ["Temporal conflict resolved by recency"],
                    }

            return None

        except Exception as e:
            logger.debug(f"Recency resolution failed: {e}")
            return None
