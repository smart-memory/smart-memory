"""Confidence management for challenged facts."""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List

from smartmemory.models.memory_item import MemoryItem

logger = logging.getLogger(__name__)


class ConfidenceManager:
    """Manages confidence scoring and decay for memory items."""

    def __init__(self, smart_memory):
        self.sm = smart_memory

    def apply_decay(
        self,
        item_id: str,
        decay_factor: float = 0.1,
        reason: str | None = None,
        conflicting_fact: str | None = None,
    ) -> bool:
        """Apply confidence decay to a challenged fact with full tracking.

        Tracks: confidence, challenged flag, challenge_count, confidence_history,
        last_challenged_at.
        """
        try:
            item = self.sm.get(item_id)
            if not item:
                logger.warning(f"Item {item_id} not found for confidence decay")
                return False

            current_confidence = item.metadata.get("confidence", 1.0)
            new_confidence = max(0.0, current_confidence - decay_factor)
            now = datetime.now(timezone.utc).isoformat()

            if "confidence_history" not in item.metadata:
                item.metadata["confidence_history"] = []

            decay_event: Dict[str, Any] = {
                "timestamp": now,
                "old_confidence": current_confidence,
                "new_confidence": new_confidence,
                "decay_factor": decay_factor,
                "reason": reason or "challenged",
            }
            if conflicting_fact:
                decay_event["conflicting_fact"] = conflicting_fact[:200]

            item.metadata["confidence_history"].append(decay_event)

            # Keep only last 20 events
            if len(item.metadata["confidence_history"]) > 20:
                item.metadata["confidence_history"] = item.metadata["confidence_history"][-20:]

            item.metadata["confidence"] = new_confidence
            item.metadata["challenged"] = True
            item.metadata["challenge_count"] = item.metadata.get("challenge_count", 0) + 1
            item.metadata["last_challenged_at"] = now

            self.sm.update(item)

            logger.info(
                f"Applied confidence decay to {item_id}: "
                f"{current_confidence:.2f} -> {new_confidence:.2f} "
                f"(challenge #{item.metadata['challenge_count']})"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to apply confidence decay: {e}")
            return False

    def get_history(self, item_id: str) -> List[Dict[str, Any]]:
        """Get the confidence decay history for an item."""
        try:
            item = self.sm.get(item_id)
            if not item:
                return []
            return item.metadata.get("confidence_history", [])
        except Exception as e:
            logger.error(f"Failed to get confidence history: {e}")
            return []

    def get_low_confidence_items(
        self,
        threshold: float = 0.5,
        memory_type: str = "semantic",
        limit: int = 50,
    ) -> List[MemoryItem]:
        """Get items with confidence below threshold, sorted lowest first."""
        try:
            results = self.sm.search("", top_k=limit * 3, memory_type=memory_type)

            low_confidence = [item for item in results if item.metadata.get("confidence", 1.0) < threshold]
            low_confidence.sort(key=lambda x: x.metadata.get("confidence", 1.0))
            return low_confidence[:limit]

        except Exception as e:
            logger.error(f"Failed to get low confidence items: {e}")
            return []
