"""Decision Confidence Evolver.

Background job that applies confidence decay to stale decisions and retracts
decisions that fall below a threshold. Follows the same pattern as
OpinionReinforcementEvolver.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from smartmemory.models.base import MemoryBaseModel
from smartmemory.plugins.base import EvolverPlugin, PluginMetadata

logger = logging.getLogger(__name__)


@dataclass
class DecisionConfidenceConfig(MemoryBaseModel):
    """Configuration for decision confidence evolution."""

    min_confidence_threshold: float = 0.1
    decay_after_days: int = 30
    decay_rate: float = 0.05
    enable_decay: bool = True


class DecisionConfidenceEvolver(EvolverPlugin):
    """Apply confidence decay to stale decisions and retract weak ones.

    Operations:
    1. Get all active decisions
    2. Apply decay to decisions with no recent activity
    3. Retract decisions that fall below confidence threshold
    """

    @classmethod
    def metadata(cls) -> PluginMetadata:
        return PluginMetadata(
            name="decision_confidence",
            version="1.0.0",
            author="SmartMemory Team",
            description="Applies confidence decay and retraction to stale decisions",
            plugin_type="evolver",
            dependencies=[],
            min_smartmemory_version="0.1.0",
            tags=["decision", "confidence", "decay"],
        )

    def __init__(self, config: DecisionConfidenceConfig | None = None):
        self.config = config or DecisionConfidenceConfig()

    def evolve(self, memory, log=None):
        """Apply confidence decay and retraction to decisions.

        Args:
            memory: SmartMemory instance.
            log: Optional logger.
        """
        log = log or logger
        cfg = self.config

        decisions = self._get_active_decisions(memory)
        if not decisions:
            log.info("No active decisions found for evolution")
            return

        log.info(f"Processing {len(decisions)} active decisions")

        decayed = 0
        retracted = 0

        for item in decisions:
            meta = item.metadata
            if meta.get("status") != "active":
                continue

            confidence = meta.get("confidence", 0.8)
            changed = False

            # Apply decay if enabled and decision is stale
            if cfg.enable_decay:
                last_activity = self._get_last_activity(meta)
                if self._should_decay(last_activity, cfg.decay_after_days):
                    confidence = max(0.0, confidence - cfg.decay_rate)
                    changed = True
                    decayed += 1

            # Retract if below threshold
            if confidence < cfg.min_confidence_threshold:
                self._retract_decision(memory, meta["decision_id"], confidence)
                retracted += 1
                continue

            # Persist changes
            if changed:
                self._update_confidence(memory, meta["decision_id"], confidence)

        log.info(f"Decision evolution complete: {decayed} decayed, {retracted} retracted")

    def _get_active_decisions(self, memory) -> list:
        """Get all active decision memories."""
        try:
            results = memory.search(query="*", memory_type="decision", top_k=500)
            return results if results else []
        except Exception:
            try:
                return memory.search(query="decision", memory_type="decision", top_k=500) or []
            except Exception as e:
                logger.warning(f"Failed to search for decisions: {e}")
                return []

    def _get_last_activity(self, meta: dict) -> datetime | None:
        """Get the most recent activity timestamp from decision metadata."""
        for field in ("last_reinforced_at", "last_contradicted_at", "updated_at"):
            val = meta.get(field)
            if val:
                if isinstance(val, str):
                    try:
                        return datetime.fromisoformat(val)
                    except (ValueError, TypeError):
                        continue
                elif isinstance(val, datetime):
                    return val
        return None

    def _should_decay(self, last_activity: datetime | None, decay_after_days: int) -> bool:
        """Check if a decision should decay based on time since last activity."""
        if last_activity is None:
            return True
        days_since = (datetime.now(timezone.utc) - last_activity).days
        return days_since > decay_after_days

    def _update_confidence(self, memory, decision_id: str, confidence: float) -> None:
        """Update a decision's confidence score."""
        try:
            memory.update_properties(decision_id, {
                "confidence": confidence,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            })
        except Exception as e:
            logger.warning(f"Failed to update decision {decision_id}: {e}")

    def _retract_decision(self, memory, decision_id: str, confidence: float) -> None:
        """Retract a decision that fell below threshold."""
        try:
            memory.update_properties(decision_id, {
                "status": "retracted",
                "confidence": confidence,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "context_snapshot": {"retraction_reason": "confidence_below_threshold"},
            })
        except Exception as e:
            logger.warning(f"Failed to retract decision {decision_id}: {e}")
