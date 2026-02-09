"""Residuation engine for pending decisions.

Pauses reasoning when data is incomplete instead of forcing low-confidence conclusions.
Tracks what's missing and auto-activates when requirements are satisfied.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from smartmemory.models.decision import Decision, PendingRequirement

logger = logging.getLogger(__name__)


class ResiduationManager:
    """Manages pending decisions that need more evidence before activation.

    Usage:
        rm = ResiduationManager(smart_memory, decision_manager=dm)
        decision = rm.create_pending("User might prefer Python", [
            {"description": "Need language usage data", "requirement_type": "evidence"}
        ])
        rm.resolve_requirement(decision.decision_id, "req_001", "mem_evidence_id")
        rm.try_activate(decision.decision_id)
    """

    def __init__(self, memory: Any, graph: Any = None, decision_manager: Any = None):
        self.memory = memory
        self.graph = graph or getattr(memory, "_graph", None)
        self.dm = decision_manager

    def create_pending(self, content: str, requirements: list[dict[str, Any]], **kwargs: Any) -> Decision:
        """Create a pending decision with explicit requirements.

        Args:
            content: The tentative decision statement.
            requirements: List of dicts with 'description' and 'requirement_type' keys.
            **kwargs: Additional Decision fields (domain, tags, etc.).

        Returns:
            The created pending Decision.
        """
        pending_reqs = [
            PendingRequirement(
                requirement_id=f"req_{uuid.uuid4().hex[:8]}",
                description=r.get("description", ""),
                requirement_type=r.get("requirement_type", "evidence"),
                query_hint=r.get("query_hint"),
            )
            for r in requirements
        ]

        decision = self.dm.create(
            content=content,
            confidence=0.0,
            **kwargs,
        )
        decision.status = "pending"
        decision.pending_requirements = pending_reqs
        self._update_decision(decision)
        return decision

    def resolve_requirement(self, decision_id: str, requirement_id: str, memory_id: str) -> bool:
        """Mark a pending requirement as resolved by a memory item.

        Args:
            decision_id: Decision containing the requirement.
            requirement_id: Requirement to resolve.
            memory_id: Memory ID that satisfies the requirement.

        Returns:
            True if requirement found and resolved, False otherwise.
        """
        decision = self.dm.get_decision(decision_id)
        if decision is None:
            logger.warning(f"Decision not found: {decision_id}")
            return False

        for req in decision.pending_requirements:
            if req.requirement_id == requirement_id:
                req.resolved = True
                req.resolved_by = memory_id
                req.resolved_at = datetime.now(timezone.utc)
                self._update_decision(decision)
                return True

        logger.warning(f"Requirement {requirement_id} not found on decision {decision_id}")
        return False

    def try_activate(self, decision_id: str) -> bool:
        """Attempt to activate a pending decision if all requirements are met.

        Args:
            decision_id: Decision to check and potentially activate.

        Returns:
            True if activated, False if still pending.
        """
        decision = self.dm.get_decision(decision_id)
        if decision is None or not decision.is_pending:
            return False

        if decision.has_unresolved_requirements:
            return False

        decision.status = "active"
        decision.confidence = 0.8
        decision.updated_at = datetime.now(timezone.utc)
        self._update_decision(decision)
        logger.info(f"Activated pending decision: {decision_id}")
        return True

    def get_pending_decisions(self, limit: int = 50) -> list[Decision]:
        """Get all pending decisions.

        Returns:
            List of decisions with status='pending'.
        """
        try:
            items = self.memory.search(query="*", memory_type="decision", top_k=limit * 2)
        except Exception:
            try:
                items = self.memory.search(query="decision", memory_type="decision", top_k=limit * 2)
            except Exception as e:
                logger.warning(f"Failed to search pending decisions: {e}")
                return []

        pending = []
        for item in items:
            d = self.dm._to_decision(item) if hasattr(self.dm, "_to_decision") else None
            if d and d.is_pending:
                pending.append(d)
                if len(pending) >= limit:
                    break
        return pending

    def _update_decision(self, decision: Decision) -> None:
        """Persist decision changes."""
        if hasattr(self.dm, "_update_decision"):
            self.dm._update_decision(decision)
        else:
            try:
                self.memory.update_properties(decision.decision_id, decision.to_dict())
            except Exception as e:
                logger.warning(f"Failed to update decision: {e}")
