"""Decision lifecycle manager.

Handles creation, supersession, retraction, and conflict detection for decisions.
Takes SmartMemory as dependency (like AssertionChallenger). All operations are synchronous.
"""

import logging
from datetime import datetime, timezone
from typing import Any

from smartmemory.models.decision import Decision
from smartmemory.models.memory_item import MemoryItem

logger = logging.getLogger(__name__)


class DecisionManager:
    """Manage decision lifecycle - creation, supersession, retraction, reinforcement.

    Usage:
        manager = DecisionManager(smart_memory)
        decision = manager.create("User prefers dark mode", decision_type="preference")
        manager.reinforce(decision.decision_id, "evidence_mem_123")
        manager.supersede(decision.decision_id, new_decision, reason="Updated preference")
    """

    def __init__(self, memory: Any, graph: Any = None):
        """Initialize with SmartMemory instance.

        Args:
            memory: SmartMemory instance for storage operations.
            graph: SmartGraph instance (defaults to memory._graph if available).
        """
        self.memory = memory
        self.graph = graph or getattr(memory, '_graph', None)

    def create(
        self,
        content: str,
        decision_type: str = "inference",
        confidence: float = 0.8,
        source_type: str = "inferred",
        source_trace_id: str | None = None,
        source_session_id: str | None = None,
        evidence_ids: list[str] | None = None,
        domain: str | None = None,
        tags: list[str] | None = None,
        context_snapshot: dict[str, Any] | None = None,
    ) -> Decision:
        """Create and store a new decision.

        Args:
            content: The decision statement.
            decision_type: One of inference, preference, classification, choice, belief, policy.
            confidence: Initial confidence (0.0-1.0).
            source_type: One of reasoning, explicit, imported, inferred.
            source_trace_id: ID of the ReasoningTrace that produced this (if any).
            source_session_id: Conversation session ID.
            evidence_ids: Memory IDs supporting this decision.
            domain: Domain tag for filtered retrieval.
            tags: Additional tags.
            context_snapshot: State at decision time.

        Returns:
            The created Decision with generated ID.
        """
        decision = Decision(
            decision_id=Decision.generate_id(),
            content=content,
            decision_type=decision_type,
            confidence=confidence,
            source_type=source_type,
            source_trace_id=source_trace_id,
            source_session_id=source_session_id,
            evidence_ids=evidence_ids or [],
            domain=domain,
            tags=tags or [],
            context_snapshot=context_snapshot,
        )

        self._store_decision(decision)

        # Create provenance edges
        if source_trace_id and self.graph:
            self.graph.add_edge(
                source_id=source_trace_id,
                target_id=decision.decision_id,
                edge_type="PRODUCED",
                properties={"confidence": confidence, "timestamp": datetime.now(timezone.utc).isoformat()},
            )

        if evidence_ids:
            for eid in evidence_ids:
                self._add_derived_from_edge(decision.decision_id, eid)

        return decision

    def get_decision(self, decision_id: str) -> Decision | None:
        """Retrieve a decision by ID.

        Args:
            decision_id: The decision ID to look up.

        Returns:
            Decision if found, None otherwise.
        """
        item = self.memory.get(decision_id)
        if item is None:
            return None
        return self._to_decision(item)

    def supersede(self, old_decision_id: str, new_decision: Decision, reason: str) -> Decision:
        """Replace an old decision with a new one.

        Marks the old decision as superseded, stores the new one, and creates
        a SUPERSEDES edge between them.

        Args:
            old_decision_id: ID of the decision being replaced.
            new_decision: The replacement decision.
            reason: Why the old decision is being superseded.

        Returns:
            The new decision (stored).

        Raises:
            ValueError: If old decision not found.
        """
        old = self.get_decision(old_decision_id)
        if old is None:
            raise ValueError(f"Decision not found: {old_decision_id}")

        # Mark old as superseded
        old.status = "superseded"
        old.superseded_by = new_decision.decision_id
        old.updated_at = datetime.now(timezone.utc)
        self._update_decision(old)

        # Generate ID if needed
        if not new_decision.decision_id:
            new_decision.decision_id = Decision.generate_id()

        # Link new to old evidence
        if old_decision_id not in new_decision.evidence_ids:
            new_decision.evidence_ids.append(old_decision_id)

        # Store new decision
        self._store_decision(new_decision)

        # Create SUPERSEDES edge
        if self.graph:
            self.graph.add_edge(
                source_id=new_decision.decision_id,
                target_id=old_decision_id,
                edge_type="SUPERSEDES",
                properties={"reason": reason, "timestamp": datetime.now(timezone.utc).isoformat()},
            )

        return new_decision

    def retract(self, decision_id: str, reason: str) -> None:
        """Retract a decision (mark as no longer valid).

        Args:
            decision_id: ID of the decision to retract.
            reason: Why the decision is being retracted.

        Raises:
            ValueError: If decision not found.
        """
        decision = self.get_decision(decision_id)
        if decision is None:
            raise ValueError(f"Decision not found: {decision_id}")

        decision.status = "retracted"
        decision.updated_at = datetime.now(timezone.utc)
        decision.context_snapshot = decision.context_snapshot or {}
        decision.context_snapshot["retraction_reason"] = reason
        self._update_decision(decision)

    def reinforce(self, decision_id: str, evidence_id: str) -> Decision:
        """Record supporting evidence for a decision.

        Args:
            decision_id: ID of the decision to reinforce.
            evidence_id: ID of the supporting memory.

        Returns:
            The updated decision.

        Raises:
            ValueError: If decision not found.
        """
        decision = self.get_decision(decision_id)
        if decision is None:
            raise ValueError(f"Decision not found: {decision_id}")

        decision.reinforce(evidence_id)
        self._update_decision(decision)

        # Add DERIVED_FROM edge for new evidence
        self._add_derived_from_edge(decision_id, evidence_id)

        return decision

    def contradict(self, decision_id: str, evidence_id: str) -> Decision:
        """Record contradicting evidence for a decision.

        Args:
            decision_id: ID of the decision to contradict.
            evidence_id: ID of the contradicting memory.

        Returns:
            The updated decision.

        Raises:
            ValueError: If decision not found.
        """
        decision = self.get_decision(decision_id)
        if decision is None:
            raise ValueError(f"Decision not found: {decision_id}")

        decision.contradict(evidence_id)
        self._update_decision(decision)

        # Add CONTRADICTS edge
        if self.graph:
            self.graph.add_edge(
                source_id=evidence_id,
                target_id=decision_id,
                edge_type="CONTRADICTS",
                properties={"detected_at": datetime.now(timezone.utc).isoformat()},
            )

        return decision

    def find_conflicts(self, decision: Decision, limit: int = 10) -> list[Decision]:
        """Find existing active decisions that may conflict with the given one.

        Uses semantic search to find similar decisions, then checks for contradiction
        based on domain and content overlap.

        Args:
            decision: The decision to check against existing ones.
            limit: Max number of candidates to evaluate.

        Returns:
            List of potentially conflicting active decisions.
        """
        try:
            similar = self.memory.search(
                query=decision.content,
                memory_type="decision",
                top_k=limit,
            )
        except Exception as e:
            logger.warning(f"Conflict search failed: {e}")
            return []

        conflicts = []
        for candidate in similar:
            candidate_decision = self._to_decision(candidate)
            if candidate_decision is None:
                continue
            # Skip self, inactive, and same-ID decisions
            if candidate_decision.decision_id == decision.decision_id:
                continue
            if not candidate_decision.is_active:
                continue
            # Same domain = higher conflict potential
            if decision.domain and candidate_decision.domain == decision.domain:
                conflicts.append(candidate_decision)
            # Even without domain match, similar content may conflict
            elif self._content_overlap(decision.content, candidate_decision.content):
                conflicts.append(candidate_decision)

        return conflicts

    # ---- Internal helpers ----

    def _store_decision(self, decision: Decision) -> str:
        """Store a decision as a MemoryItem."""
        item = MemoryItem(
            content=decision.content,
            memory_type="decision",
            item_id=decision.decision_id,
            metadata=decision.to_dict(),
        )
        return self.memory.add(item)

    def _update_decision(self, decision: Decision) -> None:
        """Update an existing decision's properties in the graph."""
        properties = decision.to_dict()
        try:
            self.memory.update_properties(decision.decision_id, properties)
        except Exception as e:
            logger.warning(f"Failed to update decision {decision.decision_id}: {e}")

    def _to_decision(self, item: Any) -> Decision | None:
        """Convert a MemoryItem (or dict) to a Decision."""
        if item is None:
            return None
        if isinstance(item, Decision):
            return item
        metadata = getattr(item, 'metadata', None)
        if isinstance(item, dict):
            metadata = item.get('metadata', item)
        if not isinstance(metadata, dict):
            return None
        # Ensure content is populated from the item
        if 'content' not in metadata or not metadata['content']:
            content = getattr(item, 'content', None) or (item.get('content') if isinstance(item, dict) else '')
            metadata['content'] = content
        return Decision.from_dict(metadata)

    def _add_derived_from_edge(self, decision_id: str, evidence_id: str) -> None:
        """Create a DERIVED_FROM edge from decision to evidence."""
        if self.graph:
            try:
                self.graph.add_edge(
                    source_id=decision_id,
                    target_id=evidence_id,
                    edge_type="DERIVED_FROM",
                    properties={"role": "evidence"},
                )
            except Exception as e:
                logger.debug(f"Failed to create DERIVED_FROM edge {decision_id} -> {evidence_id}: {e}")

    @staticmethod
    def _content_overlap(content1: str, content2: str) -> bool:
        """Simple heuristic check for content overlap between two decision statements."""
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        if not words1 or not words2:
            return False
        overlap = len(words1 & words2) / min(len(words1), len(words2))
        return overlap > 0.5
