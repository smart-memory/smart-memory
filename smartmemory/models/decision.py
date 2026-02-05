"""
Decision Model for Decision Memory

Decisions are discrete conclusions, inferences, and choices with full provenance.
They capture what the system believes, with confidence tracking and lifecycle management.

Decisions can be produced by ReasoningTraces or created directly from explicit user
statements. Confidence follows the same reinforce/contradict pattern as OpinionMetadata.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

from smartmemory.models.base import MemoryBaseModel

DecisionType = Literal[
    "inference",       # Concluded from evidence (e.g., "User prefers TypeScript")
    "preference",      # User preference detected
    "classification",  # Categorization decision
    "choice",          # Selection between options
    "belief",          # Adopted belief/opinion
    "policy",          # Rule to follow going forward
]

DecisionSource = Literal["reasoning", "explicit", "imported", "inferred"]
DecisionStatus = Literal["active", "superseded", "retracted", "pending"]


@dataclass
class PendingRequirement(MemoryBaseModel):
    """What's needed to complete a pending decision."""

    requirement_id: str = ""
    description: str = ""
    requirement_type: str = "evidence"  # evidence, confirmation, data
    query_hint: str | None = None
    resolved: bool = False
    resolved_by: str | None = None
    resolved_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "requirement_id": self.requirement_id,
            "description": self.description,
            "requirement_type": self.requirement_type,
            "query_hint": self.query_hint,
            "resolved": self.resolved,
            "resolved_by": self.resolved_by,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PendingRequirement":
        resolved_at = d.get("resolved_at")
        if isinstance(resolved_at, str):
            resolved_at = datetime.fromisoformat(resolved_at)
        return cls(
            requirement_id=d.get("requirement_id", ""),
            description=d.get("description", ""),
            requirement_type=d.get("requirement_type", "evidence"),
            query_hint=d.get("query_hint"),
            resolved=d.get("resolved", False),
            resolved_by=d.get("resolved_by"),
            resolved_at=resolved_at,
        )


@dataclass
class Decision(MemoryBaseModel):
    """A discrete conclusion, inference, or choice derived from reasoning or observation.

    Decisions are first-class memory entities that capture what the system believes,
    with full provenance tracking back to the evidence and reasoning that produced them.

    Confidence follows the same diminishing-returns pattern as OpinionMetadata:
    - reinforce(): min(1.0, confidence + (1 - confidence) * 0.1)
    - contradict(): max(0.0, confidence - confidence * 0.15)
    """

    decision_id: str = ""
    content: str = ""
    decision_type: str = "inference"  # One of DecisionType values

    # Confidence & Evidence
    confidence: float = 0.8
    evidence_ids: list[str] = field(default_factory=list)
    contradicting_ids: list[str] = field(default_factory=list)
    reinforcement_count: int = 0
    contradiction_count: int = 0

    # Provenance
    source_type: str = "inferred"  # One of DecisionSource values
    source_trace_id: str | None = None
    source_session_id: str | None = None

    # Context at decision time
    context_snapshot: dict[str, Any] | None = None

    # Lifecycle
    status: str = "active"  # One of DecisionStatus values
    superseded_by: str | None = None

    # Domain tagging (for filtered retrieval)
    domain: str | None = None
    tags: list[str] = field(default_factory=list)

    # Residuation (pending decisions)
    pending_requirements: list[PendingRequirement] = field(default_factory=list)

    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime | None = None
    last_reinforced_at: datetime | None = None
    last_contradicted_at: datetime | None = None

    @property
    def is_active(self) -> bool:
        """Whether this decision is currently active."""
        return self.status == "active"

    @property
    def is_pending(self) -> bool:
        """Whether this decision is waiting for more data."""
        return self.status == "pending"

    @property
    def has_unresolved_requirements(self) -> bool:
        """Whether any pending requirements remain unresolved."""
        return any(not r.resolved for r in self.pending_requirements)

    @property
    def has_provenance(self) -> bool:
        """Whether this decision has traceable provenance."""
        return self.source_trace_id is not None or len(self.evidence_ids) > 0

    @property
    def net_reinforcement(self) -> int:
        """Net reinforcement score (positive = supported, negative = contradicted)."""
        return self.reinforcement_count - self.contradiction_count

    @property
    def stability(self) -> float:
        """How stable this decision is (0-1). Higher = more stable."""
        total = self.reinforcement_count + self.contradiction_count
        if total == 0:
            return 0.5
        return self.reinforcement_count / total

    def reinforce(self, evidence_id: str) -> None:
        """Record supporting evidence. Confidence increases with diminishing returns."""
        self.reinforcement_count += 1
        self.last_reinforced_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)
        if evidence_id not in self.evidence_ids:
            self.evidence_ids.append(evidence_id)
        self.confidence = min(1.0, self.confidence + (1 - self.confidence) * 0.1)

    def contradict(self, evidence_id: str) -> None:
        """Record contradicting evidence. Confidence decreases proportionally."""
        self.contradiction_count += 1
        self.last_contradicted_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)
        if evidence_id not in self.contradicting_ids:
            self.contradicting_ids.append(evidence_id)
        self.confidence = max(0.0, self.confidence - self.confidence * 0.15)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for storage in MemoryItem.metadata."""
        return {
            "decision_id": self.decision_id,
            "content": self.content,
            "decision_type": self.decision_type,
            "confidence": self.confidence,
            "evidence_ids": self.evidence_ids,
            "contradicting_ids": self.contradicting_ids,
            "reinforcement_count": self.reinforcement_count,
            "contradiction_count": self.contradiction_count,
            "source_type": self.source_type,
            "source_trace_id": self.source_trace_id,
            "source_session_id": self.source_session_id,
            "context_snapshot": self.context_snapshot,
            "status": self.status,
            "superseded_by": self.superseded_by,
            "domain": self.domain,
            "tags": self.tags,
            "pending_requirements": [r.to_dict() for r in self.pending_requirements],
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "last_reinforced_at": self.last_reinforced_at.isoformat() if self.last_reinforced_at else None,
            "last_contradicted_at": self.last_contradicted_at.isoformat() if self.last_contradicted_at else None,
            "net_reinforcement": self.net_reinforcement,
            "stability": self.stability,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Decision":
        """Deserialize from dict (e.g., from MemoryItem.metadata)."""
        data = d
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = datetime.now(timezone.utc)

        updated_at = data.get("updated_at")
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at)

        last_reinforced = data.get("last_reinforced_at")
        if isinstance(last_reinforced, str):
            last_reinforced = datetime.fromisoformat(last_reinforced)

        last_contradicted = data.get("last_contradicted_at")
        if isinstance(last_contradicted, str):
            last_contradicted = datetime.fromisoformat(last_contradicted)

        pending_reqs_raw = data.get("pending_requirements", [])
        pending_requirements = [PendingRequirement.from_dict(r) for r in pending_reqs_raw]

        return cls(
            decision_id=data.get("decision_id", ""),
            content=data.get("content", ""),
            decision_type=data.get("decision_type", "inference"),
            confidence=data.get("confidence", 0.8),
            evidence_ids=data.get("evidence_ids", []),
            contradicting_ids=data.get("contradicting_ids", []),
            reinforcement_count=data.get("reinforcement_count", 0),
            contradiction_count=data.get("contradiction_count", 0),
            source_type=data.get("source_type", "inferred"),
            source_trace_id=data.get("source_trace_id"),
            source_session_id=data.get("source_session_id"),
            context_snapshot=data.get("context_snapshot"),
            status=data.get("status", "active"),
            superseded_by=data.get("superseded_by"),
            domain=data.get("domain"),
            tags=data.get("tags", []),
            pending_requirements=pending_requirements,
            created_at=created_at,
            updated_at=updated_at,
            last_reinforced_at=last_reinforced,
            last_contradicted_at=last_contradicted,
        )

    @staticmethod
    def generate_id() -> str:
        """Generate a unique decision ID."""
        return f"dec_{uuid.uuid4().hex[:12]}"
