"""Data models for the reasoning module.

These live here (not in challenger.py) to avoid circular imports between
challenger.py and the detection/resolution strategy packages.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List

from smartmemory.models.memory_item import MemoryItem


class ConflictType(Enum):
    """Types of conflicts that can be detected."""

    DIRECT_CONTRADICTION = "direct_contradiction"  # A is B vs A is not B
    TEMPORAL_CONFLICT = "temporal_conflict"  # Was X, now Y
    NUMERIC_MISMATCH = "numeric_mismatch"  # Value differs
    ENTITY_CONFUSION = "entity_confusion"  # Same name, different entities
    PARTIAL_OVERLAP = "partial_overlap"  # Some claims conflict


class ResolutionStrategy(Enum):
    """Strategies for resolving conflicts."""

    KEEP_EXISTING = "keep_existing"  # Trust existing fact
    ACCEPT_NEW = "accept_new"  # Replace with new fact
    KEEP_BOTH = "keep_both"  # Store both with conflict marker
    MERGE = "merge"  # Combine information
    DEFER = "defer"  # Flag for human review


@dataclass
class Conflict:
    """Represents a detected conflict between facts."""

    existing_item: MemoryItem
    existing_fact: str
    new_fact: str
    conflict_type: ConflictType
    confidence: float  # 0.0 to 1.0
    explanation: str
    suggested_resolution: ResolutionStrategy

    def to_dict(self) -> Dict[str, Any]:
        return {
            "existing_item_id": self.existing_item.item_id,
            "existing_fact": self.existing_fact,
            "new_fact": self.new_fact,
            "conflict_type": self.conflict_type.value,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "suggested_resolution": self.suggested_resolution.value,
        }


@dataclass
class ChallengeResult:
    """Result of challenging an assertion against existing knowledge."""

    new_assertion: str
    has_conflicts: bool
    conflicts: List[Conflict] = field(default_factory=list)
    related_facts: List[MemoryItem] = field(default_factory=list)
    overall_confidence: float = 1.0  # Confidence in the new assertion

    def to_dict(self) -> Dict[str, Any]:
        return {
            "new_assertion": self.new_assertion,
            "has_conflicts": self.has_conflicts,
            "conflicts": [c.to_dict() for c in self.conflicts],
            "related_facts_count": len(self.related_facts),
            "overall_confidence": self.overall_confidence,
        }


class DetectionMethod(Enum):
    """Available contradiction detection methods."""

    LLM = "llm"  # Most accurate, slowest
    GRAPH = "graph"  # Structural analysis
    EMBEDDING = "embedding"  # Semantic + polarity
    HEURISTIC = "heuristic"  # Pattern matching (least reliable)
