"""Multi-dimensional fuzzy confidence scoring.

Enhances single-number confidence with four orthogonal dimensions:
evidence, recency, consensus, and directness.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from smartmemory.models.base import MemoryBaseModel
from smartmemory.models.decision import Decision

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS = {"evidence": 0.3, "recency": 0.2, "consensus": 0.3, "directness": 0.2}
EVIDENCE_SATURATION = 5
RECENCY_HALFLIFE_DAYS = 90


@dataclass
class ConfidenceScore(MemoryBaseModel):
    """Multi-dimensional confidence breakdown."""

    evidence: float = 0.0
    recency: float = 0.0
    consensus: float = 0.0
    directness: float = 0.0

    def combined(self, weights: dict[str, float] | None = None) -> float:
        """Weighted combination of all dimensions."""
        w = weights or DEFAULT_WEIGHTS
        total_weight = sum(w.values())
        if total_weight == 0:
            return 0.0
        return (
            w.get("evidence", 0.3) * self.evidence
            + w.get("recency", 0.2) * self.recency
            + w.get("consensus", 0.3) * self.consensus
            + w.get("directness", 0.2) * self.directness
        ) / total_weight

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence": round(self.evidence, 4),
            "recency": round(self.recency, 4),
            "consensus": round(self.consensus, 4),
            "directness": round(self.directness, 4),
            "combined": round(self.combined(), 4),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ConfidenceScore":
        return cls(
            evidence=d.get("evidence", 0.0),
            recency=d.get("recency", 0.0),
            consensus=d.get("consensus", 0.0),
            directness=d.get("directness", 0.0),
        )


class FuzzyConfidenceCalculator:
    """Computes multi-dimensional confidence for decisions.

    Usage:
        calc = FuzzyConfidenceCalculator(graph)
        score = calc.calculate(decision)
        print(f"Combined: {score.combined():.2f}")
    """

    def __init__(self, graph: Any = None, weights: dict[str, float] | None = None):
        self.graph = graph
        self.weights = weights or DEFAULT_WEIGHTS

    def calculate(self, decision: Decision) -> ConfidenceScore:
        """Calculate multi-dimensional confidence for a decision.

        Args:
            decision: The decision to score.

        Returns:
            ConfidenceScore with all dimensions populated.
        """
        if decision.is_pending:
            return ConfidenceScore()

        return ConfidenceScore(
            evidence=self._evidence_score(decision),
            recency=self._recency_score(decision),
            consensus=self._consensus_score(decision),
            directness=self._directness_score(decision),
        )

    def _evidence_score(self, decision: Decision) -> float:
        """Score based on amount of supporting evidence (saturates at EVIDENCE_SATURATION)."""
        count = len(decision.evidence_ids)
        return min(1.0, count / EVIDENCE_SATURATION)

    def _recency_score(self, decision: Decision) -> float:
        """Score based on how recently the decision was created (exponential decay)."""
        if not decision.created_at:
            return 0.5
        now = datetime.now(timezone.utc)
        created = decision.created_at
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        age_days = (now - created).total_seconds() / 86400
        return 0.5 ** (age_days / RECENCY_HALFLIFE_DAYS)

    def _consensus_score(self, decision: Decision) -> float:
        """Score based on reinforcement vs contradiction ratio."""
        return decision.stability

    def _directness_score(self, decision: Decision) -> float:
        """Score based on directness of evidence (has provenance = higher)."""
        if decision.has_provenance:
            return 1.0
        return 0.5
