"""
Procedure Candidate Models for CFS-3b Recommendation Engine.

Defines data structures for procedure promotion candidates detected
from working memory patterns.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any


@dataclass
class CandidateScores:
    """Scoring breakdown for a procedure candidate."""

    recommendation_score: float = 0.0
    frequency_score: float = 0.0
    consistency_score: float = 0.0
    recency_score: float = 0.0
    agent_workflow_score: float = 0.0


@dataclass
class ProcedureCandidate:
    """
    A candidate for procedure promotion detected from working memory patterns.

    Represents a cluster of similar working memory items that could be
    promoted to a stored procedure.
    """

    cluster_id: str
    suggested_name: str
    suggested_description: str
    representative_content: str
    item_count: int
    scores: CandidateScores
    common_skills: List[str] = field(default_factory=list)
    common_tools: List[str] = field(default_factory=list)
    sample_item_ids: List[str] = field(default_factory=list)
    earliest_date: Optional[datetime] = None
    latest_date: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "cluster_id": self.cluster_id,
            "suggested_name": self.suggested_name,
            "suggested_description": self.suggested_description,
            "representative_content": self.representative_content,
            "item_count": self.item_count,
            "scores": {
                "recommendation_score": round(self.scores.recommendation_score, 4),
                "frequency_score": round(self.scores.frequency_score, 4),
                "consistency_score": round(self.scores.consistency_score, 4),
                "recency_score": round(self.scores.recency_score, 4),
                "agent_workflow_score": round(self.scores.agent_workflow_score, 4),
            },
            "common_skills": self.common_skills,
            "common_tools": self.common_tools,
            "sample_item_ids": self.sample_item_ids,
            "date_range": {
                "earliest": self.earliest_date.isoformat() if self.earliest_date else None,
                "latest": self.latest_date.isoformat() if self.latest_date else None,
            },
        }


@dataclass
class PatternDetectorConfig:
    """Configuration for pattern detection."""

    # Clustering threshold for similarity
    cluster_threshold: float = 0.75

    # Scoring weights (must sum to 1.0)
    frequency_weight: float = 0.40
    consistency_weight: float = 0.30
    recency_weight: float = 0.20
    agent_workflow_weight: float = 0.10

    # Minimum thresholds
    min_cluster_size: int = 3
    min_score: float = 0.6

    # Time window
    days_back: int = 30

    # Result limits
    max_candidates: int = 20
    max_sample_items: int = 5

    def __post_init__(self):
        """Validate configuration."""
        total_weight = (
            self.frequency_weight + self.consistency_weight + self.recency_weight + self.agent_workflow_weight
        )
        if abs(total_weight - 1.0) > 0.01:
            # Normalize weights
            self.frequency_weight /= total_weight
            self.consistency_weight /= total_weight
            self.recency_weight /= total_weight
            self.agent_workflow_weight /= total_weight
