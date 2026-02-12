"""
Procedure Recommendation Engine for CFS-3b.

Provides pattern detection and scoring for procedure promotion candidates
detected from working memory patterns.

Usage:
    ```python
    from smartmemory import SmartMemory
    from smartmemory.procedures import PatternDetector, PatternDetectorConfig

    memory = SmartMemory()
    config = PatternDetectorConfig(
        cluster_threshold=0.75,
        min_cluster_size=3,
        min_score=0.6
    )
    detector = PatternDetector(memory, config)

    candidates = detector.detect_candidates(days_back=30)
    for candidate in candidates:
        print(f"{candidate.suggested_name}: {candidate.scores.recommendation_score:.2f}")
    ```
"""

from smartmemory.procedures.models import (
    ProcedureCandidate,
    CandidateScores,
    PatternDetectorConfig,
)
from smartmemory.procedures.pattern_detector import PatternDetector
from smartmemory.procedures.candidate_scorer import CandidateScorer
from smartmemory.procedures.candidate_namer import CandidateNamer

__all__ = [
    # Main detector
    "PatternDetector",
    # Models
    "ProcedureCandidate",
    "CandidateScores",
    "PatternDetectorConfig",
    # Components (for advanced use)
    "CandidateScorer",
    "CandidateNamer",
]
