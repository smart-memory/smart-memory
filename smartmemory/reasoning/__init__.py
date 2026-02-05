"""
Reasoning module for SmartMemory.

Provides:
- AssertionChallenger: Detect contradictions between new and existing facts
- Detection strategies: LLM, Graph, Embedding, Heuristic
- Resolution strategies: Wikipedia, LLM, Grounding, Recency
- ConfidenceManager: Confidence decay and history tracking
"""

from .challenger import (
    AssertionChallenger,
    ChallengeResult,
    Conflict,
    ConflictType,
    ResolutionStrategy,
    DetectionMethod,
    should_challenge,
    FACTUAL_PATTERNS,
    SKIP_PATTERNS,
    CHALLENGEABLE_MEMORY_TYPES,
)

__all__ = [
    'AssertionChallenger',
    'ChallengeResult',
    'Conflict',
    'ConflictType',
    'ResolutionStrategy',
    'DetectionMethod',
    'should_challenge',
    'FACTUAL_PATTERNS',
    'SKIP_PATTERNS',
    'CHALLENGEABLE_MEMORY_TYPES',
]
