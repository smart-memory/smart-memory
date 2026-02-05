"""Heuristic-based contradiction detection using pattern matching (fallback)."""

import logging
import re
from typing import Optional, Set

from smartmemory.reasoning.models import Conflict, ConflictType, ResolutionStrategy

from .base import ContradictionDetector, DetectionContext

logger = logging.getLogger(__name__)

NEGATION_PATTERNS = [
    ("is not", "is"),
    ("isn't", "is"),
    ("are not", "are"),
    ("aren't", "are"),
    ("was not", "was"),
    ("wasn't", "was"),
    ("does not", "does"),
    ("doesn't", "does"),
    ("cannot", "can"),
    ("can't", "can"),
    ("never", "always"),
    ("false", "true"),
]

STOPWORDS: Set[str] = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "must", "shall",
    "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above",
    "below", "between", "under", "again", "further", "then", "once",
    "and", "but", "or", "nor", "so", "yet", "both", "either", "neither",
    "not", "only", "own", "same", "than", "too", "very", "just",
}


def _extract_common_context(text1: str, text2: str) -> Optional[str]:
    """Extract common context words between two texts."""
    common = (set(text1.split()) & set(text2.split())) - STOPWORDS
    if len(common) >= 2:
        return " ".join(common)
    return None


class HeuristicDetector(ContradictionDetector):
    """Detect contradictions using simple heuristics (least reliable, fallback)."""

    name = "heuristic"

    def detect(self, ctx: DetectionContext) -> Optional[Conflict]:
        new_lower = ctx.new_assertion.lower()
        existing_lower = ctx.existing_fact.lower()

        # Check for direct negation
        for neg, pos in NEGATION_PATTERNS:
            if neg in new_lower and pos in existing_lower and neg not in existing_lower:
                return Conflict(
                    existing_item=ctx.existing_item,
                    existing_fact=ctx.existing_fact,
                    new_fact=ctx.new_assertion,
                    conflict_type=ConflictType.DIRECT_CONTRADICTION,
                    confidence=0.7,
                    explanation=f"Detected negation pattern: '{neg}' vs '{pos}'",
                    suggested_resolution=ResolutionStrategy.DEFER,
                )
            if neg in existing_lower and pos in new_lower and neg not in new_lower:
                return Conflict(
                    existing_item=ctx.existing_item,
                    existing_fact=ctx.existing_fact,
                    new_fact=ctx.new_assertion,
                    conflict_type=ConflictType.DIRECT_CONTRADICTION,
                    confidence=0.7,
                    explanation=f"Detected negation pattern: '{pos}' vs '{neg}'",
                    suggested_resolution=ResolutionStrategy.DEFER,
                )

        # Check for numeric differences
        new_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", ctx.new_assertion))
        existing_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", ctx.existing_fact))

        if new_numbers and existing_numbers:
            common_context = _extract_common_context(new_lower, existing_lower)
            if common_context and new_numbers != existing_numbers:
                return Conflict(
                    existing_item=ctx.existing_item,
                    existing_fact=ctx.existing_fact,
                    new_fact=ctx.new_assertion,
                    conflict_type=ConflictType.NUMERIC_MISMATCH,
                    confidence=0.6,
                    explanation=f"Numeric values differ: {existing_numbers} vs {new_numbers}",
                    suggested_resolution=ResolutionStrategy.DEFER,
                )

        return None
