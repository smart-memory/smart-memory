"""Promotion flow — evaluate and promote LLM-discovered entity types.

Gate order:
1. min_name_length
2. Common word blocklist
3. min_confidence
4. min_frequency
5. min_type_consistency
6. reasoning_validation (optional, delegates to ReasoningValidator)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from smartmemory.graph.ontology_graph import OntologyGraph
    from smartmemory.pipeline.config import PromotionConfig

logger = logging.getLogger(__name__)

COMMON_WORD_BLOCKLIST: set[str] = {
    "the",
    "a",
    "an",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "shall",
    "should",
    "may",
    "might",
    "must",
    "can",
    "could",
    "and",
    "but",
    "or",
    "nor",
    "for",
    "yet",
    "so",
    "if",
    "then",
    "else",
    "when",
    "where",
    "while",
    "although",
    "because",
    "since",
    "until",
    "after",
    "before",
    "during",
    "about",
    "above",
    "below",
    "between",
    "into",
    "through",
    "with",
    "without",
    "under",
    "over",
    "from",
    "to",
    "in",
    "on",
    "at",
    "by",
    "of",
    "up",
    "out",
    "off",
    "not",
    "no",
    "yes",
    "all",
    "each",
    "every",
    "both",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "any",
    "only",
    "own",
    "same",
    "than",
    "too",
    "very",
    "just",
    "also",
    "now",
    "here",
    "there",
    "how",
    "what",
    "which",
    "who",
    "whom",
    "whose",
    "why",
    "much",
    "many",
    "new",
    "old",
    "big",
    "small",
    "long",
    "short",
    "high",
    "low",
    "good",
    "bad",
    "great",
    "little",
    "first",
    "last",
    "next",
    "back",
    "well",
    "way",
    "even",
    "still",
    "already",
    "always",
    "never",
    "often",
    "really",
    "quite",
    "rather",
    "almost",
    "enough",
    "thing",
    "things",
    "something",
    "nothing",
    "anything",
    "everything",
    "one",
    "two",
    "three",
    "part",
    "time",
    "day",
    "year",
    "people",
    "made",
    "make",
    "like",
    "know",
    "get",
    "got",
    "say",
    "said",
    "see",
    "go",
    "come",
    "take",
    "give",
    "think",
    "tell",
    "work",
    "call",
    "try",
    "need",
    "use",
    "find",
    "want",
    "seem",
    "show",
    "look",
    "set",
    "put",
    "run",
    "move",
    "play",
    "point",
    "help",
    "turn",
    "start",
    "end",
    "keep",
    "let",
    "begin",
    "fact",
    "case",
    "lot",
    "able",
    "sure",
    "real",
    "right",
    "left",
    "different",
    "important",
    "another",
    "used",
    "using",
}


@dataclass
class PromotionCandidate:
    """A candidate for entity type promotion."""

    entity_name: str
    entity_type: str
    confidence: float
    source_memory_id: str | None = None


@dataclass
class PromotionResult:
    """Result of evaluating a promotion candidate."""

    promoted: bool
    reason: str
    reasoning_trace: Any = None  # Optional ReasoningTrace


class PromotionEvaluator:
    """Evaluate candidates against statistical and semantic gates."""

    def __init__(self, ontology_graph: OntologyGraph, config: PromotionConfig):
        self._ontology = ontology_graph
        self._config = config

    def evaluate(self, candidate: PromotionCandidate) -> PromotionResult:
        """Apply gates in order. Returns PromotionResult with verdict and reason."""
        name = candidate.entity_name
        entity_type = candidate.entity_type

        # Gate 1: min_name_length
        if len(name.strip()) < self._config.min_name_length:
            return PromotionResult(
                promoted=False, reason=f"Name too short ({len(name.strip())} < {self._config.min_name_length})"
            )

        # Gate 2: common word blocklist
        if name.lower().strip() in COMMON_WORD_BLOCKLIST:
            return PromotionResult(promoted=False, reason=f"'{name}' is a common word")

        # Gate 3: min_confidence
        if candidate.confidence < self._config.min_confidence:
            return PromotionResult(
                promoted=False,
                reason=f"Confidence {candidate.confidence:.2f} < {self._config.min_confidence}",
            )

        # Gate 4: min_frequency
        freq = self._ontology.get_frequency(entity_type.title())
        if freq < self._config.min_frequency:
            return PromotionResult(
                promoted=False,
                reason=f"Frequency {freq} < {self._config.min_frequency}",
            )

        # Gate 5: min_type_consistency
        assignments = self._ontology.get_type_assignments(name.lower())
        if assignments:
            total = sum(a["count"] for a in assignments)
            same_type = sum(a["count"] for a in assignments if a["type"].lower() == entity_type.lower())
            consistency = same_type / total if total > 0 else 0.0
            if consistency < self._config.min_type_consistency:
                return PromotionResult(
                    promoted=False,
                    reason=f"Type consistency {consistency:.2f} < {self._config.min_type_consistency}",
                )

        # Gate 6: reasoning_validation (optional)
        if self._config.reasoning_validation:
            try:
                from smartmemory.ontology.reasoning_validator import ReasoningValidator

                validator = ReasoningValidator()
                stats = {"frequency": freq, "assignments": assignments}
                result = validator.validate(candidate, stats)
                if not result.is_valid:
                    return PromotionResult(
                        promoted=False,
                        reason=f"Reasoning validation rejected: {result.explanation}",
                        reasoning_trace=result.reasoning_trace,
                    )
            except ImportError:
                logger.debug("ReasoningValidator not available, skipping gate 6")

        return PromotionResult(promoted=True, reason="All gates passed")

    def promote(self, candidate: PromotionCandidate) -> None:
        """Execute promotion: promote type in ontology, create EntityPattern."""
        entity_type = candidate.entity_type.title()
        self._ontology.promote(entity_type)
        self._ontology.add_entity_pattern(
            name=candidate.entity_name.lower(),
            label=entity_type,
            confidence=candidate.confidence,
            source="promoted",
        )
        logger.info("Promoted '%s' as '%s' (confidence=%.2f)", candidate.entity_name, entity_type, candidate.confidence)
