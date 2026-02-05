"""
Assertion Challenger - Detect contradictions between new and existing facts.

This module provides:
- Contradiction detection using semantic similarity + LLM reasoning
- Confidence decay for challenged facts
- Conflict resolution strategies
- Smart triggering to only challenge when appropriate

Example:
    >>> challenger = AssertionChallenger(smart_memory)
    >>> result = challenger.challenge("Paris is the capital of Germany")
    >>> if result.has_conflicts:
    ...     for conflict in result.conflicts:
    ...         print(f"Contradicts: {conflict.existing_fact}")
"""

import logging
import re
from typing import List, Optional, Dict, Any

from smartmemory.models.memory_item import MemoryItem
from smartmemory.reasoning.models import (  # noqa: F401 - re-exported for backwards compat
    ChallengeResult,
    Conflict,
    ConflictType,
    DetectionMethod,
    ResolutionStrategy,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Smart Triggering - Determine when to challenge
# =============================================================================

# Patterns that indicate factual claims worth challenging
FACTUAL_PATTERNS = [
    r'\b(?:is|are|was|were)\s+(?:the|a|an)\b',  # "X is the Y"
    r'\b(?:is|are|was|were)\s+\d',  # "X is 42"
    r'\b(?:has|have|had)\s+\d',  # "X has 5"
    r'\b(?:equals?|=)\b',  # "X equals Y"
    r'\b(?:capital|president|CEO|founder|inventor)\s+of\b',  # "capital of X"
    r'\b(?:born|died|founded|created|invented)\s+(?:in|on)\b',  # dates
    r'\b(?:located|situated)\s+in\b',  # locations
    r'\b(?:contains?|consists?\s+of)\b',  # composition
    r'\b(?:always|never|every|all|none)\b',  # absolutes
]

# Patterns that indicate non-factual content (skip challenging)
SKIP_PATTERNS = [
    r'^(?:I|we|you)\s+(?:think|feel|believe|want|need|like|love|hate)',  # opinions
    r'^(?:maybe|perhaps|possibly|probably|might)',  # uncertainty
    r'\?$',  # questions
    r'^(?:hello|hi|hey|thanks|thank you|please|sorry)',  # greetings
    r'^(?:let\'s|let us|can you|could you|would you)',  # requests
    r'^(?:I\'m|I am)\s+(?:going|trying|working|looking)',  # actions
    r'(?:today|yesterday|tomorrow|now|currently|right now)',  # temporal/personal
]

# Memory types that should trigger challenging
CHALLENGEABLE_MEMORY_TYPES = {'semantic'}  # Only semantic facts by default


def should_challenge(
    content: str,
    memory_type: Optional[str] = None,
    source: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Determine if content should be challenged against existing knowledge.

    Smart triggering based on:
    1. Content patterns (factual claims vs opinions/questions)
    2. Memory type (semantic facts vs episodic experiences)
    3. Source trustworthiness
    4. Explicit metadata flags

    Args:
        content: The text content to evaluate
        memory_type: Type of memory being ingested
        source: Source of the content (e.g., "user", "llm", "api")
        metadata: Additional metadata that may contain challenge hints

    Returns:
        True if content should be challenged, False otherwise
    """
    metadata = metadata or {}

    # 1. Check explicit metadata flags
    if metadata.get('skip_challenge'):
        return False
    if metadata.get('force_challenge'):
        return True
    if metadata.get('trusted_source'):
        return False

    # 2. Check memory type
    if memory_type and memory_type not in CHALLENGEABLE_MEMORY_TYPES:
        logger.debug(f"Skipping challenge: memory_type '{memory_type}' not challengeable")
        return False

    # 3. Check for skip patterns (opinions, questions, etc.)
    content_lower = content.lower().strip()
    for pattern in SKIP_PATTERNS:
        if re.search(pattern, content_lower, re.IGNORECASE):
            logger.debug(f"Skipping challenge: matched skip pattern '{pattern}'")
            return False

    # 4. Check for factual patterns
    for pattern in FACTUAL_PATTERNS:
        if re.search(pattern, content_lower, re.IGNORECASE):
            logger.debug(f"Should challenge: matched factual pattern '{pattern}'")
            return True

    # 5. Check content length - very short content unlikely to be factual claim
    if len(content.split()) < 4:
        return False

    # 6. Check for high-confidence sources that should be challenged
    untrusted_sources = {'user_input', 'llm_generated', 'external_api'}
    if source in untrusted_sources:
        return True

    # Default: don't challenge (conservative)
    return False


class AssertionChallenger:
    """
    Challenges new assertions against existing knowledge to detect contradictions.

    Uses a multi-stage approach:
    1. Semantic search to find related facts
    2. Cascade of detection methods (LLM -> Graph -> Embedding -> Heuristic)
    3. Confidence scoring and resolution suggestions
    """

    def __init__(
        self,
        smart_memory,
        similarity_threshold: float = 0.6,
        max_related_facts: int = 10,
        use_llm: bool = True,
        use_graph: bool = True,
        use_embedding: bool = True,
        use_heuristic: bool = True,
    ):
        self.sm = smart_memory
        self.similarity_threshold = similarity_threshold
        self.max_related_facts = max_related_facts
        self.use_llm = use_llm
        self.use_graph = use_graph
        self.use_embedding = use_embedding
        self.use_heuristic = use_heuristic

        # Build detection cascade
        from smartmemory.reasoning.detection import (
            DetectionCascade,
            EmbeddingDetector,
            GraphDetector,
            HeuristicDetector,
            LLMDetector,
        )

        detectors = []
        if use_llm:
            detectors.append(LLMDetector())
        if use_graph:
            detectors.append(GraphDetector())
        if use_embedding:
            detectors.append(EmbeddingDetector())
        if use_heuristic:
            detectors.append(HeuristicDetector())
        self._detection_cascade = DetectionCascade(detectors)

        # Build resolution cascade
        from smartmemory.reasoning.resolution import (
            GroundingResolver,
            ResolutionCascade,
            WikipediaResolver,
            LLMResolver,
            RecencyResolver,
        )

        self._all_resolvers = {
            "wikipedia": WikipediaResolver(),
            "llm": LLMResolver(),
            "grounding": GroundingResolver(),
            "recency": RecencyResolver(),
        }
        # Default cascade used by auto_resolve
        self._resolution_cascade = ResolutionCascade(list(self._all_resolvers.values()))

        # Confidence manager
        from smartmemory.reasoning.confidence import ConfidenceManager

        self._confidence = ConfidenceManager(smart_memory)

    def challenge(
        self,
        assertion: str,
        memory_type: Optional[str] = "semantic",
        context: Optional[Dict[str, Any]] = None
    ) -> ChallengeResult:
        """Challenge an assertion against existing knowledge."""
        logger.info(f"Challenging assertion: {assertion[:100]}...")

        related_facts = self._find_related_facts(assertion, memory_type)

        if not related_facts:
            logger.debug("No related facts found, assertion is novel")
            return ChallengeResult(
                new_assertion=assertion,
                has_conflicts=False,
                related_facts=[],
                overall_confidence=1.0
            )

        logger.debug(f"Found {len(related_facts)} related facts")

        from smartmemory.reasoning.detection.base import DetectionContext

        conflicts = []
        for fact in related_facts:
            ctx = DetectionContext(
                new_assertion=assertion,
                existing_item=fact,
                existing_fact=fact.content,
                extra=context or {},
            )
            conflict = self._detection_cascade.detect(ctx)
            if conflict:
                conflicts.append(conflict)

        overall_confidence = self._calculate_confidence(conflicts)

        result = ChallengeResult(
            new_assertion=assertion,
            has_conflicts=len(conflicts) > 0,
            conflicts=conflicts,
            related_facts=related_facts,
            overall_confidence=overall_confidence
        )

        if conflicts:
            logger.warning(
                f"Found {len(conflicts)} conflicts for assertion: {assertion[:50]}..."
            )

        return result

    def _find_related_facts(
        self,
        assertion: str,
        memory_type: Optional[str]
    ) -> List[MemoryItem]:
        """Find facts semantically related to the assertion."""
        try:
            results = self.sm.search(
                assertion,
                top_k=self.max_related_facts,
                memory_type=memory_type
            )
            return results if results else []
        except Exception as e:
            logger.error(f"Error searching for related facts: {e}")
            return []

    def _calculate_confidence(self, conflicts: List[Conflict]) -> float:
        """Lower confidence = more/stronger conflicts."""
        if not conflicts:
            return 1.0
        total_conflict_weight = sum(c.confidence for c in conflicts)
        confidence = max(0.0, 1.0 - (total_conflict_weight / len(conflicts)) * 0.5)
        return round(confidence, 3)

    # -- Confidence management (delegates to ConfidenceManager) --

    def apply_confidence_decay(
        self,
        item_id: str,
        decay_factor: float = 0.1,
        reason: Optional[str] = None,
        conflicting_fact: Optional[str] = None
    ) -> bool:
        """Apply confidence decay to a challenged fact with full tracking."""
        return self._confidence.apply_decay(item_id, decay_factor, reason, conflicting_fact)

    def get_confidence_history(self, item_id: str) -> List[Dict[str, Any]]:
        """Get the confidence decay history for an item."""
        return self._confidence.get_history(item_id)

    def get_low_confidence_items(
        self,
        threshold: float = 0.5,
        memory_type: str = "semantic",
        limit: int = 50
    ) -> List[MemoryItem]:
        """Get items with confidence below threshold."""
        return self._confidence.get_low_confidence_items(threshold, memory_type, limit)

    # -- Auto-resolution (delegates to ResolutionCascade) --

    def auto_resolve(
        self,
        conflict: Conflict,
        use_wikipedia: bool = True,
        use_llm: bool = True,
    ) -> Dict[str, Any]:
        """Attempt to automatically resolve a conflict using external knowledge."""
        result: Dict[str, Any] = {
            "conflict": conflict.to_dict(),
            "auto_resolved": False,
            "resolution": None,
            "confidence": 0.0,
            "method": None,
            "evidence": None,
            "actions_taken": []
        }

        # Build filtered resolver list based on flags
        from smartmemory.reasoning.resolution import ResolutionCascade

        resolvers = []
        if use_wikipedia:
            resolvers.append(self._all_resolvers["wikipedia"])
        if use_llm:
            resolvers.append(self._all_resolvers["llm"])
        resolvers.append(self._all_resolvers["grounding"])
        if conflict.conflict_type == ConflictType.TEMPORAL_CONFLICT:
            resolvers.append(self._all_resolvers["recency"])

        cascade = ResolutionCascade(resolvers)
        cascade_result = cascade.resolve(conflict)

        if cascade_result:
            result.update(cascade_result)
            if result["auto_resolved"]:
                self._apply_resolution(conflict, result)
                return result

        result["actions_taken"].append("Auto-resolution failed, deferring to human review")
        return result

    def _apply_resolution(self, conflict: Conflict, resolution: Dict[str, Any]) -> None:
        """Apply the auto-resolution result."""
        strategy = resolution.get("resolution")

        if strategy == ResolutionStrategy.KEEP_EXISTING:
            conflict.existing_item.metadata['auto_verified'] = True
            conflict.existing_item.metadata['verification_method'] = resolution.get("method")
            conflict.existing_item.metadata['verification_evidence'] = resolution.get("evidence")
            self.sm.update(conflict.existing_item)
            resolution["actions_taken"].append("Marked existing fact as verified")

        elif strategy == ResolutionStrategy.ACCEPT_NEW:
            self.apply_confidence_decay(
                conflict.existing_item.item_id,
                decay_factor=0.4,
                reason=f"auto_resolved:{resolution.get('method', 'unknown')}",
                conflicting_fact=conflict.new_fact
            )
            conflict.existing_item.metadata['superseded_by'] = conflict.new_fact
            conflict.existing_item.metadata['superseded_reason'] = resolution.get("evidence")
            self.sm.update(conflict.existing_item)
            resolution["actions_taken"].append("Decayed existing fact, marked as superseded")

    def resolve_conflict(
        self,
        conflict: Conflict,
        strategy: Optional[ResolutionStrategy] = None,
        auto_resolve: bool = True,
    ) -> Dict[str, Any]:
        """Resolve a conflict, optionally attempting auto-resolution first."""
        if auto_resolve and strategy is None:
            auto_result = self.auto_resolve(conflict)
            if auto_result.get("auto_resolved"):
                return auto_result

        strategy = strategy or conflict.suggested_resolution

        result: Dict[str, Any] = {
            "conflict": conflict.to_dict(),
            "strategy": strategy.value,
            "auto_resolved": False,
            "actions_taken": []
        }

        if strategy == ResolutionStrategy.KEEP_EXISTING:
            result["actions_taken"].append("Rejected new assertion")

        elif strategy == ResolutionStrategy.ACCEPT_NEW:
            self.apply_confidence_decay(
                conflict.existing_item.item_id,
                decay_factor=0.5,
                reason="manual_resolution:accept_new",
                conflicting_fact=conflict.new_fact
            )
            result["actions_taken"].append(
                f"Decayed confidence of existing fact {conflict.existing_item.item_id}"
            )

        elif strategy == ResolutionStrategy.KEEP_BOTH:
            conflict.existing_item.metadata['has_conflict'] = True
            conflict.existing_item.metadata['conflicting_assertion'] = conflict.new_fact
            self.sm.update(conflict.existing_item)
            result["actions_taken"].append("Marked existing fact as having conflict")

        elif strategy == ResolutionStrategy.MERGE:
            result["actions_taken"].append("Merge requested - requires manual review")

        elif strategy == ResolutionStrategy.DEFER:
            conflict.existing_item.metadata['needs_review'] = True
            conflict.existing_item.metadata['review_reason'] = conflict.explanation
            self.sm.update(conflict.existing_item)
            result["actions_taken"].append("Flagged for human review")

        return result
