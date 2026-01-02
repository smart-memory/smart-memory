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
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Set
from enum import Enum

from smartmemory.models.memory_item import MemoryItem

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
            "suggested_resolution": self.suggested_resolution.value
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
            "overall_confidence": self.overall_confidence
        }


class DetectionMethod(Enum):
    """Available contradiction detection methods."""
    LLM = "llm"  # Most accurate, slowest
    GRAPH = "graph"  # Structural analysis
    EMBEDDING = "embedding"  # Semantic + polarity
    HEURISTIC = "heuristic"  # Pattern matching (least reliable)


class AssertionChallenger:
    """
    Challenges new assertions against existing knowledge to detect contradictions.
    
    Uses a multi-stage approach:
    1. Semantic search to find related facts
    2. Cascade of detection methods (LLM → Graph → Embedding → Heuristic)
    3. Confidence scoring and resolution suggestions
    
    Detection methods (in order of reliability):
    - LLM: Uses GPT to analyze contradiction (most accurate, slowest)
    - Graph: Analyzes knowledge graph structure for functional property conflicts
    - Embedding: High semantic similarity + opposite polarity detection
    - Heuristic: Simple pattern matching (fallback, least reliable)
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
        """
        Initialize the Assertion Challenger.
        
        Args:
            smart_memory: SmartMemory instance for searching existing facts
            similarity_threshold: Minimum similarity to consider facts related
            max_related_facts: Maximum number of related facts to check
            use_llm: Use LLM-based detection (most accurate, slowest)
            use_graph: Use graph structure analysis
            use_embedding: Use embedding similarity + polarity
            use_heuristic: Use pattern matching (fallback)
        """
        self.sm = smart_memory
        self.similarity_threshold = similarity_threshold
        self.max_related_facts = max_related_facts
        self.use_llm = use_llm
        self.use_graph = use_graph
        self.use_embedding = use_embedding
        self.use_heuristic = use_heuristic
        
    def challenge(
        self,
        assertion: str,
        memory_type: Optional[str] = "semantic",
        context: Optional[Dict[str, Any]] = None
    ) -> ChallengeResult:
        """
        Challenge an assertion against existing knowledge.
        
        Args:
            assertion: The new fact/assertion to challenge
            memory_type: Type of memory to search (default: semantic)
            context: Optional context for the assertion
            
        Returns:
            ChallengeResult with any detected conflicts
        """
        logger.info(f"Challenging assertion: {assertion[:100]}...")
        
        # 1. Find related facts using semantic search
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
        
        # 2. Detect contradictions
        conflicts = []
        for fact in related_facts:
            conflict = self._detect_contradiction(assertion, fact, context)
            if conflict:
                conflicts.append(conflict)
        
        # 3. Calculate overall confidence
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
    
    def _detect_contradiction(
        self,
        new_assertion: str,
        existing_item: MemoryItem,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[Conflict]:
        """
        Detect if the new assertion contradicts an existing fact.
        
        Uses a cascade of detection methods (most reliable first):
        1. LLM-based (if enabled) - most accurate but slowest
        2. Graph-based - uses knowledge graph structure
        3. Embedding-based - semantic similarity + polarity
        4. Heuristic - simple pattern matching (fallback)
        
        Returns the first conflict found, or None if no contradiction detected.
        """
        existing_fact = existing_item.content
        
        # 1. Try LLM-based detection (most accurate)
        if self.use_llm:
            conflict = self._detect_contradiction_llm(
                new_assertion, existing_item, existing_fact, context
            )
            if conflict:
                conflict.explanation = f"[LLM] {conflict.explanation}"
                return conflict
        
        # 2. Try graph-based detection (structural analysis)
        if self.use_graph:
            conflict = self._detect_contradiction_graph(new_assertion, existing_item)
            if conflict:
                conflict.explanation = f"[Graph] {conflict.explanation}"
                return conflict
        
        # 3. Try embedding-based detection (semantic + polarity)
        if self.use_embedding:
            conflict = self._detect_contradiction_embedding(new_assertion, existing_item)
            if conflict:
                conflict.explanation = f"[Embedding] {conflict.explanation}"
                return conflict
        
        # 4. Fall back to heuristics (least reliable)
        if self.use_heuristic:
            conflict = self._detect_contradiction_heuristic(
                new_assertion, existing_item, existing_fact
            )
            if conflict:
                conflict.explanation = f"[Heuristic] {conflict.explanation}"
                return conflict
        
        return None
    
    def _detect_contradiction_llm(
        self,
        new_assertion: str,
        existing_item: MemoryItem,
        existing_fact: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[Conflict]:
        """Use LLM to detect contradictions with reasoning."""
        try:
            from smartmemory.utils.llm import call_llm
            
            prompt = f"""Analyze if these two statements contradict each other.

Statement A (existing fact): {existing_fact}

Statement B (new assertion): {new_assertion}

Respond in JSON format:
{{
    "contradicts": true/false,
    "conflict_type": "direct_contradiction" | "temporal_conflict" | "numeric_mismatch" | "entity_confusion" | "partial_overlap" | "none",
    "confidence": 0.0-1.0,
    "explanation": "Brief explanation of why they do or don't contradict",
    "resolution": "keep_existing" | "accept_new" | "keep_both" | "merge" | "defer"
}}

Only respond with the JSON, no other text."""

            parsed_result, response = call_llm(
                user_content=prompt,
                max_output_tokens=500,
                response_format={"type": "json_object"}
            )
            
            # Parse response
            import json
            import re
            
            # Use parsed_result if available, otherwise extract JSON from response
            if parsed_result:
                result = parsed_result
            else:
                # Extract JSON from response
                json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
                if not json_match:
                    logger.warning("Could not parse LLM response as JSON")
                    return None
                
                result = json.loads(json_match.group())
            
            if not result.get("contradicts", False):
                return None
            
            conflict_type = ConflictType(result.get("conflict_type", "direct_contradiction"))
            resolution = ResolutionStrategy(result.get("resolution", "defer"))
            
            return Conflict(
                existing_item=existing_item,
                existing_fact=existing_fact,
                new_fact=new_assertion,
                conflict_type=conflict_type,
                confidence=result.get("confidence", 0.8),
                explanation=result.get("explanation", "LLM detected contradiction"),
                suggested_resolution=resolution
            )
            
        except Exception as e:
            logger.warning(f"LLM contradiction detection failed: {e}, using heuristics")
            return self._detect_contradiction_heuristic(
                new_assertion, existing_item, existing_fact
            )
    
    def _detect_contradiction_graph(
        self,
        new_assertion: str,
        existing_item: MemoryItem,
    ) -> Optional[Conflict]:
        """
        Detect contradictions using knowledge graph structure.
        
        Analyzes:
        1. Entity overlap - same entities with conflicting relations
        2. Functional properties - relations that can only have one value
        3. Mutual exclusivity - relations that can't coexist
        """
        try:
            # Get entities from existing item
            existing_entities = set()
            if hasattr(existing_item, 'metadata') and existing_item.metadata:
                # Check for extracted entities in metadata
                entities = existing_item.metadata.get('entities', [])
                for ent in entities:
                    if isinstance(ent, dict):
                        existing_entities.add(ent.get('name', '').lower())
                    elif isinstance(ent, str):
                        existing_entities.add(ent.lower())
            
            # Also extract entity-like words from content
            existing_content_words = set(existing_item.content.lower().split())
            
            # Extract from new assertion
            new_words = set(new_assertion.lower().split())
            
            # Find common entities/subjects
            common_words = existing_content_words & new_words
            # Filter out stopwords
            stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'of', 'in', 'to', 'and', 'or'}
            common_entities = common_words - stopwords
            
            if len(common_entities) < 2:
                # Not enough overlap to detect graph-based contradiction
                return None
            
            # Check for functional property conflicts (e.g., "capital of X" can only be one city)
            functional_patterns = [
                'capital of', 'president of', 'ceo of', 'founder of',
                'born in', 'died in', 'located in', 'headquarters in'
            ]
            
            existing_lower = existing_item.content.lower()
            new_lower = new_assertion.lower()
            
            for pattern in functional_patterns:
                if pattern in existing_lower and pattern in new_lower:
                    # Both mention the same functional property
                    # Extract the values after the pattern
                    import re
                    existing_match = re.search(rf'{pattern}\s+(\w+(?:\s+\w+)?)', existing_lower)
                    new_match = re.search(rf'{pattern}\s+(\w+(?:\s+\w+)?)', new_lower)
                    
                    if existing_match and new_match:
                        existing_value = existing_match.group(1)
                        new_value = new_match.group(1)
                        
                        if existing_value != new_value:
                            return Conflict(
                                existing_item=existing_item,
                                existing_fact=existing_item.content,
                                new_fact=new_assertion,
                                conflict_type=ConflictType.DIRECT_CONTRADICTION,
                                confidence=0.85,
                                explanation=f"Functional property conflict: '{pattern}' has different values: '{existing_value}' vs '{new_value}'",
                                suggested_resolution=ResolutionStrategy.DEFER
                            )
            
            return None
            
        except Exception as e:
            logger.debug(f"Graph-based contradiction detection failed: {e}")
            return None

    def _detect_contradiction_embedding(
        self,
        new_assertion: str,
        existing_item: MemoryItem,
    ) -> Optional[Conflict]:
        """
        Detect contradictions using embedding similarity + polarity analysis.
        
        High semantic similarity + opposite polarity = likely contradiction.
        """
        try:
            from smartmemory.plugins.embedding import create_embeddings
            import numpy as np
            
            # Get embeddings
            new_embedding = create_embeddings(new_assertion)
            existing_embedding = create_embeddings(existing_item.content)
            
            if new_embedding is None or existing_embedding is None:
                return None
            
            # Calculate cosine similarity
            if hasattr(new_embedding, 'tolist'):
                new_embedding = np.array(new_embedding)
            if hasattr(existing_embedding, 'tolist'):
                existing_embedding = np.array(existing_embedding)
            
            dot = np.dot(new_embedding, existing_embedding)
            norm_new = np.linalg.norm(new_embedding)
            norm_existing = np.linalg.norm(existing_embedding)
            
            if norm_new == 0 or norm_existing == 0:
                return None
            
            similarity = dot / (norm_new * norm_existing)
            
            # High similarity (>0.8) means they're about the same thing
            if similarity < 0.75:
                return None  # Not similar enough to be contradictory
            
            # Check for polarity difference using simple indicators
            new_lower = new_assertion.lower()
            existing_lower = existing_item.content.lower()
            
            # Polarity indicators
            negative_words = {'not', "n't", 'never', 'no', 'none', 'neither', 'nobody', 
                           'nothing', 'nowhere', 'cannot', "can't", "won't", "wouldn't",
                           "shouldn't", "couldn't", "doesn't", "didn't", "isn't", "aren't",
                           "wasn't", "weren't", 'false', 'incorrect', 'wrong', 'impossible'}
            
            new_has_negative = any(neg in new_lower.split() or neg in new_lower for neg in negative_words)
            existing_has_negative = any(neg in existing_lower.split() or neg in existing_lower for neg in negative_words)
            
            # XOR - one has negative, other doesn't = potential contradiction
            if new_has_negative != existing_has_negative:
                return Conflict(
                    existing_item=existing_item,
                    existing_fact=existing_item.content,
                    new_fact=new_assertion,
                    conflict_type=ConflictType.DIRECT_CONTRADICTION,
                    confidence=min(0.9, similarity),  # Confidence based on similarity
                    explanation=f"High semantic similarity ({similarity:.2f}) with opposite polarity detected",
                    suggested_resolution=ResolutionStrategy.DEFER
                )
            
            return None
            
        except Exception as e:
            logger.debug(f"Embedding-based contradiction detection failed: {e}")
            return None

    def _detect_contradiction_heuristic(
        self,
        new_assertion: str,
        existing_item: MemoryItem,
        existing_fact: str
    ) -> Optional[Conflict]:
        """
        Detect contradictions using simple heuristics (fallback).
        
        Note: This is the least reliable method. Prefer graph or embedding-based detection.
        """
        new_lower = new_assertion.lower()
        existing_lower = existing_fact.lower()
        
        # Check for direct negation
        negation_patterns = [
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
        
        for neg, pos in negation_patterns:
            # Check if one has negation and other has positive
            if (neg in new_lower and pos in existing_lower and neg not in existing_lower):
                return Conflict(
                    existing_item=existing_item,
                    existing_fact=existing_fact,
                    new_fact=new_assertion,
                    conflict_type=ConflictType.DIRECT_CONTRADICTION,
                    confidence=0.7,
                    explanation=f"Detected negation pattern: '{neg}' vs '{pos}'",
                    suggested_resolution=ResolutionStrategy.DEFER
                )
            if (neg in existing_lower and pos in new_lower and neg not in new_lower):
                return Conflict(
                    existing_item=existing_item,
                    existing_fact=existing_fact,
                    new_fact=new_assertion,
                    conflict_type=ConflictType.DIRECT_CONTRADICTION,
                    confidence=0.7,
                    explanation=f"Detected negation pattern: '{pos}' vs '{neg}'",
                    suggested_resolution=ResolutionStrategy.DEFER
                )
        
        # Check for numeric differences
        import re
        new_numbers = set(re.findall(r'\b\d+(?:\.\d+)?\b', new_assertion))
        existing_numbers = set(re.findall(r'\b\d+(?:\.\d+)?\b', existing_fact))
        
        if new_numbers and existing_numbers:
            # If both have numbers but they differ significantly
            common_context = self._extract_common_context(new_lower, existing_lower)
            if common_context and new_numbers != existing_numbers:
                return Conflict(
                    existing_item=existing_item,
                    existing_fact=existing_fact,
                    new_fact=new_assertion,
                    conflict_type=ConflictType.NUMERIC_MISMATCH,
                    confidence=0.6,
                    explanation=f"Numeric values differ: {existing_numbers} vs {new_numbers}",
                    suggested_resolution=ResolutionStrategy.DEFER
                )
        
        return None
    
    def _extract_common_context(self, text1: str, text2: str) -> Optional[str]:
        """Extract common context words between two texts."""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        # Remove common stopwords
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 
                     'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                     'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                     'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                     'as', 'into', 'through', 'during', 'before', 'after', 'above',
                     'below', 'between', 'under', 'again', 'further', 'then', 'once',
                     'and', 'but', 'or', 'nor', 'so', 'yet', 'both', 'either', 'neither',
                     'not', 'only', 'own', 'same', 'than', 'too', 'very', 'just'}
        
        common = (words1 & words2) - stopwords
        
        if len(common) >= 2:
            return " ".join(common)
        return None
    
    def _calculate_confidence(self, conflicts: List[Conflict]) -> float:
        """
        Calculate overall confidence in the new assertion.
        
        Lower confidence = more/stronger conflicts.
        """
        if not conflicts:
            return 1.0
        
        # Weight by conflict confidence
        total_conflict_weight = sum(c.confidence for c in conflicts)
        
        # More conflicts = lower confidence
        # Each conflict reduces confidence proportionally
        confidence = max(0.0, 1.0 - (total_conflict_weight / len(conflicts)) * 0.5)
        
        return round(confidence, 3)
    
    def apply_confidence_decay(
        self,
        item_id: str,
        decay_factor: float = 0.1,
        reason: Optional[str] = None,
        conflicting_fact: Optional[str] = None
    ) -> bool:
        """
        Apply confidence decay to a challenged fact with full tracking.
        
        Args:
            item_id: ID of the memory item to decay
            decay_factor: How much to reduce confidence (0.0-1.0)
            reason: Why the decay was applied
            conflicting_fact: The fact that caused the challenge
            
        Returns:
            True if decay was applied successfully
            
        Tracks:
            - confidence: Current confidence score (0.0-1.0)
            - challenged: Boolean flag
            - challenge_count: Number of times challenged
            - confidence_history: List of {timestamp, old, new, reason, conflicting_fact}
            - last_challenged_at: ISO timestamp of last challenge
        """
        from datetime import datetime
        
        try:
            item = self.sm.get(item_id)
            if not item:
                logger.warning(f"Item {item_id} not found for confidence decay")
                return False
            
            current_confidence = item.metadata.get('confidence', 1.0)
            new_confidence = max(0.0, current_confidence - decay_factor)
            now = datetime.utcnow().isoformat() + 'Z'
            
            # Initialize confidence history if not present
            if 'confidence_history' not in item.metadata:
                item.metadata['confidence_history'] = []
            
            # Record this decay event
            decay_event = {
                'timestamp': now,
                'old_confidence': current_confidence,
                'new_confidence': new_confidence,
                'decay_factor': decay_factor,
                'reason': reason or 'challenged',
            }
            if conflicting_fact:
                decay_event['conflicting_fact'] = conflicting_fact[:200]  # Truncate
            
            item.metadata['confidence_history'].append(decay_event)
            
            # Keep only last 20 events to avoid bloat
            if len(item.metadata['confidence_history']) > 20:
                item.metadata['confidence_history'] = item.metadata['confidence_history'][-20:]
            
            # Update current state
            item.metadata['confidence'] = new_confidence
            item.metadata['challenged'] = True
            item.metadata['challenge_count'] = item.metadata.get('challenge_count', 0) + 1
            item.metadata['last_challenged_at'] = now
            
            self.sm.update(item)
            
            logger.info(
                f"Applied confidence decay to {item_id}: "
                f"{current_confidence:.2f} -> {new_confidence:.2f} "
                f"(challenge #{item.metadata['challenge_count']})"
            )
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply confidence decay: {e}")
            return False
    
    def get_confidence_history(self, item_id: str) -> List[Dict[str, Any]]:
        """
        Get the confidence decay history for an item.
        
        Args:
            item_id: ID of the memory item
            
        Returns:
            List of decay events with timestamps
        """
        try:
            item = self.sm.get(item_id)
            if not item:
                return []
            
            return item.metadata.get('confidence_history', [])
        except Exception as e:
            logger.error(f"Failed to get confidence history: {e}")
            return []
    
    def get_low_confidence_items(
        self,
        threshold: float = 0.5,
        memory_type: str = "semantic",
        limit: int = 50
    ) -> List[MemoryItem]:
        """
        Get items with confidence below threshold.
        
        Useful for finding facts that have been challenged multiple times
        and may need review or removal.
        
        Args:
            threshold: Confidence threshold (items below this are returned)
            memory_type: Type of memory to search
            limit: Maximum items to return
            
        Returns:
            List of low-confidence items sorted by confidence (lowest first)
        """
        try:
            # Search for all items and filter by confidence
            # TODO: This could be optimized with a graph query
            results = self.sm.search("", top_k=limit * 3, memory_type=memory_type)
            
            low_confidence = []
            for item in results:
                confidence = item.metadata.get('confidence', 1.0)
                if confidence < threshold:
                    low_confidence.append(item)
            
            # Sort by confidence (lowest first)
            low_confidence.sort(key=lambda x: x.metadata.get('confidence', 1.0))
            
            return low_confidence[:limit]
            
        except Exception as e:
            logger.error(f"Failed to get low confidence items: {e}")
            return []
    
    def auto_resolve(
        self,
        conflict: Conflict,
        use_wikipedia: bool = True,
        use_llm: bool = True,
    ) -> Dict[str, Any]:
        """
        Attempt to automatically resolve a conflict using external knowledge.
        
        Resolution cascade:
        1. Wikipedia lookup - Check if either fact is grounded in Wikipedia
        2. LLM reasoning - Ask LLM to determine which is correct
        3. Grounding check - Check existing provenance/grounding edges
        4. Recency heuristic - Prefer more recent information (if temporal)
        
        Args:
            conflict: The conflict to resolve
            use_wikipedia: Whether to lookup Wikipedia for verification
            use_llm: Whether to use LLM for reasoning
            
        Returns:
            Dict with resolution result and confidence
        """
        result = {
            "conflict": conflict.to_dict(),
            "auto_resolved": False,
            "resolution": None,
            "confidence": 0.0,
            "method": None,
            "evidence": None,
            "actions_taken": []
        }
        
        # 1. Try Wikipedia verification
        if use_wikipedia:
            wiki_result = self._resolve_via_wikipedia(conflict)
            if wiki_result:
                result.update(wiki_result)
                if result["auto_resolved"]:
                    self._apply_resolution(conflict, result)
                    return result
        
        # 2. Try LLM reasoning with external knowledge
        if use_llm:
            llm_result = self._resolve_via_llm_reasoning(conflict)
            if llm_result:
                result.update(llm_result)
                if result["auto_resolved"]:
                    self._apply_resolution(conflict, result)
                    return result
        
        # 3. Check existing grounding/provenance
        grounding_result = self._resolve_via_grounding(conflict)
        if grounding_result:
            result.update(grounding_result)
            if result["auto_resolved"]:
                self._apply_resolution(conflict, result)
                return result
        
        # 4. Recency heuristic for temporal conflicts
        if conflict.conflict_type == ConflictType.TEMPORAL_CONFLICT:
            recency_result = self._resolve_via_recency(conflict)
            if recency_result:
                result.update(recency_result)
                if result["auto_resolved"]:
                    self._apply_resolution(conflict, result)
                    return result
        
        # Could not auto-resolve, defer to human
        result["actions_taken"].append("Auto-resolution failed, deferring to human review")
        return result
    
    def _resolve_via_wikipedia(self, conflict: Conflict) -> Optional[Dict[str, Any]]:
        """Verify facts against Wikipedia."""
        try:
            # Extract key entities from both facts
            entities = self._extract_entities_for_lookup(
                conflict.existing_fact, 
                conflict.new_fact
            )
            
            if not entities:
                return None
            
            # Try Wikipedia lookup
            from smartmemory.plugins.grounders.wikipedia import WikipediaGrounder
            grounder = WikipediaGrounder()
            
            for entity in entities[:3]:  # Limit lookups
                try:
                    wiki_result = grounder.lookup(entity)
                    if not wiki_result:
                        continue
                    
                    wiki_text = wiki_result.get('summary', '') or wiki_result.get('extract', '')
                    if not wiki_text:
                        continue
                    
                    # Check which fact aligns with Wikipedia
                    existing_match = self._text_supports_claim(wiki_text, conflict.existing_fact)
                    new_match = self._text_supports_claim(wiki_text, conflict.new_fact)
                    
                    if existing_match and not new_match:
                        return {
                            "auto_resolved": True,
                            "resolution": ResolutionStrategy.KEEP_EXISTING,
                            "confidence": 0.85,
                            "method": "wikipedia",
                            "evidence": f"Wikipedia article '{entity}' supports existing fact",
                            "actions_taken": [f"Verified via Wikipedia: {entity}"]
                        }
                    elif new_match and not existing_match:
                        return {
                            "auto_resolved": True,
                            "resolution": ResolutionStrategy.ACCEPT_NEW,
                            "confidence": 0.85,
                            "method": "wikipedia",
                            "evidence": f"Wikipedia article '{entity}' supports new fact",
                            "actions_taken": [f"Verified via Wikipedia: {entity}"]
                        }
                        
                except Exception as e:
                    logger.debug(f"Wikipedia lookup failed for '{entity}': {e}")
                    continue
            
            return None
            
        except Exception as e:
            logger.debug(f"Wikipedia resolution failed: {e}")
            return None
    
    def _resolve_via_llm_reasoning(self, conflict: Conflict) -> Optional[Dict[str, Any]]:
        """Use LLM to reason about which fact is correct."""
        try:
            from smartmemory.utils.llm import call_llm
            
            prompt = f"""You are a fact-checker. Two statements conflict with each other. 
Determine which one is more likely to be correct based on your knowledge.

Statement A (existing): {conflict.existing_fact}
Statement B (new): {conflict.new_fact}

Respond in JSON format:
{{
    "correct_statement": "A" or "B" or "unknown",
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation of why",
    "source": "What knowledge you used to determine this"
}}

If you cannot determine which is correct with at least 70% confidence, respond with "unknown".
Only respond with the JSON, no other text."""

            parsed_result, response = call_llm(
                user_content=prompt,
                max_output_tokens=500,
                response_format={"type": "json_object"}
            )
            
            # Parse response
            import json
            if parsed_result:
                result = parsed_result
            else:
                json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
                if not json_match:
                    return None
                
                result = json.loads(json_match.group())
            
            correct = result.get("correct_statement", "unknown")
            confidence = float(result.get("confidence", 0.0))
            
            # Only auto-resolve if confidence is high enough
            if confidence < 0.7 or correct == "unknown":
                return None
            
            if correct == "A":
                return {
                    "auto_resolved": True,
                    "resolution": ResolutionStrategy.KEEP_EXISTING,
                    "confidence": confidence,
                    "method": "llm_reasoning",
                    "evidence": result.get("reasoning", "LLM determined existing fact is correct"),
                    "actions_taken": [f"LLM reasoning: {result.get('source', 'general knowledge')}"]
                }
            elif correct == "B":
                return {
                    "auto_resolved": True,
                    "resolution": ResolutionStrategy.ACCEPT_NEW,
                    "confidence": confidence,
                    "method": "llm_reasoning",
                    "evidence": result.get("reasoning", "LLM determined new fact is correct"),
                    "actions_taken": [f"LLM reasoning: {result.get('source', 'general knowledge')}"]
                }
            
            return None
            
        except Exception as e:
            logger.debug(f"LLM reasoning resolution failed: {e}")
            return None
    
    def _resolve_via_grounding(self, conflict: Conflict) -> Optional[Dict[str, Any]]:
        """Check if either fact has existing grounding/provenance."""
        try:
            existing_metadata = conflict.existing_item.metadata or {}
            
            # Check for grounding edges
            grounded = existing_metadata.get('grounded_to')
            provenance = existing_metadata.get('provenance', {})
            
            # If existing fact is grounded to Wikipedia, trust it more
            if grounded or provenance.get('wikipedia_id'):
                return {
                    "auto_resolved": True,
                    "resolution": ResolutionStrategy.KEEP_EXISTING,
                    "confidence": 0.75,
                    "method": "grounding",
                    "evidence": f"Existing fact has provenance: {grounded or provenance}",
                    "actions_taken": ["Existing fact has verified grounding"]
                }
            
            # Check source reliability
            source = provenance.get('source', '')
            trusted_sources = {'wikipedia', 'wikidata', 'official', 'verified', 'authoritative'}
            if any(ts in source.lower() for ts in trusted_sources):
                return {
                    "auto_resolved": True,
                    "resolution": ResolutionStrategy.KEEP_EXISTING,
                    "confidence": 0.7,
                    "method": "grounding",
                    "evidence": f"Existing fact from trusted source: {source}",
                    "actions_taken": [f"Trusted source: {source}"]
                }
            
            return None
            
        except Exception as e:
            logger.debug(f"Grounding resolution failed: {e}")
            return None
    
    def _resolve_via_recency(self, conflict: Conflict) -> Optional[Dict[str, Any]]:
        """For temporal conflicts, prefer more recent information."""
        try:
            from datetime import datetime
            
            existing_metadata = conflict.existing_item.metadata or {}
            existing_time = existing_metadata.get('valid_start_time') or existing_metadata.get('timestamp')
            
            # If we have temporal info, newer might be better for temporal conflicts
            if existing_time:
                # Check if new fact mentions "now", "currently", "as of"
                new_lower = conflict.new_fact.lower()
                recency_indicators = ['now', 'currently', 'as of', 'today', 'recent']
                
                if any(ind in new_lower for ind in recency_indicators):
                    return {
                        "auto_resolved": True,
                        "resolution": ResolutionStrategy.ACCEPT_NEW,
                        "confidence": 0.65,
                        "method": "recency",
                        "evidence": "New fact appears to be more recent update",
                        "actions_taken": ["Temporal conflict resolved by recency"]
                    }
            
            return None
            
        except Exception as e:
            logger.debug(f"Recency resolution failed: {e}")
            return None
    
    def _extract_entities_for_lookup(self, *texts: str) -> List[str]:
        """Extract entity names from texts for external lookup."""
        entities = []
        
        for text in texts:
            # Simple extraction: capitalized words that aren't at sentence start
            words = text.split()
            for i, word in enumerate(words):
                # Skip first word (might just be capitalized for sentence)
                if i == 0:
                    continue
                # Check if capitalized (likely proper noun)
                clean_word = word.strip('.,!?";:')
                if clean_word and clean_word[0].isupper():
                    entities.append(clean_word)
        
        # Deduplicate while preserving order
        seen = set()
        unique = []
        for e in entities:
            if e.lower() not in seen:
                seen.add(e.lower())
                unique.append(e)
        
        return unique
    
    def _text_supports_claim(self, reference_text: str, claim: str) -> bool:
        """Check if reference text supports a claim (simple heuristic)."""
        ref_lower = reference_text.lower()
        claim_lower = claim.lower()
        
        # Extract key terms from claim
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'of', 'in', 'to', 'and', 'or'}
        claim_words = [w for w in claim_lower.split() if w not in stopwords and len(w) > 2]
        
        # Check how many claim words appear in reference
        matches = sum(1 for w in claim_words if w in ref_lower)
        
        # If more than 60% of key words match, consider it supporting
        if claim_words and matches / len(claim_words) > 0.6:
            return True
        
        return False
    
    def _apply_resolution(self, conflict: Conflict, resolution: Dict[str, Any]) -> None:
        """Apply the auto-resolution result."""
        strategy = resolution.get("resolution")
        
        if strategy == ResolutionStrategy.KEEP_EXISTING:
            # Mark new fact as rejected, existing as verified
            conflict.existing_item.metadata['auto_verified'] = True
            conflict.existing_item.metadata['verification_method'] = resolution.get("method")
            conflict.existing_item.metadata['verification_evidence'] = resolution.get("evidence")
            self.sm.update(conflict.existing_item)
            resolution["actions_taken"].append("Marked existing fact as verified")
            
        elif strategy == ResolutionStrategy.ACCEPT_NEW:
            # Decay existing fact confidence with tracking
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
        """
        Resolve a conflict, optionally attempting auto-resolution first.
        
        Args:
            conflict: The conflict to resolve
            strategy: Resolution strategy (uses suggested if not provided)
            auto_resolve: If True, attempt auto-resolution before using strategy
            
        Returns:
            Dict with resolution details
        """
        # Try auto-resolution first
        if auto_resolve and strategy is None:
            auto_result = self.auto_resolve(conflict)
            if auto_result.get("auto_resolved"):
                return auto_result
        
        # Fall back to manual strategy
        strategy = strategy or conflict.suggested_resolution
        
        result = {
            "conflict": conflict.to_dict(),
            "strategy": strategy.value,
            "auto_resolved": False,
            "actions_taken": []
        }
        
        if strategy == ResolutionStrategy.KEEP_EXISTING:
            # Just log, don't add new fact
            result["actions_taken"].append("Rejected new assertion")
            
        elif strategy == ResolutionStrategy.ACCEPT_NEW:
            # Mark old as superseded, add new
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
            # Mark both as potentially conflicting
            conflict.existing_item.metadata['has_conflict'] = True
            conflict.existing_item.metadata['conflicting_assertion'] = conflict.new_fact
            self.sm.update(conflict.existing_item)
            result["actions_taken"].append("Marked existing fact as having conflict")
            
        elif strategy == ResolutionStrategy.MERGE:
            # Would need more sophisticated merging logic
            result["actions_taken"].append("Merge requested - requires manual review")
            
        elif strategy == ResolutionStrategy.DEFER:
            # Flag for human review
            conflict.existing_item.metadata['needs_review'] = True
            conflict.existing_item.metadata['review_reason'] = conflict.explanation
            self.sm.update(conflict.existing_item)
            result["actions_taken"].append("Flagged for human review")
        
        return result
