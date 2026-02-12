"""CFS-2: Automatic Procedure Matching.

When content is ingested, search stored procedures for a semantic match.
If confidence meets or exceeds a threshold (default 0.85), auto-select a lighter
pipeline profile, saving tokens and time.

Opt-in by default — matching is disabled unless explicitly enabled.
"""

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Optional
from uuid import uuid4

if TYPE_CHECKING:
    from smartmemory.smart_memory import SmartMemory

logger = logging.getLogger(__name__)

# Default mapping from procedure type metadata to pipeline profile name
DEFAULT_PROFILE_MAPPING: Dict[str, str] = {
    "extraction": "quick_extract",
    "summarization": "quick_extract",
    "classification": "quick_extract",
    "lookup": "quick_extract",
    "storage": "quick_extract",
}


@dataclass
class ProcedureMatchResult:
    """Result of attempting to match ingested content against stored procedures."""

    matched: bool = False
    procedure_id: Optional[str] = None
    procedure_name: Optional[str] = None
    confidence: float = 0.0
    recommended_profile: Optional[str] = None
    threshold: float = 0.85
    match_id: str = field(default_factory=lambda: str(uuid4()))
    drift_detected: bool = False
    drift_event_id: Optional[str] = None
    effective_confidence: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "matched": self.matched,
            "procedure_id": self.procedure_id,
            "procedure_name": self.procedure_name,
            "confidence": self.confidence,
            "recommended_profile": self.recommended_profile,
            "threshold": self.threshold,
            "match_id": self.match_id,
            "drift_detected": self.drift_detected,
            "drift_event_id": self.drift_event_id,
            "effective_confidence": self.effective_confidence,
        }


@dataclass
class ProcedureMatcherConfig:
    """Configuration for the procedure matcher."""

    enabled: bool = False  # Opt-in
    confidence_threshold: float = 0.85
    max_candidates: int = 3
    profile_mapping: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_PROFILE_MAPPING))


class ProcedureMatcher:
    """Matches ingested content against stored procedural memories.

    Uses semantic search over procedural memories to find a matching procedure.
    When a match meets or exceeds the confidence threshold, recommends a lighter pipeline
    profile to reduce token spend.
    """

    def __init__(self, smart_memory: "SmartMemory", config: Optional[ProcedureMatcherConfig] = None):
        self._memory = smart_memory
        self._config = config or ProcedureMatcherConfig()

    @property
    def config(self) -> ProcedureMatcherConfig:
        return self._config

    def match(self, content: str) -> ProcedureMatchResult:
        """Search for a procedural memory that matches the given content.

        Args:
            content: The text being ingested.

        Returns:
            ProcedureMatchResult with match details (or matched=False).
        """
        if not self._config.enabled:
            return ProcedureMatchResult(threshold=self._config.confidence_threshold)

        if not content or not content.strip():
            return ProcedureMatchResult(threshold=self._config.confidence_threshold)

        try:
            candidates = self._memory.search(
                query=content[:500],
                top_k=self._config.max_candidates,
                memory_type="procedural",
            )
        except Exception as e:
            logger.debug("CFS-2: Procedure search failed: %s", e)
            return ProcedureMatchResult(threshold=self._config.confidence_threshold)

        best = self._score_candidates(content, candidates)
        if best is None:
            return ProcedureMatchResult(threshold=self._config.confidence_threshold)

        confidence = best.get("score", 0.0)
        if confidence < self._config.confidence_threshold:
            return ProcedureMatchResult(
                confidence=confidence,
                threshold=self._config.confidence_threshold,
            )

        procedure_id = best.get("item_id") or best.get("id")
        if not procedure_id:
            logger.warning("CFS-2: Best candidate above threshold has no item_id or id, skipping match")
            return ProcedureMatchResult(
                confidence=confidence,
                threshold=self._config.confidence_threshold,
            )

        procedure_name = best.get("content", "")[:100]
        recommended_profile = self._recommend_profile(best)

        return ProcedureMatchResult(
            matched=True,
            procedure_id=procedure_id,
            procedure_name=procedure_name,
            confidence=confidence,
            recommended_profile=recommended_profile,
            threshold=self._config.confidence_threshold,
        )

    def _score_candidates(self, content: str, candidates) -> Optional[dict]:  # noqa: ARG002
        """Pick the best candidate from search results.

        Reuses similarity scores already computed by the search engine.
        content is accepted for future re-ranking but currently unused.
        """
        if not candidates:
            return None

        # candidates may be a list of dicts or objects with .score
        best = None
        best_score = -1.0
        for c in candidates:
            if isinstance(c, dict):
                candidate = c
                score = c.get("score", 0.0)
            else:
                score = getattr(c, "score", 0.0)
                candidate = {
                    "item_id": getattr(c, "item_id", None),
                    "content": getattr(c, "content", ""),
                    "score": score,
                    "metadata": getattr(c, "metadata", {}),
                }

            if score > best_score:
                best_score = score
                best = candidate

        return best

    def _recommend_profile(self, procedure: dict) -> str:
        """Determine the pipeline profile to recommend for a matched procedure.

        Priority:
        1. procedure.metadata.preferred_profile (explicit override)
        2. profile_mapping[procedure_type] (type-based)
        3. "quick_extract" (fallback)
        """
        metadata = procedure.get("metadata") or {}

        preferred = metadata.get("preferred_profile")
        if preferred:
            return preferred

        procedure_type = metadata.get("procedure_type", "")
        return self._config.profile_mapping.get(procedure_type, "quick_extract")
