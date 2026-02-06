"""ReasoningValidator — LLM-based validation for entity type promotion candidates.

Only invoked when PromotionConfig.reasoning_validation=True and the candidate passes
all statistical pre-filters (frequency, consistency).
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Literal

if TYPE_CHECKING:
    from smartmemory.ontology.promotion import PromotionCandidate

logger = logging.getLogger(__name__)

VALIDATION_PROMPT = """You are an ontology expert. Evaluate whether the following entity type should be promoted
to a confirmed type in a knowledge graph.

Entity name: {entity_name}
Proposed type: {entity_type}
Observation frequency: {frequency}
Average confidence: {avg_confidence:.2f}
Type consistency: {consistency:.2f}

Respond with a JSON object:
{{"verdict": "accept" or "reject", "explanation": "brief reason"}}"""


@dataclass
class ReasoningValidationResult:
    """Result of LLM-based validation."""

    is_valid: bool
    verdict: Literal["accept", "reject"]
    explanation: str
    reasoning_trace: Any = None


class ReasoningValidator:
    """Validate promotion candidates via LLM reasoning.

    Builds a prompt with candidate stats, calls LLM for structured reasoning,
    and optionally stores the trace as a 'reasoning' memory.
    """

    def __init__(self, smart_memory=None, lm=None):
        """Args:
        smart_memory: Optional SmartMemory instance for storing reasoning traces.
        lm: Optional language model callable (for testing). If None, uses DSPy default.
        """
        self._smart_memory = smart_memory
        self._lm = lm

    def validate(self, candidate: PromotionCandidate, stats: Dict[str, Any]) -> ReasoningValidationResult:
        """Validate a promotion candidate with LLM reasoning.

        Args:
            candidate: The PromotionCandidate to validate.
            stats: Dict with 'frequency', 'assignments', and other context.

        Returns:
            ReasoningValidationResult with verdict and explanation.
        """
        frequency = stats.get("frequency", 0)
        assignments = stats.get("assignments", [])
        total = sum(a.get("count", 1) for a in assignments) if assignments else 0
        same_type = (
            sum(a.get("count", 1) for a in assignments if a.get("type", "").lower() == candidate.entity_type.lower())
            if assignments
            else 0
        )
        consistency = same_type / total if total > 0 else 1.0

        prompt = VALIDATION_PROMPT.format(
            entity_name=candidate.entity_name,
            entity_type=candidate.entity_type,
            frequency=frequency,
            avg_confidence=candidate.confidence,
            consistency=consistency,
        )

        try:
            response = self._call_llm(prompt)
            verdict, explanation = self._parse_response(response)
        except Exception as e:
            logger.warning("Reasoning validation failed, defaulting to accept: %s", e)
            return ReasoningValidationResult(
                is_valid=True, verdict="accept", explanation=f"Validation error, defaulting to accept: {e}"
            )

        # Build reasoning trace
        trace = self._build_trace(candidate, verdict, explanation, stats)

        # Store trace if SmartMemory is available
        if self._smart_memory is not None and trace is not None:
            try:
                self._smart_memory.add(
                    {
                        "content": explanation,
                        "memory_type": "reasoning",
                        "metadata": {
                            "reasoning_domain": "ontology",
                            "entity_name": candidate.entity_name,
                            "entity_type": candidate.entity_type,
                            "promotion_decision": verdict,
                            "trace_id": trace.trace_id if hasattr(trace, "trace_id") else str(uuid.uuid4()),
                        },
                    }
                )
            except Exception as e:
                logger.debug("Failed to store reasoning trace: %s", e)

        return ReasoningValidationResult(
            is_valid=(verdict == "accept"),
            verdict=verdict,
            explanation=explanation,
            reasoning_trace=trace,
        )

    def _call_llm(self, prompt: str) -> str:
        """Call LLM with the prompt. Uses injected lm or DSPy default."""
        if self._lm is not None:
            return self._lm(prompt)

        try:
            import dspy

            predictor = dspy.Predict("prompt -> response")
            result = predictor(prompt=prompt)
            return result.response
        except Exception as e:
            raise RuntimeError(f"LLM call failed: {e}") from e

    def _parse_response(self, response: str) -> tuple[str, str]:
        """Parse LLM response into (verdict, explanation)."""
        import json

        # Try JSON parse first
        try:
            data = json.loads(response)
            verdict = data.get("verdict", "accept").lower()
            explanation = data.get("explanation", "No explanation provided")
            if verdict not in ("accept", "reject"):
                verdict = "accept"
            return verdict, explanation
        except (json.JSONDecodeError, AttributeError):
            pass

        # Fallback: look for keywords
        lower = response.lower()
        if "reject" in lower:
            return "reject", response.strip()
        return "accept", response.strip()

    def _build_trace(self, candidate: PromotionCandidate, verdict: str, explanation: str, stats: Dict) -> Any:
        """Build a ReasoningTrace for the validation decision."""
        try:
            from smartmemory.models.reasoning import ReasoningTrace, ReasoningStep

            trace = ReasoningTrace(
                trace_id=str(uuid.uuid4()),
                steps=[
                    ReasoningStep(
                        type="thought",
                        content=f"Evaluating promotion of '{candidate.entity_name}' as '{candidate.entity_type}'",
                    ),
                    ReasoningStep(
                        type="observation",
                        content=f"Frequency: {stats.get('frequency', 0)}, Confidence: {candidate.confidence:.2f}",
                    ),
                    ReasoningStep(
                        type="conclusion",
                        content=f"Verdict: {verdict} — {explanation}",
                    ),
                ],
            )
            return trace
        except ImportError:
            return None
