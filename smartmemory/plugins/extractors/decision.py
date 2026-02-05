"""Decision Extractor for Decision Memory.

Extracts decision-like statements from text and reasoning traces.
Two extraction modes:
1. Text extraction: Regex-based detection of decision patterns (preference, choice, belief, etc.)
2. Trace extraction: Extracts decisions from ReasoningTrace steps of type "decision" or "conclusion"

No LLM required - uses keyword heuristics for classification.
"""

import logging
import re
from dataclasses import dataclass

from smartmemory.models.base import MemoryBaseModel
from smartmemory.models.decision import Decision
from smartmemory.models.reasoning import ReasoningTrace
from smartmemory.plugins.base import ExtractorPlugin, PluginMetadata

logger = logging.getLogger(__name__)

# Step types in ReasoningTrace that represent decisions
DECISION_STEP_TYPES = {"decision", "conclusion"}

# Patterns that indicate a decision statement in text.
# Each pattern maps to a list of (regex, decision_type) tuples.
# Regexes are applied case-insensitively.
_SENT = r"(?:^|[.!?]\s+)"  # Sentence boundary anchor
_CAP = r"([^.!?]*\b(?:{})\b[^.!?]*[.!?]?)"  # Capture group template


def _pat(keywords: str, dtype: str) -> tuple[re.Pattern[str], str]:
    """Build a (compiled_pattern, decision_type) tuple."""
    return (re.compile(_SENT + _CAP.format(keywords), re.IGNORECASE), dtype)


DECISION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    _pat(r"i prefer|my preference is|i favor", "preference"),
    _pat(r"my favorite|i like using|i like", "preference"),
    _pat(r"i decided|my decision is|i chose|i choose|i'll go with", "choice"),
    _pat(r"i believe|i think that", "belief"),
    _pat(r"we should always|always|never|the rule is", "policy"),
    _pat(r"therefore|in conclusion|thus|this means", "inference"),
    _pat(r"this is a|this falls into|this belongs to", "classification"),
]

# Keywords for classifying decision type from arbitrary content
TYPE_KEYWORDS: dict[str, list[str]] = {
    "preference": ["prefer", "preference", "favorite", "favour", "favor", "like using", "i like"],
    "choice": ["decided", "decision", "chose", "choose", "go with", "selected", "picked"],
    "belief": ["believe", "i think"],
    "policy": ["should always", "always", "never", "rule is", "must"],
    "inference": ["therefore", "this means", "in conclusion", "thus", "hence", "consequently"],
    "classification": ["this is a", "falls into", "belongs to", "categorize", "classify"],
}


@dataclass
class DecisionExtractorConfig(MemoryBaseModel):
    """Configuration for the decision extractor."""

    min_confidence: float = 0.7
    min_content_length: int = 20


class DecisionExtractor(ExtractorPlugin):
    """Extract decision statements from text and reasoning traces.

    Returns standard extractor format with decisions in metadata:
    {
        'entities': [],
        'relations': [],
        'decisions': [Decision, ...],
    }
    """

    @classmethod
    def metadata(cls) -> PluginMetadata:
        return PluginMetadata(
            name="decision",
            version="1.0.0",
            author="SmartMemory Team",
            description="Extracts decision statements from text and reasoning traces",
            plugin_type="extractor",
            dependencies=[],
            min_smartmemory_version="0.1.0",
            tags=["decision", "preference", "inference", "classification"],
        )

    def __init__(self, config: DecisionExtractorConfig | None = None):
        self.cfg = config or DecisionExtractorConfig()

    def extract(self, text: str) -> dict:
        """Extract decisions from text using regex pattern matching.

        Args:
            text: The text to extract decisions from.

        Returns:
            Dict with 'entities', 'relations', and 'decisions' keys.
        """
        result: dict = {
            "entities": [],
            "relations": [],
            "decisions": [],
        }

        if not text or len(text.strip()) < self.cfg.min_content_length:
            return result

        seen_contents: set[str] = set()

        for pattern, decision_type in DECISION_PATTERNS:
            for match in pattern.finditer(text):
                content = match.group(1).strip()
                # Remove trailing punctuation for cleaner content
                content = content.rstrip(".!?").strip()

                if len(content) < self.cfg.min_content_length:
                    continue

                # Deduplicate by normalized content
                normalized = content.lower()
                if normalized in seen_contents:
                    continue
                seen_contents.add(normalized)

                decision = Decision(
                    decision_id=Decision.generate_id(),
                    content=content,
                    decision_type=decision_type,
                    confidence=self.cfg.min_confidence,
                    source_type="inferred",
                )
                result["decisions"].append(decision)

        return result

    def extract_from_trace(self, trace: ReasoningTrace) -> list[Decision]:
        """Extract decisions from a ReasoningTrace.

        Looks at steps with type "decision" or "conclusion" and creates
        Decision objects with provenance linking back to the trace.

        Args:
            trace: The reasoning trace to extract decisions from.

        Returns:
            List of Decision objects extracted from the trace.
        """
        decisions: list[Decision] = []

        for step in trace.steps:
            if step.type not in DECISION_STEP_TYPES:
                continue

            content = step.content.strip()
            if len(content) < self.cfg.min_content_length:
                continue

            decision_type = self._classify_decision_type(content)

            decision = Decision(
                decision_id=Decision.generate_id(),
                content=content,
                decision_type=decision_type,
                confidence=self.cfg.min_confidence,
                source_type="reasoning",
                source_trace_id=trace.trace_id,
                source_session_id=trace.session_id,
            )
            decisions.append(decision)

        return decisions

    def _classify_decision_type(self, content: str) -> str:
        """Classify decision type based on keyword matching.

        Args:
            content: The decision content to classify.

        Returns:
            One of: inference, preference, classification, choice, belief, policy.
        """
        content_lower = content.lower()

        for dtype, keywords in TYPE_KEYWORDS.items():
            for keyword in keywords:
                if keyword in content_lower:
                    return dtype

        return "inference"
