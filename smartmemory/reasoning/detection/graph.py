"""Graph-based contradiction detection using knowledge graph structure."""

import logging
import re
from typing import Optional

from smartmemory.reasoning.models import Conflict, ConflictType, ResolutionStrategy

from .base import ContradictionDetector, DetectionContext

logger = logging.getLogger(__name__)

# Functional properties that can only have one value
FUNCTIONAL_PATTERNS = [
    "capital of",
    "president of",
    "ceo of",
    "founder of",
    "born in",
    "died in",
    "located in",
    "headquarters in",
]


class GraphDetector(ContradictionDetector):
    """Detect contradictions using knowledge graph structure.

    Analyses:
    1. Entity overlap - same entities with conflicting relations
    2. Functional properties - relations that can only have one value
    """

    name = "graph"

    def detect(self, ctx: DetectionContext) -> Optional[Conflict]:
        try:
            existing_entities: set[str] = set()
            if hasattr(ctx.existing_item, "metadata") and ctx.existing_item.metadata:
                entities = ctx.existing_item.metadata.get("entities", [])
                for ent in entities:
                    if isinstance(ent, dict):
                        existing_entities.add(ent.get("name", "").lower())
                    elif isinstance(ent, str):
                        existing_entities.add(ent.lower())

            existing_content_words = set(ctx.existing_item.content.lower().split())
            new_words = set(ctx.new_assertion.lower().split())

            stopwords = {"the", "a", "an", "is", "are", "was", "were", "of", "in", "to", "and", "or"}
            common_entities = (existing_content_words & new_words) - stopwords

            if len(common_entities) < 2:
                return None

            existing_lower = ctx.existing_item.content.lower()
            new_lower = ctx.new_assertion.lower()

            for pattern in FUNCTIONAL_PATTERNS:
                if pattern in existing_lower and pattern in new_lower:
                    existing_match = re.search(rf"{pattern}\s+(\w+(?:\s+\w+)?)", existing_lower)
                    new_match = re.search(rf"{pattern}\s+(\w+(?:\s+\w+)?)", new_lower)

                    if existing_match and new_match:
                        existing_value = existing_match.group(1)
                        new_value = new_match.group(1)

                        if existing_value != new_value:
                            return Conflict(
                                existing_item=ctx.existing_item,
                                existing_fact=ctx.existing_item.content,
                                new_fact=ctx.new_assertion,
                                conflict_type=ConflictType.DIRECT_CONTRADICTION,
                                confidence=0.85,
                                explanation=(
                                    f"Functional property conflict: '{pattern}' has different "
                                    f"values: '{existing_value}' vs '{new_value}'"
                                ),
                                suggested_resolution=ResolutionStrategy.DEFER,
                            )

            return None

        except Exception as e:
            logger.debug(f"Graph-based contradiction detection failed: {e}")
            return None
