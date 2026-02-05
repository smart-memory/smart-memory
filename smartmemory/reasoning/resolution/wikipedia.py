"""Resolve conflicts by verifying facts against Wikipedia."""

import logging
from typing import Any, Dict, List, Optional

from smartmemory.reasoning.challenger import Conflict, ResolutionStrategy

from .base import ConflictResolver

logger = logging.getLogger(__name__)


def _extract_entities_for_lookup(*texts: str) -> List[str]:
    """Extract entity names from texts for external lookup."""
    entities: List[str] = []
    for text in texts:
        words = text.split()
        for i, word in enumerate(words):
            if i == 0:
                continue
            clean_word = word.strip(".,!?\";:")
            if clean_word and clean_word[0].isupper():
                entities.append(clean_word)

    seen: set[str] = set()
    unique: List[str] = []
    for e in entities:
        if e.lower() not in seen:
            seen.add(e.lower())
            unique.append(e)
    return unique


def _text_supports_claim(reference_text: str, claim: str) -> bool:
    """Check if reference text supports a claim (simple heuristic)."""
    ref_lower = reference_text.lower()
    claim_lower = claim.lower()
    stopwords = {"the", "a", "an", "is", "are", "was", "were", "of", "in", "to", "and", "or"}
    claim_words = [w for w in claim_lower.split() if w not in stopwords and len(w) > 2]
    if not claim_words:
        return False
    matches = sum(1 for w in claim_words if w in ref_lower)
    return matches / len(claim_words) > 0.6


class WikipediaResolver(ConflictResolver):
    """Verify facts against Wikipedia."""

    name = "wikipedia"

    def resolve(self, conflict: Conflict) -> Optional[Dict[str, Any]]:
        try:
            entities = _extract_entities_for_lookup(conflict.existing_fact, conflict.new_fact)
            if not entities:
                return None

            from smartmemory.plugins.grounders.wikipedia import WikipediaGrounder

            grounder = WikipediaGrounder()

            for entity in entities[:3]:
                try:
                    wiki_result = grounder.lookup(entity)
                    if not wiki_result:
                        continue

                    wiki_text = wiki_result.get("summary", "") or wiki_result.get("extract", "")
                    if not wiki_text:
                        continue

                    existing_match = _text_supports_claim(wiki_text, conflict.existing_fact)
                    new_match = _text_supports_claim(wiki_text, conflict.new_fact)

                    if existing_match and not new_match:
                        return {
                            "auto_resolved": True,
                            "resolution": ResolutionStrategy.KEEP_EXISTING,
                            "confidence": 0.85,
                            "method": "wikipedia",
                            "evidence": f"Wikipedia article '{entity}' supports existing fact",
                            "actions_taken": [f"Verified via Wikipedia: {entity}"],
                        }
                    elif new_match and not existing_match:
                        return {
                            "auto_resolved": True,
                            "resolution": ResolutionStrategy.ACCEPT_NEW,
                            "confidence": 0.85,
                            "method": "wikipedia",
                            "evidence": f"Wikipedia article '{entity}' supports new fact",
                            "actions_taken": [f"Verified via Wikipedia: {entity}"],
                        }

                except Exception as e:
                    logger.debug(f"Wikipedia lookup failed for '{entity}': {e}")
                    continue

            return None

        except Exception as e:
            logger.debug(f"Wikipedia resolution failed: {e}")
            return None
