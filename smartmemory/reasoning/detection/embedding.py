"""Embedding-based contradiction detection using semantic similarity + polarity."""

import logging
from typing import Optional

from smartmemory.reasoning.challenger import (
    Conflict,
    ConflictType,
    ResolutionStrategy,
)

from .base import ContradictionDetector, DetectionContext

logger = logging.getLogger(__name__)

NEGATIVE_WORDS = frozenset({
    "not", "n't", "never", "no", "none", "neither", "nobody",
    "nothing", "nowhere", "cannot", "can't", "won't", "wouldn't",
    "shouldn't", "couldn't", "doesn't", "didn't", "isn't", "aren't",
    "wasn't", "weren't", "false", "incorrect", "wrong", "impossible",
})


class EmbeddingDetector(ContradictionDetector):
    """High semantic similarity + opposite polarity = likely contradiction."""

    name = "embedding"

    def detect(self, ctx: DetectionContext) -> Optional[Conflict]:
        try:
            from smartmemory.plugins.embedding import create_embeddings
            import numpy as np

            new_embedding = create_embeddings(ctx.new_assertion)
            existing_embedding = create_embeddings(ctx.existing_item.content)

            if new_embedding is None or existing_embedding is None:
                return None

            new_emb = np.asarray(new_embedding)
            existing_emb = np.asarray(existing_embedding)

            norm_new = np.linalg.norm(new_emb)
            norm_existing = np.linalg.norm(existing_emb)
            if norm_new == 0 or norm_existing == 0:
                return None

            similarity = float(np.dot(new_emb, existing_emb) / (norm_new * norm_existing))

            if similarity < 0.75:
                return None

            new_lower = ctx.new_assertion.lower()
            existing_lower = ctx.existing_item.content.lower()

            new_has_negative = any(neg in new_lower.split() or neg in new_lower for neg in NEGATIVE_WORDS)
            existing_has_negative = any(neg in existing_lower.split() or neg in existing_lower for neg in NEGATIVE_WORDS)

            # XOR: one has negative, other doesn't
            if new_has_negative != existing_has_negative:
                return Conflict(
                    existing_item=ctx.existing_item,
                    existing_fact=ctx.existing_item.content,
                    new_fact=ctx.new_assertion,
                    conflict_type=ConflictType.DIRECT_CONTRADICTION,
                    confidence=min(0.9, similarity),
                    explanation=f"High semantic similarity ({similarity:.2f}) with opposite polarity detected",
                    suggested_resolution=ResolutionStrategy.DEFER,
                )

            return None

        except Exception as e:
            logger.debug(f"Embedding-based contradiction detection failed: {e}")
            return None
