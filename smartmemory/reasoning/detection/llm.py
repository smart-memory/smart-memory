"""LLM-based contradiction detection."""

import json
import logging
import re
from typing import Optional

from smartmemory.reasoning.challenger import (
    Conflict,
    ConflictType,
    ResolutionStrategy,
)

from .base import ContradictionDetector, DetectionContext

logger = logging.getLogger(__name__)


class LLMDetector(ContradictionDetector):
    """Use an LLM to detect contradictions with reasoning."""

    name = "llm"

    def detect(self, ctx: DetectionContext) -> Optional[Conflict]:
        try:
            from smartmemory.utils.llm import call_llm

            prompt = f"""Analyze if these two statements contradict each other.

Statement A (existing fact): {ctx.existing_fact}

Statement B (new assertion): {ctx.new_assertion}

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
                response_format={"type": "json_object"},
            )

            if parsed_result:
                result = parsed_result
            else:
                json_match = re.search(r"\{[^{}]*\}", response, re.DOTALL)
                if not json_match:
                    logger.warning("Could not parse LLM response as JSON")
                    return None
                result = json.loads(json_match.group())

            if not result.get("contradicts", False):
                return None

            return Conflict(
                existing_item=ctx.existing_item,
                existing_fact=ctx.existing_fact,
                new_fact=ctx.new_assertion,
                conflict_type=ConflictType(result.get("conflict_type", "direct_contradiction")),
                confidence=result.get("confidence", 0.8),
                explanation=result.get("explanation", "LLM detected contradiction"),
                suggested_resolution=ResolutionStrategy(result.get("resolution", "defer")),
            )

        except Exception as e:
            logger.warning(f"LLM contradiction detection failed: {e}")
            return None
