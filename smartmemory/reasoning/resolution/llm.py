"""Resolve conflicts using LLM reasoning."""

import json
import logging
import re
from typing import Any, Dict, Optional

from smartmemory.reasoning.challenger import Conflict, ResolutionStrategy

from .base import ConflictResolver

logger = logging.getLogger(__name__)


class LLMResolver(ConflictResolver):
    """Use LLM to reason about which fact is correct."""

    name = "llm_reasoning"

    def resolve(self, conflict: Conflict) -> Optional[Dict[str, Any]]:
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
                response_format={"type": "json_object"},
            )

            if parsed_result:
                result = parsed_result
            else:
                json_match = re.search(r"\{[^{}]*\}", response, re.DOTALL)
                if not json_match:
                    return None
                result = json.loads(json_match.group())

            correct = result.get("correct_statement", "unknown")
            confidence = float(result.get("confidence", 0.0))

            if confidence < 0.7 or correct == "unknown":
                return None

            resolution = ResolutionStrategy.KEEP_EXISTING if correct == "A" else ResolutionStrategy.ACCEPT_NEW
            label = "existing" if correct == "A" else "new"

            return {
                "auto_resolved": True,
                "resolution": resolution,
                "confidence": confidence,
                "method": "llm_reasoning",
                "evidence": result.get("reasoning", f"LLM determined {label} fact is correct"),
                "actions_taken": [f"LLM reasoning: {result.get('source', 'general knowledge')}"],
            }

        except Exception as e:
            logger.debug(f"LLM reasoning resolution failed: {e}")
            return None
