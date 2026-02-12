"""Enricher LLM usage tracking (CFS-1b).

Provides thread-local accumulation of LLM token usage from enricher plugins.
Multiple enrichers may run during a single enrichment pass, so usage is
accumulated rather than replaced.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

# Thread-local storage for accumulated enricher usage
_thread_local = threading.local()


@dataclass
class EnricherUsageRecord:
    """Token usage from a single enricher LLM call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model: str = ""
    enricher_name: str = ""


@dataclass
class EnricherUsageAccumulator:
    """Accumulates usage from multiple enricher LLM calls."""

    records: list[EnricherUsageRecord] = field(default_factory=list)

    def add(
        self,
        enricher_name: str,
        prompt_tokens: int,
        completion_tokens: int,
        model: str,
    ) -> None:
        """Add a usage record from an enricher LLM call."""
        self.records.append(
            EnricherUsageRecord(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                model=model,
                enricher_name=enricher_name,
            ),
        )

    @property
    def total_prompt_tokens(self) -> int:
        """Total prompt tokens across all enricher calls."""
        return sum(r.prompt_tokens for r in self.records)

    @property
    def total_completion_tokens(self) -> int:
        """Total completion tokens across all enricher calls."""
        return sum(r.completion_tokens for r in self.records)

    @property
    def total_tokens(self) -> int:
        """Total tokens across all enricher calls."""
        return sum(r.total_tokens for r in self.records)


def get_enricher_usage() -> dict[str, Any] | None:
    """Get and clear accumulated enricher usage from this thread.

    Returns:
        Dict with total_prompt_tokens, total_completion_tokens, total_tokens,
        records (list of per-enricher usage) — or None if no recent calls.

    """
    accumulator = getattr(_thread_local, "enricher_usage", None)
    _thread_local.enricher_usage = None  # Consume-once
    if accumulator and accumulator.records:
        return {
            "total_prompt_tokens": accumulator.total_prompt_tokens,
            "total_completion_tokens": accumulator.total_completion_tokens,
            "total_tokens": accumulator.total_tokens,
            "records": [
                {
                    "enricher_name": r.enricher_name,
                    "prompt_tokens": r.prompt_tokens,
                    "completion_tokens": r.completion_tokens,
                    "total_tokens": r.total_tokens,
                    "model": r.model,
                }
                for r in accumulator.records
            ],
        }
    return None


def record_enricher_usage(
    enricher_name: str,
    prompt_tokens: int,
    completion_tokens: int,
    model: str,
) -> None:
    """Record LLM usage from an enricher call.

    Args:
        enricher_name: Name of the enricher plugin (e.g., "temporal_enricher")
        prompt_tokens: Number of prompt tokens used
        completion_tokens: Number of completion tokens used
        model: Model name (e.g., "gpt-4o-mini")

    """
    accumulator = getattr(_thread_local, "enricher_usage", None)
    if accumulator is None:
        accumulator = EnricherUsageAccumulator()
        _thread_local.enricher_usage = accumulator
    accumulator.add(enricher_name, prompt_tokens, completion_tokens, model)


def clear_enricher_usage() -> None:
    """Clear any accumulated enricher usage (for testing)."""
    _thread_local.enricher_usage = None
