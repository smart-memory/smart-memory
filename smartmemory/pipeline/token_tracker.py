"""Per-request token cost instrumentation for the pipeline.

Tracks tokens **spent** (actual LLM calls) and tokens **avoided** (cache hits,
graph lookups, disabled stages) attributed by pipeline stage.  Attached to
``PipelineState`` so each ``ingest()`` call gets its own tracker.

The existing global ``TokenTracker`` singleton in ``utils/token_tracking.py`` is
left untouched — this module supplements it with per-request, per-stage, per-tenant
attribution.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from smartmemory.utils.token_tracking import COST_PER_1K_TOKENS

logger = logging.getLogger(__name__)


@dataclass
class StageTokenRecord:
    """Token usage for a single event (spent or avoided) within a stage."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""
    reason: str = ""  # e.g. "cache_hit", "graph_lookup", "stage_disabled"

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class StageTokenSummary:
    """Aggregated token counts for one stage."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    call_count: int = 0
    models: Dict[str, int] = field(default_factory=dict)
    reasons: Dict[str, int] = field(default_factory=dict)


class PipelineTokenTracker:
    """Per-request token tracker that rides on ``PipelineState``.

    Not thread-shared — each request gets its own instance, so no locking needed.
    """

    def __init__(self, workspace_id: Optional[str] = None, profile_name: Optional[str] = None):
        self.workspace_id = workspace_id
        self.profile_name = profile_name
        self._spent: List[tuple[str, StageTokenRecord]] = []  # (stage, record)
        self._avoided: List[tuple[str, StageTokenRecord]] = []

    # ------------------------------------------------------------------ #
    # Recording
    # ------------------------------------------------------------------ #

    def record_spent(
        self,
        stage: str,
        prompt_tokens: int,
        completion_tokens: int,
        model: str = "",
    ) -> None:
        """Record tokens actually consumed by an LLM call."""
        self._spent.append(
            (
                stage,
                StageTokenRecord(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    model=model,
                ),
            )
        )

    def record_avoided(
        self,
        stage: str,
        estimated_tokens: int,
        model: str = "",
        reason: str = "cache_hit",
    ) -> None:
        """Record tokens that were *not* consumed thanks to a shortcut."""
        self._avoided.append(
            (
                stage,
                StageTokenRecord(
                    prompt_tokens=estimated_tokens,
                    completion_tokens=0,
                    model=model,
                    reason=reason,
                ),
            )
        )

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #

    def summary(self) -> Dict[str, Any]:
        """Return a full breakdown suitable for persistence / API response."""
        spent_by_stage = self._aggregate(self._spent)
        avoided_by_stage = self._aggregate(self._avoided)

        total_spent = sum(s["total_tokens"] for s in spent_by_stage.values())
        total_avoided = sum(s["total_tokens"] for s in avoided_by_stage.values())
        total = total_spent + total_avoided

        # Cost estimation (spent only — avoided is hypothetical savings)
        cost_spent = self._estimate_cost(self._spent)
        cost_avoided = self._estimate_cost(self._avoided)

        return {
            "workspace_id": self.workspace_id,
            "profile_name": self.profile_name,
            "total_spent": total_spent,
            "total_avoided": total_avoided,
            "savings_pct": round((total_avoided / total) * 100, 1) if total > 0 else 0.0,
            "cost_usd": {
                "spent": round(cost_spent, 6),
                "avoided": round(cost_avoided, 6),
            },
            "stages": {
                "spent": spent_by_stage,
                "avoided": avoided_by_stage,
            },
        }

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    @staticmethod
    def _aggregate(records: List[tuple[str, StageTokenRecord]]) -> Dict[str, Dict[str, Any]]:
        """Group records by stage into summary dicts."""
        stages: Dict[str, StageTokenSummary] = {}
        for stage, rec in records:
            if stage not in stages:
                stages[stage] = StageTokenSummary()
            s = stages[stage]
            s.prompt_tokens += rec.prompt_tokens
            s.completion_tokens += rec.completion_tokens
            s.total_tokens += rec.total_tokens
            s.call_count += 1
            if rec.model:
                s.models[rec.model] = s.models.get(rec.model, 0) + 1
            if rec.reason:
                s.reasons[rec.reason] = s.reasons.get(rec.reason, 0) + 1

        return {
            stage: {
                "prompt_tokens": s.prompt_tokens,
                "completion_tokens": s.completion_tokens,
                "total_tokens": s.total_tokens,
                "call_count": s.call_count,
                "models": dict(s.models),
                "reasons": dict(s.reasons),
            }
            for stage, s in stages.items()
        }

    @staticmethod
    def _estimate_cost(records: List[tuple[str, StageTokenRecord]]) -> float:
        """Estimate USD cost from token records using the shared pricing table."""
        total = 0.0
        for _, rec in records:
            pricing = COST_PER_1K_TOKENS.get(rec.model)
            if not pricing:
                for key in COST_PER_1K_TOKENS:
                    if rec.model.startswith(key):
                        pricing = COST_PER_1K_TOKENS[key]
                        break
            if not pricing:
                pricing = COST_PER_1K_TOKENS.get("default", {"prompt": 0.001, "completion": 0.002})
            total += (rec.prompt_tokens / 1000) * pricing["prompt"]
            total += (rec.completion_tokens / 1000) * pricing["completion"]
        return total
