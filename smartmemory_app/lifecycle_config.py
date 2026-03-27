"""Lifecycle configuration for DIST-AGENT-HOOKS-1.

Two layers:
1. Persisted defaults from [lifecycle] section in ~/.config/smartmemory/config.toml
2. Session overrides from memory_auto() MCP tool, stored in sessions/<session_id>.json
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class RecallStrategy(Enum):
    """How aggressively to inject recalled context per prompt."""

    SESSION_ONLY = "session_only"  # Orient only, no per-prompt recall
    TOPIC_CHANGE = "topic_change"  # Recall when topic diverges (default)
    EVERY_PROMPT = "every_prompt"  # Recall on every prompt (gated by trivial-skip)


@dataclass
class LifecycleConfig:
    """Configuration for the automatic memory lifecycle."""

    enabled: bool = True
    recall_strategy: RecallStrategy = RecallStrategy.TOPIC_CHANGE
    orient_budget: int = 1500  # max tokens for Orient injection
    recall_budget: int = 500  # max tokens for Recall injection
    topic_threshold: float = 0.7  # cosine similarity for topic-change detection
    observe_tool_calls: bool = True
    distill_turns: bool = True
    learn_from_errors: bool = True

    @classmethod
    def from_config(cls, raw: dict | None = None) -> LifecycleConfig:
        """Load from a [lifecycle] TOML section dict.

        Missing keys use defaults. Invalid recall_strategy falls back to topic_change.
        """
        if not raw:
            return cls()

        strategy_str = raw.get("recall_strategy", "topic_change")
        try:
            strategy = RecallStrategy(strategy_str)
        except ValueError:
            strategy = RecallStrategy.TOPIC_CHANGE

        return cls(
            enabled=raw.get("enabled", True),
            recall_strategy=strategy,
            orient_budget=raw.get("orient_budget", 1500),
            recall_budget=raw.get("recall_budget", 500),
            topic_threshold=raw.get("topic_threshold", 0.7),
            observe_tool_calls=raw.get("observe_tool_calls", True),
            distill_turns=raw.get("distill_turns", True),
            learn_from_errors=raw.get("learn_from_errors", True),
        )

    def apply_overrides(self, overrides: dict) -> LifecycleConfig:
        """Return a new config with session overrides merged in.

        Used by memory_auto() MCP tool to apply per-session config changes.
        """
        merged = {
            "enabled": overrides.get("enabled", self.enabled),
            "recall_strategy": overrides.get("recall_strategy", self.recall_strategy.value),
            "orient_budget": overrides.get("orient_budget", self.orient_budget),
            "recall_budget": overrides.get("recall_budget", self.recall_budget),
            "topic_threshold": overrides.get("topic_threshold", self.topic_threshold),
            "observe_tool_calls": overrides.get("observe_tool_calls", self.observe_tool_calls),
            "distill_turns": overrides.get("distill_turns", self.distill_turns),
            "learn_from_errors": overrides.get("learn_from_errors", self.learn_from_errors),
        }
        return LifecycleConfig.from_config(merged)
