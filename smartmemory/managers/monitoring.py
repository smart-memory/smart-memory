"""Monitoring and analytics operations extracted from SmartMemory."""

import logging

logger = logging.getLogger(__name__)


class MonitoringManager:
    """Manages summary, orphan detection, pruning, and reflection operations."""

    def __init__(self, graph, monitoring):
        self._graph = graph
        self._monitoring = monitoring

    def summary(self) -> dict:
        """Delegate summary to Monitoring submodule."""
        return self._monitoring.summary()

    def orphaned_notes(self) -> list:
        """Delegate orphaned_notes to Monitoring submodule."""
        return self._monitoring.orphaned_notes()

    def prune(self, strategy="old", days=365, **kwargs):
        """Delegate prune to Monitoring submodule."""
        return self._monitoring.prune(strategy, days, **kwargs)

    def find_old_notes(self, days: int = 365) -> list:
        """Delegate find_old_notes to Monitoring submodule."""
        return self._monitoring.find_old_notes(days)

    def self_monitor(self) -> dict:
        """Delegate self_monitor to Monitoring submodule."""
        return self._monitoring.self_monitor()

    def reflect(self, top_k: int = 5) -> dict:
        """Delegate reflect to Monitoring submodule."""
        return self._monitoring.reflect(top_k)

    def summarize(self, max_items: int = 10) -> dict:
        """Delegate summarize to Monitoring submodule."""
        return self._monitoring.summarize(max_items)
