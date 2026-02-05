"""Evolution and clustering operations extracted from SmartMemory."""

import logging
from typing import List

logger = logging.getLogger(__name__)


class EvolutionManager:
    """Manages evolution cycles, memory promotion, and clustering."""

    def __init__(self, graph, evolution_orchestrator, clustering):
        self._graph = graph
        self._evolution = evolution_orchestrator
        self._clustering = clustering

    def run_evolution_cycle(self):
        """Run a single evolution cycle with automatic scope filtering."""
        return self._evolution.run_evolution_cycle()

    def commit_working_to_episodic(self, remove_from_source: bool = True) -> List[str]:
        """Delegate evolution to EvolutionOrchestrator component."""
        return self._evolution.commit_working_to_episodic(remove_from_source)

    def commit_working_to_procedural(self, remove_from_source: bool = True) -> List[str]:
        """Delegate evolution to EvolutionOrchestrator component."""
        return self._evolution.commit_working_to_procedural(remove_from_source)

    def run_clustering(self) -> dict:
        """Run clustering to deduplicate entities."""
        return self._clustering.run()
