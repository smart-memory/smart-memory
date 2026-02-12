"""
Evolution module for SmartMemory.

Provides procedure evolution tracking, diff computation, and timeline support.
"""

from smartmemory.evolution.models import (
    ContentSnapshot,
    EventDiff,
    EventSource,
    EvolutionEvent,
    MatchStatsSnapshot,
)
from smartmemory.evolution.diff_engine import ProcedureDiffEngine
from smartmemory.evolution.store import EvolutionEventStore
from smartmemory.evolution.tracker import EvolutionTracker

# Existing exports
from smartmemory.evolution.cycle import run_evolution_cycle
from smartmemory.evolution.flow import EvolutionFlow
from smartmemory.evolution.registry import (
    EVOLVER_REGISTRY,
    EvolverRegistry,
    EvolverSpec,
    get_evolver_by_key,
    list_evolver_specs,
    register_builtin_evolvers,
)

__all__ = [
    # New evolution timeline exports
    "ContentSnapshot",
    "EventDiff",
    "EventSource",
    "EvolutionEvent",
    "MatchStatsSnapshot",
    "ProcedureDiffEngine",
    "EvolutionEventStore",
    "EvolutionTracker",
    # Existing exports
    "run_evolution_cycle",
    "EvolutionFlow",
    "EVOLVER_REGISTRY",
    "EvolverRegistry",
    "EvolverSpec",
    "get_evolver_by_key",
    "list_evolver_specs",
    "register_builtin_evolvers",
]
