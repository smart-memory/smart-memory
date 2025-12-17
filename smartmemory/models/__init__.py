# Node type imports for smartmemory.graph.nodes

# Memory types constant
from smartmemory.models.memory_item import MEMORY_TYPES, MemoryItem

# Reasoning trace models (System 2 Memory - Phase 2)
from smartmemory.models.reasoning import (
    ReasoningStep,
    ReasoningTrace,
    ReasoningEvaluation,
    TaskContext,
)

# Opinion/Observation models (Synthesis Memory - Phase 3)
from smartmemory.models.opinion import (
    OpinionMetadata,
    ObservationMetadata,
    Disposition,
)

__all__ = [
    'MEMORY_TYPES',
    'MemoryItem',
    # Phase 2: Reasoning
    'ReasoningStep',
    'ReasoningTrace',
    'ReasoningEvaluation',
    'TaskContext',
    # Phase 3: Opinion/Observation
    'OpinionMetadata',
    'ObservationMetadata',
    'Disposition',
]
