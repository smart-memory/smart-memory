"""
SmartMemory Evolvers - Memory evolution plugins.

Evolvers transform memories across types and synthesize new memories from patterns.
"""

# Base evolver
from .base import Evolver

# Type promotion evolvers
from .working_to_episodic import WorkingToEpisodicEvolver
from .episodic_to_semantic import EpisodicToSemanticEvolver
from .working_to_procedural import WorkingToProceduralEvolver
from .episodic_to_zettel import EpisodicToZettelEvolver

# Maintenance evolvers
from .episodic_decay import EpisodicDecayEvolver

# Synthesis evolvers
from .opinion_synthesis import OpinionSynthesisEvolver
from .observation_synthesis import ObservationSynthesisEvolver
from .opinion_reinforcement import OpinionReinforcementEvolver

# Decision evolvers
from .decision_confidence import DecisionConfidenceEvolver

__all__ = [
    # Base
    'Evolver',

    # Type promotion
    'WorkingToEpisodicEvolver',
    'EpisodicToSemanticEvolver',
    'WorkingToProceduralEvolver',
    'EpisodicToZettelEvolver',

    # Maintenance
    'EpisodicDecayEvolver',

    # Synthesis
    'OpinionSynthesisEvolver',
    'ObservationSynthesisEvolver',
    'OpinionReinforcementEvolver',

    # Decision
    'DecisionConfidenceEvolver',
]
