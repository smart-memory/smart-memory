"""Pipeline v2 stage wrappers.

Each module wraps an existing pipeline component as a StageCommand.
Dependencies are injected via constructor, not from state or config.
"""

from smartmemory.pipeline.stages.classify import ClassifyStage
from smartmemory.pipeline.stages.coreference import CoreferenceStageCommand
from smartmemory.pipeline.stages.extract import ExtractStage
from smartmemory.pipeline.stages.store import StoreStage
from smartmemory.pipeline.stages.link import LinkStage
from smartmemory.pipeline.stages.enrich import EnrichStage
from smartmemory.pipeline.stages.ground import GroundStage
from smartmemory.pipeline.stages.evolve import EvolveStage

__all__ = [
    "ClassifyStage",
    "CoreferenceStageCommand",
    "ExtractStage",
    "StoreStage",
    "LinkStage",
    "EnrichStage",
    "GroundStage",
    "EvolveStage",
]
