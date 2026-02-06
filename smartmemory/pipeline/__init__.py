"""Unified pipeline v2 for SmartMemory ingestion.

Replaces the three-orchestrator pattern (MemoryIngestionFlow, FastIngestionFlow,
EvolutionOrchestrator) with a single composable pipeline built from StageCommands.
"""

from smartmemory.pipeline.protocol import StageCommand
from smartmemory.pipeline.state import PipelineState
from smartmemory.pipeline.config import PipelineConfig
from smartmemory.pipeline.transport import Transport, InProcessTransport
from smartmemory.pipeline.runner import PipelineRunner

__all__ = [
    "StageCommand",
    "PipelineState",
    "PipelineConfig",
    "Transport",
    "InProcessTransport",
    "PipelineRunner",
]
