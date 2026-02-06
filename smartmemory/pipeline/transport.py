"""Transport abstraction — how stages are invoked.

InProcessTransport calls ``stage.execute()`` directly (one-liner).
Future transports (EventBusTransport, CeleryTransport) can serialize
state to a queue and execute stages in workers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from smartmemory.pipeline.config import PipelineConfig
    from smartmemory.pipeline.protocol import StageCommand
    from smartmemory.pipeline.state import PipelineState


@runtime_checkable
class Transport(Protocol):
    """Protocol for executing a stage."""

    def execute(self, stage: StageCommand, state: PipelineState, config: PipelineConfig) -> PipelineState:
        """Execute *stage* against *state* using *config* and return the new state."""
        ...


class InProcessTransport:
    """Simplest transport: direct in-process call."""

    def execute(self, stage: StageCommand, state: PipelineState, config: PipelineConfig) -> PipelineState:
        return stage.execute(state, config)
