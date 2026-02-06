"""StageCommand protocol — the unit of pipeline composition.

Any class with a matching signature satisfies the protocol (structural subtyping).
No base class inheritance required.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from smartmemory.pipeline.config import PipelineConfig
    from smartmemory.pipeline.state import PipelineState


@runtime_checkable
class StageCommand(Protocol):
    """A single pipeline stage that transforms PipelineState."""

    @property
    def name(self) -> str:
        """Unique stage identifier (e.g. 'classify', 'extract')."""
        ...

    def execute(self, state: PipelineState, config: PipelineConfig) -> PipelineState:
        """Run the stage, returning a new PipelineState.

        Must not mutate the input state — return via ``dataclasses.replace()``.
        """
        ...

    def undo(self, state: PipelineState) -> PipelineState:
        """Reverse the stage's effects, returning a new PipelineState.

        For preview mode: discard results.
        For production mode: may be a no-op for irreversible stages.
        """
        ...
