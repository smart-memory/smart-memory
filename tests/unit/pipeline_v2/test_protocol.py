"""Unit tests for StageCommand protocol (structural subtyping)."""

import pytest

pytestmark = pytest.mark.unit


import dataclasses

from smartmemory.pipeline.protocol import StageCommand
from smartmemory.pipeline.state import PipelineState
from smartmemory.pipeline.config import PipelineConfig


class _ValidStage:
    """A class that satisfies the StageCommand protocol."""

    @property
    def name(self) -> str:
        return "valid_stage"

    def execute(self, state: PipelineState, config: PipelineConfig) -> PipelineState:
        return dataclasses.replace(state)

    def undo(self, state: PipelineState) -> PipelineState:
        return dataclasses.replace(state)


class _MissingUndo:
    """A class missing the undo method."""

    @property
    def name(self) -> str:
        return "no_undo"

    def execute(self, state: PipelineState, config: PipelineConfig) -> PipelineState:
        return dataclasses.replace(state)


class _MissingName:
    """A class missing the name property."""

    def execute(self, state: PipelineState, config: PipelineConfig) -> PipelineState:
        return dataclasses.replace(state)

    def undo(self, state: PipelineState) -> PipelineState:
        return dataclasses.replace(state)


class TestStageCommandProtocol:
    """Tests for the runtime_checkable StageCommand Protocol."""

    def test_class_with_all_methods_satisfies_protocol(self):
        """A class with name, execute, and undo satisfies StageCommand."""
        stage = _ValidStage()
        assert isinstance(stage, StageCommand)

    def test_class_missing_undo_does_not_satisfy_protocol(self):
        """A class without undo does NOT satisfy StageCommand."""
        stage = _MissingUndo()
        assert not isinstance(stage, StageCommand)

    def test_class_missing_name_does_not_satisfy_protocol(self):
        """A class without a name property does NOT satisfy StageCommand."""
        stage = _MissingName()
        assert not isinstance(stage, StageCommand)

    def test_valid_stage_execute_returns_new_state(self):
        """Verify execute actually works on a conformant class."""
        stage = _ValidStage()
        state = PipelineState(text="hello")
        config = PipelineConfig()
        result = stage.execute(state, config)
        assert isinstance(result, PipelineState)
        assert result.text == "hello"

    def test_valid_stage_undo_returns_new_state(self):
        """Verify undo actually works on a conformant class."""
        stage = _ValidStage()
        state = PipelineState(text="hello")
        result = stage.undo(state)
        assert isinstance(result, PipelineState)

    def test_valid_stage_name_property(self):
        """Verify name returns the expected string."""
        stage = _ValidStage()
        assert stage.name == "valid_stage"
