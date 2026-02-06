"""Unit tests for Transport protocol and InProcessTransport."""

import dataclasses
from unittest.mock import MagicMock

import pytest

from smartmemory.pipeline.config import PipelineConfig
from smartmemory.pipeline.state import PipelineState
from smartmemory.pipeline.transport import InProcessTransport, Transport


class TestInProcessTransport:
    """Tests for InProcessTransport."""

    def test_calls_stage_execute(self):
        """InProcessTransport.execute() delegates to stage.execute()."""
        state = PipelineState(text="hello")
        config = PipelineConfig()
        expected_state = dataclasses.replace(state, text="modified")

        stage = MagicMock()
        stage.execute.return_value = expected_state

        transport = InProcessTransport()
        result = transport.execute(stage, state, config)

        stage.execute.assert_called_once_with(state, config)
        assert result is expected_state

    def test_returns_stage_result(self):
        """InProcessTransport returns whatever stage.execute() returns."""
        state = PipelineState(text="input")
        config = PipelineConfig()
        new_state = PipelineState(text="output", mode="preview")

        stage = MagicMock()
        stage.execute.return_value = new_state

        transport = InProcessTransport()
        result = transport.execute(stage, state, config)

        assert result.text == "output"
        assert result.mode == "preview"

    def test_propagates_stage_exception(self):
        """If stage.execute() raises, InProcessTransport propagates it."""
        state = PipelineState()
        config = PipelineConfig()

        stage = MagicMock()
        stage.execute.side_effect = ValueError("stage error")

        transport = InProcessTransport()
        with pytest.raises(ValueError, match="stage error"):
            transport.execute(stage, state, config)

    def test_with_mock_stage_command(self):
        """Use a MagicMock that simulates a StageCommand interface."""
        state = PipelineState(text="test data")
        config = PipelineConfig()

        mock_stage = MagicMock()
        mock_stage.name = "mock_classify"
        mock_stage.execute.return_value = dataclasses.replace(state, classified_types=["semantic"])
        mock_stage.undo.return_value = dataclasses.replace(state)

        transport = InProcessTransport()
        result = transport.execute(mock_stage, state, config)

        assert result.classified_types == ["semantic"]
        mock_stage.execute.assert_called_once()


class TestTransportProtocol:
    """Tests for Transport protocol satisfaction."""

    def test_in_process_transport_satisfies_protocol(self):
        """InProcessTransport satisfies the Transport protocol."""
        transport = InProcessTransport()
        assert isinstance(transport, Transport)

    def test_class_with_execute_satisfies_protocol(self):
        """Any class with a matching execute() satisfies Transport."""

        class CustomTransport:
            def execute(self, stage, state, config):
                return stage.execute(state, config)

        transport = CustomTransport()
        assert isinstance(transport, Transport)

    def test_class_without_execute_does_not_satisfy(self):
        """A class missing execute() does NOT satisfy Transport."""

        class NotATransport:
            def run(self, stage, state, config):
                return state

        obj = NotATransport()
        assert not isinstance(obj, Transport)
