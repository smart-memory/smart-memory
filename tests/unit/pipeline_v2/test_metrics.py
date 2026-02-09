"""Tests for PipelineMetricsEmitter."""

import pytest

pytestmark = pytest.mark.unit


from unittest.mock import MagicMock, patch
from dataclasses import replace

from smartmemory.pipeline.metrics import PipelineMetricsEmitter
from smartmemory.pipeline.state import PipelineState


def _make_state(**overrides) -> PipelineState:
    defaults = dict(
        text="test text",
        raw_metadata={},
        mode="sync",
        workspace_id="ws_test",
    )
    defaults.update(overrides)
    return PipelineState(**defaults)


# ------------------------------------------------------------------ #
# on_stage_complete tests
# ------------------------------------------------------------------ #


def test_on_stage_complete_emits_event():
    """on_stage_complete emits a stage_complete event via the spooler."""
    emitter = PipelineMetricsEmitter(workspace_id="ws_123")
    mock_spooler = MagicMock()
    emitter._spooler = mock_spooler

    state = _make_state(entities=[{"name": "Alice"}], relations=[{"type": "KNOWS"}])
    emitter.on_stage_complete("entity_ruler", 4.2, state)

    mock_spooler.emit_event.assert_called_once()
    call_kwargs = mock_spooler.emit_event.call_args
    assert call_kwargs[1]["event_type"] == "stage_complete" or call_kwargs[0][0] == "stage_complete"
    data = call_kwargs[1].get("data") or call_kwargs[0][3]
    assert data["stage_name"] == "entity_ruler"
    assert data["elapsed_ms"] == 4.2
    assert data["status"] == "success"
    assert data["entity_count"] == 1
    assert data["relation_count"] == 1
    assert data["workspace_id"] == "ws_123"


def test_on_stage_complete_with_error():
    """on_stage_complete emits error status when error is provided."""
    emitter = PipelineMetricsEmitter()
    mock_spooler = MagicMock()
    emitter._spooler = mock_spooler

    state = _make_state()
    emitter.on_stage_complete("llm_extract", 100.0, state, error=RuntimeError("LLM timeout"))

    mock_spooler.emit_event.assert_called_once()
    data = mock_spooler.emit_event.call_args[1].get("data") or mock_spooler.emit_event.call_args[0][3]
    assert data["status"] == "error"
    assert "LLM timeout" in data["error"]


def test_on_stage_complete_tracks_timing():
    """on_stage_complete accumulates stage timings for pipeline summary."""
    emitter = PipelineMetricsEmitter()
    mock_spooler = MagicMock()
    emitter._spooler = mock_spooler

    state = _make_state()
    emitter.on_stage_complete("classify", 2.1, state)
    emitter.on_stage_complete("coreference", 3.5, state)

    assert emitter._stage_timings == {"classify": 2.1, "coreference": 3.5}


# ------------------------------------------------------------------ #
# on_pipeline_complete tests
# ------------------------------------------------------------------ #


def test_on_pipeline_complete_emits_summary():
    """on_pipeline_complete emits pipeline_complete with stage breakdown."""
    emitter = PipelineMetricsEmitter(workspace_id="ws_456")
    mock_spooler = MagicMock()
    emitter._spooler = mock_spooler

    # Simulate stage completions
    state = _make_state(entities=[{"name": "X"}, {"name": "Y"}], relations=[])
    emitter.on_stage_complete("classify", 2.0, state)
    emitter.on_stage_complete("store", 10.0, state)

    # Now emit pipeline complete
    emitter.on_pipeline_complete(state)

    # Should have 3 calls: 2 stage + 1 pipeline
    assert mock_spooler.emit_event.call_count == 3
    last_call = mock_spooler.emit_event.call_args_list[-1]
    data = last_call[1].get("data") or last_call[0][3]
    assert data["event_type"] == "pipeline_complete"
    assert data["total_ms"] == 12.0
    assert data["stage_timings"]["classify"] == 2.0
    assert data["stage_timings"]["store"] == 10.0
    assert data["stages_completed"] == 2
    assert data["entity_count"] == 2
    assert data["workspace_id"] == "ws_456"


# ------------------------------------------------------------------ #
# Error resilience tests
# ------------------------------------------------------------------ #


def test_emitter_handles_redis_failure_gracefully():
    """If spooler raises, emitter swallows the error silently."""
    emitter = PipelineMetricsEmitter()
    mock_spooler = MagicMock()
    mock_spooler.emit_event.side_effect = ConnectionError("Redis down")
    emitter._spooler = mock_spooler

    state = _make_state()
    # Should not raise
    emitter.on_stage_complete("classify", 1.0, state)
    emitter.on_pipeline_complete(state)


def test_emitter_with_none_spooler_is_silent():
    """When Redis is unavailable (_get_spooler returns None), emitter is silent."""
    emitter = PipelineMetricsEmitter()
    # Patch _get_spooler to return None (simulating Redis unavailable)
    emitter._get_spooler = lambda: None

    state = _make_state()
    # Should not raise
    emitter.on_stage_complete("classify", 1.0, state)
    emitter.on_pipeline_complete(state)


# ------------------------------------------------------------------ #
# Runner integration test
# ------------------------------------------------------------------ #


def test_runner_calls_metrics_emitter_for_all_stages():
    """PipelineRunner calls on_stage_complete for each stage and on_pipeline_complete at end."""
    from smartmemory.pipeline.runner import PipelineRunner
    from smartmemory.pipeline.config import PipelineConfig

    # Build trivial stages that pass state through
    class PassthroughStage:
        def __init__(self, name):
            self.name = name

        def execute(self, state, config):
            return state

        def undo(self, state):
            return state

    stages = [PassthroughStage(f"stage_{i}") for i in range(3)]

    mock_emitter = MagicMock()
    runner = PipelineRunner(stages, metrics_emitter=mock_emitter)
    config = PipelineConfig.default(workspace_id="test")

    runner.run("test text", config)

    # on_stage_complete called 3 times (once per stage)
    assert mock_emitter.on_stage_complete.call_count == 3
    for i, call in enumerate(mock_emitter.on_stage_complete.call_args_list):
        assert call[0][0] == f"stage_{i}"  # stage_name
        assert isinstance(call[0][1], float)  # elapsed_ms

    # on_pipeline_complete called once
    mock_emitter.on_pipeline_complete.assert_called_once()


def test_runner_calls_metrics_on_error_path():
    """When a stage fails and is skipped, metrics still get emitted."""
    from smartmemory.pipeline.runner import PipelineRunner
    from smartmemory.pipeline.config import PipelineConfig

    class FailingStage:
        name = "failing_stage"

        def execute(self, state, config):
            raise RuntimeError("Boom")

        def undo(self, state):
            return state

    mock_emitter = MagicMock()
    runner = PipelineRunner([FailingStage()], metrics_emitter=mock_emitter)
    config = PipelineConfig.default(workspace_id="test")
    config.retry.on_failure = "skip"
    config.retry.max_retries = 0

    runner.run("test text", config)

    # on_stage_complete should be called with error
    mock_emitter.on_stage_complete.assert_called_once()
    call_args = mock_emitter.on_stage_complete.call_args
    assert call_args[0][0] == "failing_stage"
    assert call_args[1]["error"] is not None or call_args[0][3] is not None
