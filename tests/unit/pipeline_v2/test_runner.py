"""Unit tests for PipelineRunner orchestration."""

import dataclasses

import pytest

from smartmemory.pipeline.config import PipelineConfig, RetryConfig
from smartmemory.pipeline.runner import PipelineRunner
from smartmemory.pipeline.state import PipelineState


# ------------------------------------------------------------------ #
# Test helpers
# ------------------------------------------------------------------ #


class MockStage:
    """A test-double stage that records execution and supports side-effects."""

    def __init__(self, stage_name, side_effect=None):
        self._name = stage_name
        self.executed = False
        self.undone = False
        self.side_effect = side_effect

    @property
    def name(self):
        return self._name

    def execute(self, state, config):
        if self.side_effect:
            raise self.side_effect
        self.executed = True
        return dataclasses.replace(state)

    def undo(self, state):
        self.undone = True
        return dataclasses.replace(state)


class OrderTrackingStage:
    """A stage that appends its name to a shared list to verify ordering."""

    def __init__(self, stage_name, execution_log):
        self._name = stage_name
        self._log = execution_log

    @property
    def name(self):
        return self._name

    def execute(self, state, config):
        self._log.append(self._name)
        return dataclasses.replace(state)

    def undo(self, state):
        self._log.append(f"undo:{self._name}")
        return dataclasses.replace(state)


class FailThenSucceedStage:
    """A stage that fails a configurable number of times, then succeeds."""

    def __init__(self, stage_name, fail_count):
        self._name = stage_name
        self._fail_count = fail_count
        self._attempts = 0
        self.executed = False

    @property
    def name(self):
        return self._name

    def execute(self, state, config):
        self._attempts += 1
        if self._attempts <= self._fail_count:
            raise RuntimeError(f"Deliberate failure #{self._attempts}")
        self.executed = True
        return dataclasses.replace(state)

    def undo(self, state):
        return dataclasses.replace(state)


# ------------------------------------------------------------------ #
# Tests
# ------------------------------------------------------------------ #


class TestPipelineRunnerRun:
    """Tests for PipelineRunner.run() — full pipeline execution."""

    def test_run_executes_all_stages_in_order(self):
        """All stages execute in the order they are given."""
        log = []
        stages = [
            OrderTrackingStage("classify", log),
            OrderTrackingStage("extract", log),
            OrderTrackingStage("store", log),
        ]
        runner = PipelineRunner(stages)
        config = PipelineConfig()

        state = runner.run("hello", config)

        assert log == ["classify", "extract", "store"]
        assert state.stage_history == ["classify", "extract", "store"]

    def test_run_sets_text_on_state(self):
        """run() initializes state.text from the input string."""
        stages = [MockStage("noop")]
        runner = PipelineRunner(stages)
        config = PipelineConfig()

        state = runner.run("test content", config)
        assert state.text == "test content"

    def test_run_sets_metadata(self):
        """run() passes metadata to the initial state."""
        stages = [MockStage("noop")]
        runner = PipelineRunner(stages)
        config = PipelineConfig()

        state = runner.run("text", config, metadata={"source": "test"})
        assert state.raw_metadata == {"source": "test"}

    def test_run_sets_started_at_and_completed_at(self):
        """run() timestamps the state with started_at and completed_at."""
        stages = [MockStage("noop")]
        runner = PipelineRunner(stages)
        config = PipelineConfig()

        state = runner.run("text", config)
        assert state.started_at is not None
        assert state.completed_at is not None
        assert state.completed_at >= state.started_at


class TestPipelineRunnerRunTo:
    """Tests for PipelineRunner.run_to() — breakpoint execution."""

    def test_run_to_stops_after_named_stage(self):
        """run_to() executes stages up to and including the target."""
        log = []
        stages = [
            OrderTrackingStage("classify", log),
            OrderTrackingStage("extract", log),
            OrderTrackingStage("store", log),
        ]
        runner = PipelineRunner(stages)
        config = PipelineConfig()

        state = runner.run_to("text", config, stop_after="extract")

        assert log == ["classify", "extract"]
        assert "store" not in state.stage_history

    def test_run_to_with_first_stage(self):
        """run_to() with first stage only executes that one stage."""
        log = []
        stages = [
            OrderTrackingStage("classify", log),
            OrderTrackingStage("extract", log),
        ]
        runner = PipelineRunner(stages)
        config = PipelineConfig()

        state = runner.run_to("text", config, stop_after="classify")

        assert log == ["classify"]
        assert state.stage_history == ["classify"]

    def test_run_to_with_last_stage(self):
        """run_to() with the last stage runs all stages."""
        log = []
        stages = [
            OrderTrackingStage("classify", log),
            OrderTrackingStage("extract", log),
        ]
        runner = PipelineRunner(stages)
        config = PipelineConfig()

        runner.run_to("text", config, stop_after="extract")

        assert log == ["classify", "extract"]

    def test_run_to_invalid_stage_raises_value_error(self):
        """run_to() with an unknown stage name raises ValueError."""
        stages = [MockStage("classify"), MockStage("extract")]
        runner = PipelineRunner(stages)
        config = PipelineConfig()

        with pytest.raises(ValueError, match="not found"):
            runner.run_to("text", config, stop_after="nonexistent")


class TestPipelineRunnerRunFrom:
    """Tests for PipelineRunner.run_from() — checkpoint resumption."""

    def test_run_from_resumes_at_named_stage(self):
        """run_from() starts execution from the specified stage."""
        log = []
        stages = [
            OrderTrackingStage("classify", log),
            OrderTrackingStage("extract", log),
            OrderTrackingStage("store", log),
        ]
        runner = PipelineRunner(stages)
        config = PipelineConfig()

        # Simulate a state that already completed 'classify'
        state = PipelineState(
            text="hello",
            stage_history=["classify"],
        )
        result = runner.run_from(state, config, start_from="extract")

        assert log == ["extract", "store"]
        assert "extract" in result.stage_history
        assert "store" in result.stage_history

    def test_run_from_auto_detects_next_stage(self):
        """run_from() with start_from=None auto-detects from stage_history."""
        log = []
        stages = [
            OrderTrackingStage("classify", log),
            OrderTrackingStage("extract", log),
            OrderTrackingStage("store", log),
        ]
        runner = PipelineRunner(stages)
        config = PipelineConfig()

        # 'classify' already done, should start from 'extract'
        state = PipelineState(
            text="hello",
            stage_history=["classify"],
        )
        result = runner.run_from(state, config, start_from=None)

        assert log == ["extract", "store"]
        assert "extract" in result.stage_history
        assert "store" in result.stage_history

    def test_run_from_auto_detect_all_completed(self):
        """run_from() returns state as-is when all stages are already completed."""
        stages = [MockStage("classify"), MockStage("extract")]
        runner = PipelineRunner(stages)
        config = PipelineConfig()

        state = PipelineState(
            text="hello",
            stage_history=["classify", "extract"],
        )
        result = runner.run_from(state, config, start_from=None)

        # No stages executed; state returned as-is
        assert result is state

    def test_run_from_with_stop_after(self):
        """run_from() with stop_after limits execution range."""
        log = []
        stages = [
            OrderTrackingStage("classify", log),
            OrderTrackingStage("extract", log),
            OrderTrackingStage("store", log),
            OrderTrackingStage("link", log),
        ]
        runner = PipelineRunner(stages)
        config = PipelineConfig()

        state = PipelineState(text="hello", stage_history=["classify"])
        result = runner.run_from(state, config, start_from="extract", stop_after="store")

        assert log == ["extract", "store"]
        assert "link" not in result.stage_history


class TestPipelineRunnerUndoTo:
    """Tests for PipelineRunner.undo_to() — rollback."""

    def test_undo_to_calls_undo_in_reverse_order(self):
        """undo_to() calls undo on stages after the target, in reverse."""
        log = []
        stages = [
            OrderTrackingStage("classify", log),
            OrderTrackingStage("extract", log),
            OrderTrackingStage("store", log),
        ]
        runner = PipelineRunner(stages)

        state = PipelineState(
            text="hello",
            stage_history=["classify", "extract", "store"],
        )
        result = runner.undo_to(state, "classify")

        assert log == ["undo:store", "undo:extract"]
        assert "store" not in result.stage_history
        assert "extract" not in result.stage_history
        assert "classify" in result.stage_history

    def test_undo_to_target_not_in_history_raises(self):
        """undo_to() raises ValueError if target stage is not in history."""
        stages = [MockStage("classify"), MockStage("extract")]
        runner = PipelineRunner(stages)

        state = PipelineState(stage_history=["classify"])

        with pytest.raises(ValueError, match="not in history"):
            runner.undo_to(state, "extract")

    def test_undo_to_last_stage_undoes_nothing(self):
        """undo_to() with target=last stage in history undoes nothing."""
        log = []
        stages = [
            OrderTrackingStage("classify", log),
            OrderTrackingStage("extract", log),
        ]
        runner = PipelineRunner(stages)

        state = PipelineState(stage_history=["classify", "extract"])
        result = runner.undo_to(state, "extract")

        assert log == []
        assert result.stage_history == ["classify", "extract"]


class TestPipelineRunnerRetry:
    """Tests for retry logic in _execute_stage."""

    def test_retry_on_transient_failure(self):
        """A stage that fails once then succeeds with max_retries=1."""
        stage = FailThenSucceedStage("flaky", fail_count=1)
        runner = PipelineRunner([stage])
        config = PipelineConfig(retry=RetryConfig(max_retries=1, backoff_seconds=0.0, on_failure="abort"))

        state = runner.run("text", config)

        assert stage.executed is True
        assert "flaky" in state.stage_history

    def test_on_failure_skip(self):
        """on_failure='skip' skips the failed stage instead of raising."""
        stage = MockStage("failing", side_effect=RuntimeError("boom"))
        runner = PipelineRunner([stage])
        config = PipelineConfig(retry=RetryConfig(max_retries=0, backoff_seconds=0.0, on_failure="skip"))

        state = runner.run("text", config)

        assert "failing:skipped" in state.stage_history
        assert stage.executed is False

    def test_on_failure_abort_raises_runtime_error(self):
        """on_failure='abort' raises RuntimeError after exhausting retries."""
        stage = MockStage("failing", side_effect=RuntimeError("boom"))
        runner = PipelineRunner([stage])
        config = PipelineConfig(retry=RetryConfig(max_retries=0, backoff_seconds=0.0, on_failure="abort"))

        with pytest.raises(RuntimeError, match="failed after 1 attempts"):
            runner.run("text", config)

    def test_retry_exhaustion_then_abort(self):
        """Stage fails more times than max_retries allows, then aborts."""
        stage = FailThenSucceedStage("fragile", fail_count=5)
        runner = PipelineRunner([stage])
        config = PipelineConfig(retry=RetryConfig(max_retries=2, backoff_seconds=0.0, on_failure="abort"))

        with pytest.raises(RuntimeError, match="failed after 3 attempts"):
            runner.run("text", config)

    def test_retry_exhaustion_then_skip(self):
        """Stage fails more times than max_retries allows, then skips."""
        stage = FailThenSucceedStage("fragile", fail_count=5)
        runner = PipelineRunner([stage])
        config = PipelineConfig(retry=RetryConfig(max_retries=2, backoff_seconds=0.0, on_failure="skip"))

        state = runner.run("text", config)
        assert "fragile:skipped" in state.stage_history


class TestPipelineRunnerTimings:
    """Tests for stage timing recording."""

    def test_stage_timings_are_recorded(self):
        """Each stage's execution time is recorded in state.stage_timings."""
        stages = [MockStage("classify"), MockStage("extract")]
        runner = PipelineRunner(stages)
        config = PipelineConfig()

        state = runner.run("text", config)

        assert "classify" in state.stage_timings
        assert "extract" in state.stage_timings
        assert isinstance(state.stage_timings["classify"], float)
        assert isinstance(state.stage_timings["extract"], float)
        assert state.stage_timings["classify"] >= 0
        assert state.stage_timings["extract"] >= 0

    def test_stage_timings_in_milliseconds(self):
        """Stage timings are recorded in milliseconds (>= 0)."""
        stages = [MockStage("fast_stage")]
        runner = PipelineRunner(stages)
        config = PipelineConfig()

        state = runner.run("text", config)

        # Timing should be a non-negative float (milliseconds)
        timing = state.stage_timings["fast_stage"]
        assert timing >= 0.0


class TestPipelineRunnerEdgeCases:
    """Edge-case and integration-like tests for PipelineRunner."""

    def test_empty_pipeline(self):
        """An empty stages list produces a state with no stage_history."""
        runner = PipelineRunner([])
        config = PipelineConfig()

        state = runner.run("text", config)

        assert state.stage_history == []
        assert state.text == "text"
        assert state.completed_at is not None

    def test_single_stage_pipeline(self):
        """A pipeline with one stage executes it and records it."""
        stage = MockStage("only_stage")
        runner = PipelineRunner([stage])
        config = PipelineConfig()

        state = runner.run("text", config)

        assert stage.executed is True
        assert state.stage_history == ["only_stage"]

    def test_run_from_invalid_stage_raises_value_error(self):
        """run_from() with an unknown start_from stage raises ValueError."""
        stages = [MockStage("classify"), MockStage("extract")]
        runner = PipelineRunner(stages)
        config = PipelineConfig()

        state = PipelineState(text="hello")
        with pytest.raises(ValueError, match="not found"):
            runner.run_from(state, config, start_from="nonexistent")
