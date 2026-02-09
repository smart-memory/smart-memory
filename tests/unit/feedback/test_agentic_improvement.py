"""Unit tests for AgenticImprover and fully_automatic_agentic_loop."""

from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.unit

from smartmemory.feedback.agentic_improvement import AgenticImprover, fully_automatic_agentic_loop


class TestAgenticImprover:
    def test_record_stores_plan_and_feedbacks(self):
        improver = AgenticImprover()
        improver.record(["step1", "step2"], [1, -1])
        assert len(improver.history) == 1
        assert improver.history[0]["plan"] == ["step1", "step2"]
        assert improver.history[0]["feedbacks"] == [1, -1]

    def test_record_multiple_entries(self):
        improver = AgenticImprover()
        improver.record(["a"], [1])
        improver.record(["b"], [-1])
        assert len(improver.history) == 2

    def test_aggregate_feedback_numeric(self):
        improver = AgenticImprover()
        improver.record([], [1.0, 0.5, -0.5])
        assert improver.aggregate_feedback() == pytest.approx(1.0 / 3)

    def test_aggregate_feedback_string_good(self):
        improver = AgenticImprover()
        improver.record([], ["good", "very good", "looks good"])
        assert improver.aggregate_feedback() == pytest.approx(1.0)

    def test_aggregate_feedback_string_bad(self):
        improver = AgenticImprover()
        improver.record([], ["bad", "stop"])
        assert improver.aggregate_feedback() == pytest.approx(-1.0)

    def test_aggregate_feedback_string_neutral(self):
        improver = AgenticImprover()
        improver.record([], ["ok", "fine", "whatever"])
        assert improver.aggregate_feedback() == pytest.approx(0.0)

    def test_aggregate_feedback_mixed(self):
        improver = AgenticImprover()
        improver.record([], [1, "bad", "good", 0])
        # 1 + (-1) + 1 + 0 = 1, /4 = 0.25
        assert improver.aggregate_feedback() == pytest.approx(0.25)

    def test_aggregate_feedback_empty(self):
        improver = AgenticImprover()
        assert improver.aggregate_feedback() == 0.0

    def test_improve_triggers_update_on_negative(self):
        update_fn = MagicMock()
        improver = AgenticImprover(update_fn=update_fn)
        improver.record([], ["bad", "bad"])

        agent = MagicMock()
        store = MagicMock()
        result = improver.improve(agent, store, ["plan"], ["bad"])
        assert result == "Triggered improvement"
        update_fn.assert_called_once_with(agent, store, ["plan"], ["bad"])

    def test_improve_no_action_on_positive(self):
        update_fn = MagicMock()
        improver = AgenticImprover(update_fn=update_fn)
        improver.record([], ["good", "good"])

        result = improver.improve(MagicMock(), MagicMock(), [], [])
        assert result == "No improvement needed"
        update_fn.assert_not_called()

    def test_improve_no_update_fn(self):
        improver = AgenticImprover()
        improver.record([], ["bad"])
        result = improver.improve(MagicMock(), MagicMock(), [], [])
        assert result == "Triggered improvement"


class TestFullyAutomaticAgenticLoop:
    def test_stops_on_positive_feedback(self):
        agent = MagicMock()
        store = MagicMock()
        feedback_mgr = MagicMock()
        feedback_mgr.get_channel.return_value = lambda step, ctx, res: "good"

        improver = AgenticImprover()

        monitor = MagicMock()
        monitor.feedback_log = [{"feedback": "good"}]

        def planner_fn(agent, memory_store, goal, feedback_fn, **kwargs):
            return ["plan_step"], monitor

        plan, history, improvements = fully_automatic_agentic_loop(
            agent, store, "goal", feedback_mgr, improver, planner_fn,
            max_cycles=5, stop_on_positive=True
        )
        assert plan == ["plan_step"]
        # Should stop after 1 cycle since feedback is positive
        assert len(improvements) == 1
        assert improvements[0] == "No improvement needed"

    def test_runs_max_cycles_on_negative(self):
        agent = MagicMock()
        store = MagicMock()
        feedback_mgr = MagicMock()
        feedback_mgr.get_channel.return_value = lambda step, ctx, res: "bad"

        update_fn = MagicMock()
        improver = AgenticImprover(update_fn=update_fn)

        monitor = MagicMock()
        monitor.feedback_log = [{"feedback": "bad"}]

        def planner_fn(agent, memory_store, goal, feedback_fn, **kwargs):
            return ["plan"], monitor

        plan, history, improvements = fully_automatic_agentic_loop(
            agent, store, "goal", feedback_mgr, improver, planner_fn,
            max_cycles=3, stop_on_positive=True
        )
        assert len(improvements) == 3
        assert all(imp == "Triggered improvement" for imp in improvements)

    def test_planner_kwargs_forwarded(self):
        agent = MagicMock()
        store = MagicMock()
        feedback_mgr = MagicMock()
        feedback_mgr.get_channel.return_value = None

        improver = AgenticImprover()
        monitor = MagicMock()
        monitor.feedback_log = [{"feedback": "good"}]

        received_kwargs = {}

        def planner_fn(agent, memory_store, goal, feedback_fn, **kwargs):
            received_kwargs.update(kwargs)
            return ["plan"], monitor

        fully_automatic_agentic_loop(
            agent, store, "goal", feedback_mgr, improver, planner_fn,
            max_cycles=1, planner_kwargs={"temperature": 0.7}
        )
        assert received_kwargs["temperature"] == 0.7
