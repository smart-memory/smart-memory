"""Unit tests for agentic routines: AgenticSelfMonitor, PlanScorer, and planning functions."""

from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.unit

from smartmemory.feedback.agentic_routines import (
    AgenticSelfMonitor,
    PlanScorer,
    agentic_goal_directed_planning,
    agentic_chain_of_thought,
    agentic_weighted_chain_of_thought,
    agentic_mcts_planning,
)


# ---------------------------------------------------------------------------
# AgenticSelfMonitor
# ---------------------------------------------------------------------------
class TestAgenticSelfMonitor:
    def test_log_usage(self):
        monitor = AgenticSelfMonitor()
        monitor.log_usage("search", {"query": "test"})
        assert len(monitor.usage_log) == 1
        assert monitor.usage_log[0]["action"] == "search"
        assert monitor.usage_log[0]["details"]["query"] == "test"

    def test_log_feedback(self):
        monitor = AgenticSelfMonitor()
        monitor.log_feedback("good", {"step": 1})
        assert len(monitor.feedback_log) == 1
        assert monitor.feedback_log[0]["feedback"] == "good"

    def test_log_iteration(self):
        monitor = AgenticSelfMonitor()
        monitor.log_iteration(0, "First step done")
        assert len(monitor.iteration_log) == 1
        assert monitor.iteration_log[0]["iteration"] == 0
        assert monitor.iteration_log[0]["summary"] == "First step done"

    def test_summarize_usage(self):
        monitor = AgenticSelfMonitor()
        monitor.log_usage("search", {})
        monitor.log_usage("store", {})
        summary = monitor.summarize_usage()
        assert "Total actions: 2" in summary
        assert "search" in summary
        assert "store" in summary

    def test_summarize_feedback_empty(self):
        monitor = AgenticSelfMonitor()
        summary = monitor.summarize_feedback()
        assert "Feedback count: 0" in summary
        assert "Last: None" in summary

    def test_summarize_feedback_with_entries(self):
        monitor = AgenticSelfMonitor()
        monitor.log_feedback("bad", {})
        monitor.log_feedback("good", {})
        summary = monitor.summarize_feedback()
        assert "Feedback count: 2" in summary
        assert "Last: good" in summary

    def test_summarize_iterations_empty(self):
        monitor = AgenticSelfMonitor()
        summary = monitor.summarize_iterations()
        assert "Iterations: 0" in summary
        assert "Last: None" in summary

    def test_summarize_iterations_with_entries(self):
        monitor = AgenticSelfMonitor()
        monitor.log_iteration(0, "step 0")
        monitor.log_iteration(1, "step 1")
        summary = monitor.summarize_iterations()
        assert "Iterations: 2" in summary
        assert "Last: step 1" in summary

    def test_reset(self):
        monitor = AgenticSelfMonitor()
        monitor.log_usage("a", {})
        monitor.log_feedback("b", {})
        monitor.log_iteration(0, "c")
        monitor.reset()
        assert len(monitor.usage_log) == 0
        assert len(monitor.feedback_log) == 0
        assert len(monitor.iteration_log) == 0


# ---------------------------------------------------------------------------
# PlanScorer
# ---------------------------------------------------------------------------
class TestPlanScorer:
    def test_empty_plan_scores_zero(self):
        scorer = PlanScorer()
        assert scorer.score_plan([]) == 0

    def test_plan_length_reward(self):
        scorer = PlanScorer()
        score = scorer.score_plan(["a", "b", "c"])
        # min(3, 10) + len({"a","b","c"}) = 3 + 3 = 6
        assert score == 6

    def test_plan_length_capped_at_10(self):
        scorer = PlanScorer()
        plan = [f"step_{i}" for i in range(15)]
        score = scorer.score_plan(plan)
        # min(15, 10) + 15 unique = 10 + 15 = 25
        assert score == 25

    def test_cycle_penalty(self):
        scorer = PlanScorer()
        # Plan with duplicate: ["a", "b", "a"]
        score = scorer.score_plan(["a", "b", "a"])
        # min(3, 10) + len({"a","b"}) - 2 = 3 + 2 - 2 = 3
        assert score == 3

    def test_good_feedback_bonus(self):
        scorer = PlanScorer()
        score_no_fb = scorer.score_plan(["a", "b"])
        scorer2 = PlanScorer()
        score_good = scorer2.score_plan(["a", "b"], feedbacks=["good"])
        assert score_good == score_no_fb + 2

    def test_bad_feedback_penalty(self):
        scorer = PlanScorer()
        score_no_fb = scorer.score_plan(["a", "b"])
        scorer2 = PlanScorer()
        score_bad = scorer2.score_plan(["a", "b"], feedbacks=["bad"])
        assert score_bad == score_no_fb - 2

    def test_scores_accumulated(self):
        scorer = PlanScorer()
        scorer.score_plan(["a"])
        scorer.score_plan(["b", "c"])
        assert len(scorer.scores) == 2

    def test_feedback_accumulated(self):
        scorer = PlanScorer()
        scorer.score_plan(["a"], feedbacks=["good"])
        scorer.score_plan(["b"], feedbacks=["bad"])
        assert len(scorer.feedback) == 2


# ---------------------------------------------------------------------------
# agentic_goal_directed_planning
# ---------------------------------------------------------------------------
class TestAgenticGoalDirectedPlanning:
    def _make_store(self, results_per_query=None):
        store = MagicMock()
        if results_per_query is None:
            store.search.return_value = []
        else:
            store.search.side_effect = results_per_query
        return store

    def test_empty_search_results_stops(self):
        store = self._make_store()
        plan, monitor = agentic_goal_directed_planning(None, store, "goal", max_steps=5)
        assert plan == ["goal"]
        assert len(monitor.usage_log) >= 1

    def test_greedy_strategy_picks_top(self):
        store = self._make_store([
            [{"key": "step2", "content": "next"}],
            [],  # stop
        ])
        plan, monitor = agentic_goal_directed_planning(None, store, "start", max_steps=3, strategy="greedy")
        assert "start" in plan
        assert "step2" in plan

    def test_breadth_strategy_adds_all(self):
        store = self._make_store([
            [{"key": "a"}, {"key": "b"}, {"key": "c"}],
            [],
        ])
        plan, monitor = agentic_goal_directed_planning(None, store, "root", max_steps=2, strategy="breadth")
        assert "root" in plan
        assert "a" in plan
        assert "b" in plan
        assert "c" in plan

    def test_depth_strategy_picks_first_unexplored(self):
        store = self._make_store([
            [{"key": "deep1"}, {"key": "deep2"}],
            [],
        ])
        plan, monitor = agentic_goal_directed_planning(None, store, "root", max_steps=2, strategy="depth")
        assert "deep1" in plan
        # deep2 should NOT be added in depth-first
        assert "deep2" not in plan

    def test_feedback_fn_called(self):
        store = self._make_store([[{"key": "next"}], []])
        feedback_calls = []

        def feedback_fn(step, context, results):
            feedback_calls.append(step)
            return "good"

        plan, monitor = agentic_goal_directed_planning(
            None, store, "goal", max_steps=2, feedback_fn=feedback_fn
        )
        assert len(feedback_calls) >= 1

    def test_stop_feedback_halts_planning(self):
        store = self._make_store([
            [{"key": "step2"}],
            [{"key": "step3"}],
            [{"key": "step4"}],
        ])

        def feedback_fn(step, context, results):
            return "stop" if step >= 1 else "continue"

        plan, monitor = agentic_goal_directed_planning(
            None, store, "goal", max_steps=5, feedback_fn=feedback_fn
        )
        # Should stop after step 1 due to "stop" feedback
        assert len(plan) <= 3

    def test_scorer_called_at_end(self):
        store = self._make_store()
        scorer = PlanScorer()
        plan, monitor = agentic_goal_directed_planning(
            None, store, "goal", max_steps=1, scorer=scorer
        )
        assert len(scorer.scores) == 1

    def test_avoids_cycles(self):
        store = MagicMock()
        # Always returns the start goal again
        store.search.return_value = [{"key": "goal"}]
        plan, monitor = agentic_goal_directed_planning(
            None, store, "goal", max_steps=3, strategy="greedy"
        )
        # "goal" already explored, so greedy won't add it again
        assert plan.count("goal") == 1


# ---------------------------------------------------------------------------
# agentic_chain_of_thought
# ---------------------------------------------------------------------------
class TestAgenticChainOfThought:
    def test_basic_chain(self):
        store = MagicMock()
        store.search.return_value = [{"content": "next thought"}]
        thoughts = agentic_chain_of_thought(None, store, "initial prompt", steps=3)
        assert len(thoughts) == 3
        assert thoughts[0]["step"] == 0
        assert thoughts[0]["prompt"] == "initial prompt"

    def test_prompt_evolves_from_results(self):
        store = MagicMock()
        store.search.side_effect = [
            [{"content": "thought A"}],
            [{"content": "thought B"}],
            [{"content": "thought C"}],
        ]
        thoughts = agentic_chain_of_thought(None, store, "start", steps=3)
        assert thoughts[1]["prompt"] == "thought A"
        assert thoughts[2]["prompt"] == "thought B"

    def test_empty_results_keep_prompt(self):
        store = MagicMock()
        store.search.return_value = []
        thoughts = agentic_chain_of_thought(None, store, "stuck", steps=2)
        assert thoughts[0]["prompt"] == "stuck"
        assert thoughts[1]["prompt"] == "stuck"

    def test_single_step(self):
        store = MagicMock()
        store.search.return_value = [{"content": "only"}]
        thoughts = agentic_chain_of_thought(None, store, "q", steps=1)
        assert len(thoughts) == 1


# ---------------------------------------------------------------------------
# agentic_weighted_chain_of_thought
# ---------------------------------------------------------------------------
class TestAgenticWeightedChainOfThought:
    def test_basic_weighted_chain(self):
        store = MagicMock()
        store.search.return_value = [{"content": "result A"}, {"content": "result B"}]
        thoughts = agentic_weighted_chain_of_thought(None, store, "start", steps=2)
        assert len(thoughts) == 2
        assert thoughts[0]["step"] == 0
        assert thoughts[0]["prompt"] == "start"

    def test_default_weights_are_uniform(self):
        store = MagicMock()
        store.search.return_value = [{"content": "a"}, {"content": "b"}]
        thoughts = agentic_weighted_chain_of_thought(None, store, "q", steps=1)
        assert thoughts[0]["weights"] == [1.0, 1.0]

    def test_custom_weight_fn(self):
        store = MagicMock()
        store.search.return_value = [
            {"content": "low", "score": 0.1},
            {"content": "high", "score": 0.9},
        ]
        weight_fn = lambda r: r["score"]
        thoughts = agentic_weighted_chain_of_thought(
            None, store, "start", steps=2, weight_fn=weight_fn
        )
        # First step picks "high" (max weight), so second step prompt should be "high"
        assert thoughts[1]["prompt"] == "high"

    def test_empty_results_keep_prompt(self):
        store = MagicMock()
        store.search.return_value = []
        thoughts = agentic_weighted_chain_of_thought(None, store, "stuck", steps=2)
        assert thoughts[0]["prompt"] == "stuck"
        assert thoughts[1]["prompt"] == "stuck"
        assert thoughts[0]["weights"] == []

    def test_single_step(self):
        store = MagicMock()
        store.search.return_value = [{"content": "only"}]
        thoughts = agentic_weighted_chain_of_thought(None, store, "q", steps=1)
        assert len(thoughts) == 1


# ---------------------------------------------------------------------------
# agentic_mcts_planning
# ---------------------------------------------------------------------------
class TestAgenticMCTSPlanning:
    def test_returns_best_plan_and_score(self):
        store = MagicMock()
        store.search.return_value = [{"key": "node_a"}, {"key": "node_b"}]
        plan, score = agentic_mcts_planning(store, "root", max_depth=3, simulations=5)
        assert isinstance(plan, list)
        assert isinstance(score, (int, float))
        assert "root" in plan

    def test_empty_search_returns_start_only(self):
        store = MagicMock()
        store.search.return_value = []
        plan, score = agentic_mcts_planning(store, "root", max_depth=3, simulations=3)
        assert plan == ["root"]

    def test_custom_scorer(self):
        store = MagicMock()
        store.search.return_value = [{"key": "a"}]
        scorer = PlanScorer()
        plan, score = agentic_mcts_planning(store, "root", simulations=2, scorer=scorer)
        assert len(scorer.scores) >= 1

    def test_avoids_cycles_in_plan(self):
        store = MagicMock()
        # Always returns same key — should detect cycle and stop
        store.search.return_value = [{"key": "loop"}]
        plan, score = agentic_mcts_planning(store, "root", max_depth=5, simulations=1)
        # "loop" should appear at most once after "root"
        assert plan.count("loop") <= 1
