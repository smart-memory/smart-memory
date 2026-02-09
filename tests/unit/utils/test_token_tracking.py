"""Unit tests for utils.token_tracking — LLM token usage tracking."""

from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.unit

from smartmemory.utils.token_tracking import (
    TokenUsage,
    AggregatedUsage,
    TokenTracker,
    get_global_tracker,
    track_usage,
    get_usage,
    reset_usage,
    estimate_cost,
    COST_PER_1K_TOKENS,
)


# ---------------------------------------------------------------------------
# TokenUsage dataclass
# ---------------------------------------------------------------------------
class TestTokenUsage:
    def test_defaults(self):
        u = TokenUsage()
        assert u.prompt_tokens == 0
        assert u.completion_tokens == 0
        assert u.total_tokens == 0
        assert u.model == ""

    def test_auto_total(self):
        u = TokenUsage(prompt_tokens=100, completion_tokens=50)
        assert u.total_tokens == 150

    def test_explicit_total(self):
        u = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=200)
        assert u.total_tokens == 200

    def test_timestamp_set(self):
        u = TokenUsage()
        assert u.timestamp is not None


# ---------------------------------------------------------------------------
# AggregatedUsage
# ---------------------------------------------------------------------------
class TestAggregatedUsage:
    def test_add_single(self):
        agg = AggregatedUsage()
        agg.add(TokenUsage(prompt_tokens=10, completion_tokens=5, model="gpt-4o"))
        assert agg.prompt_tokens == 10
        assert agg.completion_tokens == 5
        assert agg.total_tokens == 15
        assert agg.call_count == 1
        assert agg.models_used == {"gpt-4o": 1}

    def test_add_multiple(self):
        agg = AggregatedUsage()
        agg.add(TokenUsage(prompt_tokens=10, completion_tokens=5, model="gpt-4o"))
        agg.add(TokenUsage(prompt_tokens=20, completion_tokens=10, model="gpt-4o"))
        agg.add(TokenUsage(prompt_tokens=5, completion_tokens=3, model="gpt-5"))
        assert agg.call_count == 3
        assert agg.models_used == {"gpt-4o": 2, "gpt-5": 1}
        assert agg.total_tokens == 53

    def test_to_dict(self):
        agg = AggregatedUsage()
        agg.add(TokenUsage(prompt_tokens=10, completion_tokens=5, model="m"))
        d = agg.to_dict()
        assert d["prompt_tokens"] == 10
        assert d["completion_tokens"] == 5
        assert d["call_count"] == 1
        assert isinstance(d["models_used"], dict)


# ---------------------------------------------------------------------------
# TokenTracker
# ---------------------------------------------------------------------------
class TestTokenTracker:
    def test_track_explicit_tokens(self):
        t = TokenTracker()
        t.track(prompt_tokens=100, completion_tokens=50, model="gpt-4o")
        usage = t.get_usage()
        assert usage.total_tokens == 150
        assert usage.call_count == 1

    def test_track_from_dict_response(self):
        t = TokenTracker()
        resp = {
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "model": "gpt-4o",
        }
        t.track(response=resp)
        assert t.get_usage().total_tokens == 15

    def test_track_from_object_response(self):
        t = TokenTracker()
        resp = MagicMock()
        resp.usage.prompt_tokens = 20
        resp.usage.completion_tokens = 10
        resp.usage.total_tokens = 30
        resp.model = "gpt-5"
        t.track(response=resp)
        assert t.get_usage().total_tokens == 30

    def test_disabled_tracking(self):
        t = TokenTracker()
        t.disable()
        t.track(prompt_tokens=100, completion_tokens=50)
        assert t.get_usage().total_tokens == 0

    def test_enable_after_disable(self):
        t = TokenTracker()
        t.disable()
        t.enable()
        t.track(prompt_tokens=10, completion_tokens=5)
        assert t.get_usage().total_tokens == 15

    def test_reset(self):
        t = TokenTracker()
        t.track(prompt_tokens=100, completion_tokens=50)
        t.reset()
        assert t.get_usage().total_tokens == 0
        assert t.get_usage().call_count == 0

    def test_get_history(self):
        t = TokenTracker()
        t.track(prompt_tokens=10, completion_tokens=5, model="a")
        t.track(prompt_tokens=20, completion_tokens=10, model="b")
        history = t.get_history()
        assert len(history) == 2
        assert history[0].model == "a"
        assert history[1].model == "b"

    def test_str(self):
        t = TokenTracker()
        t.track(prompt_tokens=100, completion_tokens=50)
        s = str(t)
        assert "150" in s
        assert "1 calls" in s


# ---------------------------------------------------------------------------
# Global tracker functions
# ---------------------------------------------------------------------------
class TestGlobalTracker:
    @pytest.fixture(autouse=True)
    def _reset(self):
        reset_usage()
        yield
        reset_usage()

    def test_track_and_get(self):
        track_usage(prompt_tokens=10, completion_tokens=5, model="gpt-4o")
        usage = get_usage()
        assert usage["total_tokens"] == 15

    def test_get_global_tracker_instance(self):
        tracker = get_global_tracker()
        assert isinstance(tracker, TokenTracker)


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------
class TestEstimateCost:
    def test_known_model(self):
        usage = AggregatedUsage(prompt_tokens=1000, completion_tokens=500)
        cost = estimate_cost(usage, model="gpt-4o")
        expected = (1000 / 1000) * 0.005 + (500 / 1000) * 0.015
        assert cost == pytest.approx(expected)

    def test_default_model(self):
        usage = AggregatedUsage(prompt_tokens=1000, completion_tokens=500)
        cost = estimate_cost(usage, model="unknown-model")
        pricing = COST_PER_1K_TOKENS["default"]
        expected = (1000 / 1000) * pricing["prompt"] + (500 / 1000) * pricing["completion"]
        assert cost == pytest.approx(expected)

    def test_zero_tokens(self):
        usage = AggregatedUsage()
        assert estimate_cost(usage) == 0.0

    def test_prefix_match(self):
        usage = AggregatedUsage(prompt_tokens=1000, completion_tokens=500)
        cost = estimate_cost(usage, model="gpt-5-turbo")
        pricing = COST_PER_1K_TOKENS["gpt-5"]
        expected = (1000 / 1000) * pricing["prompt"] + (500 / 1000) * pricing["completion"]
        assert cost == pytest.approx(expected)
