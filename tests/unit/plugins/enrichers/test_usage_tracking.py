"""Unit tests for enricher LLM usage tracking (CFS-1b)."""

import threading

import pytest

pytestmark = pytest.mark.unit


class TestEnricherUsageTracking:
    """Tests for enricher usage accumulation and retrieval."""

    def test_record_and_get_single_usage(self):
        """Single enricher usage is recorded and retrieved."""
        from smartmemory.plugins.enrichers.usage_tracking import (
            clear_enricher_usage,
            get_enricher_usage,
            record_enricher_usage,
        )

        clear_enricher_usage()

        record_enricher_usage(
            enricher_name="temporal_enricher",
            prompt_tokens=500,
            completion_tokens=200,
            model="gpt-4o-mini",
        )

        usage = get_enricher_usage()
        assert usage is not None
        assert usage["total_prompt_tokens"] == 500
        assert usage["total_completion_tokens"] == 200
        assert usage["total_tokens"] == 700
        assert len(usage["records"]) == 1
        assert usage["records"][0]["enricher_name"] == "temporal_enricher"
        assert usage["records"][0]["model"] == "gpt-4o-mini"

    def test_accumulate_multiple_enricher_calls(self):
        """Multiple enricher calls accumulate in a single pass."""
        from smartmemory.plugins.enrichers.usage_tracking import (
            clear_enricher_usage,
            get_enricher_usage,
            record_enricher_usage,
        )

        clear_enricher_usage()

        # Simulate temporal enricher
        record_enricher_usage("temporal_enricher", 300, 150, "gpt-4o-mini")

        # Simulate link expansion enricher
        record_enricher_usage("link_expansion_enricher", 800, 400, "gpt-4o-mini")

        usage = get_enricher_usage()
        assert usage is not None
        assert usage["total_prompt_tokens"] == 1100
        assert usage["total_completion_tokens"] == 550
        assert usage["total_tokens"] == 1650
        assert len(usage["records"]) == 2

    def test_consume_once_behavior(self):
        """get_enricher_usage clears after retrieval (consume-once)."""
        from smartmemory.plugins.enrichers.usage_tracking import (
            clear_enricher_usage,
            get_enricher_usage,
            record_enricher_usage,
        )

        clear_enricher_usage()

        record_enricher_usage("temporal_enricher", 100, 50, "gpt-4o-mini")

        # First call returns usage
        usage1 = get_enricher_usage()
        assert usage1 is not None

        # Second call returns None
        usage2 = get_enricher_usage()
        assert usage2 is None

    def test_returns_none_when_no_usage(self):
        """Returns None when no enricher calls were made."""
        from smartmemory.plugins.enrichers.usage_tracking import (
            clear_enricher_usage,
            get_enricher_usage,
        )

        clear_enricher_usage()

        usage = get_enricher_usage()
        assert usage is None

    def test_thread_isolation(self):
        """Usage is isolated per thread."""
        from smartmemory.plugins.enrichers.usage_tracking import (
            clear_enricher_usage,
            get_enricher_usage,
            record_enricher_usage,
        )

        results = {}

        def thread_a():
            clear_enricher_usage()
            record_enricher_usage("enricher_a", 100, 50, "model-a")
            results["a"] = get_enricher_usage()

        def thread_b():
            clear_enricher_usage()
            record_enricher_usage("enricher_b", 200, 100, "model-b")
            results["b"] = get_enricher_usage()

        t1 = threading.Thread(target=thread_a)
        t2 = threading.Thread(target=thread_b)

        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert results["a"]["total_prompt_tokens"] == 100
        assert results["a"]["records"][0]["enricher_name"] == "enricher_a"
        assert results["b"]["total_prompt_tokens"] == 200
        assert results["b"]["records"][0]["enricher_name"] == "enricher_b"


class TestEnricherUsageAccumulator:
    """Tests for EnricherUsageAccumulator dataclass."""

    def test_accumulator_totals(self):
        """Accumulator correctly sums token counts."""
        from smartmemory.plugins.enrichers.usage_tracking import EnricherUsageAccumulator

        acc = EnricherUsageAccumulator()
        acc.add("enricher1", 100, 50, "model1")
        acc.add("enricher2", 200, 100, "model2")

        assert acc.total_prompt_tokens == 300
        assert acc.total_completion_tokens == 150
        assert acc.total_tokens == 450
        assert len(acc.records) == 2

    def test_accumulator_empty(self):
        """Empty accumulator returns zeros."""
        from smartmemory.plugins.enrichers.usage_tracking import EnricherUsageAccumulator

        acc = EnricherUsageAccumulator()

        assert acc.total_prompt_tokens == 0
        assert acc.total_completion_tokens == 0
        assert acc.total_tokens == 0
        assert len(acc.records) == 0
