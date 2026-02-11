"""Unit tests for PipelineTokenTracker (CFS-1)."""

import pytest

from smartmemory.pipeline.token_tracker import PipelineTokenTracker, StageTokenRecord


class TestRecordSpent:
    """Tests for record_spent() accumulation."""

    def test_single_spent_record(self):
        tracker = PipelineTokenTracker(workspace_id="ws1")
        tracker.record_spent("llm_extract", prompt_tokens=500, completion_tokens=200, model="gpt-4o-mini")

        summary = tracker.summary()
        assert summary["total_spent"] == 700
        assert summary["total_avoided"] == 0
        assert "llm_extract" in summary["stages"]["spent"]
        stage = summary["stages"]["spent"]["llm_extract"]
        assert stage["prompt_tokens"] == 500
        assert stage["completion_tokens"] == 200
        assert stage["total_tokens"] == 700
        assert stage["call_count"] == 1
        assert stage["models"] == {"gpt-4o-mini": 1}

    def test_multiple_spent_same_stage(self):
        tracker = PipelineTokenTracker()
        tracker.record_spent("llm_extract", 100, 50, "gpt-4o-mini")
        tracker.record_spent("llm_extract", 200, 80, "gpt-4o-mini")

        summary = tracker.summary()
        stage = summary["stages"]["spent"]["llm_extract"]
        assert stage["prompt_tokens"] == 300
        assert stage["completion_tokens"] == 130
        assert stage["total_tokens"] == 430
        assert stage["call_count"] == 2

    def test_spent_across_stages(self):
        tracker = PipelineTokenTracker()
        tracker.record_spent("llm_extract", 500, 200, "gpt-4o-mini")
        tracker.record_spent("ontology_constrain", 100, 50, "gpt-4o-mini")

        summary = tracker.summary()
        assert summary["total_spent"] == 850
        assert len(summary["stages"]["spent"]) == 2

    def test_multiple_models(self):
        tracker = PipelineTokenTracker()
        tracker.record_spent("llm_extract", 500, 200, "gpt-4o-mini")
        tracker.record_spent("llm_extract", 300, 100, "llama-3.3-70b-versatile")

        stage = tracker.summary()["stages"]["spent"]["llm_extract"]
        assert stage["models"] == {"gpt-4o-mini": 1, "llama-3.3-70b-versatile": 1}


class TestRecordAvoided:
    """Tests for record_avoided() with reason attribution."""

    def test_cache_hit_avoided(self):
        tracker = PipelineTokenTracker()
        tracker.record_avoided("llm_extract", 800, model="gpt-4o-mini", reason="cache_hit")

        summary = tracker.summary()
        assert summary["total_avoided"] == 800
        stage = summary["stages"]["avoided"]["llm_extract"]
        assert stage["reasons"] == {"cache_hit": 1}

    def test_graph_lookup_avoided(self):
        tracker = PipelineTokenTracker()
        tracker.record_avoided("ground", 200, reason="graph_lookup")

        stage = tracker.summary()["stages"]["avoided"]["ground"]
        assert stage["reasons"] == {"graph_lookup": 1}

    def test_stage_disabled_avoided(self):
        tracker = PipelineTokenTracker()
        tracker.record_avoided("llm_extract", 800, model="gpt-4o-mini", reason="stage_disabled")

        stage = tracker.summary()["stages"]["avoided"]["llm_extract"]
        assert stage["reasons"] == {"stage_disabled": 1}

    def test_multiple_reasons_same_stage(self):
        tracker = PipelineTokenTracker()
        tracker.record_avoided("llm_extract", 800, reason="cache_hit")
        tracker.record_avoided("llm_extract", 600, reason="stage_disabled")

        stage = tracker.summary()["stages"]["avoided"]["llm_extract"]
        assert stage["total_tokens"] == 1400
        assert stage["reasons"] == {"cache_hit": 1, "stage_disabled": 1}


class TestSummary:
    """Tests for summary() aggregation, percentages, and cost."""

    def test_savings_percentage(self):
        tracker = PipelineTokenTracker()
        tracker.record_spent("llm_extract", 500, 200, "gpt-4o-mini")  # 700 spent
        tracker.record_avoided("llm_extract", 700, reason="cache_hit")  # 700 avoided

        summary = tracker.summary()
        assert summary["savings_pct"] == 50.0

    def test_zero_tokens_no_division_error(self):
        tracker = PipelineTokenTracker()
        summary = tracker.summary()
        assert summary["total_spent"] == 0
        assert summary["total_avoided"] == 0
        assert summary["savings_pct"] == 0.0

    def test_cost_estimation_gpt4o_mini(self):
        tracker = PipelineTokenTracker()
        # 1000 prompt tokens @ $0.00015/1k = $0.00015
        # 500 completion tokens @ $0.0006/1k = $0.0003
        tracker.record_spent("llm_extract", 1000, 500, "gpt-4o-mini")

        summary = tracker.summary()
        assert summary["cost_usd"]["spent"] == pytest.approx(0.00045, abs=1e-6)

    def test_cost_estimation_unknown_model_uses_default(self):
        tracker = PipelineTokenTracker()
        tracker.record_spent("llm_extract", 1000, 500, "some-unknown-model")

        summary = tracker.summary()
        # default: prompt=0.001, completion=0.002
        # 1000/1000 * 0.001 + 500/1000 * 0.002 = 0.001 + 0.001 = 0.002
        assert summary["cost_usd"]["spent"] == pytest.approx(0.002, abs=1e-6)

    def test_workspace_and_profile_in_summary(self):
        tracker = PipelineTokenTracker(workspace_id="ws-abc", profile_name="quick_extract")
        summary = tracker.summary()
        assert summary["workspace_id"] == "ws-abc"
        assert summary["profile_name"] == "quick_extract"

    def test_avoided_cost_estimated_separately(self):
        tracker = PipelineTokenTracker()
        tracker.record_spent("llm_extract", 500, 200, "gpt-4o-mini")
        tracker.record_avoided("llm_extract", 800, model="gpt-4o-mini", reason="cache_hit")

        summary = tracker.summary()
        assert summary["cost_usd"]["spent"] > 0
        assert summary["cost_usd"]["avoided"] > 0
        # Avoided uses prompt-only pricing (800 * 0.00015/1k = 0.00012)
        # Spent uses prompt + completion pricing
        assert summary["cost_usd"]["avoided"] == pytest.approx(0.00012, abs=1e-6)


class TestStageTokenRecord:
    """Tests for StageTokenRecord dataclass."""

    def test_total_tokens_property(self):
        rec = StageTokenRecord(prompt_tokens=100, completion_tokens=50)
        assert rec.total_tokens == 150

    def test_zero_tokens(self):
        rec = StageTokenRecord()
        assert rec.total_tokens == 0

    def test_reason_field(self):
        rec = StageTokenRecord(reason="cache_hit")
        assert rec.reason == "cache_hit"
