"""Unit tests for EnrichStage."""

from unittest.mock import MagicMock, patch

import pytest

from smartmemory.pipeline.config import EnrichConfig, PipelineConfig
from smartmemory.pipeline.state import PipelineState
from smartmemory.pipeline.stages.enrich import EnrichStage

pytestmark = pytest.mark.unit


class TestEnrichStage:
    """Tests for the enrich pipeline stage."""

    def _make_stage(self, run_return=None):
        """Build an EnrichStage with a mocked EnrichmentPipeline."""
        pipeline = MagicMock()
        pipeline.run_enrichment.return_value = run_return or {}
        return EnrichStage(pipeline), pipeline

    def test_enrich_preview_mode_noop(self):
        """In preview mode, state is returned unchanged."""
        stage, pipeline = self._make_stage()
        state = PipelineState(text="Preview enrichment.", item_id="p1")
        config = PipelineConfig.preview()

        result = stage.execute(state, config)

        assert result is state
        pipeline.run_enrichment.assert_not_called()

    def test_enrich_sets_enrichments(self):
        """Enrichment results are stored in state.enrichments."""
        stage, pipeline = self._make_stage(run_return={"sentiment": "positive", "topics": ["AI"]})
        state = PipelineState(text="AI is great.", item_id="item_1")
        config = PipelineConfig.default()

        result = stage.execute(state, config)

        assert result.enrichments == {"sentiment": "positive", "topics": ["AI"]}

    def test_enrich_handles_failure(self):
        """When run_enrichment raises, enrichments default to empty dict."""
        stage, pipeline = self._make_stage()
        pipeline.run_enrichment.side_effect = RuntimeError("enrichment failed")

        state = PipelineState(text="Failing enrichment.", item_id="item_1")
        config = PipelineConfig.default()

        result = stage.execute(state, config)

        assert result.enrichments == {}

    def test_enrich_passes_enricher_names(self):
        """When config specifies enricher_names, they appear in the context."""
        stage, pipeline = self._make_stage(run_return={"sentiment": "neutral"})
        state = PipelineState(text="Sentiment test.", item_id="item_1")
        config = PipelineConfig.default()
        config.enrich = EnrichConfig(enricher_names=["sentiment"])

        stage.execute(state, config)

        pipeline.run_enrichment.assert_called_once()
        context = pipeline.run_enrichment.call_args[0][0]
        assert context["enricher_names"] == ["sentiment"]

    def test_enrich_no_enricher_names(self):
        """When enricher_names is None, it is not included in context."""
        stage, pipeline = self._make_stage(run_return={})
        state = PipelineState(text="No enricher names.", item_id="item_1")
        config = PipelineConfig.default()

        stage.execute(state, config)

        context = pipeline.run_enrichment.call_args[0][0]
        assert "enricher_names" not in context

    def test_undo_clears_enrichments(self):
        """Undo resets enrichments to empty dict."""
        stage, _ = self._make_stage()
        state = PipelineState(
            text="Content.",
            enrichments={"sentiment": "positive"},
        )

        result = stage.undo(state)

        assert result.enrichments == {}


class TestEnrichStageTokenTracking:
    """Tests for enricher LLM token tracking in EnrichStage (CFS-1b)."""

    def _make_stage(self, run_return=None):
        """Build an EnrichStage with a mocked EnrichmentPipeline."""
        pipeline = MagicMock()
        pipeline.run_enrichment.return_value = run_return or {}
        return EnrichStage(pipeline), pipeline

    def test_track_enricher_usage_single_enricher(self):
        """Token usage from a single enricher is recorded."""
        from smartmemory.pipeline.token_tracker import PipelineTokenTracker

        stage, _ = self._make_stage()
        tracker = PipelineTokenTracker(workspace_id="ws-test")
        state = PipelineState(text="Test content.", item_id="item_1", token_tracker=tracker)
        config = PipelineConfig.default()

        # Mock enricher usage
        mock_usage = {
            "total_prompt_tokens": 500,
            "total_completion_tokens": 200,
            "total_tokens": 700,
            "records": [
                {
                    "enricher_name": "temporal_enricher",
                    "prompt_tokens": 500,
                    "completion_tokens": 200,
                    "total_tokens": 700,
                    "model": "gpt-4o-mini",
                }
            ],
        }

        with patch(
            "smartmemory.plugins.enrichers.usage_tracking.get_enricher_usage",
            return_value=mock_usage,
        ):
            stage.execute(state, config)

        summary = tracker.summary()
        assert "enrich" in summary["stages"]["spent"]
        enrich_spent = summary["stages"]["spent"]["enrich"]
        assert enrich_spent["prompt_tokens"] == 500
        assert enrich_spent["completion_tokens"] == 200
        assert enrich_spent["total_tokens"] == 700

    def test_track_enricher_usage_multiple_enrichers(self):
        """Token usage from multiple enrichers is accumulated."""
        from smartmemory.pipeline.token_tracker import PipelineTokenTracker

        stage, _ = self._make_stage()
        tracker = PipelineTokenTracker(workspace_id="ws-test")
        state = PipelineState(text="Test content.", item_id="item_1", token_tracker=tracker)
        config = PipelineConfig.default()

        # Mock usage from two enrichers
        mock_usage = {
            "total_prompt_tokens": 900,
            "total_completion_tokens": 400,
            "total_tokens": 1300,
            "records": [
                {
                    "enricher_name": "temporal_enricher",
                    "prompt_tokens": 400,
                    "completion_tokens": 150,
                    "total_tokens": 550,
                    "model": "gpt-4o-mini",
                },
                {
                    "enricher_name": "link_expansion_enricher",
                    "prompt_tokens": 500,
                    "completion_tokens": 250,
                    "total_tokens": 750,
                    "model": "gpt-4o-mini",
                },
            ],
        }

        with patch(
            "smartmemory.plugins.enrichers.usage_tracking.get_enricher_usage",
            return_value=mock_usage,
        ):
            stage.execute(state, config)

        summary = tracker.summary()
        enrich_spent = summary["stages"]["spent"]["enrich"]
        # Each record is recorded separately, so call_count should be 2
        assert enrich_spent["call_count"] == 2
        assert enrich_spent["prompt_tokens"] == 900
        assert enrich_spent["completion_tokens"] == 400

    def test_track_enricher_usage_no_tracker(self):
        """When no token tracker, tracking is silently skipped."""
        stage, _ = self._make_stage()
        state = PipelineState(text="Test content.", item_id="item_1", token_tracker=None)
        config = PipelineConfig.default()

        # Should not raise even with no tracker
        with patch(
            "smartmemory.plugins.enrichers.usage_tracking.get_enricher_usage",
            return_value={"records": [{"prompt_tokens": 100}]},
        ):
            result = stage.execute(state, config)

        assert result.enrichments == {}

    def test_track_enricher_usage_no_usage_data(self):
        """When no usage data, tracking is silently skipped."""
        from smartmemory.pipeline.token_tracker import PipelineTokenTracker

        stage, _ = self._make_stage()
        tracker = PipelineTokenTracker(workspace_id="ws-test")
        state = PipelineState(text="Test content.", item_id="item_1", token_tracker=tracker)
        config = PipelineConfig.default()

        with patch(
            "smartmemory.plugins.enrichers.usage_tracking.get_enricher_usage",
            return_value=None,
        ):
            stage.execute(state, config)

        summary = tracker.summary()
        assert summary["total_spent"] == 0

    def test_track_enricher_usage_on_failure(self):
        """Token tracking happens even when enrichment fails."""
        from smartmemory.pipeline.token_tracker import PipelineTokenTracker

        stage, pipeline = self._make_stage()
        pipeline.run_enrichment.side_effect = RuntimeError("enrichment failed")

        tracker = PipelineTokenTracker(workspace_id="ws-test")
        state = PipelineState(text="Test content.", item_id="item_1", token_tracker=tracker)
        config = PipelineConfig.default()

        mock_usage = {
            "total_prompt_tokens": 300,
            "total_completion_tokens": 100,
            "total_tokens": 400,
            "records": [
                {
                    "enricher_name": "temporal_enricher",
                    "prompt_tokens": 300,
                    "completion_tokens": 100,
                    "total_tokens": 400,
                    "model": "gpt-4o-mini",
                }
            ],
        }

        with patch(
            "smartmemory.plugins.enrichers.usage_tracking.get_enricher_usage",
            return_value=mock_usage,
        ):
            stage.execute(state, config)

        # Tracking should still happen via finally block
        summary = tracker.summary()
        assert summary["total_spent"] == 400
