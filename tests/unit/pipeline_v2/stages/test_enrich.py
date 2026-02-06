"""Unit tests for EnrichStage."""

from unittest.mock import MagicMock

from smartmemory.pipeline.config import PipelineConfig, EnrichConfig
from smartmemory.pipeline.state import PipelineState
from smartmemory.pipeline.stages.enrich import EnrichStage


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
