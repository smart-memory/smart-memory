"""Unit tests for ClassifyStage."""

from unittest.mock import MagicMock, patch

from smartmemory.pipeline.config import PipelineConfig
from smartmemory.pipeline.state import PipelineState
from smartmemory.pipeline.stages.classify import ClassifyStage


class TestClassifyStage:
    """Tests for the classify pipeline stage."""

    def _make_stage(self, classify_return=None):
        """Build a ClassifyStage with a mocked MemoryIngestionFlow."""
        flow = MagicMock()
        flow.classify_item.return_value = classify_return or []
        return ClassifyStage(flow), flow

    def test_classify_sets_memory_type_from_first_type(self):
        """When classify_item returns multiple types, memory_type is the first."""
        stage, flow = self._make_stage(classify_return=["episodic", "semantic"])
        state = PipelineState(text="Claude Code is an AI coding assistant.")
        config = PipelineConfig.default()

        result = stage.execute(state, config)

        assert result.memory_type == "episodic"
        assert result.classified_types == ["episodic", "semantic"]

    def test_classify_preserves_existing_memory_type(self):
        """If the state already has a memory_type, it is not overwritten."""
        stage, flow = self._make_stage(classify_return=["episodic", "semantic"])
        state = PipelineState(
            text="Claude Code is an AI coding assistant.",
            memory_type="procedural",
        )
        config = PipelineConfig.default()

        result = stage.execute(state, config)

        assert result.memory_type == "procedural"
        assert result.classified_types == ["episodic", "semantic"]

    def test_classify_defaults_to_semantic(self):
        """When classify_item returns empty list and no existing type, default to semantic."""
        stage, flow = self._make_stage(classify_return=[])
        state = PipelineState(text="Some content.")
        config = PipelineConfig.default()

        result = stage.execute(state, config)

        assert result.memory_type == "semantic"
        assert result.classified_types == []

    @patch("smartmemory.memory.pipeline.config.ClassificationConfig")
    @patch("smartmemory.models.memory_item.MemoryItem")
    def test_classify_passes_config_to_legacy(self, mock_item_cls, mock_legacy_cls):
        """Verify classify_item is called with correctly-mapped legacy config."""
        stage, flow = self._make_stage(classify_return=["semantic"])

        mock_item_cls.return_value = MagicMock()
        mock_legacy_conf = MagicMock()
        mock_legacy_cls.return_value = mock_legacy_conf

        state = PipelineState(text="Test content.")
        config = PipelineConfig.default()
        config.classify.content_analysis_enabled = True
        config.classify.default_confidence = 0.85
        config.classify.inferred_confidence = 0.6

        stage.execute(state, config)

        mock_legacy_cls.assert_called_once_with(
            content_analysis_enabled=True,
            default_confidence=0.85,
            inferred_confidence=0.6,
        )
        flow.classify_item.assert_called_once_with(mock_item_cls.return_value, mock_legacy_conf)

    def test_undo_clears_classification(self):
        """Undo resets classified_types and memory_type."""
        stage, _ = self._make_stage()
        state = PipelineState(
            text="Some content.",
            classified_types=["episodic"],
            memory_type="episodic",
        )

        result = stage.undo(state)

        assert result.classified_types == []
        assert result.memory_type is None
