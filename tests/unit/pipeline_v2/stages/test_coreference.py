"""Unit tests for CoreferenceStageCommand."""

from unittest.mock import MagicMock, patch

from smartmemory.pipeline.config import PipelineConfig, CoreferenceConfig
from smartmemory.pipeline.state import PipelineState
from smartmemory.pipeline.stages.coreference import CoreferenceStageCommand


class TestCoreferenceStageCommand:
    """Tests for the coreference pipeline stage."""

    def _make_stage(self):
        return CoreferenceStageCommand()

    def test_coreference_disabled_returns_original_text(self):
        """When coreference is disabled, resolved_text equals the original text."""
        stage = self._make_stage()
        state = PipelineState(text="He went to the store.")
        config = PipelineConfig.default()
        config.coreference = CoreferenceConfig(enabled=False)

        result = stage.execute(state, config)

        assert result.resolved_text == "He went to the store."

    def test_coreference_short_text_returns_original(self):
        """Text shorter than min_text_length is returned unchanged."""
        stage = self._make_stage()
        state = PipelineState(text="Short.")
        config = PipelineConfig.default()
        config.coreference = CoreferenceConfig(enabled=True, min_text_length=50)

        result = stage.execute(state, config)

        assert result.resolved_text == "Short."

    def test_coreference_resolves_text(self):
        """Successful resolution returns the resolved text."""
        stage = self._make_stage()

        mock_result = MagicMock()
        mock_result.resolved_text = "John went to the store. John bought milk."
        mock_result.skipped = False
        mock_result.skip_reason = None
        mock_result.chains = [["John", "He"]]
        mock_result.replacements_made = 1

        mock_coref_instance = MagicMock()
        mock_coref_instance.run.return_value = mock_result

        mock_coref_cls = MagicMock(return_value=mock_coref_instance)
        mock_config_cls = MagicMock()

        long_text = "John went to the store. He bought milk. " * 3
        state = PipelineState(text=long_text)
        config = PipelineConfig.default()

        # Patch the imports that happen inside execute()'s try block
        with patch.dict(
            "sys.modules",
            {
                "smartmemory.memory.pipeline.stages.coreference": MagicMock(CoreferenceStage=mock_coref_cls),
                "smartmemory.memory.pipeline.config": MagicMock(CoreferenceConfig=mock_config_cls),
            },
        ):
            result = stage.execute(state, config)

        assert result.resolved_text == "John went to the store. John bought milk."

    def test_coreference_stores_result_in_context(self):
        """Coreference result (chains, replacements) is stored in _context."""
        stage = self._make_stage()

        mock_result = MagicMock()
        mock_result.resolved_text = "Resolved content."
        mock_result.skipped = False
        mock_result.skip_reason = None
        mock_result.chains = [["Alice", "she"]]
        mock_result.replacements_made = 2

        mock_coref_instance = MagicMock()
        mock_coref_instance.run.return_value = mock_result

        mock_coref_cls = MagicMock(return_value=mock_coref_instance)
        mock_config_cls = MagicMock()

        long_text = "Alice went home. She liked it there. " * 3
        state = PipelineState(text=long_text)
        config = PipelineConfig.default()

        with patch.dict(
            "sys.modules",
            {
                "smartmemory.memory.pipeline.stages.coreference": MagicMock(CoreferenceStage=mock_coref_cls),
                "smartmemory.memory.pipeline.config": MagicMock(CoreferenceConfig=mock_config_cls),
            },
        ):
            result = stage.execute(state, config)

        coref = result._context["coreference_result"]
        assert coref["skipped"] is False
        assert coref["chains"] == [["Alice", "she"]]
        assert coref["replacements_made"] == 2

    def test_coreference_graceful_on_error(self):
        """When the legacy import raises, resolved_text falls back to original."""
        stage = self._make_stage()

        long_text = "John went to the store. He bought milk. " * 3
        state = PipelineState(text=long_text)
        config = PipelineConfig.default()

        # Make the import inside execute() raise an ImportError
        bad_module = MagicMock()
        bad_module.CoreferenceStage = MagicMock(side_effect=RuntimeError("model not available"))

        with patch.dict(
            "sys.modules",
            {
                "smartmemory.memory.pipeline.stages.coreference": bad_module,
                "smartmemory.memory.pipeline.config": MagicMock(),
            },
        ):
            result = stage.execute(state, config)

        assert result.resolved_text == long_text

    def test_undo_clears_resolved_text(self):
        """Undo resets resolved_text to None."""
        stage = self._make_stage()
        state = PipelineState(text="Original.", resolved_text="Resolved.")

        result = stage.undo(state)

        assert result.resolved_text is None
