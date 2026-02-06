"""Unit tests for ExtractStage."""

from unittest.mock import MagicMock, patch

from smartmemory.pipeline.config import PipelineConfig
from smartmemory.pipeline.state import PipelineState
from smartmemory.pipeline.stages.extract import ExtractStage


class TestExtractStage:
    """Tests for the extract pipeline stage."""

    def _make_stage(self, extract_return=None):
        """Build an ExtractStage with a mocked ExtractionPipeline."""
        pipeline = MagicMock()
        pipeline.extract_semantics.return_value = extract_return or {}
        return ExtractStage(pipeline), pipeline

    def test_extract_sets_entities_and_relations(self):
        """Extraction result with entities and relations populates state."""
        entities = [{"name": "Claude", "type": "AI"}]
        relations = [{"source": "Claude", "relation": "is_a", "target": "AI"}]
        stage, pipeline = self._make_stage(extract_return={"entities": entities, "relations": relations})

        state = PipelineState(text="Claude is an AI assistant.")
        config = PipelineConfig.default()

        result = stage.execute(state, config)

        assert result.entities == entities
        assert result.relations == relations

    def test_extract_reads_nodes_key_fallback(self):
        """When result has 'nodes' instead of 'entities', still populates state."""
        nodes = [{"name": "GPT", "type": "Model"}]
        stage, pipeline = self._make_stage(extract_return={"nodes": nodes, "relations": []})

        state = PipelineState(text="GPT is a language model.")
        config = PipelineConfig.default()

        result = stage.execute(state, config)

        assert result.entities == nodes

    def test_extract_uses_resolved_text(self):
        """When state has resolved_text, the MemoryItem content uses it."""
        stage, pipeline = self._make_stage(extract_return={"entities": [], "relations": []})

        state = PipelineState(
            text="He is smart.",
            resolved_text="John is smart.",
        )
        config = PipelineConfig.default()

        with patch("smartmemory.models.memory_item.MemoryItem") as mock_item_cls:
            mock_item_cls.return_value = MagicMock()
            stage.execute(state, config)

            mock_item_cls.assert_called_once_with(
                content="John is smart.",
                memory_type="semantic",
                metadata={},
            )

    def test_extract_passes_coreference_chains(self):
        """Coreference chains from _context are passed as conversation_context."""
        stage, pipeline = self._make_stage(extract_return={"entities": []})

        state = PipelineState(
            text="Some text.",
            _context={"coreference_result": {"chains": [["John", "he"]]}},
        )
        config = PipelineConfig.default()

        stage.execute(state, config)

        call_kwargs = pipeline.extract_semantics.call_args
        assert call_kwargs.kwargs["conversation_context"] == {"coreference_chains": [["John", "he"]]}

    def test_extract_handles_empty_result(self):
        """When extraction returns empty dict, entities and relations are empty."""
        stage, pipeline = self._make_stage(extract_return={})

        state = PipelineState(text="Empty extraction test.")
        config = PipelineConfig.default()

        result = stage.execute(state, config)

        assert result.entities == []
        assert result.relations == []

    def test_undo_clears_extraction(self):
        """Undo resets entities, relations, rejected, and promotion_candidates."""
        stage, _ = self._make_stage()
        state = PipelineState(
            text="Content.",
            entities=[{"name": "A"}],
            relations=[{"source": "A"}],
            rejected=[{"name": "B"}],
            promotion_candidates=[{"name": "C"}],
        )

        result = stage.undo(state)

        assert result.entities == []
        assert result.relations == []
        assert result.rejected == []
        assert result.promotion_candidates == []
