"""Unit tests for GroundStage."""

import pytest

pytestmark = pytest.mark.unit


from unittest.mock import MagicMock, patch

from smartmemory.pipeline.config import PipelineConfig, EnrichConfig, WikidataConfig
from smartmemory.pipeline.state import PipelineState
from smartmemory.pipeline.stages.ground import GroundStage


class TestGroundStage:
    """Tests for the ground pipeline stage."""

    def _make_stage(self):
        """Build a GroundStage with a mocked SmartMemory instance."""
        memory = MagicMock()
        memory._graph = MagicMock()
        memory._grounding = MagicMock()
        return GroundStage(memory), memory

    def _patch_ground_imports(self, grounder_mock=None):
        """Patch the locally-imported WikipediaGrounder and MemoryItem.

        Returns a context manager that injects mock modules for the local
        imports inside GroundStage.execute().
        """
        mock_grounders_module = MagicMock()
        if grounder_mock is not None:
            mock_grounders_module.WikipediaGrounder = grounder_mock
        else:
            mock_grounders_module.WikipediaGrounder = MagicMock()

        return patch.dict(
            "sys.modules",
            {
                "smartmemory.plugins.grounders": mock_grounders_module,
            },
        )

    def test_ground_preview_mode_noop(self):
        """In preview mode, state is returned unchanged."""
        stage, memory = self._make_stage()
        state = PipelineState(
            text="Preview grounding.",
            entities=[{"name": "Test"}],
        )
        config = PipelineConfig.preview()

        result = stage.execute(state, config)

        assert result is state

    def test_ground_disabled_noop(self):
        """When wikidata grounding is disabled, state is returned unchanged."""
        stage, memory = self._make_stage()
        state = PipelineState(
            text="Disabled grounding.",
            entities=[{"name": "Test"}],
        )
        config = PipelineConfig.default()
        config.enrich = EnrichConfig(wikidata=WikidataConfig(enabled=False))

        result = stage.execute(state, config)

        assert result is state

    def test_ground_no_entities_noop(self):
        """When there are no entities, state is returned unchanged."""
        stage, memory = self._make_stage()
        state = PipelineState(text="No entities here.", entities=[])
        config = PipelineConfig.default()

        result = stage.execute(state, config)

        assert result is state

    def test_ground_stores_provenance_in_context(self):
        """Provenance candidates from the grounder are stored in _context."""
        stage, memory = self._make_stage()

        provenance_data = [{"entity": "Python", "url": "https://en.wikipedia.org/wiki/Python", "score": 0.95}]
        mock_grounder_instance = MagicMock()
        mock_grounder_instance.ground.return_value = provenance_data
        mock_grounder_cls = MagicMock(return_value=mock_grounder_instance)

        state = PipelineState(
            text="Python is a programming language.",
            entities=[{"name": "Python", "type": "Language"}],
            entity_ids={"Python": "e1"},
            item_id="item_1",
        )
        config = PipelineConfig.default()

        with self._patch_ground_imports(grounder_mock=mock_grounder_cls):
            result = stage.execute(state, config)

        # GroundStage instantiates WikipediaGrounder directly — it does not delegate
        # through memory._grounding. Asserting on memory._grounding.ground would be
        # an implementation-detail check, not a behavioral one. Assert behavior only.
        assert result._context["provenance_candidates"] == provenance_data

    def test_ground_handles_exception(self):
        """When grounding raises an exception, the original state is returned."""
        stage, memory = self._make_stage()

        # Make WikipediaGrounder constructor raise
        mock_grounder_cls = MagicMock(side_effect=RuntimeError("grounder not available"))

        state = PipelineState(
            text="Error grounding test.",
            entities=[{"name": "Test"}],
            item_id="item_1",
        )
        config = PipelineConfig.default()

        with self._patch_ground_imports(grounder_mock=mock_grounder_cls):
            result = stage.execute(state, config)

        assert result is state
        assert "provenance_candidates" not in result._context

    def test_undo_removes_provenance(self):
        """Undo removes provenance_candidates from _context."""
        stage, _ = self._make_stage()
        state = PipelineState(
            text="Content.",
            _context={"provenance_candidates": [{"entity": "Test"}], "other": "data"},
        )

        result = stage.undo(state)

        assert "provenance_candidates" not in result._context
        assert result._context["other"] == "data"
