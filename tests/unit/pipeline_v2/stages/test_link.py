"""Unit tests for LinkStage."""

import pytest

pytestmark = pytest.mark.unit


from unittest.mock import MagicMock

from smartmemory.pipeline.config import PipelineConfig
from smartmemory.pipeline.state import PipelineState
from smartmemory.pipeline.stages.link import LinkStage


class TestLinkStage:
    """Tests for the link pipeline stage."""

    def _make_stage(self):
        """Build a LinkStage with a mocked Linking instance."""
        linking = MagicMock()
        return LinkStage(linking), linking

    def test_link_preview_mode_noop(self):
        """In preview mode, state is returned unchanged."""
        stage, linking = self._make_stage()
        state = PipelineState(text="Preview linking.", item_id="p1")
        config = PipelineConfig.preview()

        result = stage.execute(state, config)

        assert result is state
        linking.link_new_item.assert_not_called()

    def test_link_calls_link_new_item(self):
        """link_new_item is called with a context dict containing item, entities, etc."""
        stage, linking = self._make_stage()
        state = PipelineState(
            text="Link test content.",
            item_id="item_1",
            entity_ids={"Alice": "e1"},
            entities=[{"name": "Alice"}],
            relations=[{"source": "Alice", "target": "Bob"}],
        )
        config = PipelineConfig.default()

        stage.execute(state, config)

        linking.link_new_item.assert_called_once()
        context = linking.link_new_item.call_args[0][0]
        assert "item" in context
        assert context["entity_ids"] == {"Alice": "e1"}
        assert context["entities"] == [{"name": "Alice"}]
        assert context["relations"] == [{"source": "Alice", "target": "Bob"}]

    def test_link_captures_links_from_context(self):
        """When link_new_item mutates context['links'], state captures them."""
        stage, linking = self._make_stage()

        def add_links(context):
            context["links"] = {"similar": ["item_2", "item_3"]}

        linking.link_new_item.side_effect = add_links

        state = PipelineState(text="Content to link.", item_id="item_1")
        config = PipelineConfig.default()

        result = stage.execute(state, config)

        assert result.links == {"similar": ["item_2", "item_3"]}

    def test_link_no_links_added(self):
        """When link_new_item does not add links, state.links is empty dict."""
        stage, linking = self._make_stage()
        state = PipelineState(text="No links found.", item_id="item_1")
        config = PipelineConfig.default()

        result = stage.execute(state, config)

        assert result.links == {}

    def test_undo_clears_links(self):
        """Undo resets links to empty dict."""
        stage, _ = self._make_stage()
        state = PipelineState(
            text="Content.",
            links={"similar": ["item_2"]},
        )

        result = stage.undo(state)

        assert result.links == {}
