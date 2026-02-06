"""Unit tests for StoreStage."""

from unittest.mock import MagicMock, patch

from smartmemory.pipeline.config import PipelineConfig
from smartmemory.pipeline.state import PipelineState
from smartmemory.pipeline.stages.store import StoreStage


class TestStoreStage:
    """Tests for the store pipeline stage."""

    def _make_stage(self, add_return=None):
        """Build a StoreStage with a mocked SmartMemory instance."""
        memory = MagicMock()
        memory._crud.add.return_value = add_return or "item_123"
        return StoreStage(memory), memory

    def _patch_storage_imports(self):
        """Patch the locally-imported StoragePipeline and IngestionObserver."""
        return patch.dict(
            "sys.modules",
            {
                "smartmemory.memory.ingestion.storage": MagicMock(),
                "smartmemory.memory.ingestion.observer": MagicMock(),
            },
        )

    def test_store_preview_mode(self):
        """In preview mode, item_id is set to 'preview_item' without calling CRUD."""
        stage, memory = self._make_stage()
        state = PipelineState(text="Preview content.")
        config = PipelineConfig.preview()

        result = stage.execute(state, config)

        assert result.item_id == "preview_item"
        memory._crud.add.assert_not_called()

    def test_store_dict_result(self):
        """When _crud.add returns a dict, item_id and entity_ids are extracted."""
        stage, memory = self._make_stage(add_return={"memory_node_id": "abc", "entity_node_ids": ["e1", "e2"]})
        state = PipelineState(
            text="Store test.",
            entities=[
                {"name": "Alice", "metadata": {"name": "Alice"}},
                {"name": "Bob", "metadata": {"name": "Bob"}},
            ],
        )
        config = PipelineConfig.default()

        with self._patch_storage_imports():
            result = stage.execute(state, config)

        assert result.item_id == "abc"
        assert result.entity_ids == {"Alice": "e1", "Bob": "e2"}

    def test_store_string_result(self):
        """When _crud.add returns a string, it is used as item_id."""
        stage, memory = self._make_stage(add_return="simple_id")
        state = PipelineState(text="Simple store.")
        config = PipelineConfig.default()

        with self._patch_storage_imports():
            result = stage.execute(state, config)

        assert result.item_id == "simple_id"

    def test_store_maps_entity_ids(self):
        """Entity names are mapped to their created IDs or fallback IDs."""
        stage, memory = self._make_stage(add_return={"memory_node_id": "m1", "entity_node_ids": ["eid_0"]})
        state = PipelineState(
            text="Entity mapping test.",
            entities=[
                {"name": "Entity1", "metadata": {"name": "Entity1"}},
                {"name": "Entity2", "metadata": {"name": "Entity2"}},
            ],
        )
        config = PipelineConfig.default()

        with self._patch_storage_imports():
            result = stage.execute(state, config)

        assert result.entity_ids["Entity1"] == "eid_0"
        # Second entity falls back to generated ID
        assert result.entity_ids["Entity2"] == "m1_entity_1"

    def test_undo_clears_storage(self):
        """Undo resets item_id and entity_ids."""
        stage, _ = self._make_stage()
        state = PipelineState(
            text="Content.",
            item_id="some_id",
            entity_ids={"A": "a1"},
        )

        result = stage.undo(state)

        assert result.item_id is None
        assert result.entity_ids == {}
