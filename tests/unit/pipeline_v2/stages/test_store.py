"""Unit tests for StoreStage."""

import pytest

pytestmark = pytest.mark.unit


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

    def test_store_sets_extraction_status_in_metadata(self):
        """extraction_status from state appears in MemoryItem metadata passed to _crud.add()."""
        stage, memory = self._make_stage(add_return="item_456")
        state = PipelineState(
            text="Extraction status test.",
            extraction_status="ruler_only",
        )
        config = PipelineConfig.default()

        with self._patch_storage_imports():
            stage.execute(state, config)

        # Inspect the MemoryItem passed to _crud.add
        call_args = memory._crud.add.call_args
        item_arg = call_args[0][0]
        assert item_arg.metadata["extraction_status"] == "ruler_only"

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


class TestStoreStageRunIdInjection:
    """Tests for run_id injection in StoreStage."""

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

    def test_run_id_injected_into_metadata(self):
        """run_id from raw_metadata should be injected into MemoryItem metadata."""
        stage, memory = self._make_stage(add_return="item_789")
        state = PipelineState(
            text="Content with run_id.",
            raw_metadata={"run_id": "test-run-abc-123"},
        )
        config = PipelineConfig.default()

        with self._patch_storage_imports():
            stage.execute(state, config)

        # Inspect the MemoryItem passed to _crud.add
        call_args = memory._crud.add.call_args
        item_arg = call_args[0][0]
        assert item_arg.metadata["run_id"] == "test-run-abc-123"

    def test_run_id_absent_when_not_provided(self):
        """When run_id is not in raw_metadata, metadata should not have run_id key."""
        stage, memory = self._make_stage(add_return="item_999")
        state = PipelineState(
            text="Content without run_id.",
            raw_metadata={"some_other_key": "value"},
        )
        config = PipelineConfig.default()

        with self._patch_storage_imports():
            stage.execute(state, config)

        # Inspect the MemoryItem passed to _crud.add
        call_args = memory._crud.add.call_args
        item_arg = call_args[0][0]
        assert "run_id" not in item_arg.metadata

    def test_run_id_coexists_with_extraction_status(self):
        """Both run_id and extraction_status should appear in metadata when present."""
        stage, memory = self._make_stage(add_return="item_combo")
        state = PipelineState(
            text="Content with both.",
            raw_metadata={"run_id": "run-xyz"},
            extraction_status="full",
        )
        config = PipelineConfig.default()

        with self._patch_storage_imports():
            stage.execute(state, config)

        call_args = memory._crud.add.call_args
        item_arg = call_args[0][0]
        assert item_arg.metadata["run_id"] == "run-xyz"
        assert item_arg.metadata["extraction_status"] == "full"
