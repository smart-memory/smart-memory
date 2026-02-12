"""Unit tests for StoreStage."""

from unittest.mock import MagicMock, patch

import pytest

from smartmemory.pipeline.config import PipelineConfig
from smartmemory.pipeline.state import PipelineState
from smartmemory.pipeline.stages.store import StoreStage

pytestmark = pytest.mark.unit


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


class TestStoreStageEmbeddingTokenTracking:
    """Tests for embedding token tracking in StoreStage (CFS-1a)."""

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

    def test_track_embedding_spent_on_api_call(self):
        """When embedding is not cached, tokens are recorded as spent."""
        from smartmemory.pipeline.token_tracker import PipelineTokenTracker

        stage, _ = self._make_stage()
        tracker = PipelineTokenTracker(workspace_id="ws-test")
        state = PipelineState(text="Test content.", token_tracker=tracker)
        config = PipelineConfig.default()

        # Mock embedding usage: API call with actual tokens
        mock_usage = {
            "prompt_tokens": 150,
            "total_tokens": 150,
            "model": "text-embedding-ada-002",
            "cached": False,
        }

        with self._patch_storage_imports():
            with patch(
                "smartmemory.plugins.embedding.get_last_embedding_usage",
                return_value=mock_usage,
            ):
                stage.execute(state, config)

        summary = tracker.summary()
        assert "store" in summary["stages"]["spent"]
        store_spent = summary["stages"]["spent"]["store"]
        assert store_spent["prompt_tokens"] == 150
        assert store_spent["completion_tokens"] == 0
        assert store_spent["models"] == {"text-embedding-ada-002": 1}

    def test_track_embedding_avoided_on_cache_hit(self):
        """When embedding is cached, tokens are recorded as avoided."""
        from smartmemory.pipeline.token_tracker import PipelineTokenTracker

        stage, _ = self._make_stage()
        tracker = PipelineTokenTracker(workspace_id="ws-test")
        state = PipelineState(text="Test content.", token_tracker=tracker)
        config = PipelineConfig.default()

        # Mock embedding usage: cache hit
        mock_usage = {
            "prompt_tokens": 0,
            "total_tokens": 0,
            "model": "text-embedding-ada-002",
            "cached": True,
        }

        with self._patch_storage_imports():
            with patch(
                "smartmemory.plugins.embedding.get_last_embedding_usage",
                return_value=mock_usage,
            ):
                stage.execute(state, config)

        summary = tracker.summary()
        assert "store" in summary["stages"]["avoided"]
        store_avoided = summary["stages"]["avoided"]["store"]
        # Should use estimated average tokens
        assert store_avoided["total_tokens"] == StoreStage._AVG_EMBEDDING_TOKENS
        assert store_avoided["reasons"] == {"cache_hit": 1}

    def test_track_embedding_no_tracker(self):
        """When no token tracker, tracking is silently skipped."""
        stage, _ = self._make_stage()
        state = PipelineState(text="Test content.", token_tracker=None)
        config = PipelineConfig.default()

        # Should not raise even with no tracker
        with self._patch_storage_imports():
            with patch(
                "smartmemory.plugins.embedding.get_last_embedding_usage",
                return_value={"prompt_tokens": 100, "cached": False},
            ):
                result = stage.execute(state, config)

        assert result.item_id is not None

    def test_track_embedding_no_usage_data(self):
        """When no usage data is available, tracking is silently skipped."""
        from smartmemory.pipeline.token_tracker import PipelineTokenTracker

        stage, _ = self._make_stage()
        tracker = PipelineTokenTracker(workspace_id="ws-test")
        state = PipelineState(text="Test content.", token_tracker=tracker)
        config = PipelineConfig.default()

        with self._patch_storage_imports():
            with patch(
                "smartmemory.plugins.embedding.get_last_embedding_usage",
                return_value=None,
            ):
                stage.execute(state, config)

        summary = tracker.summary()
        assert summary["total_spent"] == 0
        assert summary["total_avoided"] == 0

    def test_track_embedding_import_error(self):
        """When embedding module import fails, tracking is silently skipped."""
        from smartmemory.pipeline.token_tracker import PipelineTokenTracker

        stage, _ = self._make_stage()
        tracker = PipelineTokenTracker(workspace_id="ws-test")
        state = PipelineState(text="Test content.", token_tracker=tracker)
        config = PipelineConfig.default()

        # Force ImportError in _track_embedding_usage by making the import fail
        def mock_track_that_catches_import():
            # This test verifies that ImportError is handled gracefully
            # We need to actually trigger the import failure
            pass

        with self._patch_storage_imports():
            # Force the embedding module to raise ImportError when accessed
            import sys

            original = sys.modules.get("smartmemory.plugins.embedding")
            sys.modules["smartmemory.plugins.embedding"] = None
            try:
                stage.execute(state, config)
            finally:
                if original is not None:
                    sys.modules["smartmemory.plugins.embedding"] = original
                else:
                    del sys.modules["smartmemory.plugins.embedding"]

        summary = tracker.summary()
        assert summary["total_spent"] == 0
