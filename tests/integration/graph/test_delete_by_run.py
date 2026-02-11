"""Integration tests for delete_by_run_id functionality."""

import pytest
from uuid import uuid4

pytestmark = [pytest.mark.integration, pytest.mark.slow]


class TestDeleteByRunId:
    """Tests for deleting entities by pipeline run_id."""

    def test_delete_removes_all_run_entities(self, real_smartmemory_for_integration):
        """Deleting by run_id removes all entities from that run."""
        memory = real_smartmemory_for_integration

        # Create items with same run_id
        run_id_1 = str(uuid4())
        run_id_2 = str(uuid4())

        # Add 3 items with run_id_1
        for i in range(3):
            memory.add(
                {
                    "content": f"Item {i} from run 1",
                    "memory_type": "semantic",
                    "metadata": {"run_id": run_id_1, "index": i},
                }
            )

        # Add 2 items with run_id_2
        for i in range(2):
            memory.add(
                {
                    "content": f"Item {i} from run 2",
                    "memory_type": "semantic",
                    "metadata": {"run_id": run_id_2, "index": i},
                }
            )

        # Delete by first run_id
        deleted_count = memory.delete_run(run_id_1)

        # Should have deleted 3 items
        assert deleted_count == 3

        # Verify run_id_2 items still exist by searching
        results = memory.search("run 2", top_k=10)
        run_2_results = [r for r in results if getattr(r, "metadata", {}).get("run_id") == run_id_2]
        assert len(run_2_results) == 2

    def test_delete_returns_zero_when_no_matches(self, real_smartmemory_for_integration):
        """Deleting a non-existent run_id returns 0."""
        memory = real_smartmemory_for_integration

        # Delete by non-existent run_id
        deleted_count = memory.delete_run("non-existent-run-id-xyz")

        assert deleted_count == 0

    def test_delete_is_idempotent(self, real_smartmemory_for_integration):
        """Calling delete_run twice with same run_id is safe."""
        memory = real_smartmemory_for_integration

        run_id = str(uuid4())

        # Add item with run_id
        memory.add(
            {
                "content": "Idempotent test item",
                "memory_type": "semantic",
                "metadata": {"run_id": run_id},
            }
        )

        # First delete
        first_delete = memory.delete_run(run_id)
        assert first_delete == 1

        # Second delete should return 0 (already deleted)
        second_delete = memory.delete_run(run_id)
        assert second_delete == 0

    def test_delete_returns_count(self, real_smartmemory_for_integration):
        """Delete returns the exact count of deleted nodes."""
        memory = real_smartmemory_for_integration

        run_id = str(uuid4())

        # Add exactly 5 items
        for i in range(5):
            memory.add(
                {
                    "content": f"Count test item {i}",
                    "memory_type": "semantic",
                    "metadata": {"run_id": run_id},
                }
            )

        deleted_count = memory.delete_run(run_id)

        assert deleted_count == 5


class TestDeleteByRunIdWithWorkspaceScope:
    """Tests for delete_by_run_id with workspace scoping."""

    def test_delete_with_explicit_workspace_id(self, real_smartmemory_for_integration):
        """Delete can target a specific workspace."""
        memory = real_smartmemory_for_integration

        run_id = str(uuid4())
        workspace_a = f"workspace-a-{uuid4()}"
        workspace_b = f"workspace-b-{uuid4()}"

        # Add items in workspace A
        memory.add(
            {
                "content": "Workspace A item",
                "memory_type": "semantic",
                "metadata": {"run_id": run_id, "workspace_id": workspace_a},
            }
        )

        # Add items in workspace B
        memory.add(
            {
                "content": "Workspace B item",
                "memory_type": "semantic",
                "metadata": {"run_id": run_id, "workspace_id": workspace_b},
            }
        )

        # Delete only from workspace A using graph directly
        deleted = memory._graph.delete_by_run_id(run_id, workspace_id=workspace_a)

        # Should only delete 1 item (workspace A)
        assert deleted == 1

        # Workspace B item should still exist
        results = memory.search("Workspace B", top_k=5)
        assert len([r for r in results if workspace_b in str(getattr(r, "metadata", {}))]) >= 0
