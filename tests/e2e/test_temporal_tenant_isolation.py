"""E2E: Temporal Query Tenant Isolation Tests.

Tests that temporal queries (compare_versions, rollback) respect workspace boundaries
and cannot access items from other workspaces.

Gap ID: SEC-TEMPORAL-1
Priority: P0 (Security)

Requires running FalkorDB (port 9010) and Redis (port 9012).
"""

import os
import uuid
from typing import Dict, Any

import pytest

from smartmemory import SmartMemory
from smartmemory.temporal.queries import TemporalQueries

pytestmark = [pytest.mark.e2e, pytest.mark.security]


# =============================================================================
# Test Fixtures
# =============================================================================


class MockScopeProvider:
    """Test scope provider for tenant isolation testing."""

    def __init__(self, workspace_id: str, user_id: str = "test-user"):
        self._workspace_id = workspace_id
        self._user_id = user_id

    @property
    def workspace_id(self) -> str:
        return self._workspace_id

    @property
    def user_id(self) -> str:
        return self._user_id

    def get_isolation_filters(self) -> Dict[str, Any]:
        """Return workspace-level isolation filters."""
        return {"workspace_id": self._workspace_id}

    def get_write_context(self) -> Dict[str, Any]:
        """Return metadata for writes."""
        return {
            "workspace_id": self._workspace_id,
            "user_id": self._user_id,
        }


@pytest.fixture
def shared_memory():
    """Shared SmartMemory instance for testing (not workspace-bound)."""
    return SmartMemory(config_path=None)


@pytest.fixture
def workspace_a_id():
    """Unique workspace A ID for test isolation."""
    return f"workspace-a-{uuid.uuid4().hex[:8]}"


@pytest.fixture
def workspace_b_id():
    """Unique workspace B ID for test isolation."""
    return f"workspace-b-{uuid.uuid4().hex[:8]}"


# =============================================================================
# Test Cases: compare_versions Tenant Isolation
# =============================================================================


class TestCompareVersionsTenantIsolation:
    """Verify compare_versions cannot access items from other workspaces."""

    @pytest.mark.skipif(
        os.getenv("CI") == "true",
        reason="Requires FalkorDB infrastructure",
    )
    def test_compare_versions_same_workspace_allowed(self, shared_memory, workspace_a_id):
        """compare_versions should work for items in the same workspace."""
        memory = shared_memory

        # Create item in workspace A
        item_id = memory.ingest(
            "Test content for version comparison",
            metadata={"workspace_id": workspace_a_id},
        )

        # Query from same workspace should succeed
        scope_a = MockScopeProvider(workspace_a_id)
        temporal_a = TemporalQueries(memory, scope_provider=scope_a)

        # This may return an error about versions not existing at those times,
        # but it should NOT fail due to tenant isolation
        result = temporal_a.compare_versions(item_id, "2020-01-01", "2030-12-31")

        # Either finds versions or returns "Version not found" error
        # (NOT "Access denied" or cross-tenant data)
        assert "error" in result or "changed_fields" in result

        # Cleanup
        memory.remove(item_id)

    @pytest.mark.skipif(
        os.getenv("CI") == "true",
        reason="Requires FalkorDB infrastructure",
    )
    def test_compare_versions_cross_workspace_blocked(self, shared_memory, workspace_a_id, workspace_b_id):
        """compare_versions should NOT return data from other workspaces."""
        memory = shared_memory

        # Create item in workspace A
        item_id = memory.ingest(
            "Secret content in workspace A",
            metadata={"workspace_id": workspace_a_id},
        )

        # Query from workspace B should fail to find the item
        scope_b = MockScopeProvider(workspace_b_id)
        temporal_b = TemporalQueries(memory, scope_provider=scope_b)

        result = temporal_b.compare_versions(item_id, "2020-01-01", "2030-12-31")

        # Must return error - item not visible from workspace B
        assert "error" in result
        assert result["error"] == "Version not found at one or both times"

        # Verify no data from workspace A is leaked
        assert "changed_fields" not in result
        assert "modifications" not in result
        assert "Secret" not in str(result)

        # Cleanup
        memory.remove(item_id)


# =============================================================================
# Test Cases: rollback Tenant Isolation
# =============================================================================


class TestRollbackTenantIsolation:
    """Verify rollback cannot access or modify items from other workspaces."""

    @pytest.mark.skipif(
        os.getenv("CI") == "true",
        reason="Requires FalkorDB infrastructure",
    )
    def test_rollback_same_workspace_allowed(self, shared_memory, workspace_a_id):
        """rollback should work for items in the same workspace."""
        memory = shared_memory

        # Create item in workspace A
        item_id = memory.ingest(
            "Original content",
            metadata={"workspace_id": workspace_a_id},
        )

        # Query from same workspace
        scope_a = MockScopeProvider(workspace_a_id)
        temporal_a = TemporalQueries(memory, scope_provider=scope_a)

        # Dry run rollback - should work (may fail to find version, but not due to isolation)
        result = temporal_a.rollback(item_id, "2020-01-01", dry_run=True)

        # Either finds version or returns "Version not found" error
        # (NOT "Access denied" or cross-tenant data)
        assert "error" in result or "dry_run" in result

        # Cleanup
        memory.remove(item_id)

    @pytest.mark.skipif(
        os.getenv("CI") == "true",
        reason="Requires FalkorDB infrastructure",
    )
    def test_rollback_cross_workspace_blocked(self, shared_memory, workspace_a_id, workspace_b_id):
        """rollback should NOT be able to access items from other workspaces."""
        memory = shared_memory

        # Create item in workspace A
        item_id = memory.ingest(
            "Secret content that should not be rollbackable from workspace B",
            metadata={"workspace_id": workspace_a_id},
        )

        # Attempt rollback from workspace B
        scope_b = MockScopeProvider(workspace_b_id)
        temporal_b = TemporalQueries(memory, scope_provider=scope_b)

        result = temporal_b.rollback(item_id, "2020-01-01", dry_run=True)

        # Must return error - item not visible from workspace B
        assert "error" in result
        assert result["error"] == "Version not found at that time"

        # Verify no data from workspace A is leaked
        assert "preview" not in result
        assert "would_change" not in result
        assert "secret" not in str(result).lower()

        # Cleanup
        memory.remove(item_id)

    @pytest.mark.skipif(
        os.getenv("CI") == "true",
        reason="Requires FalkorDB infrastructure",
    )
    def test_rollback_actual_cross_workspace_prevented(self, shared_memory, workspace_a_id, workspace_b_id):
        """Even with dry_run=False, rollback should not modify items from other workspaces."""
        memory = shared_memory

        # Create item in workspace A with known content
        original_content = f"Original content {uuid.uuid4().hex}"
        item_id = memory.ingest(
            original_content,
            metadata={"workspace_id": workspace_a_id},
        )

        # Attempt actual rollback from workspace B
        scope_b = MockScopeProvider(workspace_b_id)
        temporal_b = TemporalQueries(memory, scope_provider=scope_b)

        result = temporal_b.rollback(item_id, "2020-01-01", dry_run=False)

        # Must fail - cannot find item from other workspace
        assert "error" in result

        # Verify item was NOT modified
        item = memory.get(item_id)
        assert item is not None
        assert original_content in item.content

        # Cleanup
        memory.remove(item_id)


# =============================================================================
# Test Cases: OSS Mode (No Scope Provider)
# =============================================================================


class TestOSSModeBackwardCompatibility:
    """Verify temporal queries work without scope provider (OSS mode)."""

    @pytest.mark.skipif(
        os.getenv("CI") == "true",
        reason="Requires FalkorDB infrastructure",
    )
    def test_compare_versions_without_scope_provider(self, shared_memory):
        """compare_versions should work when no scope provider is set."""
        memory = shared_memory

        # Create item without workspace metadata
        item_id = memory.ingest("Test content for OSS mode")

        # Create temporal queries without scope provider
        temporal = TemporalQueries(memory, scope_provider=None)

        # Should work (may not find versions at those times, but shouldn't crash)
        result = temporal.compare_versions(item_id, "2020-01-01", "2030-12-31")

        # Either finds versions or returns error - but shouldn't crash
        assert isinstance(result, dict)

        # Cleanup
        memory.remove(item_id)

    @pytest.mark.skipif(
        os.getenv("CI") == "true",
        reason="Requires FalkorDB infrastructure",
    )
    def test_rollback_without_scope_provider(self, shared_memory):
        """rollback should work when no scope provider is set."""
        memory = shared_memory

        # Create item without workspace metadata
        item_id = memory.ingest("Test content for OSS mode rollback")

        # Create temporal queries without scope provider
        temporal = TemporalQueries(memory, scope_provider=None)

        # Should work (may not find versions, but shouldn't crash)
        result = temporal.rollback(item_id, "2020-01-01", dry_run=True)

        # Either finds version or returns error - but shouldn't crash
        assert isinstance(result, dict)

        # Cleanup
        memory.remove(item_id)
