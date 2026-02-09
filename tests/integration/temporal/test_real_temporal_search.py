"""
Real integration tests for temporal search functionality using the full backend stack.
"""

import pytest


pytestmark = pytest.mark.integration
from datetime import datetime, timezone, timedelta
import time
from smartmemory.models.memory_item import MemoryItem
from smartmemory.temporal.queries import TemporalQueries
from smartmemory.temporal.version_tracker import VersionTracker

@pytest.mark.integration
class TestRealTemporalSearch:
    """Integration tests for temporal search using real backends."""
    
    @pytest.fixture(autouse=True)
    def setup_teardown(self, real_smartmemory_for_integration):
        """Setup and teardown for each test."""
        self.memory = real_smartmemory_for_integration
        # Ensure we have a version tracker initialized
        self.memory.version_tracker = VersionTracker(self.memory._graph)
        self.temporal_queries = TemporalQueries(self.memory)
        yield
        try:
            self.memory.clear()
        except Exception as e:
            print(f"Cleanup warning: {e}")

    def test_temporal_version_creation_and_retrieval(self):
        """Test creating versions and retrieving them via temporal queries."""
        
        # 1. Create initial memory
        item = MemoryItem(
            content="Initial content v1",
            memory_type="semantic",
            
            metadata={"version": 1}
        )
        item_id = self.memory.add(item)
        
        # Manually create a version entry for v1 (since add() might not auto-version in all configs)
        # Note: In a full system, update() would handle this, but we want explicit control for testing
        v1 = self.memory.version_tracker.create_version(
            item_id=item_id,
            content=item.content,
            metadata=item.metadata,
            change_reason="initial creation"
        )
        
        # 2. Update memory (create v2)
        time.sleep(1) # Ensure time difference
        v2_content = "Updated content v2"
        v2_metadata = {"version": 2}
        
        v2 = self.memory.version_tracker.create_version(
            item_id=item_id,
            content=v2_content,
            metadata=v2_metadata,
            change_reason="update"
        )
        
        # 3. Verify history
        history = self.temporal_queries.get_history(item_id)
        assert len(history) >= 2

        # Verify order (newest first)
        # memory.add() does NOT auto-create versions — only explicit create_version() does
        assert history[0].version == 2
        assert history[1].version == 1
        assert history[0].content == v2_content
        assert history[1].content == item.content
        
    def test_search_temporal_time_range(self):
        """Test searching within specific time ranges."""
        
        # Create items with different timestamps
        # We'll simulate this by creating versions with explicit transaction times
        # Note: Since we can't easily override creation time in the real backend without sleeping,
        # we will use the fact that we create them sequentially.
        
        item_id = "time_range_test_item"
        
        # V1
        v1 = self.memory.version_tracker.create_version(
            item_id=item_id,
            content="Content Phase 1",
            change_reason="phase 1"
        )
        t1 = datetime.now(timezone.utc)
        time.sleep(2)
        
        # V2
        v2 = self.memory.version_tracker.create_version(
            item_id=item_id,
            content="Content Phase 2",
            change_reason="phase 2"
        )
        t2 = datetime.now(timezone.utc)
        
        # Search covering only T1
        # We need to ensure our search window captures V1 but excludes V2
        # Since search_temporal relies on vector search + temporal filtering
        # We need to make sure the item is indexed in vector store first
        # In this test setup, we are manually creating versions, but search_temporal
        # also does a memory.search() first. So we need the item to exist in memory.
        
        self.memory.add(MemoryItem(content="Content Phase 2", item_id=item_id, ))
        
        # Search 
        start_time = (t1 - timedelta(seconds=5)).isoformat()
        end_time = (t1 + timedelta(seconds=1)).isoformat()
        
        results = self.temporal_queries.search_temporal(
            "Content",
            start_time=start_time,
            end_time=end_time
        )
        
        # Should find the item, and specifically filter to V1
        # Note: search_temporal logic gets the item from vector search, then looks up versions
        # that overlap with the window.
        
        filtered_versions = [r['version'] for r in results if r['item_id'] == item_id]
        assert 1 in filtered_versions
        assert 2 not in filtered_versions

    def test_get_changes(self):
        """Test change detection between versions."""
        item_id = "diff_test_item"
        
        # V1
        self.memory.version_tracker.create_version(
            item_id=item_id,
            content="Original text",
            metadata={"tag": "A"}
        )
        time.sleep(1)
        
        # V2
        self.memory.version_tracker.create_version(
            item_id=item_id,
            content="Modified text",
            metadata={"tag": "B", "new_field": "added"}
        )
        
        changes = self.temporal_queries.get_changes(item_id)
        assert len(changes) > 0
        
        latest_change = changes[0] # Most recent change logic depends on get_changes implementation (it returns list ordered by time)
        # Actually get_changes uses get_history which sorts newest first? 
        # Let's check implementation: "changes ordered by time" usually means chronological.
        # get_changes iterates len(versions)-1. versions are newest first.
        # It compares versions[i] (new) and versions[i+1] (old).
        
        assert latest_change.change_type == 'updated'
        assert 'content' in latest_change.changed_fields
        assert 'metadata.tag' in latest_change.changed_fields
        assert 'metadata.new_field' in latest_change.changed_fields

    def test_at_time_query(self):
        """Test retrieving state at a specific point in time."""
        item_id = "at_time_test"
        
        # Ensure the node exists in the graph with Memory label FIRST
        # This ensures at_time query (MATCH (m:Memory)) can find it
        self.memory.graph.add_node(item_id, {"content": "Initial State", "item_id": item_id}, memory_type="semantic")
        
        # T0: V1
        self.memory.version_tracker.create_version(
            item_id=item_id,
            content="State at T0"
        )
        t0 = datetime.now(timezone.utc)
        time.sleep(2)
        
        # T1: V2
        self.memory.version_tracker.create_version(
            item_id=item_id,
            content="State at T1"
        )
        t1 = datetime.now(timezone.utc)
        
        # Query at T0 (should see V1)
        # We pass a time slightly after T0 creation but before T1
        query_time = t0 + timedelta(seconds=1)
        
        results = self.temporal_queries.at_time(query_time.isoformat())
        
        # Find our item
        item_version = next((i for i in results if i.item_id == item_id), None)
        
        assert item_version is not None
        assert item_version.content == "State at T0"
        assert item_version.version_number == 1
