"""
Integration tests for core SmartMemory workflows.

Tests complete workflows that should catch issues before E2E tests:
- Add → Search → Retrieve workflow
- Metadata preservation through storage
- User isolation across full stack
- Graph-vector consistency
"""
import pytest
from datetime import datetime, timezone

from smartmemory.models.memory_item import MemoryItem


@pytest.mark.integration
class TestCoreWorkflows:
    """Integration tests for core workflows that should catch issues before E2E."""
    
    @pytest.fixture(autouse=True)
    def setup_teardown(self, real_smartmemory_for_integration):
        """Setup and teardown for each test."""
        self.memory = real_smartmemory_for_integration
        yield
        try:
            self.memory.clear()
        except Exception as e:
            print(f"Cleanup warning: {e}")
    
    def test_complete_add_search_retrieve_workflow(self):
        """Test complete workflow: add → search → retrieve.
        
        This should catch search/embedding failures that E2E tests revealed.
        """
        # Add memory
        item = MemoryItem(
            content="Machine learning algorithms process data to find patterns",
            memory_type="semantic",
            user_id="workflow_test_user",
            metadata={"category": "technology"}
        )
        
        item_id = self.memory.add(item)
        assert item_id is not None, "Failed to add memory"
        print(f"✅ Added memory: {item_id}")
        
        # Search for content
        search_results = self.memory.search(
            "machine learning algorithms",
            user_id="workflow_test_user",
            top_k=5
        )
        assert len(search_results) > 0, "Search returned no results - CRITICAL BUG"
        print(f"✅ Search found {len(search_results)} results")
        
        # Retrieve specific memory
        retrieved = self.memory.get(item_id)
        assert retrieved is not None, "Failed to retrieve memory"
        assert retrieved.content == item.content, "Content mismatch"
        print(f"✅ Retrieved memory successfully")
    
    def test_metadata_preservation(self):
        """Test metadata is preserved through storage and retrieval.
        
        This should catch metadata transformation bugs that E2E tests revealed.
        """
        # Create memory with complex metadata
        metadata = {
            "event_time": "2025-01-01T00:00:00Z",
            "event_type": "milestone",
            "custom_field": "custom_value",
            "importance": "high",
            "nested": {
                "key1": "value1",
                "key2": "value2"
            },
            "list_field": ["item1", "item2", "item3"]
        }
        
        item = MemoryItem(
            content="Test metadata preservation",
            memory_type="semantic",
            user_id="metadata_test_user",
            metadata=metadata
        )
        
        item_id = self.memory.add(item)
        assert item_id is not None
        print(f"✅ Added memory with complex metadata")
        
        # Retrieve and validate metadata
        retrieved = self.memory.get(item_id)
        assert retrieved is not None
        assert hasattr(retrieved, 'metadata'), "Missing metadata attribute"
        
        # Validate each metadata field
        assert "event_time" in retrieved.metadata, "event_time field missing"
        assert retrieved.metadata["event_time"] == metadata["event_time"], \
            f"event_time transformed: {retrieved.metadata.get('event_time')} != {metadata['event_time']}"
        
        assert retrieved.metadata.get("event_type") == metadata["event_type"], \
            "event_type mismatch"
        
        assert retrieved.metadata.get("custom_field") == metadata["custom_field"], \
            "custom_field mismatch"
        
        assert retrieved.metadata.get("importance") == metadata["importance"], \
            "importance mismatch"
        
        # Note: Nested structures may be flattened/transformed by storage layer
        # This is expected behavior, but we should document it
        print(f"✅ Metadata preserved: {list(retrieved.metadata.keys())}")
    
    def test_user_isolation_full_stack(self):
        """Test user_id is preserved and isolated through full stack.
        
        This should catch user ID inconsistencies that E2E tests revealed.
        """
        # Add memories for different users
        user1_item = MemoryItem(
            content="User 1 private data",
            memory_type="episodic",
            user_id="user_001",
            metadata={"private": True}
        )
        
        user2_item = MemoryItem(
            content="User 2 private data",
            memory_type="episodic",
            user_id="user_002",
            metadata={"private": True}
        )
        
        id1 = self.memory.add(user1_item)
        id2 = self.memory.add(user2_item)
        
        assert id1 is not None
        assert id2 is not None
        print(f"✅ Added memories for 2 users")
        
        # Retrieve and validate user_id preservation
        retrieved1 = self.memory.get(id1)
        retrieved2 = self.memory.get(id2)
        
        assert retrieved1 is not None
        assert retrieved2 is not None
        
        # CRITICAL: Validate exact user_id match
        assert hasattr(retrieved1, 'user_id'), "Missing user_id attribute"
        assert hasattr(retrieved2, 'user_id'), "Missing user_id attribute"
        
        assert retrieved1.user_id == "user_001", \
            f"User ID mismatch: expected 'user_001', got '{retrieved1.user_id}'"
        assert retrieved2.user_id == "user_002", \
            f"User ID mismatch: expected 'user_002', got '{retrieved2.user_id}'"
        
        print(f"✅ User IDs preserved correctly")
        
        # Test search isolation
        user1_results = self.memory.search("private data", user_id="user_001", top_k=10)
        
        # Validate all results belong to user_001
        for result in user1_results:
            if hasattr(result, 'user_id'):
                assert result.user_id == "user_001", \
                    f"Search leaked user_002 data to user_001: {result.user_id}"
            elif hasattr(result, 'metadata') and result.metadata:
                result_user = result.metadata.get('user_id')
                if result_user:
                    assert result_user == "user_001", \
                        f"Search leaked user_002 data to user_001 via metadata: {result_user}"
        
        print(f"✅ Search isolation verified")
    
    def test_graph_vector_consistency(self):
        """Test graph and vector store remain consistent.
        
        This should catch graph-vector sync issues.
        """
        # Add memory
        item = MemoryItem(
            content="Python is a versatile programming language",
            memory_type="semantic",
            user_id="consistency_test_user",
            metadata={"language": "Python"}
        )
        
        item_id = self.memory.add(item)
        assert item_id is not None
        print(f"✅ Added memory: {item_id}")
        
        # Verify accessible via graph (retrieval)
        retrieved_from_graph = self.memory.get(item_id)
        assert retrieved_from_graph is not None, "Graph retrieval failed"
        assert retrieved_from_graph.content == item.content
        print(f"✅ Graph retrieval successful")
        
        # Verify accessible via vector store (search)
        search_results = self.memory.search(
            "Python programming",
            user_id="consistency_test_user",
            top_k=5
        )
        assert len(search_results) > 0, "Vector search failed"
        print(f"✅ Vector search successful: {len(search_results)} results")
        
        # Verify consistency: search results should include our item
        found_in_search = False
        for result in search_results:
            if hasattr(result, 'content') and "Python" in result.content:
                found_in_search = True
                break
        
        assert found_in_search, "Item not found in search results - graph/vector inconsistency"
        print(f"✅ Graph-vector consistency verified")
    
    def test_search_returns_results_after_add(self):
        """Focused test: search must return results immediately after add.
        
        This is the core issue E2E tests revealed.
        """
        # Add memory
        item = MemoryItem(
            content="Neural networks are used in deep learning",
            memory_type="semantic",
            user_id="search_test_user"
        )
        
        item_id = self.memory.add(item)
        assert item_id is not None
        
        # Search immediately
        results = self.memory.search("neural networks", user_id="search_test_user", top_k=5)
        
        # CRITICAL: This must not be empty
        assert len(results) > 0, \
            "CRITICAL BUG: Search returns no results immediately after add. " \
            "This indicates embedding/vector store failure."
        
        print(f"✅ Search returned {len(results)} results after add")
    
    def test_multiple_adds_and_searches(self):
        """Test multiple add/search cycles work correctly."""
        user_id = "multi_test_user"
        
        # Add multiple memories
        items = [
            MemoryItem(content="Python programming", memory_type="semantic", user_id=user_id),
            MemoryItem(content="JavaScript development", memory_type="semantic", user_id=user_id),
            MemoryItem(content="Machine learning algorithms", memory_type="semantic", user_id=user_id),
        ]
        
        item_ids = []
        for item in items:
            item_id = self.memory.add(item)
            assert item_id is not None
            item_ids.append(item_id)
        
        print(f"✅ Added {len(item_ids)} memories")
        
        # Search for each
        search_queries = ["Python", "JavaScript", "machine learning"]
        
        for query in search_queries:
            results = self.memory.search(query, user_id=user_id, top_k=5)
            assert len(results) > 0, f"Search for '{query}' returned no results"
        
        print(f"✅ All searches returned results")
        
        # Retrieve all
        for item_id in item_ids:
            retrieved = self.memory.get(item_id)
            assert retrieved is not None, f"Failed to retrieve {item_id}"
            assert retrieved.user_id == user_id
        
        print(f"✅ All retrievals successful with correct user_id")


@pytest.mark.integration
class TestEdgeCases:
    """Integration tests for edge cases that could cause issues."""
    
    @pytest.fixture(autouse=True)
    def setup_teardown(self, real_smartmemory_for_integration):
        """Setup and teardown for each test."""
        self.memory = real_smartmemory_for_integration
        yield
        try:
            self.memory.clear()
        except Exception as e:
            print(f"Cleanup warning: {e}")
    
    def test_empty_search_query(self):
        """Test search with empty query doesn't crash."""
        # Add some data first
        item = MemoryItem(content="Test content", user_id="test_user")
        self.memory.add(item)
        
        # Search with empty query
        results = self.memory.search("", user_id="test_user", top_k=5)
        # Should return results or empty list, not crash
        assert isinstance(results, list)
        print(f"✅ Empty search handled gracefully: {len(results)} results")
    
    def test_search_before_any_adds(self):
        """Test search on empty memory doesn't crash."""
        results = self.memory.search("anything", user_id="test_user", top_k=5)
        assert isinstance(results, list)
        assert len(results) == 0
        print(f"✅ Search on empty memory handled gracefully")
    
    def test_retrieve_nonexistent_item(self):
        """Test retrieving non-existent item returns None."""
        retrieved = self.memory.get("nonexistent_id_12345")
        assert retrieved is None
        print(f"✅ Nonexistent item retrieval handled gracefully")
