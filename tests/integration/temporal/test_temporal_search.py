"""
Integration tests for temporal search functionality.
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, MagicMock

from smartmemory.temporal.queries import TemporalQueries
from smartmemory.temporal.version_tracker import VersionTracker
from smartmemory.models.memory_item import MemoryItem


class TestTemporalSearchIntegration:
    """Integration tests for temporal search."""
    
    @pytest.fixture
    def mock_memory(self):
        """Create mock memory system."""
        memory = Mock()
        memory.graph = Mock()
        memory.graph.execute_query = Mock(return_value=[])
        memory.search = Mock(return_value=[])
        return memory
    
    @pytest.fixture
    def temporal_queries(self, mock_memory):
        """Create temporal queries instance."""
        return TemporalQueries(mock_memory)
    
    def test_temporal_queries_initialization(self, temporal_queries):
        """Test temporal queries initializes correctly."""
        assert temporal_queries is not None
        assert temporal_queries.memory is not None
        assert temporal_queries.version_tracker is not None
        assert isinstance(temporal_queries.version_tracker, VersionTracker)
    
    def test_search_temporal_basic(self, temporal_queries, mock_memory):
        """Test basic temporal search."""
        # Mock search results with proper attributes
        mock_result = Mock()
        mock_result.item_id = "test123"
        mock_result.content = "Test content with Python"
        mock_result.metadata = {"topic": "programming"}
        mock_result.score = 0.95
        
        # Also support dict-style access
        mock_result.get = lambda key, default=None: {
            'item_id': "test123",
            'content': "Test content with Python",
            'metadata': {"topic": "programming"}
        }.get(key, default)
        
        mock_memory.search.return_value = [mock_result]
        
        # Mock version tracker to return empty (will use fallback)
        temporal_queries.version_tracker = None  # Force fallback path
        
        results = temporal_queries.search_temporal(
            "Python",
            start_time="2024-09-01",
            end_time="2024-09-30"
        )
        
        assert isinstance(results, list)
        assert len(results) == 1
        assert results[0]['item_id'] == "test123"
        assert results[0]['content'] == "Test content with Python"
        assert results[0]['relevance_score'] == 0.95
    
    def test_search_temporal_with_versions(self, temporal_queries, mock_memory):
        """Test temporal search with version history."""
        from smartmemory.temporal.version_tracker import Version
        
        # Mock search results
        mock_result = Mock()
        mock_result.item_id = "test123"
        mock_result.content = "Current content"
        mock_result.metadata = {}
        mock_result.score = 0.9
        
        mock_memory.search.return_value = [mock_result]
        
        # Mock versions
        now = datetime.now(timezone.utc)
        versions = [
            Version(
                item_id="test123",
                version_number=2,
                content="Updated content",
                metadata={"version": 2},
                valid_time_start=now - timedelta(days=5),
                transaction_time_start=now - timedelta(days=5)
            ),
            Version(
                item_id="test123",
                version_number=1,
                content="Original content",
                metadata={"version": 1},
                valid_time_start=now - timedelta(days=10),
                transaction_time_start=now - timedelta(days=10)
            )
        ]
        
        temporal_queries.version_tracker.get_versions = Mock(return_value=versions)
        
        results = temporal_queries.search_temporal(
            "content",
            start_time=(now - timedelta(days=15)).isoformat(),
            end_time=now.isoformat()
        )
        
        assert len(results) == 2
        assert results[0]['version'] == 2
        assert results[1]['version'] == 1
        assert 'transaction_time' in results[0]
        assert 'valid_time_start' in results[0]
    
    def test_search_temporal_time_range_filtering(self, temporal_queries, mock_memory):
        """Test that time range filtering works correctly."""
        from smartmemory.temporal.version_tracker import Version
        
        mock_result = Mock()
        mock_result.item_id = "test123"
        mock_result.content = "Test"
        mock_result.score = 1.0
        
        mock_memory.search.return_value = [mock_result]
        
        now = datetime.now(timezone.utc)
        
        # Create versions outside and inside time range
        all_versions = [
            Version(
                item_id="test123",
                version_number=3,
                content="Recent",
                transaction_time_start=now - timedelta(days=1)  # Inside range
            ),
            Version(
                item_id="test123",
                version_number=2,
                content="Middle",
                transaction_time_start=now - timedelta(days=5)  # Inside range
            ),
            Version(
                item_id="test123",
                version_number=1,
                content="Old",
                transaction_time_start=now - timedelta(days=20)  # Outside range
            )
        ]
        
        # Mock to return only versions in range
        temporal_queries.version_tracker.get_versions = Mock(
            return_value=[v for v in all_versions if v.version_number >= 2]
        )
        
        results = temporal_queries.search_temporal(
            "test",
            start_time=(now - timedelta(days=7)).isoformat(),
            end_time=now.isoformat()
        )
        
        # Should only get versions 2 and 3
        assert len(results) == 2
        version_numbers = [r['version'] for r in results]
        assert 3 in version_numbers
        assert 2 in version_numbers
        assert 1 not in version_numbers
    
    def test_at_time_query(self, temporal_queries, mock_memory):
        """Test querying memories at a specific time."""
        from smartmemory.temporal.version_tracker import Version
        
        now = datetime.now(timezone.utc)
        target_time = now - timedelta(days=5)
        
        # Mock graph query to return item IDs
        mock_memory.graph.execute_query.return_value = [
            {'item_id': 'item1'},
            {'item_id': 'item2'}
        ]
        
        # Mock version tracker
        def mock_get_version_at_time(item_id, time):
            if item_id == 'item1':
                return Version(
                    item_id='item1',
                    version_number=1,
                    content='Item 1 content',
                    transaction_time_start=now - timedelta(days=10)
                )
            return None
        
        temporal_queries.version_tracker.get_version_at_time = Mock(
            side_effect=mock_get_version_at_time
        )
        
        results = temporal_queries.at_time(target_time.isoformat())
        
        assert isinstance(results, list)
        assert len(results) == 1
        assert results[0].item_id == 'item1'
    
    def test_search_temporal_with_filters(self, temporal_queries, mock_memory):
        """Test temporal search with additional filters."""
        mock_result = Mock()
        mock_result.item_id = "test123"
        mock_result.content = "Test"
        mock_result.score = 1.0
        
        mock_memory.search.return_value = [mock_result]
        temporal_queries.version_tracker.get_versions = Mock(return_value=[])
        
        results = temporal_queries.search_temporal(
            "test",
            start_time="2024-09-01",
            filters={"user_id": "alice", "memory_type": "semantic"}
        )
        
        # Verify filters were passed to search
        mock_memory.search.assert_called_once()
        call_kwargs = mock_memory.search.call_args[1]
        assert call_kwargs.get('user_id') == "alice"
        assert call_kwargs.get('memory_type') == "semantic"
    
    def test_search_temporal_no_results(self, temporal_queries, mock_memory):
        """Test temporal search with no results."""
        mock_memory.search.return_value = []
        
        results = temporal_queries.search_temporal("nonexistent")
        
        assert isinstance(results, list)
        assert len(results) == 0
    
    def test_search_temporal_error_handling(self, temporal_queries, mock_memory):
        """Test error handling in temporal search."""
        # Make search raise an exception
        mock_memory.search.side_effect = Exception("Search failed")
        
        results = temporal_queries.search_temporal("test")
        
        # Should return empty list on error
        assert isinstance(results, list)
        assert len(results) == 0


class TestTemporalSearchPerformance:
    """Performance tests for temporal search."""
    
    @pytest.fixture
    def mock_memory(self):
        """Create mock memory system."""
        memory = Mock()
        memory.graph = Mock()
        memory.graph.execute_query = Mock(return_value=[])
        memory.search = Mock(return_value=[])
        return memory
    
    @pytest.fixture
    def temporal_queries(self, mock_memory):
        """Create temporal queries instance."""
        return TemporalQueries(mock_memory)
    
    def test_search_temporal_with_many_versions(self, temporal_queries, mock_memory):
        """Test temporal search performance with many versions."""
        from smartmemory.temporal.version_tracker import Version
        
        # Create mock result
        mock_result = Mock()
        mock_result.item_id = "test123"
        mock_result.content = "Test"
        mock_result.score = 1.0
        
        mock_memory.search.return_value = [mock_result]
        
        # Create many versions
        now = datetime.now(timezone.utc)
        versions = [
            Version(
                item_id="test123",
                version_number=i,
                content=f"Version {i}",
                transaction_time_start=now - timedelta(days=100-i)
            )
            for i in range(1, 51)  # 50 versions
        ]
        
        temporal_queries.version_tracker.get_versions = Mock(return_value=versions)
        
        results = temporal_queries.search_temporal(
            "test",
            start_time=(now - timedelta(days=100)).isoformat(),
            end_time=now.isoformat()
        )
        
        # Should handle many versions efficiently
        assert len(results) == 50
        assert all('version' in r for r in results)
