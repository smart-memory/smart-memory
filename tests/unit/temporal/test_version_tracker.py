"""
Unit tests for version tracking system.
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, MagicMock

from smartmemory.temporal.version_tracker import VersionTracker, Version


class TestVersion:
    """Test Version dataclass."""
    
    def test_version_creation(self):
        """Test creating a version."""
        now = datetime.now(timezone.utc)
        version = Version(
            item_id="test123",
            version_number=1,
            content="Test content",
            metadata={"key": "value"},
            valid_time_start=now,
            transaction_time_start=now
        )
        
        assert version.item_id == "test123"
        assert version.version_number == 1
        assert version.content == "Test content"
        assert version.metadata == {"key": "value"}
        assert version.valid_time_start == now
        assert version.transaction_time_start == now
    
    def test_version_to_dict(self):
        """Test converting version to dictionary."""
        now = datetime.now(timezone.utc)
        version = Version(
            item_id="test123",
            version_number=1,
            content="Test content",
            valid_time_start=now,
            transaction_time_start=now
        )
        
        data = version.to_dict()
        
        assert isinstance(data, dict)
        assert data['item_id'] == "test123"
        assert data['version_number'] == 1
        assert isinstance(data['valid_time_start'], str)  # Should be ISO string
        assert isinstance(data['transaction_time_start'], str)
    
    def test_version_from_dict(self):
        """Test creating version from dictionary."""
        now = datetime.now(timezone.utc)
        data = {
            'item_id': "test123",
            'version_number': 1,
            'content': "Test content",
            'metadata': {},
            'valid_time_start': now.isoformat(),
            'transaction_time_start': now.isoformat(),
            'valid_time_end': None,
            'transaction_time_end': None,
            'changed_by': None,
            'change_reason': None,
            'previous_version': None
        }
        
        version = Version.from_dict(data)
        
        assert version.item_id == "test123"
        assert version.version_number == 1
        assert isinstance(version.valid_time_start, datetime)
        assert isinstance(version.transaction_time_start, datetime)
    
    def test_version_is_current(self):
        """Test checking if version is current."""
        now = datetime.now(timezone.utc)
        
        # Current version (no transaction_time_end)
        current = Version(
            item_id="test123",
            version_number=1,
            content="Test",
            transaction_time_start=now,
            transaction_time_end=None
        )
        assert current.is_current() is True
        
        # Closed version
        closed = Version(
            item_id="test123",
            version_number=1,
            content="Test",
            transaction_time_start=now,
            transaction_time_end=now + timedelta(hours=1)
        )
        assert closed.is_current() is False
    
    def test_version_is_valid_at(self):
        """Test checking if version is valid at a time."""
        now = datetime.now(timezone.utc)
        
        version = Version(
            item_id="test123",
            version_number=1,
            content="Test",
            valid_time_start=now,
            valid_time_end=now + timedelta(days=1)
        )
        
        # Before valid time
        assert version.is_valid_at(now - timedelta(hours=1)) is False
        
        # During valid time
        assert version.is_valid_at(now + timedelta(hours=12)) is True
        
        # After valid time
        assert version.is_valid_at(now + timedelta(days=2)) is False
        
        # Open-ended (no end time)
        open_version = Version(
            item_id="test123",
            version_number=1,
            content="Test",
            valid_time_start=now,
            valid_time_end=None
        )
        assert open_version.is_valid_at(now + timedelta(days=100)) is True


class TestVersionTracker:
    """Test VersionTracker class."""
    
    @pytest.fixture
    def mock_graph(self):
        """Create mock graph backend."""
        graph = Mock()
        graph.add_node = Mock(return_value=True)
        graph.add_edge = Mock(return_value=True)
        graph.update_node = Mock(return_value=True)
        graph.execute_query = Mock(return_value=[])
        return graph
    
    @pytest.fixture
    def tracker(self, mock_graph):
        """Create version tracker with mock graph."""
        return VersionTracker(mock_graph)
    
    def test_create_first_version(self, tracker, mock_graph):
        """Test creating the first version of an item."""
        # Mock no existing versions
        mock_graph.execute_query.return_value = []
        
        version = tracker.create_version(
            item_id="test123",
            content="First version",
            metadata={"author": "alice"}
        )
        
        assert version.item_id == "test123"
        assert version.version_number == 1
        assert version.content == "First version"
        assert version.metadata["author"] == "alice"
        assert version.is_current() is True
        
        # Verify graph calls
        assert mock_graph.add_node.called
        assert mock_graph.add_edge.called
    
    def test_create_second_version(self, tracker, mock_graph):
        """Test creating a second version."""
        # Mock existing version
        existing_version = Version(
            item_id="test123",
            version_number=1,
            content="First version",
            transaction_time_start=datetime.now(timezone.utc),
            transaction_time_end=None
        )
        
        # Mock query to return existing version
        mock_graph.execute_query.return_value = [
            {'v': existing_version.to_dict()}
        ]
        
        # Create second version
        version = tracker.create_version(
            item_id="test123",
            content="Second version",
            changed_by="bob",
            change_reason="Update"
        )
        
        assert version.version_number == 2
        assert version.content == "Second version"
        assert version.changed_by == "bob"
        assert version.change_reason == "Update"
        assert version.previous_version == 1
    
    def test_get_versions_empty(self, tracker, mock_graph):
        """Test getting versions when none exist."""
        mock_graph.execute_query.return_value = []
        
        versions = tracker.get_versions("nonexistent")
        
        assert isinstance(versions, list)
        assert len(versions) == 0
    
    def test_get_current_version(self, tracker, mock_graph):
        """Test getting current version."""
        # Mock current version
        current = Version(
            item_id="test123",
            version_number=2,
            content="Current",
            transaction_time_start=datetime.now(timezone.utc),
            transaction_time_end=None
        )
        
        mock_graph.execute_query.return_value = [
            {'v': current.to_dict()}
        ]
        
        version = tracker.get_current_version("test123")
        
        assert version is not None
        assert version.version_number == 2
        assert version.is_current() is True
    
    def test_get_version_at_time(self, tracker, mock_graph):
        """Test getting version at specific time."""
        now = datetime.now(timezone.utc)
        
        # Mock versions
        v1 = Version(
            item_id="test123",
            version_number=1,
            content="Version 1",
            transaction_time_start=now - timedelta(hours=2),
            transaction_time_end=now - timedelta(hours=1)
        )
        
        v2 = Version(
            item_id="test123",
            version_number=2,
            content="Version 2",
            transaction_time_start=now - timedelta(hours=1),
            transaction_time_end=None
        )
        
        mock_graph.execute_query.return_value = [
            {'v': v1.to_dict()},
            {'v': v2.to_dict()}
        ]
        
        # Get version 90 minutes ago (should be v1)
        version = tracker.get_version_at_time(
            "test123",
            now - timedelta(minutes=90)
        )
        
        assert version is not None
        assert version.version_number == 1
    
    def test_compare_versions(self, tracker, mock_graph):
        """Test comparing two versions."""
        now = datetime.now(timezone.utc)
        
        v1 = Version(
            item_id="test123",
            version_number=1,
            content="Original content",
            metadata={"key1": "value1"},
            transaction_time_start=now - timedelta(hours=1)
        )
        
        v2 = Version(
            item_id="test123",
            version_number=2,
            content="Updated content",
            metadata={"key1": "value1", "key2": "value2"},
            transaction_time_start=now,
            changed_by="alice",
            change_reason="Update"
        )
        
        mock_graph.execute_query.return_value = [
            {'v': v1.to_dict()},
            {'v': v2.to_dict()}
        ]
        
        comparison = tracker.compare_versions("test123", 1, 2)
        
        assert comparison['item_id'] == "test123"
        assert comparison['version1'] == 1
        assert comparison['version2'] == 2
        assert comparison['content_changed'] is True
        assert 'metadata_changes' in comparison
        assert comparison['changed_by'] == "alice"
        assert comparison['change_reason'] == "Update"
    
    def test_version_caching(self, tracker, mock_graph):
        """Test that versions are cached."""
        mock_graph.execute_query.return_value = [
            {'v': Version(
                item_id="test123",
                version_number=1,
                content="Test",
                transaction_time_start=datetime.now(timezone.utc)
            ).to_dict()}
        ]
        
        # First call
        versions1 = tracker.get_versions("test123")
        
        # Second call (should use cache)
        versions2 = tracker.get_versions("test123")
        
        # Should only query once
        assert mock_graph.execute_query.call_count == 1
        assert versions1 == versions2
