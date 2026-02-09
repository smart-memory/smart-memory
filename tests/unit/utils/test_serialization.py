"""
Unit tests for serialization utilities.
Tests MemoryItemSerializer and helper functions for edge cases and correctness.
"""
import pytest


pytestmark = pytest.mark.unit
from smartmemory.utils.serialization import MemoryItemSerializer, ensure_public_fields
from smartmemory.models.memory_item import MemoryItem

class TestMemoryItemSerializer:
    """Test cases for MemoryItemSerializer."""

    def test_to_storage_flattening(self):
        """Test that metadata is correctly flattened."""
        item = MemoryItem(
            content="Test Content",
            metadata={
                "simple": "value",
                "nested": {"key": "nested_value"},
                "deep": {"level1": {"level2": "deep_value"}}
            }
        )
        
        storage_data = MemoryItemSerializer.to_storage(item)
        
        assert storage_data["content"] == "Test Content"
        assert storage_data["simple"] == "value"
        assert storage_data["nested__key"] == "nested_value"
        assert storage_data["deep__level1__level2"] == "deep_value"
        assert "metadata" not in storage_data

    def test_to_storage_system_field_protection(self):
        """Test that metadata cannot overwrite system fields."""
        item = MemoryItem(
            content="System Content",
            metadata={
                "content": "Metadata Content",  # Should be ignored/dropped from top level
            }
        )
        
        storage_data = MemoryItemSerializer.to_storage(item)
        
        # System fields should be preserved
        assert storage_data["content"] == "System Content"
        # Metadata fields that clash are not in top level
        # They should have been filtered out by the safe_metadata logic
        assert "content" in storage_data
        assert storage_data["content"] != "Metadata Content"

    def test_from_storage_unflattening(self):
        """Test that storage data is correctly unflattened into MemoryItem."""
        data = {
            "content": "Test Content",
            "simple": "value",
            "nested__key": "nested_value",
            "deep__level1__level2": "deep_value"
        }
        
        item = MemoryItemSerializer.from_storage(MemoryItem, data)
        
        assert item.content == "Test Content"
        assert item.metadata["simple"] == "value"
        assert item.metadata["nested"]["key"] == "nested_value"
        assert item.metadata["deep"]["level1"]["level2"] == "deep_value"

    def test_roundtrip_complex_data(self):
        """Test roundtrip serialization with complex data types."""
        item = MemoryItem(
            content="Complex Content",
            metadata={
                "list_data": [1, 2, 3],
                "dict_data": {"a": 1, "b": 2},
                "mixed": [{"x": 1}, [4, 5]]
            }
        )
        
        assert MemoryItemSerializer.validate_roundtrip(item)

    def test_internal_field_filtering(self):
        """Test that internal fields are not serialized."""
        item = MemoryItem(content="Test")
        # Simulate internal field existence if it's not there by default
        # But MemoryItem might have _immutable_fields
        
        storage_data = MemoryItemSerializer.to_storage(item)
        
        assert "_immutable_fields" not in storage_data
        assert "content" in storage_data

    def test_ensure_public_fields(self):
        """Test ensure_public_fields utility."""
        data = {
            "public": "value",
            "_immutable_fields": ["some", "fields"]
        }
        
        clean = ensure_public_fields(data)
        assert "public" in clean
        assert "_immutable_fields" not in clean
        
    def test_unknown_fields_to_metadata(self):
        """Test that unknown top-level fields in dict are moved to metadata on serialization."""
        # This scenario happens if we manually construct a dict that has extra fields 
        # and pass it to to_storage? No, to_storage takes a MemoryItem.
        # But if MemoryItem.to_dict() returns extra fields (e.g. dynamic attributes), 
        # they should go to metadata.
        
        class ExtendedMemoryItem(MemoryItem):
            def to_dict(self):
                d = super().to_dict()
                d['extra_attr'] = 'extra_value'
                return d
                
        item = ExtendedMemoryItem(content="Test")
        storage_data = MemoryItemSerializer.to_storage(item)
        
        # extra_attr is not a SYSTEM_FIELD, so it should be in metadata (flat)
        assert storage_data['extra_attr'] == 'extra_value'
        # When deserializing, it should end up in metadata
        
        restored = MemoryItemSerializer.from_storage(MemoryItem, storage_data)
        assert 'extra_attr' in restored.metadata
        assert restored.metadata['extra_attr'] == 'extra_value'
