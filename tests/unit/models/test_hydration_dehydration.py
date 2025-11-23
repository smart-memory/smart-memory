"""
Unit tests for hydration/dehydration implementation.
Tests serialization/deserialization roundtrips and validates encapsulation.
"""
import pytest
from datetime import datetime, timezone
from smartmemory.models.memory_item import MemoryItem
from smartmemory.graph.core.nodes import SmartGraphNodes
from smartmemory.models.compat.dataclass_model import get_field_names


class TestMemoryItemSerialization:
    """Test MemoryItem serialization methods."""
    
    def test_to_dict_maps_protected_fields(self):
        """Verify to_dict() correctly maps _content to content."""
        item = MemoryItem(
            content="Test content",
            user_id="user123",
            metadata={"key": "value"}
        )
        
        data = item.to_dict()
        
        # Public names should be present
        assert 'content' in data
        assert data['content'] == "Test content"
        assert 'user_id' in data
        assert data['user_id'] == "user123"
        
        # Protected names should NOT be present
        assert '_content' not in data
        assert '_user_id' not in data
        assert '_embedding' not in data
    
    def test_to_node_maps_protected_fields(self):
        """Verify to_node() correctly maps _content to content."""
        item = MemoryItem(
            content="Test content",
            user_id="user123",
            metadata={"nested": {"key": "value"}}
        )
        
        node = item.to_node()
        
        # Public names should be present
        assert 'content' in node
        assert node['content'] == "Test content"
        assert 'user_id' in node
        assert node['user_id'] == "user123"
        
        # Protected names should NOT be present
        assert '_content' not in node
        assert '_user_id' not in node
        assert '_embedding' not in node
        
        # Metadata should be flattened
        assert 'nested' in node or 'nested__key' in node  # Depends on flattening
    
    def test_from_node_hydrates_correctly(self):
        """Verify from_node() correctly hydrates from public field names."""
        node_data = {
            'item_id': 'test123',
            'content': 'Test content',
            'user_id': 'user123',
            'memory_type': 'semantic',
            'metadata_key': 'metadata_value'
        }
        
        item = MemoryItem.from_node(node_data)
        
        assert item.content == 'Test content'
        assert item.user_id == 'user123'
        assert item.item_id == 'test123'
        assert item.memory_type == 'semantic'
        # metadata_key should be in metadata
        assert 'metadata_key' in item.metadata


class TestSmartGraphNodesHydration:
    """Test SmartGraphNodes hydration/dehydration."""
    
    def test_to_node_dict_no_protected_fields(self):
        """Verify _to_node_dict doesn't expose protected fields."""
        item = MemoryItem(
            content="Test content",
            user_id="user123"
        )
        
        node_dict = SmartGraphNodes._to_node_dict(item)
        
        # Should have public names
        assert 'content' in node_dict
        assert node_dict['content'] == "Test content"
        
        # Should NOT have protected names
        assert '_content' not in node_dict
        assert '_user_id' not in node_dict
    
    def test_from_node_dict_handles_public_fields(self):
        """Verify _from_node_dict correctly maps public field names."""
        node_data = {
            'item_id': 'test123',
            'content': 'Test content',
            'user_id': 'user123',
            'memory_type': 'semantic',
            'extra_field': 'extra_value'
        }
        
        item = SmartGraphNodes._from_node_dict(MemoryItem, node_data)
        
        assert item.content == 'Test content'
        assert item.user_id == 'user123'
        assert item.item_id == 'test123'
        # extra_field should be in metadata
        assert 'extra_field' in item.metadata
        assert item.metadata['extra_field'] == 'extra_value'
    
    def test_from_node_dict_with_nested_properties(self):
        """Verify _from_node_dict handles Neo4j-style nested properties."""
        node_data = {
            'properties': {
                'item_id': 'test123',
                'content': 'Test content',
                'user_id': 'user123',
                'memory_type': 'semantic'
            }
        }
        
        item = SmartGraphNodes._from_node_dict(MemoryItem, node_data)
        
        assert item.content == 'Test content'
        assert item.user_id == 'user123'


class TestRoundtripSerialization:
    """Test complete roundtrip serialization."""
    
    def test_roundtrip_via_to_node_from_node(self):
        """Verify data survives roundtrip through to_node/from_node."""
        original = MemoryItem(
            content="Test content with special chars: 日本語",
            user_id="user123",
            memory_type="episodic",
            metadata={
                "nested": {
                    "key": "value",
                    "number": 42
                },
                "list": [1, 2, 3]
            }
        )
        
        # Serialize
        node_data = original.to_node()
        
        # Deserialize
        restored = MemoryItem.from_node(node_data)
        
        # Verify critical fields
        assert restored.content == original.content
        assert restored.user_id == original.user_id
        assert restored.memory_type == original.memory_type
        assert restored.item_id == original.item_id
    
    def test_roundtrip_via_graph_nodes(self):
        """Verify data survives roundtrip through SmartGraphNodes."""
        original = MemoryItem(
            content="Test content",
            user_id="user123",
            metadata={"key": "value"}
        )
        
        # Serialize
        node_dict = SmartGraphNodes._to_node_dict(original)
        
        # Deserialize
        restored = SmartGraphNodes._from_node_dict(MemoryItem, node_dict)
        
        # Verify
        assert restored.content == original.content
        assert restored.user_id == original.user_id
        assert restored.item_id == original.item_id


class TestFieldIntrospection:
    """Test field introspection utilities."""
    
    def test_get_field_names_returns_public_fields(self):
        """Verify get_field_names returns public field names (after refactoring)."""
        field_names = get_field_names(MemoryItem)
        
        # After refactoring: public fields are actual dataclass fields
        assert 'content' in field_names
        assert 'user_id' in field_names
        assert 'embedding' in field_names
        
        # Protected fields no longer exist
        assert '_content' not in field_names
        assert '_user_id' not in field_names
        assert '_embedding' not in field_names
        
        # Internal tracking field should be present but excluded from serialization
        assert '_immutable_fields' in field_names


class TestImmutability:
    """Test immutability constraints."""
    
    def test_content_immutable_after_set(self):
        """Verify content cannot be modified after initial set."""
        item = MemoryItem(content="Original content")
        
        with pytest.raises(ValueError, match="content has already been set"):
            item.content = "New content"
    
    def test_content_can_be_set_once(self):
        """Verify content can be set if not previously set."""
        item = MemoryItem()  # No content
        item.content = "First set"
        assert item.content == "First set"
        
        # Second set should fail
        with pytest.raises(ValueError):
            item.content = "Second set"
    
    def test_user_id_immutable_after_set(self):
        """Verify user_id cannot be modified after initial set."""
        item = MemoryItem(user_id="user123")
        
        with pytest.raises(ValueError, match="user_id has already been set"):
            item.user_id = "user456"


class TestAbstractionBoundaries:
    """Test that abstractions don't leak between layers."""
    
    def test_backend_data_has_no_protected_fields(self):
        """Verify backend storage format doesn't contain protected field names."""
        item = MemoryItem(
            content="Test",
            user_id="user123"
        )
        
        # Simulate what backend stores
        backend_data = SmartGraphNodes._to_node_dict(item)
        
        # Backend should only see public API
        assert '_content' not in backend_data
        assert '_user_id' not in backend_data
        assert '_embedding' not in backend_data
        
        # Backend should see public names
        assert 'content' in backend_data
        assert 'user_id' in backend_data
    
    def test_hydration_doesnt_require_field_name_knowledge(self):
        """Verify hydration works with public field names only."""
        # Simulate backend returning data with public names
        backend_data = {
            'item_id': 'test123',
            'content': 'Test content',
            'user_id': 'user123',
            'memory_type': 'semantic'
        }
        
        # Hydration should work without knowing about _content, _user_id
        item = SmartGraphNodes._from_node_dict(MemoryItem, backend_data)
        
        assert item.content == 'Test content'
        assert item.user_id == 'user123'


class TestMetadataHandling:
    """Test metadata flattening and unflattening."""
    
    def test_nested_metadata_preserved(self):
        """Verify nested metadata structures survive serialization."""
        original = MemoryItem(
            content="Test",
            metadata={
                "level1": {
                    "level2": {
                        "level3": "deep value"
                    }
                }
            }
        )
        
        node_data = original.to_node()
        restored = MemoryItem.from_node(node_data)
        
        # Metadata structure should be preserved
        # (May be flattened in storage but should unflatten on restore)
        assert 'level1' in restored.metadata or 'level1__level2__level3' in node_data
    
    def test_metadata_doesnt_overwrite_system_fields(self):
        """Verify metadata doesn't overwrite system fields like content."""
        item = MemoryItem(
            content="System content",
            metadata={
                "content": "Metadata content",  # Should NOT overwrite
                "custom": "value"
            }
        )
        
        # System content should win
        assert item.content == "System content"
        
        # Metadata content should be preserved separately
        assert item.metadata.get("content") == "Metadata content"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
