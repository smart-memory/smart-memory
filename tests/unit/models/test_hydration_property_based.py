"""
Property-based tests for hydration/dehydration using hypothesis.

These tests generate random MemoryItems and verify that serialization
is always reversible, catching edge cases that manual tests might miss.
"""
import pytest
from hypothesis import given, strategies as st, assume
from datetime import datetime, timezone
from smartmemory.models.memory_item import MemoryItem
from smartmemory.graph.core.nodes import SmartGraphNodes
from smartmemory.utils.serialization import MemoryItemSerializer


# Custom strategies for MemoryItem fields
@st.composite
def memory_item_strategy(draw):
    """Generate random but valid MemoryItem instances."""
    content = draw(st.text(min_size=0, max_size=1000))
    user_id = draw(st.one_of(st.none(), st.text(min_size=1, max_size=100)))
    memory_type = draw(st.sampled_from(['semantic', 'episodic', 'procedural']))
    
    # Generate nested metadata with various types
    metadata = draw(st.dictionaries(
        keys=st.text(min_size=1, max_size=50, alphabet=st.characters(
            blacklist_categories=('Cs',),  # Exclude surrogates
            blacklist_characters='\x00'  # Exclude null bytes
        )),
        values=st.one_of(
            st.text(max_size=100),
            st.integers(),
            st.floats(allow_nan=False, allow_infinity=False),
            st.booleans(),
            st.lists(st.text(max_size=50), max_size=10)
        ),
        max_size=20
    ))
    
    # Optionally add nested metadata
    if draw(st.booleans()):
        nested_key = draw(st.text(min_size=1, max_size=20))
        nested_value = draw(st.dictionaries(
            keys=st.text(min_size=1, max_size=20),
            values=st.text(max_size=50),
            max_size=5
        ))
        metadata[nested_key] = nested_value
    
    return MemoryItem(
        content=content,
        user_id=user_id,
        memory_type=memory_type,
        metadata=metadata
    )


class TestPropertyBasedRoundtrip:
    """Property-based tests for serialization roundtrips."""
    
    @given(memory_item_strategy())
    def test_to_dict_from_dict_roundtrip(self, item):
        """Property: to_dict() followed by reconstruction preserves data."""
        # Serialize
        data = item.to_dict()
        
        # Verify no internal fields leaked
        assert '_immutable_fields' not in data
        
        # Deserialize via from_node (which uses dict input)
        restored = MemoryItem.from_node(data)
        
        # Verify critical fields preserved
        assert restored.content == item.content
        assert restored.user_id == item.user_id
        assert restored.memory_type == item.memory_type
        assert restored.item_id == item.item_id
    
    @given(memory_item_strategy())
    def test_to_node_from_node_roundtrip(self, item):
        """Property: to_node() followed by from_node() preserves data."""
        # Serialize
        node_data = item.to_node()
        
        # Deserialize
        restored = MemoryItem.from_node(node_data)
        
        # Verify critical fields preserved
        assert restored.content == item.content
        assert restored.user_id == item.user_id
        assert restored.memory_type == item.memory_type
        assert restored.item_id == item.item_id
    
    @given(memory_item_strategy())
    def test_graph_nodes_roundtrip(self, item):
        """Property: SmartGraphNodes serialization is reversible."""
        # Serialize
        node_dict = SmartGraphNodes._to_node_dict(item)
        
        # Verify no internal fields
        assert '_immutable_fields' not in node_dict
        
        # Deserialize
        restored = SmartGraphNodes._from_node_dict(MemoryItem, node_dict)
        
        # Verify critical fields preserved
        assert restored.content == item.content
        assert restored.user_id == item.user_id
        assert restored.memory_type == item.memory_type
        assert restored.item_id == item.item_id
    
    @given(memory_item_strategy())
    def test_serializer_roundtrip(self, item):
        """Property: MemoryItemSerializer roundtrip preserves data."""
        # Use the centralized serializer
        storage_data = MemoryItemSerializer.to_storage(item)
        
        # Verify no internal fields leaked
        assert '_immutable_fields' not in storage_data
        
        # Deserialize
        restored = MemoryItemSerializer.from_storage(MemoryItem, storage_data)
        
        # Verify critical fields preserved
        assert restored.content == item.content
        assert restored.user_id == item.user_id
        assert restored.memory_type == item.memory_type
        assert restored.item_id == item.item_id
    
    @given(memory_item_strategy())
    def test_serializer_validate_roundtrip(self, item):
        """Property: Serializer validation always passes for valid items."""
        # This should never raise
        assert MemoryItemSerializer.validate_roundtrip(item) is True


class TestPropertyBasedMetadata:
    """Property-based tests for metadata handling."""
    
    @given(
        st.dictionaries(
            keys=st.text(min_size=1, max_size=50),
            values=st.one_of(
                st.text(max_size=100),
                st.integers(),
                st.dictionaries(
                    keys=st.text(min_size=1, max_size=20),
                    values=st.text(max_size=50),
                    max_size=5
                )
            ),
            max_size=20
        )
    )
    def test_metadata_with_underscores_preserved(self, metadata):
        """Property: Metadata keys with single underscores are preserved."""
        # Add keys with single underscores
        metadata['user_name'] = 'test'
        metadata['api_key'] = 'secret'
        
        item = MemoryItem(content="test", metadata=metadata)
        
        # Roundtrip
        node_data = item.to_node()
        restored = MemoryItem.from_node(node_data)
        
        # Verify underscore keys preserved
        assert 'user_name' in str(restored.metadata) or 'user_name' in node_data
        assert 'api_key' in str(restored.metadata) or 'api_key' in node_data
    
    @given(memory_item_strategy())
    def test_metadata_never_overwrites_system_fields(self, item):
        """Property: Metadata never overwrites system fields like content."""
        # Add conflicting keys to metadata
        item.metadata['content'] = 'METADATA_CONTENT'
        item.metadata['user_id'] = 'METADATA_USER'
        item.metadata['item_id'] = 'METADATA_ID'
        
        # System fields should win
        assert item.content != 'METADATA_CONTENT'
        assert item.user_id != 'METADATA_USER'
        assert item.item_id != 'METADATA_ID'
        
        # Roundtrip should preserve system fields
        node_data = item.to_node()
        restored = MemoryItem.from_node(node_data)
        
        assert restored.content == item.content
        assert restored.user_id == item.user_id
        assert restored.item_id == item.item_id


class TestPropertyBasedImmutability:
    """Property-based tests for immutability constraints."""
    
    @given(
        st.text(min_size=1, max_size=100),
        st.text(min_size=1, max_size=100)
    )
    def test_content_immutability(self, first_content, second_content):
        """Property: Content cannot be changed after initial set."""
        assume(first_content != second_content)  # Only test different values
        
        item = MemoryItem(content=first_content)
        
        with pytest.raises(ValueError, match="content has already been set"):
            item.content = second_content
    
    @given(
        st.text(min_size=1, max_size=100),
        st.text(min_size=1, max_size=100)
    )
    def test_user_id_immutability(self, first_user_id, second_user_id):
        """Property: User ID cannot be changed after initial set."""
        assume(first_user_id != second_user_id)
        
        item = MemoryItem(user_id=first_user_id)
        
        with pytest.raises(ValueError, match="user_id has already been set"):
            item.user_id = second_user_id
    
    @given(memory_item_strategy())
    def test_idempotent_setting(self, item):
        """Property: Setting same value multiple times is idempotent."""
        original_content = item.content
        original_user_id = item.user_id
        
        # Setting same value should not raise
        item.content = original_content
        item.user_id = original_user_id
        
        # Values unchanged
        assert item.content == original_content
        assert item.user_id == original_user_id


class TestPropertyBasedEdgeCases:
    """Property-based tests for edge cases."""
    
    @given(st.text(max_size=10000))
    def test_large_content(self, content):
        """Property: Large content values are handled correctly."""
        item = MemoryItem(content=content)
        
        # Roundtrip
        node_data = item.to_node()
        restored = MemoryItem.from_node(node_data)
        
        assert restored.content == content
    
    @given(st.text(alphabet=st.characters(min_codepoint=0x1F300, max_codepoint=0x1F6FF)))
    def test_unicode_content(self, content):
        """Property: Unicode content (emojis, etc.) is preserved."""
        item = MemoryItem(content=content)
        
        # Roundtrip
        node_data = item.to_node()
        restored = MemoryItem.from_node(node_data)
        
        assert restored.content == content
    
    @given(st.text(min_size=0, max_size=0))
    def test_empty_content(self, content):
        """Property: Empty content is handled correctly."""
        item = MemoryItem(content=content)
        
        assert item.content == ""
        
        # Roundtrip
        node_data = item.to_node()
        restored = MemoryItem.from_node(node_data)
        
        assert restored.content == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--hypothesis-show-statistics"])
