"""
Serialization protocols and utilities for SmartMemory.

Provides a standardized interface for serializing/deserializing memory items
to/from storage formats, ensuring consistent handling across backends.
"""
from typing import Protocol, Dict, Any, TypeVar, Type
from smartmemory.utils import flatten_dict, unflatten_dict


T = TypeVar('T', bound='Serializable')


class Serializable(Protocol):
    """Protocol for objects that can be serialized to/from storage format."""
    
    def to_storage(self) -> Dict[str, Any]:
        """
        Convert object to storage-ready dictionary.
        
        Returns flattened dict with public field names only.
        Protected fields must be mapped to public names.
        """
        ...
    
    @classmethod
    def from_storage(cls: Type[T], data: Dict[str, Any]) -> T:
        """
        Reconstruct object from storage dictionary.
        
        Args:
            data: Flattened dictionary from storage
            
        Returns:
            Reconstructed object instance
        """
        ...


class MemoryItemSerializer:
    """
    Centralized serialization logic for MemoryItem.
    
    Single source of truth for converting between MemoryItem objects
    and storage format, ensuring reversible transformations.
    """
    
    # System fields that should remain at top level (not in metadata)
    SYSTEM_FIELDS = {
        'item_id', 'content', 'memory_type', 'group_id',
        'valid_start_time', 'valid_end_time', 'transaction_time',
        'embedding', 'entities', 'relations', 'node_category'
    }
    
    @classmethod
    def to_storage(cls, item: Any) -> Dict[str, Any]:
        """
        Serialize MemoryItem to storage format.
        
        Process:
        1. Extract all fields via to_dict()
        2. Separate system fields from metadata
        3. Flatten metadata with __ separator
        4. Return flat dict ready for backend storage
        
        Args:
            item: MemoryItem instance
            
        Returns:
            Flattened dictionary with public field names
        """
        # Get dict representation
        data = item.to_dict()
        
        # Validate no internal tracking fields leaked
        internal_fields = {'_immutable_fields'}
        leaked = internal_fields & set(data.keys())
        if leaked:
            raise ValueError(f"Internal fields leaked in serialization: {leaked}")
        
        # Separate system fields from metadata
        system_data = {}
        metadata = data.get('metadata', {})
        
        for key, value in data.items():
            if key in cls.SYSTEM_FIELDS:
                system_data[key] = value
            elif key != 'metadata':
                # Unknown fields go to metadata
                metadata[key] = value
        
        # Flatten metadata with __ separator
        if metadata:
            flattened_metadata = flatten_dict(metadata, sep='__')
            # Prevent metadata from overwriting system fields
            # System fields are canonical; metadata keys that clash are ignored/dropped from top level
            safe_metadata = {k: v for k, v in flattened_metadata.items() if k not in system_data}
            system_data.update(safe_metadata)
        
        return system_data
    
    @classmethod
    def from_storage(cls, item_class: Type[T], data: Dict[str, Any]) -> T:
        """
        Deserialize storage data to MemoryItem.
        
        Process:
        1. Separate system fields from metadata fields
        2. Unflatten metadata fields with __ separator
        3. Construct MemoryItem with public field names
        
        Args:
            item_class: MemoryItem class or subclass
            data: Flattened dictionary from storage
            
        Returns:
            MemoryItem instance
        """
        # Separate system fields from metadata
        system_fields = {}
        metadata_fields = {}
        
        for key, value in data.items():
            if key in cls.SYSTEM_FIELDS:
                system_fields[key] = value
            else:
                # Everything else is metadata (may be flattened)
                metadata_fields[key] = value
        
        # Unflatten metadata
        if metadata_fields:
            metadata = unflatten_dict(metadata_fields, sep='__')
            system_fields['metadata'] = metadata
        
        # Construct object using public field names
        return item_class(**system_fields)
    
    @classmethod
    def validate_roundtrip(cls, item: Any) -> bool:
        """
        Validate that serialization is reversible.
        
        Args:
            item: MemoryItem instance
            
        Returns:
            True if roundtrip preserves data
            
        Raises:
            AssertionError if roundtrip fails
        """
        # Serialize
        storage_data = cls.to_storage(item)
        
        # Deserialize
        restored = cls.from_storage(type(item), storage_data)
        
        # Validate critical fields
        assert restored.content == item.content, "Content mismatch"
        assert restored.item_id == item.item_id, "Item ID mismatch"
        assert restored.memory_type == item.memory_type, "Memory type mismatch"
        
        # Validate no internal tracking fields in storage
        internal_fields = {'_immutable_fields'}
        leaked = internal_fields & set(storage_data.keys())
        assert not leaked, f"Internal fields in storage: {leaked}"
        
        return True


def ensure_public_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure dictionary doesn't contain internal tracking fields.
    
    Removes internal fields like _immutable_fields that shouldn't be serialized.
    
    Args:
        data: Dictionary potentially containing internal field names
        
    Returns:
        Dictionary with only public fields
    """
    result = data.copy()
    
    # Remove internal tracking fields
    result.pop('_immutable_fields', None)
    
    return result
