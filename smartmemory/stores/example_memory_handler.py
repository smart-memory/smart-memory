"""
Example memory handler demonstrating the BaseHandler[MemoryItem] + Store composition pattern.
Shows how handlers stay store-agnostic while stores self-configure.
"""

from datetime import datetime, timezone
from typing import Optional, Union, List, Any, Dict

from smartmemory.models.memory_item import MemoryItem
from smartmemory.stores.base import BaseHandler
from smartmemory.stores.json_store import JSONStore


class ExampleMemoryHandler(BaseHandler[MemoryItem]):
    """Example memory handler implementing BaseHandler[MemoryItem] with store delegation.
    
    Demonstrates:
    - Clean handler-store separation
    - Store-agnostic handler implementation  
    - Domain logic (MemoryItem) vs primitive storage (Dict)
    - No mixins, no isinstance checks
    """

    def __init__(self, store: BaseHandler[Dict]):
        """Initialize handler with any store implementing BaseHandler[Dict].
        
        Args:
            store: Any store implementing BaseHandler[Dict] (JSONStore, MongoStore, etc.)
        """
        self.store = store

    def add(self, item: Union[MemoryItem, Any], **kwargs) -> Union[str, MemoryItem, None]:
        """Add a memory item with domain logic and validation."""
        # Domain logic: Convert to MemoryItem
        memory_item = self._normalize_to_memory_item(item)
        if not memory_item:
            return None

        # Domain logic: Validation
        if not self._validate_memory_item(memory_item):
            return None

        # Convert to primitive dict for store
        primitive_dict = memory_item.to_dict()

        # Delegate to store's BaseHandler[Dict] interface
        result = self.store.add(primitive_dict, **kwargs)

        # Return based on what store returned
        if isinstance(result, str):
            return result  # Store returned item_id
        elif result:
            return memory_item  # Store returned dict, we return the MemoryItem

        return None

    def get(self, item_id: str, **kwargs) -> Optional[MemoryItem]:
        """Get a memory item with domain conversion."""
        # Delegate to store's BaseHandler[Dict] interface
        primitive_dict = self.store.get(item_id, **kwargs)

        if not primitive_dict:
            return None

        # Convert primitive dict back to domain MemoryItem
        return self._dict_to_memory_item(primitive_dict)

    def update(self, item: Union[MemoryItem, Any], **kwargs) -> Union[bool, MemoryItem]:
        """Update a memory item with domain logic."""
        # Domain logic: Convert to MemoryItem
        memory_item = self._normalize_to_memory_item(item)
        if not memory_item:
            return False

        # Domain logic: Validation
        if not memory_item.item_id:
            return False

        # Convert to primitive dict for store
        primitive_dict = memory_item.to_dict()

        # Delegate to store's BaseHandler[Dict] interface
        result = self.store.update(primitive_dict, **kwargs)

        # Convert result back to domain type
        if isinstance(result, bool):
            return result
        elif isinstance(result, dict):
            return self._dict_to_memory_item(result)

        return False

    def delete(self, item_id: str, **kwargs) -> bool:
        """Delete a memory item."""
        # Simple delegation - no domain logic needed for delete
        return self.store.delete(item_id, **kwargs)

    def search(self, query: Any, **kwargs) -> List[MemoryItem]:
        """Search memory items with domain conversion."""
        # Delegate to store's BaseHandler[Dict] interface
        primitive_results = self.store.search(query, **kwargs)

        # Convert primitive dicts back to domain MemoryItems
        memory_items = []
        for primitive_dict in primitive_results:
            memory_item = self._dict_to_memory_item(primitive_dict)
            if memory_item:
                memory_items.append(memory_item)

        return memory_items

    def clear(self, **kwargs) -> bool:
        """Clear all memory items."""
        return self.store.clear(**kwargs)

    # Domain logic helpers (not part of BaseHandler interface)

    def _normalize_to_memory_item(self, item: Union[MemoryItem, Any]) -> Optional[MemoryItem]:
        """Convert various inputs to MemoryItem - domain logic."""
        if isinstance(item, MemoryItem):
            return item

        if hasattr(item, 'to_memory_item'):
            return item.to_memory_item()

        if isinstance(item, dict):
            return MemoryItem(**item)

        if isinstance(item, str):
            return MemoryItem(content=item)

        return None

    def _validate_memory_item(self, item: MemoryItem) -> bool:
        """Validate MemoryItem - domain logic."""
        if not item:
            return False

        if not hasattr(item, 'content') or not item.content:
            return False

        if not hasattr(item, 'metadata') or not isinstance(item.metadata, dict):
            return False

        return True

    def _dict_to_memory_item(self, data: Dict) -> Optional[MemoryItem]:
        """Convert primitive dict to MemoryItem - domain logic."""
        try:
            return MemoryItem.from_dict(data)
        except Exception:
            # Fallback: create MemoryItem from dict keys
            return MemoryItem(
                item_id=data.get('id'),
                content=data.get('content', ''),
                metadata=data.get('metadata', {})
            )

    # Memory-specific domain methods (not part of BaseHandler)

    def archive_item(self, item_id: str, reason: str = "manual") -> bool:
        """Archive a memory item (domain-specific operation)."""
        item = self.get(item_id)
        if not item:
            return False

        # Add archival metadata
        item.metadata['archived'] = True
        item.metadata['archive_reason'] = reason
        item.metadata['archive_timestamp'] = datetime.now(timezone.utc).isoformat()

        # Update the item
        result = self.update(item)
        return bool(result)

    def get_archived_items(self) -> List[MemoryItem]:
        """Get all archived memory items (domain-specific query)."""
        # This could be optimized with store-specific queries, but stays generic
        all_items = self.search("")  # Get all items
        return [item for item in all_items if item.metadata.get('archived', False)]


# Usage Examples demonstrating the pattern:

def create_episodic_memory_handler() -> ExampleMemoryHandler:
    """Create handler optimized for episodic memory."""
    # Store self-configures for episodic use case
    episodic_store = JSONStore(data_dir="./data/episodic", optimize_for="episodic")
    return ExampleMemoryHandler(episodic_store)


def create_semantic_memory_handler() -> ExampleMemoryHandler:
    """Create handler optimized for semantic memory."""
    # Same handler class, store self-configures differently
    semantic_store = JSONStore(data_dir="./data/semantic", optimize_for="semantic")
    return ExampleMemoryHandler(semantic_store)


def create_ontology_memory_handler() -> ExampleMemoryHandler:
    """Create handler optimized for ontology memory."""
    # Same handler class, store self-configures for ontology
    ontology_store = JSONStore(data_dir="./data/ontology", optimize_for="ontology")
    return ExampleMemoryHandler(ontology_store)


# Example usage:
def example_usage():
    """Demonstrate the pattern in action."""

    # Create different handlers with self-configuring stores
    episodic = create_episodic_memory_handler()
    semantic = create_semantic_memory_handler()

    # Same handler interface, different optimizations
    memory = MemoryItem(content="Example memory", metadata={"type": "test"})

    # Episodic store: auto-timestamps, temporal sorting
    episodic_id = episodic.add(memory)

    # Semantic store: content indexing, deduplication  
    semantic_id = semantic.add(memory)

    # Same handler methods work with any store
    retrieved_episodic = episodic.get(episodic_id)
    retrieved_semantic = semantic.get(semantic_id)

    # Domain-specific methods work regardless of store
    episodic.archive_item(episodic_id, "example cleanup")
    archived_items = episodic.get_archived_items()

    return {
        'episodic_item': retrieved_episodic,
        'semantic_item': retrieved_semantic,
        'archived_count': len(archived_items)
    }
