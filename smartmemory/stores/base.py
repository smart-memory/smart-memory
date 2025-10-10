"""
Base Handler Interface

Provides a flexible common CRUD interface for all store handlers.
Supports both sync/async operations and flexible return types.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Union, Generic, TypeVar

# Generic type for items handled by the BaseHandler
TItem = TypeVar("TItem")


class BaseHandler(ABC, Generic[TItem]):
    """Flexible base handler interface for all store operations
    
    Supports both sync and async implementations with flexible signatures.
    Implementations can choose appropriate return types and parameter patterns.
    """

    @abstractmethod
    def add(self, item: TItem, **kwargs) -> Union[str, TItem, None]:
        """Add an item to the store
        
        Args:
            item: Item to add (format depends on implementation)
            **kwargs: Additional options (item_id, properties, memory_type, etc.)
            
        Returns:
            Item ID or created object, or None on failure
        """
        pass

    @abstractmethod
    def get(self, item_id: str, **kwargs) -> Optional[TItem]:
        """Get an item from the store
        
        Args:
            item_id: ID of item to retrieve
            **kwargs: Additional options
            
        Returns:
            Item if found, None otherwise
        """
        pass

    @abstractmethod
    def update(self, item: TItem, **kwargs) -> Union[bool, TItem]:
        """Update an item in the store
        
        Args:
            item: Item to update (must include identifier) OR item_id as separate param
            **kwargs: Additional options (properties, write_mode, etc.)
            
        Returns:
            Success status or updated object
        """
        pass

    @abstractmethod
    def delete(self, item_id: str, **kwargs) -> bool:
        """Delete an item from the store
        
        Args:
            item_id: ID of item to delete
            **kwargs: Additional options
            
        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    def search(self, query: Any, **kwargs) -> List[TItem]:
        """Search for items in the store
        
        Args:
            query: Search query (format depends on implementation)
            **kwargs: Additional search options (top_k, filters, memory_type, etc.)
            
        Returns:
            List of matching items
        """
        pass

    def clear(self, **kwargs) -> bool:
        """Clear all items from the store (optional)
        
        Args:
            **kwargs: Additional options
            
        Returns:
            True if successful, False otherwise
        """
        raise NotImplementedError("clear() not implemented")


class BaseHandlerImpl(BaseHandler[TItem], ABC):
    """Base handler implementation with built-in multi-store progressive fallback support.
    
    Supports both single store and multi-store configurations:
    - Single store: Works exactly as before
    - Multiple stores: Automatically tries stores in sequence until one succeeds
    
    All operations are synchronous for simplicity and consistency.
    """

    def __init__(self, stores: Optional[Union['BaseHandler[Any]', List['BaseHandler[Any]']]] = None):
        """Initialize with single store, multiple stores, or config-based loading.
        
        Args:
            stores: Single store, list of stores, or None for subclass-specific loading
        """
        if stores is None:
            self.stores = self._load_stores_from_config()
        elif isinstance(stores, list):
            self.stores = stores
        else:
            self.stores = [stores]

    def _load_stores_from_config(self) -> List['BaseHandlerImpl[Any]']:
        """Load stores from configuration. Override in subclasses."""
        raise NotImplementedError(
            "Subclass must implement _load_stores_from_config() or provide explicit stores"
        )

    def _try_stores(self, operation_name: str, operation_func, *args, **kwargs):
        """Try operation on stores in sequence until one succeeds."""
        for i, store in enumerate(self.stores):
            try:
                result = operation_func(store, *args, **kwargs)
                if result is not None and (not isinstance(result, (bool, list)) or result):
                    return result
            except Exception as e:
                logging.warning(f"Store {i} failed {operation_name}: {e}")
                if i == len(self.stores) - 1:  # Last store, re-raise
                    raise
                continue
        return None

    def _to_dict(self, item: TItem) -> dict:
        """Convert domain object to dict for store."""
        if hasattr(item, 'to_dict'):
            return item.to_dict()
        elif isinstance(item, dict):
            return item
        else:
            # Fallback for objects without to_dict()
            return item.__dict__ if hasattr(item, '__dict__') else {}

    def _from_dict(self, data: dict) -> TItem:
        """Convert dict from store back to domain object."""
        # Try to get the original type from the generic type parameter
        # This is a simplified approach - in practice, you might need type hints or other mechanisms
        if hasattr(self, '_domain_type') and hasattr(self._domain_type, 'from_dict'):
            return self._domain_type.from_dict(data)
        else:
            # Fallback - return the dict as-is
            return data

    # Sync methods with progressive fallback and automatic domain conversion
    def add(self, item: TItem, **kwargs) -> Union[str, TItem, None]:
        """Add with progressive fallback and automatic domain conversion."""
        # Convert domain object to dict for store
        item_dict = self._to_dict(item)
        result = self._try_stores('add', lambda store: store.add(item_dict, **kwargs))

        # Return based on what store returned
        if isinstance(result, str):
            return result  # Store returned item_id
        elif isinstance(result, dict):
            return self._from_dict(result)  # Convert back to domain type
        elif result:
            return item  # Store returned something truthy, return original item
        return None

    def get(self, item_id: str, **kwargs) -> Optional[TItem]:
        """Get with progressive fallback and automatic domain conversion."""
        result = self._try_stores('get', lambda store: store.get(item_id, **kwargs))
        if isinstance(result, dict):
            return self._from_dict(result)
        return result

    def update(self, item: TItem, **kwargs) -> Union[bool, TItem]:
        """Update with progressive fallback and automatic domain conversion."""
        # Convert domain object to dict for store
        item_dict = self._to_dict(item)
        result = self._try_stores('update', lambda store: store.update(item_dict, **kwargs))

        # Convert result back to domain type if needed
        if isinstance(result, dict):
            return self._from_dict(result)
        return result

    def delete(self, item_id: str, **kwargs) -> bool:
        """Delete with progressive fallback."""
        result = self._try_stores('delete', lambda store: store.delete(item_id, **kwargs))
        return bool(result)

    def search(self, query: Any, **kwargs) -> List[TItem]:
        """Search with progressive fallback and automatic domain conversion."""
        result = self._try_stores('search', lambda store: store.search(query, **kwargs))
        if not result:
            return []

        # Convert list of dicts back to domain objects
        converted_results = []
        for item in result:
            if isinstance(item, dict):
                converted_results.append(self._from_dict(item))
            else:
                converted_results.append(item)
        return converted_results

    def clear(self, **kwargs) -> bool:
        """Clear all stores."""
        success = True
        for i, store in enumerate(self.stores):
            try:
                result = store.clear(**kwargs)
                if not result:
                    success = False
            except Exception as e:
                logging.warning(f"Store {i} failed to clear: {e}")
                success = False
        return success
