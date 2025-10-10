"""
Mixins providing common functionality for memory classes to eliminate code duplication.
"""

import logging
from abc import abstractmethod
from datetime import timezone
from typing import Optional, List, Union, Any

from smartmemory.models.memory_item import MemoryItem
from smartmemory.stores.base import BaseHandler

logger = logging.getLogger(__name__)


class CRUDMixin(BaseHandler[MemoryItem]):
    """Mixin providing standard CRUD operations for all memory types."""

    def add(self, item: Union[MemoryItem, Any], **kwargs) -> Union[str, MemoryItem, None]:
        """
        Unified add operation with consistent error handling and logging.
        Implements BaseHandler.add() with memory-specific enhancements.
        
        Args:
            item: MemoryItem or convertible object
            **kwargs: Additional options
            
        Returns:
            MemoryItem for memory operations, or item_id string for compatibility
        """
        # Convert to MemoryItem if needed
        if not isinstance(item, MemoryItem):
            if hasattr(item, 'to_memory_item'):
                item = item.to_memory_item()
            elif isinstance(item, dict):
                item = MemoryItem(**item)
            elif isinstance(item, str):
                item = MemoryItem(content=item)
            else:
                logger.error(f"Cannot convert {type(item)} to MemoryItem")
                return None

        if not item:
            logger.error("Cannot add: item is None")
            return None

        try:
            result = self._add_impl(item, **kwargs)
            if result:
                logger.info(f"Added {self._get_memory_type()} item: {item.item_id}")
            return result
        except Exception as e:
            logger.error(f"Failed to add {self._get_memory_type()} item {item.item_id}: {e}")
            return None

    def get(self, item_id: str, **kwargs) -> Optional[MemoryItem]:
        """
        Unified get operation with consistent error handling.
        Implements BaseHandler.get() with memory-specific enhancements.
        """
        if not item_id:
            logger.error("Cannot get: item_id is None or empty")
            return None

        try:
            return self._get_impl(item_id)
        except Exception as e:
            logger.error(f"Failed to get {self._get_memory_type()} item {item_id}: {e}")
            return None

    def update(self, item: Union[MemoryItem, Any], **kwargs) -> Union[bool, MemoryItem]:
        """
        Unified update operation with consistent error handling and logging.
        Implements BaseHandler.update() with memory-specific enhancements.
        
        Args:
            item: MemoryItem to update (must include item_id)
            **kwargs: Additional options
            
        Returns:
            Updated MemoryItem or success boolean
        """
        # Convert to MemoryItem if needed
        if not isinstance(item, MemoryItem):
            if hasattr(item, 'to_memory_item'):
                item = item.to_memory_item()
            elif isinstance(item, dict):
                item = MemoryItem(**item)
            else:
                logger.error(f"Cannot convert {type(item)} to MemoryItem for update")
                return False

        if not item or not hasattr(item, 'item_id') or not item.item_id:
            logger.error("Cannot update: item missing item_id")
            return False

        try:
            result = self._update_impl(item, **kwargs)
            if result:
                logger.info(f"Updated {self._get_memory_type()} item: {item.item_id}")
            return result
        except Exception as e:
            logger.error(f"Failed to update {self._get_memory_type()} item {item.item_id}: {e}")
            return False

    def delete(self, item_id: str, **kwargs) -> bool:
        """
        Unified delete operation with consistent error handling and logging.
        Implements BaseHandler.delete() with memory-specific enhancements.
        """
        if not item_id:
            logger.error("Cannot delete: item_id is None or empty")
            return False

        try:
            result = self._remove_impl(item_id)
            if result:
                logger.info(f"Deleted {self._get_memory_type()} item: {item_id}")
            return result
        except Exception as e:
            logger.error(f"Failed to delete {self._get_memory_type()} item {item_id}: {e}")
            return False

    def clear(self, **kwargs) -> bool:
        """
        Unified clear operation with consistent error handling and logging.
        Implements BaseHandler.clear() with memory-specific enhancements.
        """
        try:
            self._clear_impl()
            logger.info(f"Cleared {self._get_memory_type()} memory")
            return True
        except Exception as e:
            logger.error(f"Failed to clear {self._get_memory_type()} memory: {e}")
            return False

    def search(self, query: Any, **kwargs) -> List[MemoryItem]:
        """
        Unified search operation with consistent error handling.
        Implements BaseHandler.search() with memory-specific enhancements.
        """
        if not query:
            logger.warning("Empty query provided to search")
            return []

        # Convert query to string if needed
        query_str = str(query) if not isinstance(query, str) else query

        try:
            return self._search_impl(query_str, **kwargs)
        except Exception as e:
            logger.error(f"Failed to search {self._get_memory_type()} memory: {e}")
            return []

    # Legacy aliases for backward compatibility
    def remove(self, key: str) -> bool:
        """Legacy alias for delete() - for backward compatibility."""
        return self.delete(key)

    # Abstract methods for subclasses to implement
    @abstractmethod
    def _add_impl(self, item: MemoryItem, **kwargs) -> Optional[MemoryItem]:
        """Implement type-specific add logic."""
        pass

    @abstractmethod
    def _get_impl(self, key: str) -> Optional[MemoryItem]:
        """Implement type-specific get logic."""
        pass

    @abstractmethod
    def _update_impl(self, item: MemoryItem, **kwargs) -> Union[bool, MemoryItem]:
        """Implement type-specific update logic."""
        pass

    def _remove_impl(self, key: str) -> bool:
        """
        Implement type-specific remove logic.
        
        Default implementation raises NotImplementedError.
        Override in subclasses that support removal.
        """
        raise NotImplementedError(f"Remove not implemented for {self._get_memory_type()}")

    def _clear_impl(self):
        """
        Implement type-specific clear logic.
        
        Default implementation delegates to store if available.
        """
        if hasattr(self, 'store') and self.store and hasattr(self.store, 'clear'):
            self.store.clear()
        else:
            raise NotImplementedError(f"Clear not implemented for {self._get_memory_type()}")

    def _search_impl(self, query: str, **kwargs) -> List[MemoryItem]:
        """
        Implement type-specific search logic.
        
        Default implementation delegates to store if available.
        """
        if hasattr(self, 'store') and self.store and hasattr(self.store, 'search'):
            return self.store.search(query, **kwargs)
        else:
            return []

    def _get_memory_type(self) -> str:
        """Get the memory type name for logging."""
        return getattr(self, '_memory_type', self.__class__.__name__.replace('Memory', '').lower())


class StoreBackedMixin:
    """Mixin for memory classes that use a store backend."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if hasattr(self, '_store_clazz') and self._store_clazz:
            self.store = self._store_clazz(*args, **kwargs)
        else:
            self.store = None

    def _ensure_store(self):
        """Ensure store is available, raise error if not."""
        if not self.store:
            raise RuntimeError(f"No store available for {self.__class__.__name__}")


class ArchivingMixin:
    """Mixin providing consistent archiving functionality."""

    def archive(self, item: MemoryItem, reason: str = "manual") -> bool:
        """
        Archive a memory item with consistent metadata.
        
        Archives rather than deletes for safety and auditability.
        """
        if not item:
            logger.error("Cannot archive: item is None")
            return False

        try:
            # Add archival metadata
            item.metadata['archived'] = True
            item.metadata['archive_reason'] = reason
            item.metadata['archive_timestamp'] = self._get_current_timestamp()

            # Update the item
            if hasattr(self, 'update') and callable(self.update):
                result = self.update(item)
            else:
                # Fallback: just mark as archived
                result = True

            if result:
                logger.info(f"Archived {self._get_memory_type()} item: {item.item_id}")

            return result
        except Exception as e:
            logger.error(f"Failed to archive {self._get_memory_type()} item {item.item_id}: {e}")
            return False

    def _get_current_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime
        return datetime.now(timezone.utc).isoformat()


class ValidationMixin:
    """Mixin providing input validation for memory operations."""

    def _validate_memory_item(self, item: MemoryItem) -> bool:
        """Validate that a MemoryItem is properly formed."""
        if not item:
            return False

        if not hasattr(item, 'item_id') or not item.item_id:
            logger.warning("MemoryItem missing item_id")
            return False

        if not hasattr(item, 'content'):
            logger.warning("MemoryItem missing content")
            return False

        if not hasattr(item, 'metadata') or not isinstance(item.metadata, dict):
            logger.warning("MemoryItem missing or invalid metadata")
            return False

        return True

    def _validate_key(self, key: str) -> bool:
        """Validate that a key is valid."""
        return bool(key and isinstance(key, str) and key.strip())


class ConfigurableMixin:
    """Mixin providing consistent configuration handling."""

    def __init__(self, config=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config or {}

    def get_config_value(self, key: str, default=None):
        """Get a configuration value with fallback to default."""
        return self.config.get(key, default)

    def set_config_value(self, key: str, value):
        """Set a configuration value."""
        self.config[key] = value


class MemoryMixin(CRUDMixin, StoreBackedMixin, ArchivingMixin, ValidationMixin, ConfigurableMixin):
    """
    Unified mixin combining all common memory functionality.
    
    Memory classes can inherit from this to get all standard behavior
    and only need to implement type-specific logic.
    """

    def add(self, item: MemoryItem, **kwargs) -> Union[str, MemoryItem, None]:
        """Add with validation."""
        if not self._validate_memory_item(item):
            return None
        return super().add(item, **kwargs)

    def get(self, item_id: str, **kwargs) -> Optional[MemoryItem]:
        """Get with validation."""
        if not self._validate_key(item_id):
            return None
        return super().get(item_id)

    def delete(self, item_id: str, **kwargs) -> bool:
        """Remove with validation."""
        if not self._validate_key(item_id):
            return False
        return super().delete(item_id)
