"""
Optional mixins for memory graph implementations.
Provides reusable functionality that graph types can choose to use.
"""

import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any

from smartmemory.models.memory_item import MemoryItem

logger = logging.getLogger(__name__)


class GraphErrorHandlingMixin:
    """
    Mixin providing consistent error handling for graph operations.
    Optional - graph types can use their own error handling if needed.
    """

    def _handle_operation_error(self, operation: str, key: str, error: Exception) -> None:
        """
        Standard error handling for graph operations.
        
        Args:
            operation: Name of the operation (add, get, remove, etc.)
            key: Item key involved in the operation
            error: Exception that occurred
        """
        graph_type = getattr(self, '__class__', type(self)).__name__
        logger.error(f"{graph_type}.{operation} failed for key {key}: {error}")

        # Add error to item metadata if available
        if hasattr(self, '_last_error'):
            self._last_error = {
                'operation': operation,
                'key': key,
                'error': str(error),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

    def handle_error(self, message: str) -> None:
        """
        Handle error with logging and return None for failed operations.
        
        Args:
            message: Error message to log
            
        Returns:
            None: Always returns None to indicate failure
        """
        graph_type = getattr(self, '__class__', type(self)).__name__
        logger.error(f"{graph_type}: {message}")
        return None

    def _validate_key(self, key: str, operation: str) -> bool:
        """
        Validate key for graph operations.
        
        Args:
            key: Key to validate
            operation: Operation name for error reporting
            
        Returns:
            bool: True if valid
        """
        if not key or not isinstance(key, str) or not key.strip():
            self._handle_operation_error(operation, str(key), ValueError("Invalid key"))
            return False
        return True

    def _validate_item(self, item: MemoryItem, operation: str) -> bool:
        """
        Validate MemoryItem for graph operations.
        
        Args:
            item: Item to validate
            operation: Operation name for error reporting
            
        Returns:
            bool: True if valid
        """
        if not item:
            self._handle_operation_error(operation, "None", ValueError("Item is None"))
            return False

        if not hasattr(item, 'item_id') or not item.item_id:
            self._handle_operation_error(operation, "unknown", ValueError("Item missing item_id"))
            return False

        return True


class GraphLoggingMixin:
    """
    Mixin providing consistent logging for graph operations.
    Optional - graph types can implement their own logging.
    """

    def log_operation(self, operation: str, message: str, **kwargs) -> None:
        """
        Log graph operation with structured message.
        
        Args:
            operation: Operation type (add, get, remove, etc.)
            message: Log message
            **kwargs: Additional context
        """
        graph_type = getattr(self, '__class__', type(self)).__name__
        logger.info(f"{graph_type}.{operation}: {message}")

        # Log additional context if provided
        if kwargs:
            logger.debug(f"{graph_type}.{operation} context: {kwargs}")

    def _log_operation(self, operation: str, key: str, success: bool = True, **kwargs) -> None:
        """
        Log graph operation with consistent format.
        
        Args:
            operation: Operation name
            key: Item key
            success: Whether operation succeeded
            **kwargs: Additional context
        """
        graph_type = getattr(self, '__class__', type(self)).__name__

        if success:
            logger.info(f"{graph_type}.{operation} succeeded for key {key}")
        else:
            logger.warning(f"{graph_type}.{operation} failed for key {key}")

        # Log additional context if provided
        if kwargs:
            logger.debug(f"{graph_type}.{operation} context: {kwargs}")

    def _log_stats(self, stats: Dict[str, Any]) -> None:
        """
        Log graph statistics.
        
        Args:
            stats: Statistics dictionary
        """
        graph_type = getattr(self, '__class__', type(self)).__name__
        logger.info(f"{graph_type} stats: {stats}")


class GraphValidationMixin:
    """
    Mixin providing validation utilities for graph operations.
    Optional - graph types can implement custom validation.
    """

    def validate_item(self, item) -> bool:
        """
        Validate item for graph operations.
        
        Args:
            item: Item to validate (MemoryItem or dict)
            
        Returns:
            bool: True if valid
        """
        if not item:
            return False

        # Handle MemoryItem validation
        if hasattr(item, 'item_id'):
            return bool(item.item_id and hasattr(item, 'content'))

        # Handle dict validation
        if isinstance(item, dict):
            return bool(item.get('item_id') or item.get('id'))

        return True  # Allow other types to pass through

    def _validate_memory_item_structure(self, item: MemoryItem) -> bool:
        """
        Validate basic MemoryItem structure.
        
        Args:
            item: MemoryItem to validate
            
        Returns:
            bool: True if structure is valid
        """
        required_fields = ['item_id', 'content', 'type']

        for field in required_fields:
            if not hasattr(item, field):
                logger.warning(f"MemoryItem missing required field: {field}")
                return False

            value = getattr(item, field)
            if value is None or (isinstance(value, str) and not value.strip()):
                logger.warning(f"MemoryItem has empty required field: {field}")
                return False

        return True

    def _validate_metadata(self, metadata: Dict[str, Any]) -> bool:
        """
        Validate metadata structure.
        
        Args:
            metadata: Metadata dictionary
            
        Returns:
            bool: True if valid
        """
        if not isinstance(metadata, dict):
            logger.warning("Metadata must be a dictionary")
            return False

        # Check for reserved keys that might cause issues
        reserved_keys = ['_internal', '_system', '_graph']
        for key in metadata:
            if key.startswith('_') and key in reserved_keys:
                logger.warning(f"Metadata contains reserved key: {key}")
                return False

        return True


class GraphPerformanceMixin:
    """
    Mixin providing performance monitoring for graph operations.
    Optional - graph types can implement their own monitoring.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._operation_stats = {
            'add_count': 0,
            'get_count': 0,
            'remove_count': 0,
            'search_count': 0,
            'total_operations': 0
        }

    def performance_context(self, operation: str):
        """
        Context manager for performance monitoring.
        
        Args:
            operation: Operation name to monitor
            
        Returns:
            Context manager that tracks operation timing
        """
        from contextlib import contextmanager
        import time

        @contextmanager
        def _context():
            start_time = time.time()
            try:
                yield
            finally:
                duration = time.time() - start_time
                self._track_operation(operation)
                logger.debug(f"Operation {operation} took {duration:.3f}s")

        return _context()

    def _track_operation(self, operation: str) -> None:
        """
        Track operation for performance monitoring.
        
        Args:
            operation: Operation name
        """
        if hasattr(self, '_operation_stats'):
            key = f"{operation}_count"
            if key in self._operation_stats:
                self._operation_stats[key] += 1
            self._operation_stats['total_operations'] += 1

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics.
        
        Returns:
            Dict with performance stats
        """
        if hasattr(self, '_operation_stats'):
            return self._operation_stats.copy()
        return {}

    def reset_performance_stats(self) -> None:
        """Reset performance statistics."""
        if hasattr(self, '_operation_stats'):
            for key in self._operation_stats:
                self._operation_stats[key] = 0


class GraphCachingMixin:
    """
    Mixin providing simple caching for graph operations.
    Optional - graph types can implement more sophisticated caching.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache = {}
        self._cache_enabled = kwargs.get('enable_cache', True)
        self._cache_max_size = kwargs.get('cache_max_size', 1000)

    def _cache_get(self, key: str) -> Optional[MemoryItem]:
        """
        Get item from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached MemoryItem or None
        """
        if not self._cache_enabled or not hasattr(self, '_cache'):
            return None
        return self._cache.get(key)

    def _cache_put(self, key: str, item: MemoryItem) -> None:
        """
        Put item in cache.
        
        Args:
            key: Cache key
            item: MemoryItem to cache
        """
        if not self._cache_enabled or not hasattr(self, '_cache'):
            return

        # Simple LRU: remove oldest if at max size
        if len(self._cache) >= self._cache_max_size:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

        self._cache[key] = item

    def _cache_remove(self, key: str) -> None:
        """
        Remove item from cache.
        
        Args:
            key: Cache key to remove
        """
        if hasattr(self, '_cache') and key in self._cache:
            del self._cache[key]

    def _cache_clear(self) -> None:
        """Clear all cached items."""
        if hasattr(self, '_cache'):
            self._cache.clear()
