import json
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from smartmemory.graph.smartgraph import SmartGraph
from smartmemory.models.memory_item import MemoryItem

logger = logging.getLogger(__name__)


class BaseMemoryGraph(ABC):
    """
    Abstract base class for all memory stores.
    Defines the required interface for consistency and future-proofing.
    Includes shared edge property serialization to eliminate code duplication.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        # Initialize with MemoryItem class for consistency
        from smartmemory.models.memory_item import MemoryItem
        self.graph = SmartGraph(item_cls=MemoryItem)

    def _serialize_edge_properties(self, properties: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Serialize edge properties to avoid unhashable dict errors.
        
        Converts nested dicts and lists to JSON strings to ensure all edge properties
        are hashable and can be stored in graph backends.
        
        Args:
            properties: Dictionary of edge properties to serialize
            
        Returns:
            Dictionary with serialized properties
        """
        if not properties:
            return {}

        serialized = {}
        for key, value in properties.items():
            if isinstance(value, dict):
                # Serialize nested dicts as JSON strings
                serialized[key] = json.dumps(value, sort_keys=True)
            elif isinstance(value, (list, tuple)):
                # Serialize lists/tuples as JSON strings
                serialized[key] = json.dumps(value, sort_keys=True)
            else:
                # Keep primitive types as-is
                serialized[key] = value

        return serialized

    def _add_edge_safe(self, source_id: str, target_id: str, edge_type: str,
                       properties: Optional[Dict[str, Any]] = None,
                       memory_type: Optional[str] = None,
                       valid_time: Optional[tuple] = None,
                       transaction_time: Optional[tuple] = None) -> None:
        """
        Add an edge with automatic property serialization.
        
        NOTE: Method name is misleading - this should fail fast, not silently handle errors.
        Edge creation failures are critical and should propagate to callers.
        
        Args:
            source_id: Source node ID
            target_id: Target node ID  
            edge_type: Type of edge relationship
            properties: Edge properties to serialize
            memory_type: Type of memory (semantic, episodic, etc.)
            valid_time: Valid time tuple for temporal graphs
            transaction_time: Transaction time tuple for temporal graphs
            
        Raises:
            Exception: If edge creation fails
        """
        serialized_props = self._serialize_edge_properties(properties)
        self.graph.add_edge(
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            properties=serialized_props,
            memory_type=memory_type,
            valid_time=valid_time,
            transaction_time=transaction_time
        )

    @abstractmethod
    def add(self, item: MemoryItem, **kwargs):
        """Add an item to the memory store."""
        pass

    @abstractmethod
    def search(self, query: str, **kwargs) -> List[MemoryItem]:
        """Search for items relevant to the query. May use keyword or semantic search."""
        pass

    def get(self, key: str):
        """Retrieve an item by unique key with consistent error handling."""
        if not key:
            logger.error(f"Cannot get from {self.__class__.__name__}: key is None or empty")
            return None

        try:
            node = self.graph.get_node(key)
            if node:
                # Convert node dict back to MemoryItem if possible
                if hasattr(self.graph, '_from_node_dict'):
                    from smartmemory.models.memory_item import MemoryItem
                    return self.graph._from_node_dict(MemoryItem, node)
                return node
            return None
        except Exception as e:
            logger.error(f"Failed to retrieve item {key} from {self.__class__.__name__}: {e}")
            return None

    def remove(self, key: str) -> bool:
        """Remove an item by key with consistent error handling."""
        if not key:
            logger.error(f"Cannot remove from {self.__class__.__name__}: key is None or empty")
            return False

        try:
            self.graph.remove_node(key)
            logger.info(f"Removed item {key} from {self.__class__.__name__}")
            return True
        except Exception as e:
            logger.error(f"Failed to remove item {key} from {self.__class__.__name__}: {e}")
            return False

    def clear(self):
        """Clear the memory store. If self.graph exists, delegate to self.graph.clear(). Otherwise, raise NotImplementedError."""
        if hasattr(self, "graph") and self.graph is not None and hasattr(self.graph, "clear"):
            self.graph.clear()
        else:
            raise NotImplementedError("Clear not implemented for this memory store and no graph backend found.")
