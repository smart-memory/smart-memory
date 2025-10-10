"""
Memory graph interfaces for hybrid architecture.
Provides minimal common interface while preserving per-type flexibility.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from smartmemory.models.memory_item import MemoryItem


@dataclass
class GraphRelation:
    """Represents a relation/edge in the graph."""
    source_id: str
    target_id: str
    relation_type: str
    properties: Dict[str, Any]


@dataclass
class GraphData:
    """Structured data for graph storage."""
    node_id: str
    node_properties: Dict[str, Any]
    relations: List[GraphRelation]


class MemoryGraphInterface(ABC):
    """
    Minimal common interface for all memory graph types.
    
    Defines only essential operations that all graph types must support.
    Type-specific features should be implemented as additional methods.
    """

    @abstractmethod
    def add(self, item: MemoryItem, **kwargs) -> str:
        """
        Add a memory item to the graph.
        
        Args:
            item: MemoryItem to add
            **kwargs: Type-specific parameters
            
        Returns:
            str: Item ID of added item
        """
        pass

    @abstractmethod
    def get(self, key: str) -> Optional[MemoryItem]:
        """
        Retrieve a memory item by key.
        
        Args:
            key: Item ID to retrieve
            
        Returns:
            MemoryItem if found, None otherwise
        """
        pass

    @abstractmethod
    def remove(self, key: str) -> bool:
        """
        Remove a memory item by key.
        
        Args:
            key: Item ID to remove
            
        Returns:
            bool: True if removed, False if not found
        """
        pass

    @abstractmethod
    def search(self, query: str, **kwargs) -> List[MemoryItem]:
        """
        Search for memory items.
        
        Args:
            query: Search query
            **kwargs: Type-specific search parameters
            
        Returns:
            List of matching MemoryItems
        """
        pass

    @abstractmethod
    def clear(self) -> bool:
        """
        Clear all items from the graph.
        
        Returns:
            bool: True if successful
        """
        pass

    def get_stats(self) -> Dict[str, Any]:
        """
        Get graph statistics. Optional implementation.
        
        Returns:
            Dict with graph statistics
        """
        return {
            'type': self.__class__.__name__,
            'node_count': 0,
            'edge_count': 0
        }


class MemoryItemConverter(ABC):
    """
    Abstract base for type-specific MemoryItem converters.
    
    Each graph type can implement its own conversion logic
    while maintaining a consistent interface.
    """

    @abstractmethod
    def to_graph_data(self, item: MemoryItem) -> 'GraphData':
        """
        Convert MemoryItem to graph-specific format.
        
        Args:
            item: MemoryItem to convert
            
        Returns:
            Dict in graph-specific format
        """
        pass

    @abstractmethod
    def from_graph_format(self, data: Dict[str, Any]) -> MemoryItem:
        """
        Convert graph-specific format to MemoryItem.
        
        Args:
            data: Graph-specific data dict
            
        Returns:
            MemoryItem instance
        """
        pass

    def validate_item(self, item: MemoryItem) -> bool:
        """
        Validate MemoryItem for this graph type.
        Default implementation checks basic requirements.
        
        Args:
            item: MemoryItem to validate
            
        Returns:
            bool: True if valid
        """
        if not item or not hasattr(item, 'item_id') or not item.item_id:
            return False
        if not hasattr(item, 'content'):
            return False
        return True
