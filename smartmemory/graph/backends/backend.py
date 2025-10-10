from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple


class SmartGraphBackend(ABC):
    @abstractmethod
    def add_node(self, item_id: str, properties: Dict[str, Any], valid_time: Optional[Tuple] = None, transaction_time: Optional[Tuple] = None, memory_type: Optional[str] = None):
        """Add a node with properties, bi-temporal info, and memory type."""
        pass

    @abstractmethod
    def clear(self):
        """
        Remove all nodes and edges from the graph.
        """
        pass

    @abstractmethod
    def add_edge(self,
                 source_id: str,
                 target_id: str,
                 edge_type: str,
                 properties: Dict[str, Any],
                 valid_time: Optional[Tuple] = None,
                 transaction_time: Optional[Tuple] = None,
                 memory_type: Optional[str] = None):
        """Add an edge with properties, bi-temporal info, and memory type."""
        pass

    @abstractmethod
    def get_node(self, item_id: str, as_of_time: Optional[str] = None) -> Dict[str, Any]:
        """Get a node by ID, optionally as of a specific time."""
        pass

    @abstractmethod
    def get_neighbors(self, item_id: str, edge_type: Optional[str] = None, as_of_time: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get neighboring nodes, optionally filtered by edge type and time."""
        pass

    @abstractmethod
    def remove_node(self, item_id: str):
        """Remove a node by ID."""
        pass

    @abstractmethod
    def remove_edge(self, source_id: str, target_id: str, edge_type: Optional[str] = None):
        """Remove an edge by source, target, and optionally edge type."""
        pass

    @abstractmethod
    def search_nodes(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find nodes matching query properties (type, time, etc)."""
        pass

    @abstractmethod
    def serialize(self) -> Any:
        """Serialize the graph (for export, backup, or test snapshot)."""
        pass

    @abstractmethod
    def deserialize(self, data: Any):
        """Load the graph from a serialized format."""
        pass
