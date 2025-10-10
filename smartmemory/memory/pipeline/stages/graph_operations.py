"""
Graph operations component for SmartMemory.
Provides proper abstraction layer for low-level graph operations.
"""
import logging
from typing import Optional, List, Any

logger = logging.getLogger(__name__)


class GraphOperations:
    """
    Provides abstraction layer for graph operations.
    Eliminates direct graph access from SmartMemory.
    """

    def __init__(self, graph):
        """
        Initialize with graph backend.
        
        Args:
            graph: The SmartGraph instance to operate on
        """
        self._graph = graph

    def clear_all(self) -> bool:
        """
        Clear all data from the graph backend.
        Provides proper abstraction for graph clearing.
        """
        try:
            if hasattr(self._graph, "clear"):
                self._graph.clear()
                logger.info("Successfully cleared all graph data")
                return True
            else:
                logger.warning("Graph backend does not support clear operation")
                return False
        except Exception as e:
            logger.error(f"Failed to clear graph data: {e}")
            return False

    def get_node_count(self) -> int:
        """Get total number of nodes in the graph."""
        try:
            if hasattr(self._graph, "nodes"):
                return len(self._graph.nodes())
            return 0
        except Exception as e:
            logger.error(f"Failed to get node count: {e}")
            return 0

    def get_edge_count(self) -> int:
        """Get total number of edges in the graph."""
        try:
            if hasattr(self._graph, "edges"):
                return len(self._graph.edges())
            return 0
        except Exception as e:
            logger.error(f"Failed to get edge count: {e}")
            return 0

    def get_neighbors(self, node_id: str, edge_type: Optional[str] = None) -> List[Any]:
        """
        Get neighboring nodes for a given node.
        
        Args:
            node_id: ID of the node to get neighbors for
            edge_type: Optional edge type filter
            
        Returns:
            List of neighboring nodes
        """
        try:
            if hasattr(self._graph, "get_neighbors"):
                return self._graph.get_neighbors(node_id, edge_type=edge_type)
            return []
        except Exception as e:
            logger.error(f"Failed to get neighbors for {node_id}: {e}")
            return []

    def node_exists(self, node_id: str) -> bool:
        """Check if a node exists in the graph."""
        try:
            node = self._graph.get_node(node_id)
            return node is not None
        except Exception as e:
            logger.error(f"Failed to check if node {node_id} exists: {e}")
            return False

    def get_graph_stats(self) -> dict:
        """Get comprehensive graph statistics."""
        return {
            'node_count': self.get_node_count(),
            'edge_count': self.get_edge_count(),
            'backend_type': type(self._graph).__name__
        }
