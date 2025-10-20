import importlib
from typing import Any, Dict, List, Optional, Tuple

from smartmemory.graph.core.edges import SmartGraphEdges
from smartmemory.graph.core.nodes import SmartGraphNodes
from smartmemory.graph.core.search import SmartGraphSearch
from smartmemory.models.memory_item import MemoryItem
from smartmemory.utils import get_config


class SmartGraph:
    """
    Unified graph API for agentic memory. Converts backend-native dicts to the canonical models (item_cls) provided at construction.
    Backend is selected based on the 'backend_class' key in config.json['graph_db'] (default: FalkorDBBackend).
    """

    def __init__(self, item_cls=None, enable_caching=True, cache_size=1000):
        if item_cls is None:
            item_cls = MemoryItem
        self.item_cls = item_cls
        backend_cls = self._get_backend_class()
        self.backend = backend_cls()

        # Initialize submodules
        self.nodes = SmartGraphNodes(self.backend, item_cls, enable_caching, cache_size)
        self.edges = SmartGraphEdges(self.backend, self.nodes, enable_caching, cache_size)
        self.search = SmartGraphSearch(self.backend, self.nodes, enable_caching, cache_size)

        # Performance caching (for backward compatibility)
        self.enable_caching = enable_caching
        self.cache_size = cache_size

    @staticmethod
    def _get_backend_class():
        """
        Resolve the backend class from config. Import directly from backends module.
        """
        graph_cfg = get_config('graph_db')
        backend_class_name = graph_cfg.get('backend_class', 'FalkorDBBackend')

        # Map backend class names to their module paths
        backend_modules = {
            'FalkorDBBackend': 'smartmemory.graph.backends.falkordb',
            'Neo4jBackend': 'smartmemory.graph.backends.neo4j',
            # Add other backends as needed
        }

        module_path = backend_modules.get(backend_class_name)
        if not module_path:
            raise ImportError(f"Unknown backend class '{backend_class_name}'. Available: {list(backend_modules.keys())}")

        try:
            module = importlib.import_module(module_path)
            return getattr(module, backend_class_name)
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Failed to import backend class '{backend_class_name}' from '{module_path}': {e}")

    def _emit_graph_stats(self, operation: str, details: Dict[str, Any], delta_nodes: Optional[int] = None, delta_edges: Optional[int] = None) -> None:
        """Best-effort emission of graph stats update events."""
        # Delegate to nodes module for consistency
        self.nodes._emit_graph_stats(operation, details, delta_nodes, delta_edges)

    def clear(self):
        """Remove all nodes and edges from the graph."""
        # Capture pre-clear counts if possible
        pre_nodes: Optional[int] = None
        pre_edges: Optional[int] = None
        try:
            if hasattr(self.backend, 'get_counts'):
                counts = self.backend.get_counts()  # type: ignore[attr-defined]
                if isinstance(counts, dict):
                    pre_nodes = counts.get('node_count')
                    pre_edges = counts.get('edge_count')
            else:
                if hasattr(self.backend, 'get_node_count'):
                    pre_nodes = self.backend.get_node_count()  # type: ignore[attr-defined]
                elif hasattr(self.backend, 'get_all_nodes'):
                    nodes = self.backend.get_all_nodes()  # type: ignore[attr-defined]
                    pre_nodes = len(nodes) if nodes is not None else None
                if hasattr(self.backend, 'get_edge_count'):
                    pre_edges = self.backend.get_edge_count()  # type: ignore[attr-defined]
        except Exception:
            pass

        result = self.backend.clear()

        # Clear all submodule caches
        self.nodes.clear_cache()
        self.edges.clear_cache() if hasattr(self.edges, 'clear_cache') else None
        self.search.clear_cache()

        try:
            self._emit_graph_stats(
                "clear",
                details={"pre_counts": {"node_count": pre_nodes, "edge_count": pre_edges}},
                delta_nodes=(-pre_nodes if isinstance(pre_nodes, int) else None),
                delta_edges=(-pre_edges if isinstance(pre_edges, int) else None),
            )
        except Exception:
            pass
        return result

    def add_node(self,
                 item_id: Optional[str],
                 properties: Dict[str, Any],
                 valid_time: Optional[Tuple] = None,
                 transaction_time: Optional[Tuple] = None,
                 memory_type: Optional[str] = None,
                 is_global: bool = False):
        """Add a node to the graph."""
        return self.nodes.add_node(item_id, properties, valid_time, transaction_time, memory_type, is_global)

    def add_dual_node(self, item_id: str, memory_properties: Dict[str, Any], memory_type: str, entity_nodes: List[Dict[str, Any]] = None, is_global: bool = False):
        """Add a dual-node structure through the backend."""
        return self.nodes.add_dual_node(item_id, memory_properties, memory_type, entity_nodes, is_global)

    @staticmethod
    def _to_node_dict(obj):
        """Convert object to node dictionary."""
        return SmartGraphNodes._to_node_dict(obj)

    @staticmethod
    def _from_node_dict(item_cls, node):
        """Convert node dictionary to object."""
        return SmartGraphNodes._from_node_dict(item_cls, node)

    def _validate_edge_structural(self,
                                  source_id: str,
                                  target_id: str,
                                  edge_type: str,
                                  properties: Optional[Dict[str, Any]]) -> bool:
        """Lightweight structural validation for edges."""
        return self.edges._validate_edge_structural(source_id, target_id, edge_type, properties)

    def add_edge(self,
                 source_id: str,
                 target_id: str,
                 edge_type: str,
                 properties: Dict[str, Any],
                 valid_time: Optional[Tuple] = None,
                 transaction_time: Optional[Tuple] = None,
                 memory_type: Optional[str] = None):
        """Add an edge to the graph."""
        return self.edges.add_edge(source_id, target_id, edge_type, properties, valid_time, transaction_time, memory_type)

    def add_triple(self, triple: 'Triple', properties: Dict[str, Any] = None, **kwargs):
        """Add a triple (Triple models) to the graph."""
        return self.edges.add_triple(triple, properties, **kwargs)

    def get_node(self, item_id: str, as_of_time: Optional[str] = None):
        """Get a node by ID."""
        return self.nodes.get_node(item_id, as_of_time)

    def get_neighbors(self, item_id: str, edge_type: Optional[str] = None, as_of_time: Optional[str] = None):
        """Get neighbors of a node."""
        return self.nodes.get_neighbors(item_id, edge_type, as_of_time)

    def get_edges_for_node(self, item_id: str):
        """Get all edges (relationships) involving a specific node."""
        return self.edges.get_edges_for_node(item_id)

    def search_nodes(self, query: Dict[str, Any]):
        """Search for nodes matching the query dict."""
        return self.search.search_nodes(query)

    def search(self, query_str: str, top_k: int = 5, **kwargs):
        """Enhanced search method using vector embeddings as primary method with text-based fallbacks."""
        return self.search.search(query_str, top_k, **kwargs)

    def _search_with_vector_embeddings(self, query_str: str, top_k: int = 5, **kwargs):
        """Primary search method using vector embeddings for semantic similarity."""
        return self.search._search_with_vector_embeddings(query_str, top_k, **kwargs)

    def _search_with_regex(self, query_str: str, top_k: int = 5, **kwargs):
        """Primary search method using FalkorDB/Cypher-compatible text search."""
        return self.search._search_with_regex(query_str, top_k, **kwargs)

    def _search_text_falkordb(self, query_str: str):
        """FalkorDB-compatible text search using CONTAINS operator with multi-word support."""
        return self.search._search_text_falkordb(query_str)

    def _search_with_simple_contains(self, query_str: str, top_k: int = 5, **kwargs):
        """Fallback search using simple contains logic."""
        return self.search._search_with_simple_contains(query_str, top_k, **kwargs)

    def _search_with_keyword_matching(self, query_str: str, top_k: int = 5, **kwargs):
        """Fallback search using keyword matching."""
        return self.search._search_with_keyword_matching(query_str, top_k, **kwargs)

    def _get_all_nodes_fallback(self, query_str: str, top_k: int = 5, **kwargs):
        """Final fallback - just return all available nodes."""
        return self.search._get_all_nodes_fallback(query_str, top_k, **kwargs)

    def get_all_node_ids(self):
        """Return all node IDs for compatibility with memory store iteration."""
        return self.nodes.nodes()

    def nodes(self):
        """Return all node IDs for compatibility with memory store iteration."""
        return self.nodes.nodes()

    def get_all_nodes(self):
        """Get all nodes in the graph (full node data)."""
        if hasattr(self.backend, 'get_all_nodes'):
            return self.backend.get_all_nodes()
        else:
            # Fallback: get all node IDs and fetch each node
            node_ids = self.nodes.nodes()
            nodes = []
            for node_id in node_ids:
                node = self.get_node(node_id)
                if node:
                    nodes.append(node)
            return nodes

    def get_incoming_neighbors(self, item_id: str, edge_type: Optional[str] = None):
        """Get incoming neighbors (nodes that link TO this node)."""
        edges = self.get_edges_for_node(item_id)
        incoming_neighbors = []
        for edge in edges:
            # If this node is the target, add the source as incoming neighbor
            if edge.get('target') == item_id:
                if edge_type is None or edge.get('type') == edge_type:
                    source_node = self.get_node(edge.get('source'))
                    if source_node:
                        incoming_neighbors.append(source_node)
        return incoming_neighbors

    def remove_node(self, item_id: str):
        """Remove a node from the graph."""
        return self.nodes.remove_node(item_id)

    def remove_edge(self, source_id: str, target_id: str, edge_type: Optional[str] = None):
        """Remove an edge from the graph."""
        return self.edges.remove_edge(source_id, target_id, edge_type)

    def serialize(self) -> Any:
        return self.backend.serialize()

    def deserialize(self, data: Any):
        return self.backend.deserialize(data)

    def _manage_cache_size(self):
        """Manage cache size by removing oldest entries when cache is full."""
        # Delegate to submodules
        self.nodes._manage_cache_size()
        self.search._manage_cache_size()

    def clear_cache(self):
        """Clear all caches."""
        self.nodes.clear_cache()
        self.search.clear_cache()

    def get_all_edges(self):
        """Get all edges in the graph."""
        if hasattr(self.backend, 'get_all_edges'):
            return self.backend.get_all_edges()
        else:
            # Fallback: return empty list if backend doesn't support it
            return []

    def get_cache_stats(self):
        """Get cache performance statistics."""
        if not self.enable_caching:
            return {"caching_enabled": False}

        # Combine stats from all submodules
        node_stats = self.nodes.get_cache_stats()
        search_stats = self.search.get_cache_stats()

        return {
            "caching_enabled": True,
            "nodes": node_stats,
            "search": search_stats,
            "max_cache_size": self.cache_size
        }
