"""
Unified base classes that consolidate the patterns from BaseMemory and BaseMemoryGraph.
"""

from abc import ABC, abstractmethod
from typing import Optional, List

from smartmemory.memory.mixins import MemoryMixin
from smartmemory.models.memory_item import MemoryItem


class MemoryBase(MemoryMixin, ABC):
    """
    Unified base class consolidating BaseMemory and BaseMemoryGraph patterns.
    
    Provides all common functionality through mixins and requires only
    type-specific implementation methods.
    """

    def __init__(self, memory_type: str = None, store_class=None, config=None, *args, **kwargs):
        # Set memory type for logging and identification
        self._memory_type = memory_type or self.__class__.__name__.replace('Memory', '').lower()

        # Set store class for StoreBackedMixin
        if store_class:
            self._store_clazz = store_class

        # Initialize all mixins
        super().__init__(config=config, *args, **kwargs)

    # Abstract methods that subclasses must implement
    @abstractmethod
    def _add_impl(self, item: MemoryItem, **kwargs) -> Optional[MemoryItem]:
        """Implement type-specific add logic."""
        pass

    @abstractmethod
    def _get_impl(self, key: str) -> Optional[MemoryItem]:
        """Implement type-specific get logic."""
        pass

    # Optional methods that subclasses can override
    def _remove_impl(self, key: str) -> bool:
        """
        Implement type-specific remove logic.
        Default delegates to store if available.
        """
        if hasattr(self, 'store') and self.store and hasattr(self.store, 'remove'):
            return self.store.delete(key)
        raise NotImplementedError(f"Remove not implemented for {self._memory_type}")

    def _search_impl(self, query: str, **kwargs) -> List[MemoryItem]:
        """
        Implement type-specific search logic.
        Default delegates to store if available.
        """
        if hasattr(self, 'store') and self.store and hasattr(self.store, 'search'):
            return self.store.search(query, **kwargs)
        return []

    def _clear_impl(self):
        """
        Implement type-specific clear logic.
        Default delegates to store if available.
        """
        if hasattr(self, 'store') and self.store and hasattr(self.store, 'clear'):
            self.store.clear()
        else:
            raise NotImplementedError(f"Clear not implemented for {self._memory_type}")


class GraphBackedMemory(MemoryBase):
    """
    Base class for memory types that use a graph backend.
    
    Consolidates the common pattern from all graph-backed memory stores.
    """

    def __init__(self, memory_type: str = None, graph_class=None, config=None, *args, **kwargs):
        super().__init__(memory_type=memory_type, config=config, *args, **kwargs)

        # Initialize graph backend
        if graph_class:
            from smartmemory.models.memory_item import MemoryItem
            self.graph = graph_class(item_cls=MemoryItem)
        elif hasattr(self, '_graph_clazz') and self._graph_clazz:
            from smartmemory.models.memory_item import MemoryItem
            self.graph = self._graph_clazz(item_cls=MemoryItem)
        else:
            # Default to SmartGraph
            from smartmemory.graph.smartgraph import SmartGraph
            from smartmemory.models.memory_item import MemoryItem
            self.graph = SmartGraph(item_cls=MemoryItem)

    def _add_impl(self, item: MemoryItem, **kwargs) -> Optional[MemoryItem]:
        """Default graph-based add implementation."""
        try:
            # Use the graph store's add method (each store has its own interface)
            if hasattr(self.graph, 'add'):
                result = self.graph.add(item, **kwargs)
                return result if result else item
            else:
                # Fallback: try add_node if available
                node = self.graph.add_node(
                    item_id=item.item_id,
                    properties={
                        "content": item.content,
                        **item.metadata
                    },
                    memory_type=self._memory_type
                )
                item.metadata["_node"] = node
                return item
        except Exception as e:
            raise RuntimeError(f"Failed to add to graph: {e}")

    def _get_impl(self, key: str) -> Optional[MemoryItem]:
        """Default graph-based get implementation."""
        try:
            # Use the graph store's get method
            if hasattr(self.graph, 'get'):
                result = self.graph.get(key)
                return result
            else:
                # Fallback: try get_node if available
                node = self.graph.get_node(key)
                if node is None:
                    return None

                return MemoryItem(
                    item_id=key,
                    content=node.get("content", ""),
                    metadata=node
                )
        except Exception as e:
            raise RuntimeError(f"Failed to get from graph: {e}")

    def _remove_impl(self, key: str) -> bool:
        """Default graph-based remove implementation."""
        try:
            # Use the graph store's remove method
            if hasattr(self.graph, 'remove'):
                return self.graph.remove(key)
            else:
                # Fallback: try remove_node if available
                return self.graph.remove_node(key)
        except Exception as e:
            raise RuntimeError(f"Failed to remove from graph: {e}")

    def _search_impl(self, query: str, top_k: int = 5, **kwargs) -> List[MemoryItem]:
        """Default graph-based search implementation."""
        try:
            # Use the graph store's search method
            if hasattr(self.graph, 'search'):
                results = self.graph.search(query, top_k=top_k, **kwargs)
                # If results are already MemoryItems, return them
                if results and isinstance(results[0], MemoryItem):
                    return results
                # Otherwise convert nodes to MemoryItems
                return [
                    MemoryItem(
                        item_id=getattr(node, "item_id", ""),
                        content=getattr(node, "content", ""),
                        metadata=getattr(node, "metadata", {})
                    )
                    for node in results
                ]
            else:
                # Fallback: try search_nodes if available
                nodes = self.graph.search_nodes(
                    query=query,
                    limit=top_k,
                    memory_type=self._memory_type,
                    **kwargs
                )

                return [
                    MemoryItem(
                        item_id=node.get("item_id", ""),
                        content=node.get("content", ""),
                        metadata=node
                    )
                    for node in nodes
                ]
        except Exception as e:
            raise RuntimeError(f"Failed to search graph: {e}")

    def _clear_impl(self):
        """Default graph-based clear implementation."""
        if hasattr(self.graph, 'clear'):
            self.graph.clear()


class VectorBackedMemory(MemoryBase):
    """
    Base class for memory types that use a vector store backend.
    
    Provides common vector store patterns.
    """

    def __init__(self, memory_type: str = None, vector_store_class=None, config=None, *args, **kwargs):
        super().__init__(memory_type=memory_type, config=config, *args, **kwargs)

        # Initialize vector store
        if vector_store_class:
            self.vector_store = vector_store_class()
        else:
            from smartmemory.stores.vector.vector_store import VectorStore
            self.vector_store = VectorStore()

    def _add_impl(self, item: MemoryItem, **kwargs) -> Optional[MemoryItem]:
        """Default vector-based add implementation."""
        try:
            # Ensure embedding exists
            if not hasattr(item, 'embedding') or item.embedding is None:
                # Generate embedding if needed
                if hasattr(self, 'embedding_fn') and self.embedding_fn:
                    item.embedding = self.embedding_fn(str(item.content))

            # Add to vector store
            self.vector_store.add(
                item.item_id,
                item.embedding,
                metadata=item.metadata
            )
            return item
        except Exception as e:
            raise RuntimeError(f"Failed to add to vector store: {e}")

    def _get_impl(self, key: str) -> Optional[MemoryItem]:
        """Default vector-based get implementation."""
        try:
            # Vector stores typically don't support direct key lookup
            # This would need to be implemented based on specific vector store capabilities
            return None
        except Exception:
            return None

    def _search_impl(self, query: str, **kwargs) -> List[MemoryItem]:
        """Default vector-based search implementation."""
        try:
            top_k = kwargs.get('top_k', 5)
            results = self.vector_store.search(query, top_k=top_k)

            return [
                MemoryItem(
                    item_id=result.get('id', ''),
                    content=result.get('content', ''),
                    metadata=result.get('metadata') or {}
                )
                for result in results
            ]
        except Exception:
            return []


class HybridMemory(GraphBackedMemory, VectorBackedMemory):
    """
    Base class for memory types that use both graph and vector backends.
    
    Combines graph and vector store capabilities.
    """

    def __init__(self, memory_type: str = None, config=None, *args, **kwargs):
        # Initialize both backends
        GraphBackedMemory.__init__(self, memory_type=memory_type, config=config, *args, **kwargs)
        VectorBackedMemory.__init__(self, memory_type=memory_type, config=config, *args, **kwargs)

    def _add_impl(self, item: MemoryItem, **kwargs) -> Optional[MemoryItem]:
        """Add to both graph and vector store."""
        # Add to graph first
        result = GraphBackedMemory._add_impl(self, item, **kwargs)
        if result:
            # Then add to vector store
            try:
                VectorBackedMemory._add_impl(self, item, **kwargs)
            except Exception:
                # Vector store add failed, but graph succeeded
                pass
        return result

    def _search_impl(self, query: str, **kwargs) -> List[MemoryItem]:
        """Hybrid search combining vector and graph results."""
        vector_results = VectorBackedMemory._search_impl(self, query, **kwargs)
        graph_results = GraphBackedMemory._search_impl(self, query, **kwargs)

        # Combine and deduplicate results
        seen_ids = set()
        combined_results = []

        for result in vector_results + graph_results:
            if result.item_id not in seen_ids:
                seen_ids.add(result.item_id)
                combined_results.append(result)

        return combined_results
