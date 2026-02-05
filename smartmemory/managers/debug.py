"""Debug and troubleshooting operations extracted from SmartMemory."""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class DebugManager:
    """Debug search, item inspection, and search repair."""

    def __init__(self, graph, search, scope_provider=None):
        self._graph = graph
        self._search = search
        self.scope_provider = scope_provider

    def debug_search(self, query: str, top_k: int = 5) -> dict:
        """Debug search functionality with detailed logging."""
        debug_info: Dict[str, Any] = {
            "query": query,
            "top_k": top_k,
            "graph_backend": str(type(self._graph.backend)),
            "search_component": str(type(self._search)),
            "results": [],
        }

        graph_results = self._graph.search(query, top_k=top_k)
        debug_info["graph_search_count"] = len(graph_results)
        debug_info["graph_search_results"] = [
            {
                "item_id": getattr(r, "item_id", "No ID"),
                "content_preview": str(getattr(r, "content", "No content"))[:50] + "...",
                "type": str(type(r)),
            }
            for r in graph_results[:3]
        ]

        search_results = self._search.search(query, top_k=top_k)
        debug_info["search_component_count"] = len(search_results)
        debug_info["search_component_results"] = [
            {
                "item_id": getattr(r, "item_id", "No ID"),
                "content_preview": str(getattr(r, "content", "No content"))[:50] + "...",
                "type": str(type(r)),
            }
            for r in search_results[:3]
        ]

        debug_info["results"] = search_results
        return debug_info

    def get_all_items_debug(self) -> Dict[str, Any]:
        """Get all items for debugging, automatically filtered by ScopeProvider."""
        debug_info: Dict[str, Any] = {
            "total_items": 0,
            "items_by_type": {},
            "sample_items": [],
        }

        filters = {}
        if self.scope_provider:
            filters = self.scope_provider.get_isolation_filters()

        if hasattr(self._graph, "nodes"):
            all_node_ids = self._graph.nodes.nodes()
            debug_info["total_node_ids"] = len(all_node_ids)

            seen_ids: set[str] = set()
            for node_id in all_node_ids:
                item = self._graph.get_node(node_id)
                if not item:
                    continue

                item_id = getattr(item, "item_id", node_id)
                if item_id in seen_ids:
                    continue

                if filters:
                    item_metadata = getattr(item, "metadata", {})
                    skip_item = False
                    for filter_key, filter_value in filters.items():
                        item_value = getattr(item, filter_key, None) or item_metadata.get(filter_key)
                        if filter_value and item_value != filter_value:
                            skip_item = True
                            break
                    if skip_item:
                        continue

                seen_ids.add(item_id)
                debug_info["total_items"] += 1
                item_type = getattr(item, "memory_type", "unknown")
                debug_info["items_by_type"][item_type] = debug_info["items_by_type"].get(item_type, 0) + 1

                if len(debug_info["sample_items"]) < 3:
                    debug_info["sample_items"].append(
                        {
                            "item_id": item_id,
                            "content_preview": str(getattr(item, "content", "No content"))[:50] + "...",
                            "memory_type": item_type,
                            "type": str(type(item)),
                        }
                    )

        return debug_info

    def fix_search_if_broken(self):
        """Attempt to fix search functionality if it's broken.

        Returns:
            Tuple of (fix_info dict, new Search instance) so the caller can
            update its own reference without reaching into our internals.
        """
        from smartmemory.memory.pipeline.stages.search import Search

        fix_info: Dict[str, Any] = {"fixes_applied": [], "test_search_count": 0}

        self._search = Search(self._graph)
        fix_info["fixes_applied"].append("Reinitialized search component")

        if hasattr(self._graph, "clear_cache"):
            self._graph.clear_cache()
            fix_info["fixes_applied"].append("Cleared graph cache")

        test_results = self._search.search("test", top_k=1)
        fix_info["test_search_count"] = len(test_results)

        return fix_info, self._search

    def clear(self):
        """Clear all memory from ALL storage backends comprehensively."""
        logger.info("Clearing all memory storage backends...")

        # 1. Clear the graph backend (FalkorDB)
        from smartmemory.memory.pipeline.stages.graph_operations import GraphOperations

        graph_ops = GraphOperations(self._graph)
        graph_ops.clear_all()
        logger.info("Cleared Graph Database (FalkorDB)")

        # 2. Clear the vector database
        from smartmemory.stores.vector.vector_store import VectorStore

        try:
            vector_store = VectorStore()
            vector_store.clear()
            logger.info("Cleared Vector Store")
        except Exception as e:
            logger.warning(f"Vector Store clear failed: {e}")

        # 3. Clear ALL Redis cache types
        from smartmemory.utils.cache import get_cache

        cache = get_cache()

        cache_types = ["embedding", "search", "entity_extraction", "similarity", "graph_query"]
        total_cleared = 0
        for cache_type in cache_types:
            cleared_count = cache.clear_type(cache_type)
            total_cleared += cleared_count

        pattern = f"{cache.prefix}:*"
        remaining_keys = cache.redis.keys(pattern)
        if remaining_keys:
            cache.redis.delete(*remaining_keys)
            total_cleared += len(remaining_keys)

        logger.info(f"Cleared Redis Cache ({total_cleared} keys)")
        logger.info("All memory storage backends cleared successfully!")
        return True
