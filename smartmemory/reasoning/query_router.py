"""Query router for cost-optimized retrieval.

Routes deterministic queries to Cypher (free) and semantic queries to LLM search (costs tokens).
"""

import logging
import re
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class QueryType(Enum):
    SYMBOLIC = "symbolic"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"


# Patterns that indicate symbolic (graph-traversal) queries
SYMBOLIC_PATTERNS = [
    re.compile(r"^(who|what|where|when)\b", re.IGNORECASE),
    re.compile(r"\b(list all|count|how many)\b", re.IGNORECASE),
    re.compile(r"\b(is .+ a|does .+ have)\b", re.IGNORECASE),
]

# Patterns that indicate semantic (embedding-search) queries
SEMANTIC_PATTERNS = [
    re.compile(r"\b(similar|related|about|like|regarding)\b", re.IGNORECASE),
    re.compile(r"\b(find|search|look for)\b", re.IGNORECASE),
]

# Patterns that indicate hybrid (both) queries
HYBRID_PATTERNS = [
    re.compile(r"^(why|how|explain)\b", re.IGNORECASE),
    re.compile(r"\b(reason|cause|because|decision)\b", re.IGNORECASE),
]


class QueryRouter:
    """Routes queries to the cheapest effective retrieval method.

    Usage:
        router = QueryRouter(smart_memory)
        result = router.route("Who created SmartMemory?")
    """

    def __init__(self, memory: Any, graph: Any = None):
        self.memory = memory
        self.graph = graph or getattr(memory, "_graph", None)

    def classify(self, query: str) -> QueryType:
        """Classify a query as symbolic, semantic, or hybrid.

        Args:
            query: The user's query string.

        Returns:
            QueryType indicating the recommended retrieval method.
        """
        for pattern in HYBRID_PATTERNS:
            if pattern.search(query):
                return QueryType.HYBRID

        # Check semantic before symbolic - semantic keywords override question words
        # e.g. "What's related to X?" is semantic despite starting with "What"
        for pattern in SEMANTIC_PATTERNS:
            if pattern.search(query):
                return QueryType.SEMANTIC

        for pattern in SYMBOLIC_PATTERNS:
            if pattern.search(query):
                return QueryType.SYMBOLIC

        return QueryType.SEMANTIC

    def route(self, query: str, top_k: int = 10) -> dict[str, Any]:
        """Route and execute a query via the appropriate method.

        Args:
            query: The user's query string.
            top_k: Max results to return.

        Returns:
            Dict with query_type, results, and metadata.
        """
        query_type = self.classify(query)

        if query_type == QueryType.SYMBOLIC:
            results = self._symbolic_query(query, top_k)
        elif query_type == QueryType.SEMANTIC:
            results = self._semantic_query(query, top_k)
        else:
            symbolic = self._symbolic_query(query, top_k)
            semantic = self._semantic_query(query, top_k)
            results = self._merge_results(symbolic, semantic, top_k)

        return {
            "query_type": query_type.value,
            "query": query,
            "results": results,
            "result_count": len(results),
        }

    def _symbolic_query(self, query: str, top_k: int) -> list:
        """Execute a graph-based lookup."""
        if not self.graph:
            return []
        try:
            cypher = "MATCH (n) WHERE toLower(n.content) CONTAINS toLower($query) RETURN n LIMIT $limit"
            rows = self.graph.backend.execute_cypher(cypher, {"query": query, "limit": top_k})
            return rows or []
        except Exception as e:
            logger.debug(f"Symbolic query failed: {e}")
            return []

    def _semantic_query(self, query: str, top_k: int) -> list:
        """Execute an embedding-based search."""
        try:
            return self.memory.search(query, top_k=top_k) or []
        except Exception as e:
            logger.debug(f"Semantic query failed: {e}")
            return []

    def _merge_results(self, symbolic: list, semantic: list, top_k: int) -> list:
        """Merge results from symbolic and semantic queries, deduplicating by item_id."""
        seen: set[Any] = set()
        merged = []
        for result in symbolic + semantic:
            item_id = getattr(result, "item_id", None) or (
                result.get("item_id") if isinstance(result, dict) else id(result)
            )
            if item_id not in seen:
                seen.add(item_id)
                merged.append(result)
                if len(merged) >= top_k:
                    break
        return merged
