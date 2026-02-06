"""EntityPairCache — Redis read-through cache for known entity-pair relations.

Avoids redundant LLM calls for entity pairs that have been seen before.
Cache key: smartmemory:entity_pair:{workspace_id}:{sorted(a,b)}
TTL: 30 minutes (configurable).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_TTL_SECONDS = 1800  # 30 minutes


class EntityPairCache:
    """Redis read-through cache for entity-pair relations.

    Lookup order:
    1. Check Redis cache
    2. On miss → query data graph for known relations
    3. On graph hit → write to Redis with TTL
    4. Return [{relation_type, confidence}] or None
    """

    def __init__(self, graph=None, redis_client=None, ttl: int = DEFAULT_TTL_SECONDS):
        self._graph = graph
        self._redis = redis_client
        self._ttl = ttl

    def _cache_key(self, entity_a: str, entity_b: str, workspace_id: str) -> str:
        pair = ":".join(sorted([entity_a.lower(), entity_b.lower()]))
        return f"smartmemory:entity_pair:{workspace_id}:{pair}"

    def lookup(self, entity_a: str, entity_b: str, workspace_id: str) -> Optional[List[Dict[str, Any]]]:
        """Look up cached relations between two entities.

        Returns:
            List of {relation_type, confidence} dicts, or None if no relations found.
        """
        key = self._cache_key(entity_a, entity_b, workspace_id)

        # Step 1: Check Redis cache
        if self._redis is not None:
            try:
                cached = self._redis.get(key)
                if cached is not None:
                    data = json.loads(cached)
                    return data if data else None
            except Exception as e:
                logger.debug("Redis cache read failed for %s: %s", key, e)

        # Step 2: Query graph for known relations
        relations = self._query_graph(entity_a, entity_b)

        # Step 3: Write to Redis on graph hit
        if relations and self._redis is not None:
            try:
                self._redis.set(key, json.dumps(relations), ex=self._ttl)
            except Exception as e:
                logger.debug("Redis cache write failed for %s: %s", key, e)

        return relations if relations else None

    def invalidate(self, entity_a: str, entity_b: str, workspace_id: str) -> None:
        """Delete Redis cache key when new relation stored."""
        if self._redis is None:
            return
        key = self._cache_key(entity_a, entity_b, workspace_id)
        try:
            self._redis.delete(key)
        except Exception as e:
            logger.debug("Redis cache invalidation failed for %s: %s", key, e)

    def _query_graph(self, entity_a: str, entity_b: str) -> List[Dict[str, Any]]:
        """Query the data graph for relations between two entities."""
        if self._graph is None:
            return []
        try:
            result = self._graph.query(
                "MATCH (a)-[r]->(b) "
                "WHERE toLower(a.name) = $a AND toLower(b.name) = $b "
                "RETURN type(r) AS relation_type, COALESCE(r.confidence, 0.5) AS confidence",
                params={"a": entity_a.lower(), "b": entity_b.lower()},
            )
            if not result:
                return []
            return [{"relation_type": row[0], "confidence": float(row[1])} for row in result]
        except Exception as e:
            logger.debug("Graph query failed for pair (%s, %s): %s", entity_a, entity_b, e)
            return []
