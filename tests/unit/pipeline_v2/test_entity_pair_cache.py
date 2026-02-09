"""Unit tests for EntityPairCache."""

import json
import pytest


pytestmark = pytest.mark.unit

from smartmemory.ontology.entity_pair_cache import EntityPairCache


class MockRedis:
    """Minimal in-memory Redis mock for get/set/delete."""

    def __init__(self):
        self._store: dict[str, str] = {}

    def get(self, key: str):
        return self._store.get(key)

    def set(self, key: str, value: str, ex: int = None):
        self._store[key] = value

    def delete(self, key: str):
        self._store.pop(key, None)


class MockGraph:
    """Mock graph that returns pre-configured relations."""

    def __init__(self, relations=None):
        self._relations = relations or []

    def query(self, cypher, params=None):
        if self._relations:
            return [[r["relation_type"], r["confidence"]] for r in self._relations]
        return []


@pytest.fixture
def redis():
    return MockRedis()


@pytest.fixture
def cache(redis):
    return EntityPairCache(redis_client=redis)


# ------------------------------------------------------------------ #
# Basic lookup
# ------------------------------------------------------------------ #


def test_lookup_returns_none_when_empty(cache):
    result = cache.lookup("Alice", "Bob", "ws1")
    assert result is None


def test_lookup_returns_cached_value(cache, redis):
    key = cache._cache_key("Alice", "Bob", "ws1")
    redis.set(key, json.dumps([{"relation_type": "KNOWS", "confidence": 0.9}]))

    result = cache.lookup("Alice", "Bob", "ws1")
    assert result is not None
    assert len(result) == 1
    assert result[0]["relation_type"] == "KNOWS"


def test_lookup_cache_key_is_order_independent(cache):
    key_ab = cache._cache_key("Alice", "Bob", "ws1")
    key_ba = cache._cache_key("Bob", "Alice", "ws1")
    assert key_ab == key_ba


# ------------------------------------------------------------------ #
# Graph fallback
# ------------------------------------------------------------------ #


def test_lookup_falls_through_to_graph(redis):
    graph = MockGraph(relations=[{"relation_type": "WORKS_AT", "confidence": 0.8}])
    cache = EntityPairCache(graph=graph, redis_client=redis)

    result = cache.lookup("Alice", "Acme", "ws1")
    assert result is not None
    assert result[0]["relation_type"] == "WORKS_AT"


def test_graph_result_is_cached_in_redis(redis):
    graph = MockGraph(relations=[{"relation_type": "USES", "confidence": 0.7}])
    cache = EntityPairCache(graph=graph, redis_client=redis)

    cache.lookup("Dev", "Python", "ws1")

    # Should now be in Redis
    key = cache._cache_key("Dev", "Python", "ws1")
    assert redis.get(key) is not None


# ------------------------------------------------------------------ #
# Invalidation
# ------------------------------------------------------------------ #


def test_invalidate_removes_cache_entry(cache, redis):
    key = cache._cache_key("A", "B", "ws1")
    redis.set(key, json.dumps([{"relation_type": "RELATED", "confidence": 0.5}]))

    cache.invalidate("A", "B", "ws1")
    assert redis.get(key) is None


# ------------------------------------------------------------------ #
# No Redis
# ------------------------------------------------------------------ #


def test_lookup_works_without_redis():
    graph = MockGraph(relations=[{"relation_type": "MENTORS", "confidence": 0.6}])
    cache = EntityPairCache(graph=graph, redis_client=None)

    result = cache.lookup("Alice", "Bob", "ws1")
    assert result is not None
    assert result[0]["relation_type"] == "MENTORS"


def test_invalidate_noop_without_redis():
    cache = EntityPairCache(redis_client=None)
    cache.invalidate("A", "B", "ws1")  # Should not raise
