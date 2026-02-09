"""Unit tests for DebugManager."""

from unittest.mock import MagicMock, patch, PropertyMock

import pytest

pytestmark = pytest.mark.unit

from smartmemory.managers.debug import DebugManager


@pytest.fixture
def mock_graph():
    g = MagicMock()
    g.backend = MagicMock()
    g.search.return_value = []
    g.get_node.return_value = None
    g.clear_cache = MagicMock()
    # nodes sub-object
    g.nodes = MagicMock()
    g.nodes.nodes.return_value = []
    return g


@pytest.fixture
def mock_search():
    s = MagicMock()
    s.search.return_value = []
    return s


@pytest.fixture
def manager(mock_graph, mock_search):
    return DebugManager(mock_graph, mock_search)


@pytest.fixture
def manager_with_scope(mock_graph, mock_search):
    scope = MagicMock()
    scope.get_isolation_filters.return_value = {"tenant_id": "t1", "user_id": "u1"}
    return DebugManager(mock_graph, mock_search, scope_provider=scope)


class TestDebugSearch:
    def test_returns_debug_info_structure(self, manager):
        result = manager.debug_search("test query")
        assert result["query"] == "test query"
        assert result["top_k"] == 5
        assert "graph_backend" in result
        assert "search_component" in result
        assert "graph_search_count" in result
        assert "search_component_count" in result

    def test_custom_top_k(self, manager, mock_graph, mock_search):
        manager.debug_search("query", top_k=10)
        mock_graph.search.assert_called_once_with("query", top_k=10)
        mock_search.search.assert_called_once_with("query", top_k=10)

    def test_with_graph_results(self, manager, mock_graph):
        item = MagicMock()
        item.item_id = "item_1"
        item.content = "Some content about testing"
        mock_graph.search.return_value = [item]

        result = manager.debug_search("test")
        assert result["graph_search_count"] == 1
        assert result["graph_search_results"][0]["item_id"] == "item_1"

    def test_with_search_results(self, manager, mock_search):
        item = MagicMock()
        item.item_id = "item_2"
        item.content = "Search result content"
        mock_search.search.return_value = [item]

        result = manager.debug_search("test")
        assert result["search_component_count"] == 1
        assert result["results"] == [item]

    def test_content_preview_truncated(self, manager, mock_graph):
        item = MagicMock()
        item.item_id = "item_long"
        item.content = "A" * 200
        mock_graph.search.return_value = [item]

        result = manager.debug_search("test")
        preview = result["graph_search_results"][0]["content_preview"]
        # 50 chars + "..."
        assert len(preview) == 53

    def test_limits_preview_to_3_items(self, manager, mock_graph):
        items = []
        for i in range(5):
            item = MagicMock()
            item.item_id = f"item_{i}"
            item.content = f"Content {i}"
            items.append(item)
        mock_graph.search.return_value = items

        result = manager.debug_search("test")
        assert result["graph_search_count"] == 5
        assert len(result["graph_search_results"]) == 3


class TestGetAllItemsDebug:
    def test_empty_graph(self, manager):
        result = manager.get_all_items_debug()
        assert result["total_items"] == 0
        assert result["items_by_type"] == {}
        assert result["sample_items"] == []

    def test_counts_items_by_type(self, manager, mock_graph):
        node1 = MagicMock()
        node1.item_id = "n1"
        node1.memory_type = "semantic"
        node1.content = "Fact 1"
        node1.metadata = {}

        node2 = MagicMock()
        node2.item_id = "n2"
        node2.memory_type = "episodic"
        node2.content = "Event 1"
        node2.metadata = {}

        node3 = MagicMock()
        node3.item_id = "n3"
        node3.memory_type = "semantic"
        node3.content = "Fact 2"
        node3.metadata = {}

        mock_graph.nodes.nodes.return_value = ["n1", "n2", "n3"]
        mock_graph.get_node.side_effect = [node1, node2, node3]

        result = manager.get_all_items_debug()
        assert result["total_items"] == 3
        assert result["items_by_type"]["semantic"] == 2
        assert result["items_by_type"]["episodic"] == 1

    def test_deduplicates_by_item_id(self, manager, mock_graph):
        node = MagicMock()
        node.item_id = "same_id"
        node.memory_type = "semantic"
        node.content = "Duplicate"
        node.metadata = {}

        mock_graph.nodes.nodes.return_value = ["same_id", "same_id"]
        mock_graph.get_node.return_value = node

        result = manager.get_all_items_debug()
        assert result["total_items"] == 1

    def test_skips_none_nodes(self, manager, mock_graph):
        mock_graph.nodes.nodes.return_value = ["n1", "n2"]
        mock_graph.get_node.side_effect = [None, None]

        result = manager.get_all_items_debug()
        assert result["total_items"] == 0

    def test_sample_items_limited_to_3(self, manager, mock_graph):
        nodes = []
        for i in range(5):
            n = MagicMock()
            n.item_id = f"n_{i}"
            n.memory_type = "semantic"
            n.content = f"Content {i}"
            n.metadata = {}
            nodes.append(n)

        mock_graph.nodes.nodes.return_value = [f"n_{i}" for i in range(5)]
        mock_graph.get_node.side_effect = nodes

        result = manager.get_all_items_debug()
        assert result["total_items"] == 5
        assert len(result["sample_items"]) == 3

    def test_scope_provider_filters_items(self, manager_with_scope, mock_graph):
        matching = MagicMock()
        matching.item_id = "match"
        matching.memory_type = "semantic"
        matching.content = "Matching"
        matching.metadata = {}
        matching.tenant_id = "t1"
        matching.user_id = "u1"

        non_matching = MagicMock()
        non_matching.item_id = "no_match"
        non_matching.memory_type = "semantic"
        non_matching.content = "Wrong tenant"
        non_matching.metadata = {}
        non_matching.tenant_id = "t2"
        non_matching.user_id = "u2"

        mock_graph.nodes.nodes.return_value = ["match", "no_match"]
        mock_graph.get_node.side_effect = [matching, non_matching]

        result = manager_with_scope.get_all_items_debug()
        assert result["total_items"] == 1

    def test_no_scope_provider_returns_all(self, manager, mock_graph):
        node = MagicMock()
        node.item_id = "n1"
        node.memory_type = "semantic"
        node.content = "Content"
        node.metadata = {}

        mock_graph.nodes.nodes.return_value = ["n1"]
        mock_graph.get_node.return_value = node

        result = manager.get_all_items_debug()
        assert result["total_items"] == 1


class TestFixSearchIfBroken:
    @patch("smartmemory.memory.pipeline.stages.search.Search")
    def test_reinitializes_search(self, MockSearch, manager, mock_graph):
        new_search = MagicMock()
        new_search.search.return_value = []
        MockSearch.return_value = new_search

        fix_info, returned_search = manager.fix_search_if_broken()
        MockSearch.assert_called_once_with(mock_graph)
        assert "Reinitialized search component" in fix_info["fixes_applied"]
        assert returned_search is new_search

    @patch("smartmemory.memory.pipeline.stages.search.Search")
    def test_clears_graph_cache_if_available(self, MockSearch, manager, mock_graph):
        MockSearch.return_value = MagicMock(search=MagicMock(return_value=[]))
        manager.fix_search_if_broken()
        mock_graph.clear_cache.assert_called_once()

    @patch("smartmemory.memory.pipeline.stages.search.Search")
    def test_no_clear_cache_if_missing(self, MockSearch, mock_search):
        graph = MagicMock(spec=[])  # no clear_cache attribute
        mgr = DebugManager(graph, mock_search)
        MockSearch.return_value = MagicMock(search=MagicMock(return_value=[]))
        fix_info, _ = mgr.fix_search_if_broken()
        assert "Cleared graph cache" not in fix_info["fixes_applied"]

    @patch("smartmemory.memory.pipeline.stages.search.Search")
    def test_test_search_count(self, MockSearch, manager):
        new_search = MagicMock()
        new_search.search.return_value = [MagicMock(), MagicMock()]
        MockSearch.return_value = new_search

        fix_info, _ = manager.fix_search_if_broken()
        assert fix_info["test_search_count"] == 2


class TestClear:
    @patch("smartmemory.utils.cache.get_cache")
    @patch("smartmemory.stores.vector.vector_store.VectorStore")
    @patch("smartmemory.memory.pipeline.stages.graph_operations.GraphOperations")
    def test_clears_all_backends(self, MockGraphOps, MockVectorStore, mock_get_cache, manager):
        mock_cache = MagicMock()
        mock_cache.clear_type.return_value = 3
        mock_cache.prefix = "sm"
        mock_cache.redis.keys.return_value = []
        mock_get_cache.return_value = mock_cache

        MockVectorStore.return_value = MagicMock()

        result = manager.clear()
        assert result is True
        MockGraphOps.assert_called_once()
        MockGraphOps.return_value.clear_all.assert_called_once()

    @patch("smartmemory.utils.cache.get_cache")
    @patch("smartmemory.stores.vector.vector_store.VectorStore")
    @patch("smartmemory.memory.pipeline.stages.graph_operations.GraphOperations")
    def test_clears_all_cache_types(self, MockGraphOps, MockVectorStore, mock_get_cache, manager):
        mock_cache = MagicMock()
        mock_cache.clear_type.return_value = 2
        mock_cache.prefix = "sm"
        mock_cache.redis.keys.return_value = []
        mock_get_cache.return_value = mock_cache
        MockVectorStore.return_value = MagicMock()

        manager.clear()
        expected_types = ["embedding", "search", "entity_extraction", "similarity", "graph_query"]
        assert mock_cache.clear_type.call_count == len(expected_types)
        for cache_type in expected_types:
            mock_cache.clear_type.assert_any_call(cache_type)

    @patch("smartmemory.utils.cache.get_cache")
    @patch("smartmemory.stores.vector.vector_store.VectorStore")
    @patch("smartmemory.memory.pipeline.stages.graph_operations.GraphOperations")
    def test_clears_remaining_redis_keys(self, MockGraphOps, MockVectorStore, mock_get_cache, manager):
        mock_cache = MagicMock()
        mock_cache.clear_type.return_value = 0
        mock_cache.prefix = "sm"
        remaining = [b"sm:leftover_1", b"sm:leftover_2"]
        mock_cache.redis.keys.return_value = remaining
        mock_get_cache.return_value = mock_cache
        MockVectorStore.return_value = MagicMock()

        manager.clear()
        mock_cache.redis.delete.assert_called_once_with(*remaining)

    @patch("smartmemory.utils.cache.get_cache")
    @patch("smartmemory.stores.vector.vector_store.VectorStore")
    @patch("smartmemory.memory.pipeline.stages.graph_operations.GraphOperations")
    def test_vector_store_failure_logged_not_raised(self, MockGraphOps, MockVectorStore, mock_get_cache, manager):
        MockVectorStore.return_value.clear.side_effect = RuntimeError("Vector down")
        mock_cache = MagicMock()
        mock_cache.clear_type.return_value = 0
        mock_cache.prefix = "sm"
        mock_cache.redis.keys.return_value = []
        mock_get_cache.return_value = mock_cache

        # Should not raise — vector store failure is logged as warning
        result = manager.clear()
        assert result is True
