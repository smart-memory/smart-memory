"""Unit tests for FalkorDBBackend bulk write methods (UNWIND Cypher batching)."""

from smartmemory.graph.backends.falkordb import FalkorDBBackend


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeResultSet:
    """Mimics FalkorDB query result with a result_set attribute."""

    def __init__(self, count: int):
        # result_set is a list of rows; each row is a list of values.
        # The UNWIND queries return [[cnt]] where cnt is count(n) or count(r).
        self.result_set = [[count]]


class FakeGraph:
    """Records every query() call and returns a configurable result."""

    def __init__(self):
        self.calls: list = []

    def query(self, cypher: str, params: dict | None = None):
        self.calls.append((cypher, params))
        # Return count = len(batch) so tests can verify totals.
        batch = (params or {}).get("batch", [])
        return FakeResultSet(len(batch))


class FakeScopeProvider:
    """Returns a fixed write context."""

    def get_write_context(self):
        return {"workspace_id": "ws_test", "user_id": "u_test"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_backend() -> tuple[FalkorDBBackend, FakeGraph]:
    """Create a FalkorDBBackend without hitting a real DB."""
    b = object.__new__(FalkorDBBackend)
    g = FakeGraph()
    b.graph = g
    b.scope_provider = FakeScopeProvider()
    return b, g


# ---------------------------------------------------------------------------
# TestAddNodesBulk
# ---------------------------------------------------------------------------


class TestAddNodesBulk:
    def test_empty_list_returns_zero(self):
        b, g = _make_backend()
        assert b.add_nodes_bulk([]) == 0
        assert g.calls == []

    def test_single_label_single_batch(self):
        b, g = _make_backend()
        nodes = [
            {"item_id": "n1", "memory_type": "code", "content": "hello"},
            {"item_id": "n2", "memory_type": "code", "content": "world"},
        ]
        result = b.add_nodes_bulk(nodes)
        assert result == 2
        assert len(g.calls) == 1
        cypher, params = g.calls[0]
        assert "UNWIND" in cypher
        assert ":Code" in cypher  # capitalize() + sanitize_label() -> "Code"
        assert len(params["batch"]) == 2

    def test_multiple_labels_multiple_queries(self):
        b, g = _make_backend()
        nodes = [
            {"item_id": "n1", "memory_type": "code"},
            {"item_id": "n2", "memory_type": "function"},
            {"item_id": "n3", "memory_type": "code"},
        ]
        result = b.add_nodes_bulk(nodes)
        assert result == 3
        assert len(g.calls) == 2  # one query per label

    def test_write_context_injected(self):
        b, g = _make_backend()
        nodes = [{"item_id": "n1", "memory_type": "code"}]
        b.add_nodes_bulk(nodes)
        _, params = g.calls[0]
        props = params["batch"][0]["props"]
        assert props["workspace_id"] == "ws_test"
        assert props["user_id"] == "u_test"

    def test_chunking_with_small_batch_size(self):
        b, g = _make_backend()
        nodes = [{"item_id": f"n{i}", "memory_type": "code"} for i in range(5)]
        result = b.add_nodes_bulk(nodes, batch_size=2)
        # 5 nodes / batch_size=2 -> 3 chunks (2, 2, 1)
        assert len(g.calls) == 3
        assert result == 5

    def test_label_sanitization(self):
        b, g = _make_backend()
        nodes = [{"item_id": "n1", "memory_type": "my-type"}]
        b.add_nodes_bulk(nodes)
        cypher, _ = g.calls[0]
        assert ":My_type" in cypher  # capitalize + hyphens -> underscores

    def test_default_label_when_missing(self):
        b, g = _make_backend()
        nodes = [{"item_id": "n1"}]  # no memory_type
        b.add_nodes_bulk(nodes)
        cypher, _ = g.calls[0]
        assert ":Node" in cypher

    def test_skips_nodes_without_item_id(self):
        b, g = _make_backend()
        nodes = [
            {"item_id": "n1", "memory_type": "code"},
            {"memory_type": "code"},  # no item_id
            {"item_id": "", "memory_type": "code"},  # empty item_id
        ]
        result = b.add_nodes_bulk(nodes)
        assert result == 1  # only n1 written
        _, params = g.calls[0]
        assert len(params["batch"]) == 1

    def test_filters_invalid_properties(self):
        b, g = _make_backend()
        nodes = [{"item_id": "n1", "memory_type": "code", "name": "test", "empty": "", "embedding": [1, 2]}]
        b.add_nodes_bulk(nodes)
        _, params = g.calls[0]
        props = params["batch"][0]["props"]
        assert "name" in props
        assert "empty" not in props  # empty strings filtered
        assert "embedding" not in props  # embedding key filtered

    def test_is_global_skips_write_context(self):
        b, g = _make_backend()
        nodes = [{"item_id": "n1", "memory_type": "code", "name": "global_entity"}]
        b.add_nodes_bulk(nodes, is_global=True)
        _, params = g.calls[0]
        props = params["batch"][0]["props"]
        assert "workspace_id" not in props
        assert "user_id" not in props
        assert props["name"] == "global_entity"

    def test_is_global_false_injects_write_context(self):
        """Explicit is_global=False should behave identically to the default."""
        b, g = _make_backend()
        nodes = [{"item_id": "n1", "memory_type": "code"}]
        b.add_nodes_bulk(nodes, is_global=False)
        _, params = g.calls[0]
        props = params["batch"][0]["props"]
        assert props["workspace_id"] == "ws_test"
        assert props["user_id"] == "u_test"


# ---------------------------------------------------------------------------
# TestAddEdgesBulk
# ---------------------------------------------------------------------------


class TestAddEdgesBulk:
    def test_empty_list_returns_zero(self):
        b, g = _make_backend()
        assert b.add_edges_bulk([]) == 0
        assert g.calls == []

    def test_single_type_single_batch(self):
        b, g = _make_backend()
        edges = [
            ("a", "b", "IMPORTS", {"weight": 1}),
            ("c", "d", "IMPORTS", {"weight": 2}),
        ]
        result = b.add_edges_bulk(edges)
        assert result == 2
        assert len(g.calls) == 1
        cypher, params = g.calls[0]
        assert "UNWIND" in cypher
        assert ":IMPORTS" in cypher
        assert len(params["batch"]) == 2

    def test_multiple_types_multiple_queries(self):
        b, g = _make_backend()
        edges = [
            ("a", "b", "IMPORTS", {}),
            ("c", "d", "CALLS", {}),
        ]
        result = b.add_edges_bulk(edges)
        assert result == 2
        assert len(g.calls) == 2

    def test_edge_type_sanitized(self):
        b, g = _make_backend()
        edges = [("a", "b", "relates-to", {})]
        b.add_edges_bulk(edges)
        cypher, _ = g.calls[0]
        assert ":RELATES_TO" in cypher

    def test_empty_edge_type_falls_back_to_related(self):
        b, g = _make_backend()
        edges = [("a", "b", "", {})]
        b.add_edges_bulk(edges)
        cypher, _ = g.calls[0]
        assert ":RELATED" in cypher

    def test_write_context_injected(self):
        b, g = _make_backend()
        edges = [("a", "b", "IMPORTS", {"weight": 1})]
        b.add_edges_bulk(edges)
        _, params = g.calls[0]
        props = params["batch"][0]["props"]
        assert props["workspace_id"] == "ws_test"
        assert props["user_id"] == "u_test"

    def test_chunking_with_small_batch_size(self):
        b, g = _make_backend()
        edges = [("a", "b", "IMPORTS", {}) for _ in range(5)]
        result = b.add_edges_bulk(edges, batch_size=2)
        assert len(g.calls) == 3  # 5 / 2 -> 3 chunks
        assert result == 5

    def test_is_global_skips_write_context_and_match_scoping(self):
        b, g = _make_backend()
        edges = [("a", "b", "IMPORTS", {"weight": 1})]
        b.add_edges_bulk(edges, is_global=True)
        cypher, params = g.calls[0]
        # Props should NOT have workspace_id or user_id
        props = params["batch"][0]["props"]
        assert "workspace_id" not in props
        assert "user_id" not in props
        # MATCH clause should NOT scope to workspace_id
        assert "workspace_id" not in cypher

    def test_is_global_false_injects_write_context(self):
        """Explicit is_global=False should behave identically to the default."""
        b, g = _make_backend()
        edges = [("a", "b", "IMPORTS", {})]
        b.add_edges_bulk(edges, is_global=False)
        _, params = g.calls[0]
        props = params["batch"][0]["props"]
        assert props["workspace_id"] == "ws_test"
        assert props["user_id"] == "u_test"
