"""Tests for CORE-OBS-3: SmartGraph write methods emit graph_mutation events.

Verifies that add_node, add_edge, add_nodes_bulk, add_edges_bulk, and remove_node
all emit graph_mutation events via emit_event() so the Graph Viewer live feed
can display enricher/evolver mutations in real-time.
"""

from unittest.mock import MagicMock, patch, call
import pytest


@pytest.fixture
def mock_backend():
    """Create a mock graph backend with required methods."""
    backend = MagicMock()
    backend.add_nodes_bulk.return_value = 5
    backend.add_edges_bulk.return_value = 3
    return backend


@pytest.fixture
def graph(mock_backend):
    """Create a SmartGraph with mock backend, bypassing config resolution."""
    with patch("smartmemory.graph.smartgraph.get_config", return_value={"backend_class": "FalkorDBBackend"}):
        with patch("smartmemory.graph.smartgraph.SmartGraph._get_backend_class") as mock_cls:
            mock_cls.return_value = lambda **kwargs: mock_backend
            from smartmemory.graph.smartgraph import SmartGraph
            sg = SmartGraph()
    return sg


class TestAddNodeEventEmission:
    """add_node() must emit a graph_mutation event."""

    def test_add_node_emits_event(self, graph):
        graph.nodes.add_node = MagicMock(return_value={"item_id": "test-123"})

        with patch("smartmemory.graph.smartgraph.emit_event") as mock_emit:
            graph.add_node("test-123", {"content": "hello world", "memory_type": "semantic"}, memory_type="semantic")

        mock_emit.assert_called_once_with(
            "graph_mutation", "graph", "add_node",
            {
                "memory_id": "test-123",
                "memory_type": "semantic",
                "label": "hello world"[:40],
                "content": "hello world"[:200],
            },
        )

    def test_add_node_event_does_not_break_on_failure(self, graph):
        """emit_event failure must not propagate to caller."""
        graph.nodes.add_node = MagicMock(return_value={"item_id": "test-456"})

        with patch("smartmemory.graph.smartgraph.emit_event", side_effect=RuntimeError("redis down")):
            result = graph.add_node("test-456", {"content": "test"}, memory_type="working")

        assert result == {"item_id": "test-456"}


class TestAddEdgeEventEmission:
    """add_edge() must emit a graph_mutation event."""

    def test_add_edge_emits_event(self, graph):
        graph.edges.add_edge = MagicMock(return_value={"edge_created": True})

        with patch("smartmemory.graph.smartgraph.emit_event") as mock_emit:
            graph.add_edge("src-1", "tgt-1", "RELATES_TO", {})

        mock_emit.assert_called_once_with(
            "graph_mutation", "graph", "add_edge",
            {
                "source_id": "src-1",
                "target_id": "tgt-1",
                "edge_type": "RELATES_TO",
            },
        )

    def test_add_edge_event_does_not_break_on_failure(self, graph):
        graph.edges.add_edge = MagicMock(return_value={"edge_created": True})

        with patch("smartmemory.graph.smartgraph.emit_event", side_effect=RuntimeError("redis down")):
            result = graph.add_edge("src-1", "tgt-1", "RELATES_TO", {})

        assert result == {"edge_created": True}


class TestBulkEventEmission:
    """Bulk operations must emit summary events."""

    def test_add_nodes_bulk_emits_event(self, graph, mock_backend):
        with patch("smartmemory.graph.smartgraph.emit_event") as mock_emit:
            count = graph.add_nodes_bulk([{"item_id": f"n{i}"} for i in range(5)])

        assert count == 5
        mock_emit.assert_called_once_with(
            "graph_mutation", "graph", "add_nodes_bulk", {"count": 5}
        )

    def test_add_edges_bulk_emits_event(self, graph, mock_backend):
        edges = [("a", "b", "REL", {}) for _ in range(3)]
        with patch("smartmemory.graph.smartgraph.emit_event") as mock_emit:
            count = graph.add_edges_bulk(edges)

        assert count == 3
        mock_emit.assert_called_once_with(
            "graph_mutation", "graph", "add_edges_bulk", {"count": 3}
        )

    def test_bulk_event_does_not_break_on_failure(self, graph, mock_backend):
        with patch("smartmemory.graph.smartgraph.emit_event", side_effect=RuntimeError("redis down")):
            count = graph.add_nodes_bulk([{"item_id": "n1"}])
        assert count == 5


class TestRemoveNodeEventEmission:
    """remove_node() must emit a graph_mutation event."""

    def test_remove_node_emits_event(self, graph):
        graph.nodes.remove_node = MagicMock(return_value=True)

        with patch("smartmemory.graph.smartgraph.emit_event") as mock_emit:
            graph.remove_node("del-123")

        mock_emit.assert_called_once_with(
            "graph_mutation", "graph", "delete_node", {"memory_id": "del-123"}
        )

    def test_remove_node_event_does_not_break_on_failure(self, graph):
        graph.nodes.remove_node = MagicMock(return_value=True)

        with patch("smartmemory.graph.smartgraph.emit_event", side_effect=RuntimeError("redis down")):
            result = graph.remove_node("del-456")

        assert result is True


class TestNoDuplicateEmissionFromDualNode:
    """add_dual_node has its own _emit_dual_node_events — add_node must not double-emit."""

    def test_add_node_and_dual_node_use_separate_paths(self, graph):
        """Verify add_node and add_dual_node are independent code paths."""
        # add_dual_node calls self.nodes.add_dual_node (submodule), not self.add_node (top-level)
        # So our new emit_event in add_node() won't fire for dual node operations
        graph.nodes.add_dual_node = MagicMock(return_value={"item_id": "dual-1"})

        with patch("smartmemory.graph.smartgraph.emit_event") as mock_emit:
            graph.add_dual_node("dual-1", {"content": "test"}, "semantic", [])

        # The only emit_event calls should be from _emit_dual_node_events, not from add_node
        add_node_calls = [
            c for c in mock_emit.call_args_list
            if len(c.args) >= 3 and c.args[2] == "add_node"
            and c.args[0] == "graph_mutation"
            and (c.args[3] or {}).get("memory_type") == "semantic"  # from _emit_dual_node_events
        ]
        # _emit_dual_node_events emits the memory node — that's expected.
        # What we're verifying is that our new add_node() emit doesn't also fire.
        # If add_dual_node called self.add_node(), we'd see TWO add_node events.
        # But it calls self.nodes.add_dual_node() instead, so only _emit_dual_node_events fires.
        assert len(add_node_calls) <= 1  # At most the one from _emit_dual_node_events
