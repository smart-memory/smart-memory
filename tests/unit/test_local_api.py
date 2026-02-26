"""Unit tests for DIST-LITE-4: local_api.py.

Tests cover:
  - GET /graph/full — nodes are flattened (no nested properties key)
  - GET /graph/full — top-level fields present: label, node_category, entity_type
  - POST /graph/edges — deduplication (same edge not returned twice)
  - GET /list — pagination envelope keys
  - GET /{id}/neighbors — response key is "neighbors" not "nodes"
  - GET /{id} — 404 when backend.get_node() returns None
  - DELETE /{id} — 405 (read-only)
  - DELETE /graph/nodes/{id} — 405 (read-only)

All tests mock _get_backend() to avoid touching the filesystem.
"""
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from smartmemory_pkg.local_api import api


@pytest.fixture()
def client():
    return TestClient(api)


# ---------------------------------------------------------------------------
# Helper: canned serialize() output (nested properties — as serialize() returns)
# ---------------------------------------------------------------------------

def _make_serialize_node(
    item_id: str = "node-1",
    memory_type: str = "semantic",
    label: str = "Test label",
    node_category: str = "memory",
    entity_type: str = None,
    extra_props: dict = None,
) -> dict:
    """Return a node dict in the shape serialize() produces (properties nested)."""
    props = {"label": label, "content": "Test content", "confidence": 0.9}
    if node_category is not None:
        props["node_category"] = node_category
    if entity_type is not None:
        props["entity_type"] = entity_type
    if extra_props:
        props.update(extra_props)
    return {
        "item_id": item_id,
        "memory_type": memory_type,
        "valid_from": None,
        "valid_to": None,
        "created_at": "2026-02-26T00:00:00",
        "properties": props,
    }


def _make_edge(source: str, target: str, edge_type: str = "related_to") -> dict:
    return {
        "source_id": source,
        "target_id": target,
        "edge_type": edge_type,
        "memory_type": "semantic",
        "valid_from": None,
        "valid_to": None,
        "created_at": "2026-02-26T00:00:00",
        "properties": {},
    }


def _make_flat_node(item_id: str = "node-1", memory_type: str = "semantic") -> dict:
    """Return a node dict in the flat shape _row_to_node() produces."""
    return {
        "item_id": item_id,
        "memory_type": memory_type,
        "label": "Flat label",
        "content": "Flat content",
        "confidence": 0.9,
        "node_category": "memory",
        "entity_type": None,
        "created_at": "2026-02-26T00:00:00",
        "valid_from": None,
        "valid_to": None,
    }


# ---------------------------------------------------------------------------
# GET /graph/full
# ---------------------------------------------------------------------------


class TestGetGraphFull:
    def test_returns_200(self, client):
        mock_backend = MagicMock()
        mock_backend.serialize.return_value = {
            "nodes": [_make_serialize_node()],
            "edges": [],
        }
        with patch("smartmemory_pkg.local_api._get_backend", return_value=mock_backend):
            r = client.get("/graph/full")
        assert r.status_code == 200

    def test_envelope_keys_present(self, client):
        mock_backend = MagicMock()
        mock_backend.serialize.return_value = {"nodes": [], "edges": []}
        with patch("smartmemory_pkg.local_api._get_backend", return_value=mock_backend):
            body = client.get("/graph/full").json()
        assert "nodes" in body
        assert "edges" in body
        assert "node_count" in body
        assert "edge_count" in body

    def test_node_fields_at_top_level(self, client):
        """label, node_category, entity_type must be at top level — not nested under properties."""
        node = _make_serialize_node(
            item_id="abc",
            label="My label",
            node_category="entity",
            entity_type="Person",
        )
        mock_backend = MagicMock()
        mock_backend.serialize.return_value = {"nodes": [node], "edges": []}
        with patch("smartmemory_pkg.local_api._get_backend", return_value=mock_backend):
            body = client.get("/graph/full").json()

        assert len(body["nodes"]) == 1
        n = body["nodes"][0]
        assert n["label"] == "My label"
        assert n["node_category"] == "entity"
        assert n["entity_type"] == "Person"
        assert n["item_id"] == "abc"
        assert n["memory_type"] == "semantic"

    def test_no_nested_properties_key(self, client):
        """After flattening, the node must not contain a 'properties' key."""
        node = _make_serialize_node()
        mock_backend = MagicMock()
        mock_backend.serialize.return_value = {"nodes": [node], "edges": []}
        with patch("smartmemory_pkg.local_api._get_backend", return_value=mock_backend):
            body = client.get("/graph/full").json()

        n = body["nodes"][0]
        assert "properties" not in n

    def test_node_count_matches(self, client):
        nodes = [_make_serialize_node(item_id=f"node-{i}") for i in range(3)]
        edges = [_make_edge("node-0", "node-1")]
        mock_backend = MagicMock()
        mock_backend.serialize.return_value = {"nodes": nodes, "edges": edges}
        with patch("smartmemory_pkg.local_api._get_backend", return_value=mock_backend):
            body = client.get("/graph/full").json()
        assert body["node_count"] == 3
        assert body["edge_count"] == 1


# ---------------------------------------------------------------------------
# POST /graph/edges
# ---------------------------------------------------------------------------


class TestGetEdgesBulk:
    def test_returns_edges_key(self, client):
        mock_backend = MagicMock()
        mock_backend.get_edges_for_node.return_value = []
        with patch("smartmemory_pkg.local_api._get_backend", return_value=mock_backend):
            r = client.post("/graph/edges", json={"node_ids": ["node-1"]})
        assert r.status_code == 200
        assert "edges" in r.json()

    def test_deduplication_same_edge_returned_once(self, client):
        """Two node_ids sharing the same edge must return that edge exactly once."""
        shared_edge = _make_edge("node-A", "node-B", "related_to")

        def _get_edges(node_id):
            # Both node-A and node-B return the same shared_edge
            return [shared_edge]

        mock_backend = MagicMock()
        mock_backend.get_edges_for_node.side_effect = _get_edges
        with patch("smartmemory_pkg.local_api._get_backend", return_value=mock_backend):
            r = client.post("/graph/edges", json={"node_ids": ["node-A", "node-B"]})

        edges = r.json()["edges"]
        assert len(edges) == 1, f"Expected 1 deduplicated edge, got {len(edges)}"

    def test_distinct_edges_all_returned(self, client):
        """Two different edges are both included in the response."""
        edge_ab = _make_edge("node-A", "node-B", "related_to")
        edge_bc = _make_edge("node-B", "node-C", "related_to")

        def _get_edges(node_id):
            if node_id == "node-A":
                return [edge_ab]
            if node_id == "node-B":
                return [edge_bc]
            return []

        mock_backend = MagicMock()
        mock_backend.get_edges_for_node.side_effect = _get_edges
        with patch("smartmemory_pkg.local_api._get_backend", return_value=mock_backend):
            r = client.post("/graph/edges", json={"node_ids": ["node-A", "node-B"]})

        edges = r.json()["edges"]
        assert len(edges) == 2

    def test_empty_node_ids_returns_no_edges(self, client):
        mock_backend = MagicMock()
        mock_backend.get_edges_for_node.return_value = []
        with patch("smartmemory_pkg.local_api._get_backend", return_value=mock_backend):
            r = client.post("/graph/edges", json={"node_ids": []})
        assert r.json()["edges"] == []


# ---------------------------------------------------------------------------
# GET /list
# ---------------------------------------------------------------------------


class TestListMemories:
    def test_envelope_keys(self, client):
        mock_backend = MagicMock()
        mock_backend.serialize.return_value = {"nodes": [], "edges": []}
        with patch("smartmemory_pkg.local_api._get_backend", return_value=mock_backend):
            body = client.get("/list").json()
        assert "items" in body
        assert "total" in body
        assert "limit" in body
        assert "offset" in body

    def test_pagination(self, client):
        nodes = [_make_serialize_node(item_id=f"node-{i}") for i in range(5)]
        mock_backend = MagicMock()
        mock_backend.serialize.return_value = {"nodes": nodes, "edges": []}
        with patch("smartmemory_pkg.local_api._get_backend", return_value=mock_backend):
            body = client.get("/list?limit=2&offset=1").json()
        assert body["total"] == 5
        assert body["limit"] == 2
        assert body["offset"] == 1
        assert len(body["items"]) == 2


# ---------------------------------------------------------------------------
# GET /{memory_id}/neighbors
# ---------------------------------------------------------------------------


class TestGetNeighbors:
    def test_response_key_is_neighbors_not_nodes(self, client):
        """Response must use 'neighbors' key — adapter reads res?.neighbors || []."""
        mock_backend = MagicMock()
        mock_backend.get_neighbors.return_value = [_make_flat_node("node-2")]
        mock_backend.get_edges_for_node.return_value = [_make_edge("node-1", "node-2")]
        with patch("smartmemory_pkg.local_api._get_backend", return_value=mock_backend):
            r = client.get("/node-1/neighbors")
        assert r.status_code == 200
        body = r.json()
        assert "neighbors" in body, "Response must contain 'neighbors' key"
        assert "nodes" not in body, "Response must NOT use 'nodes' key"

    def test_edges_key_present(self, client):
        mock_backend = MagicMock()
        mock_backend.get_neighbors.return_value = []
        mock_backend.get_edges_for_node.return_value = []
        with patch("smartmemory_pkg.local_api._get_backend", return_value=mock_backend):
            body = client.get("/node-1/neighbors").json()
        assert "edges" in body

    def test_neighbors_are_not_flattened(self, client):
        """get_neighbors() output is already flat via _row_to_node() — confirm passthrough."""
        flat_neighbor = _make_flat_node("node-2")
        mock_backend = MagicMock()
        mock_backend.get_neighbors.return_value = [flat_neighbor]
        mock_backend.get_edges_for_node.return_value = []
        with patch("smartmemory_pkg.local_api._get_backend", return_value=mock_backend):
            body = client.get("/node-1/neighbors").json()
        # The flat node has item_id at top level — confirm it passes through unchanged
        neighbor = body["neighbors"][0]
        assert neighbor["item_id"] == "node-2"


# ---------------------------------------------------------------------------
# GET /{memory_id}
# ---------------------------------------------------------------------------


class TestGetMemoryItem:
    def test_returns_node_when_found(self, client):
        flat = _make_flat_node("node-1")
        mock_backend = MagicMock()
        mock_backend.get_node.return_value = flat
        with patch("smartmemory_pkg.local_api._get_backend", return_value=mock_backend):
            r = client.get("/node-1")
        assert r.status_code == 200
        assert r.json()["item_id"] == "node-1"

    def test_returns_404_when_not_found(self, client):
        mock_backend = MagicMock()
        mock_backend.get_node.return_value = None
        with patch("smartmemory_pkg.local_api._get_backend", return_value=mock_backend):
            r = client.get("/nonexistent-id")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# DELETE endpoints (Option B — 405 read-only)
# ---------------------------------------------------------------------------


class TestDeleteEndpoints:
    def test_delete_memory_node_returns_405(self, client):
        r = client.delete("/some-memory-id")
        assert r.status_code == 405

    def test_delete_entity_node_returns_405(self, client):
        r = client.delete("/graph/nodes/some-entity-id")
        assert r.status_code == 405


# ---------------------------------------------------------------------------
# Unconfigured → HTTP 503 (DIST-LITE-5: _get_mem() conversion)
# ---------------------------------------------------------------------------


class TestUnconfiguredReturns503:
    """_get_mem() must convert UnconfiguredError → HTTP 503 (not 500).

    Exercises the load-bearing behavior introduced by DIST-LITE-5: FastAPI
    would otherwise surface RuntimeError as 500 with no actionable message.
    """

    def test_graph_full_returns_503_when_unconfigured(self, client):
        from smartmemory_pkg.config import UnconfiguredError
        with patch(
            "smartmemory_pkg.local_api.get_memory",
            side_effect=UnconfiguredError("not configured"),
        ):
            r = client.get("/graph/full")
        assert r.status_code == 503
        assert "smartmemory setup" in r.json()["detail"].lower()

    def test_graph_edges_returns_503_when_unconfigured(self, client):
        from smartmemory_pkg.config import UnconfiguredError
        with patch(
            "smartmemory_pkg.local_api.get_memory",
            side_effect=UnconfiguredError("not configured"),
        ):
            r = client.post("/graph/edges", json={"node_ids": []})
        assert r.status_code == 503

    def test_memory_item_returns_503_when_unconfigured(self, client):
        from smartmemory_pkg.config import UnconfiguredError
        with patch(
            "smartmemory_pkg.local_api.get_memory",
            side_effect=UnconfiguredError("not configured"),
        ):
            r = client.get("/some-id")
        assert r.status_code == 503

    def test_graph_full_returns_400_on_invalid_mode_env_var(self, client):
        """SMARTMEMORY_MODE=<typo> raises ValueError → _get_mem() converts to HTTP 400."""
        with patch(
            "smartmemory_pkg.local_api.get_memory",
            side_effect=ValueError("Invalid SMARTMEMORY_MODE='remtoe'. Expected one of: local, remote"),
        ):
            r = client.get("/graph/full")
        assert r.status_code == 400
        assert "misconfigured" in r.json()["detail"].lower()
