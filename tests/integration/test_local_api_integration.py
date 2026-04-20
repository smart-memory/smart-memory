"""Integration tests for DIST-LITE-4: local_api.py with real SQLiteBackend.

Uses a real SQLiteBackend(":memory:") via TestClient mounted at /memory.
No mocks — exercises the full stack from HTTP through SQLite.

Mounting pattern (mirrors viewer_server.py mount):
    app = FastAPI()
    app.mount("/memory", local_api)
    client = TestClient(app)
"""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from smartmemory_app.local_api import api as local_api


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def backend():
    """Real in-memory SQLiteBackend for each test."""
    from smartmemory.graph.backends.sqlite import SQLiteBackend
    b = SQLiteBackend(db_path=":memory:")
    yield b
    b.close()


@pytest.fixture()
def client(backend, monkeypatch):
    """TestClient with local_api mounted at /memory, backed by the in-memory SQLiteBackend."""
    import smartmemory_app.local_api as _mod
    monkeypatch.setattr(_mod, "_get_backend", lambda: backend)

    wrapper = FastAPI()
    wrapper.mount("/memory", local_api)
    return TestClient(wrapper)


def _add_node(backend, item_id: str, label: str, memory_type: str = "semantic") -> str:
    """Add a node to the backend and return its item_id."""
    backend.add_node(
        item_id=item_id,
        properties={"label": label, "content": f"Content for {label}"},
        memory_type=memory_type,
    )
    return item_id


def _add_edge(backend, source_id: str, target_id: str, edge_type: str = "related_to") -> None:
    backend.add_edge(
        source_id=source_id,
        target_id=target_id,
        edge_type=edge_type,
        properties={},
        memory_type="semantic",
    )


# ---------------------------------------------------------------------------
# GET /memory/graph/full
# ---------------------------------------------------------------------------


class TestGetGraphFullIntegration:
    def test_returns_200_on_empty_graph(self, client):
        r = client.get("/memory/graph/full")
        assert r.status_code == 200

    def test_empty_graph_returns_zero_counts(self, client):
        body = client.get("/memory/graph/full").json()
        assert body["node_count"] == 0
        assert body["edge_count"] == 0
        assert body["nodes"] == []
        assert body["edges"] == []

    def test_nodes_have_item_id_and_label_at_top_level(self, client, backend):
        _add_node(backend, "n1", "Test Memory", "episodic")
        body = client.get("/memory/graph/full").json()
        assert body["node_count"] == 1
        n = body["nodes"][0]
        assert n["item_id"] == "n1"
        assert n["label"] == "Test Memory"
        assert n["memory_type"] == "episodic"

    def test_nodes_have_no_nested_properties_key(self, client, backend):
        """_flatten_node() must hoist all viewer fields — no 'properties' key on result."""
        _add_node(backend, "n1", "Flat check")
        body = client.get("/memory/graph/full").json()
        n = body["nodes"][0]
        assert "properties" not in n, "Node must not contain nested 'properties' after flattening"

    def test_three_nodes_ingested(self, client, backend):
        for i in range(3):
            _add_node(backend, f"n{i}", f"Label {i}")
        body = client.get("/memory/graph/full").json()
        assert body["node_count"] == 3

    def test_edge_appears_in_full_graph(self, client, backend):
        _add_node(backend, "n1", "Node 1")
        _add_node(backend, "n2", "Node 2")
        _add_edge(backend, "n1", "n2")
        body = client.get("/memory/graph/full").json()
        assert body["edge_count"] == 1
        assert body["edges"][0]["source_id"] == "n1"
        assert body["edges"][0]["target_id"] == "n2"


# ---------------------------------------------------------------------------
# POST /memory/graph/edges
# ---------------------------------------------------------------------------


class TestGetEdgesBulkIntegration:
    def test_returns_200(self, client, backend):
        _add_node(backend, "n1", "Node 1")
        r = client.post("/memory/graph/edges", json={"node_ids": ["n1"]})
        assert r.status_code == 200
        assert "edges" in r.json()

    def test_empty_node_ids_returns_empty_edges(self, client):
        r = client.post("/memory/graph/edges", json={"node_ids": []})
        assert r.status_code == 200
        assert r.json()["edges"] == []

    def test_deduplication_shared_edge(self, client, backend):
        """Querying both endpoints of an edge returns that edge exactly once."""
        _add_node(backend, "n1", "Node 1")
        _add_node(backend, "n2", "Node 2")
        _add_edge(backend, "n1", "n2")

        r = client.post("/memory/graph/edges", json={"node_ids": ["n1", "n2"]})
        edges = r.json()["edges"]
        assert len(edges) == 1, f"Expected 1 deduplicated edge, got {len(edges)}"


# ---------------------------------------------------------------------------
# GET /memory/{id}
# ---------------------------------------------------------------------------


class TestGetMemoryItemIntegration:
    def test_returns_200_for_existing_node(self, client, backend):
        _add_node(backend, "existing-node", "Existing")
        r = client.get("/memory/existing-node")
        assert r.status_code == 200

    def test_returns_404_for_unknown_id(self, client):
        r = client.get("/memory/this-id-does-not-exist")
        assert r.status_code == 404

    def test_node_fields_correct(self, client, backend):
        _add_node(backend, "my-id", "My Label", "pending")
        body = client.get("/memory/my-id").json()
        assert body["item_id"] == "my-id"
        assert body["memory_type"] == "pending"
        # get_node() returns _row_to_node() output — label is spread to top level
        assert body["label"] == "My Label"


# ---------------------------------------------------------------------------
# GET /memory/{id}/neighbors
# ---------------------------------------------------------------------------


class TestGetNeighborsIntegration:
    def test_returns_200(self, client, backend):
        _add_node(backend, "n1", "Node 1")
        r = client.get("/memory/n1/neighbors")
        assert r.status_code == 200

    def test_response_contains_neighbors_key(self, client, backend):
        _add_node(backend, "n1", "Node 1")
        body = client.get("/memory/n1/neighbors").json()
        assert "neighbors" in body, "Response must contain 'neighbors' key"
        assert "nodes" not in body, "Response must NOT use 'nodes' key"

    def test_neighbors_includes_linked_node(self, client, backend):
        _add_node(backend, "n1", "Source")
        _add_node(backend, "n2", "Target")
        _add_edge(backend, "n1", "n2")

        body = client.get("/memory/n1/neighbors").json()
        neighbor_ids = [n["item_id"] for n in body["neighbors"]]
        assert "n2" in neighbor_ids

    def test_edges_key_present(self, client, backend):
        _add_node(backend, "n1", "Node 1")
        body = client.get("/memory/n1/neighbors").json()
        assert "edges" in body
