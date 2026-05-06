"""DIST-OBSIDIAN-LITE-1 — daemon HTTP contract tests against SDK shapes.

Pins the request/response shapes the smartmemory-obsidian plugin (via
smartmemory-sdk-js) sends, so the daemon stays compatible with the SDK
contract used by the cloud service. Each test sends the exact body the
SDK produces and asserts the response is what the plugin parses.

All tests mock the SmartMemory layer to avoid filesystem/backend setup.
"""
import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from smartmemory_app.local_api import api


@pytest.fixture()
def client():
    return TestClient(api)


# Helper — make a plain mock SmartMemory instance that's NOT a RemoteMemory.
def _make_local_mem(**overrides):
    mem = MagicMock()
    # Default no-op behaviors — override per-test
    mem.update_properties.return_value = None
    mem.delete.return_value = True
    for k, v in overrides.items():
        setattr(mem, k, v)
    return mem


# ---------------------------------------------------------------------------
# Task 1 — POST /ingest accepts SDK contract
# ---------------------------------------------------------------------------


class TestIngestSDKContract:
    """SDK MemoryAPI.ingest body: {content, profile_name, extractor_name, context}."""

    def test_ingest_accepts_extractor_name(self, client, monkeypatch):
        """SDK sends extractor_name='llm'; daemon must accept and return {item_id}."""
        # Force no-LLM branch for determinism
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        with patch("smartmemory_app.storage.ingest", return_value="item-123"):
            r = client.post("/ingest", json={
                "content": "Hello world",
                "profile_name": None,
                "extractor_name": "llm",
                "context": {},
            })
        assert r.status_code == 200, r.text
        assert r.json() == {"item_id": "item-123"}

    def test_ingest_two_tier_with_llm(self, client, monkeypatch):
        """has_llm branch: enqueues to async queue, still returns {item_id}."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
        with patch("smartmemory_app.storage.ingest",
                   return_value={"item_id": "item-456", "entity_ids": {}, "queued": True}):
            r = client.post("/ingest", json={
                "content": "Hello",
                "extractor_name": "llm",
            })
        assert r.status_code == 200, r.text
        assert r.json() == {"item_id": "item-456"}

    def test_ingest_two_tier_without_llm(self, client, monkeypatch):
        """no-LLM branch: full sync pipeline, returns {item_id}."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        with patch("smartmemory_app.storage.ingest", return_value="item-789"):
            r = client.post("/ingest", json={
                "content": "Hello",
                "extractor_name": "llm",  # ignored without LLM key
            })
        assert r.status_code == 200, r.text
        assert r.json() == {"item_id": "item-789"}


# ---------------------------------------------------------------------------
# Task 2 — POST /search accepts SDK contract; returns {items: [...]}
# ---------------------------------------------------------------------------


class TestSearchSDKContract:
    """SDK MemoryAPI.search body: {query, top_k, enable_hybrid, memory_type?}."""

    def test_search_accepts_enable_hybrid_and_memory_type(self, client):
        with patch("smartmemory_app.storage.search", return_value=[{"item_id": "x"}]):
            r = client.post("/search", json={
                "query": "atlas",
                "top_k": 5,
                "enable_hybrid": True,
                "memory_type": "semantic",
            })
        assert r.status_code == 200, r.text
        body = r.json()
        assert body == {"items": [{"item_id": "x"}]}

    def test_search_response_is_object_not_bare_list(self, client):
        """DIST-OBSIDIAN-LITE-1: response must be {items: []} for SDK parity."""
        with patch("smartmemory_app.storage.search", return_value=[]):
            r = client.post("/search", json={"query": "q", "top_k": 5})
        body = r.json()
        assert isinstance(body, dict)
        assert "items" in body
        assert body["items"] == []

    def test_search_memory_type_folds_into_filters(self, client):
        captured = {}
        def _spy(query, top_k, filters=None):
            captured["filters"] = filters
            return []
        with patch("smartmemory_app.storage.search", side_effect=_spy):
            client.post("/search", json={
                "query": "q",
                "top_k": 5,
                "memory_type": "decision",
            })
        assert captured["filters"] is not None
        assert captured["filters"].get("memory_type") == "decision"

    def test_search_filters_param_still_works(self, client):
        captured = {}
        def _spy(query, top_k, filters=None):
            captured["filters"] = filters
            return []
        with patch("smartmemory_app.storage.search", side_effect=_spy):
            client.post("/search", json={
                "query": "q",
                "top_k": 5,
                "filters": {"project": "atlas"},
            })
        assert captured["filters"] == {"project": "atlas"}


# ---------------------------------------------------------------------------
# Task 3 — PATCH /{id}
# ---------------------------------------------------------------------------


class TestPatchEndpoint:
    """SDK MemoryAPI.update body: {content?, metadata?, properties?, write_mode?}."""

    def test_patch_content_only(self, client):
        mem = _make_local_mem()
        with patch("smartmemory_app.local_api._get_mem", return_value=mem), \
             patch("smartmemory_app.remote_backend.RemoteMemory", new=type("Stub", (), {})):
            r = client.patch("/some-id", json={"content": "new content"})
        assert r.status_code == 200, r.text
        assert r.json() == {"item_id": "some-id", "updated": True}
        # update_properties called with content folded into properties
        args, kwargs = mem.update_properties.call_args
        assert args[0] == "some-id"
        assert args[1].get("content") == "new content"

    def test_patch_metadata_flat_merge(self, client):
        mem = _make_local_mem()
        with patch("smartmemory_app.local_api._get_mem", return_value=mem), \
             patch("smartmemory_app.remote_backend.RemoteMemory", new=type("Stub", (), {})):
            r = client.patch("/some-id", json={"metadata": {"source_path": "vault/note.md"}})
        assert r.status_code == 200
        args, _kwargs = mem.update_properties.call_args
        assert args[1].get("source_path") == "vault/note.md"

    def test_patch_properties_direct(self, client):
        mem = _make_local_mem()
        with patch("smartmemory_app.local_api._get_mem", return_value=mem), \
             patch("smartmemory_app.remote_backend.RemoteMemory", new=type("Stub", (), {})):
            r = client.patch("/some-id", json={
                "properties": {"key": "val"},
                "content": "ignored when properties is set",
            })
        assert r.status_code == 200
        args, _kwargs = mem.update_properties.call_args
        # properties wins per CORE-CRUD-UPDATE-1; content shouldn't override
        assert args[1]["key"] == "val"
        # content folds in only if not already in properties (it isn't here)
        assert "content" in args[1]

    def test_patch_write_mode_replace(self, client):
        mem = _make_local_mem()
        with patch("smartmemory_app.local_api._get_mem", return_value=mem), \
             patch("smartmemory_app.remote_backend.RemoteMemory", new=type("Stub", (), {})):
            r = client.patch("/some-id", json={
                "properties": {"k": "v"},
                "write_mode": "replace",
            })
        assert r.status_code == 200
        _args, kwargs = mem.update_properties.call_args
        assert kwargs.get("write_mode") == "replace"

    def test_patch_404_on_unknown_id(self, client):
        mem = _make_local_mem()
        mem.update_properties.side_effect = ValueError("Node bad-id not found in graph.")
        with patch("smartmemory_app.local_api._get_mem", return_value=mem), \
             patch("smartmemory_app.remote_backend.RemoteMemory", new=type("Stub", (), {})):
            r = client.patch("/bad-id", json={"content": "x"})
        assert r.status_code == 404

    def test_patch_remote_mode_returns_501(self, client):
        from smartmemory_app.remote_backend import RemoteMemory
        remote = MagicMock(spec=RemoteMemory)
        with patch("smartmemory_app.local_api._get_mem", return_value=remote):
            r = client.patch("/some-id", json={"content": "x"})
        assert r.status_code == 501


# ---------------------------------------------------------------------------
# Task 4 — DELETE /{id} lifted from 405
# ---------------------------------------------------------------------------


class TestDeleteEndpoint:
    """Already partly covered by TestDeleteEndpoints in test_local_api.py;
    these add SDK-shape coverage for the lifted policy."""

    def test_delete_remote_mode_returns_501(self, client):
        from smartmemory_app.remote_backend import RemoteMemory
        remote = MagicMock(spec=RemoteMemory)
        with patch("smartmemory_app.local_api._get_mem", return_value=remote):
            r = client.delete("/some-id")
        assert r.status_code == 501

    def test_delete_uses_smart_memory_delete_not_backend(self, client):
        """Cascade requirement — must go through mem.delete() so vector store
        and Vec_* nodes are cleaned up. crud.py:288 docs the cascade."""
        mem = _make_local_mem()
        with patch("smartmemory_app.local_api._get_mem", return_value=mem), \
             patch("smartmemory_app.remote_backend.RemoteMemory", new=type("Stub", (), {})):
            r = client.delete("/some-id")
        assert r.status_code == 204
        mem.delete.assert_called_once_with("some-id")


# ---------------------------------------------------------------------------
# Task 5 — /neighbors direction enrichment
# ---------------------------------------------------------------------------


def _edge(src, tgt, et="RELATED"):
    return {
        "source_id": src, "target_id": tgt, "edge_type": et,
        "memory_type": "semantic", "valid_from": None, "valid_to": None,
        "created_at": "2026-04-30T00:00:00", "properties": {},
    }


class TestNeighborsDirection:
    def test_outgoing_supersedes(self, client):
        """A -[SUPERSEDES]-> B; query A; assert outgoing direction."""
        backend = MagicMock()
        backend.get_edges_for_node.return_value = [_edge("A", "B", "SUPERSEDES")]
        with patch("smartmemory_app.local_api._get_backend", return_value=backend):
            body = client.get("/A/neighbors").json()
        neighbors = body["neighbors"]
        assert len(neighbors) == 1
        assert neighbors[0] == {
            "item_id": "B", "link_type": "SUPERSEDES", "direction": "outgoing"
        }

    def test_incoming_supersedes(self, client):
        """Same edge, query B; assert incoming direction."""
        backend = MagicMock()
        backend.get_edges_for_node.return_value = [_edge("A", "B", "SUPERSEDES")]
        with patch("smartmemory_app.local_api._get_backend", return_value=backend):
            body = client.get("/B/neighbors").json()
        neighbors = body["neighbors"]
        assert len(neighbors) == 1
        assert neighbors[0]["direction"] == "incoming"
        assert neighbors[0]["item_id"] == "A"

    def test_dedupe_same_neighbor_same_link_same_direction(self, client):
        """Duplicate edges in storage shouldn't produce duplicate neighbors."""
        backend = MagicMock()
        backend.get_edges_for_node.return_value = [
            _edge("A", "B", "SUPERSEDES"),
            _edge("A", "B", "SUPERSEDES"),
        ]
        with patch("smartmemory_app.local_api._get_backend", return_value=backend):
            body = client.get("/A/neighbors").json()
        assert len(body["neighbors"]) == 1

    def test_self_loop_skipped(self, client):
        """An edge from A to A shouldn't yield a neighbor."""
        backend = MagicMock()
        backend.get_edges_for_node.return_value = [_edge("A", "A", "WEIRD")]
        with patch("smartmemory_app.local_api._get_backend", return_value=backend):
            body = client.get("/A/neighbors").json()
        assert body["neighbors"] == []

    def test_edges_passthrough(self, client):
        """Top-level `edges` field still present (backward-compat)."""
        edges = [_edge("A", "B"), _edge("C", "A")]
        backend = MagicMock()
        backend.get_edges_for_node.return_value = edges
        with patch("smartmemory_app.local_api._get_backend", return_value=backend):
            body = client.get("/A/neighbors").json()
        assert body["edges"] == edges

    def test_skips_malformed_edge_with_no_link_type(self, client):
        """Edges missing edge_type/link_type must not produce link_type:None
        entries that downstream clients can't classify."""
        backend = MagicMock()
        bad_edge = {
            "source_id": "A", "target_id": "B",
            "memory_type": "semantic", "valid_from": None, "valid_to": None,
            "created_at": "2026-04-30T00:00:00", "properties": {},
        }
        good_edge = _edge("A", "C", "RELATED")
        backend.get_edges_for_node.return_value = [bad_edge, good_edge]
        with patch("smartmemory_app.local_api._get_backend", return_value=backend):
            body = client.get("/A/neighbors").json()
        assert len(body["neighbors"]) == 1
        assert body["neighbors"][0]["item_id"] == "C"


# ---------------------------------------------------------------------------
# PATCH ValueError narrowing
# ---------------------------------------------------------------------------


class TestPatchValueErrorNarrowing:
    def test_validation_value_error_bubbles_not_404(self):
        """ValueErrors with non-'not found' messages must not be mis-attributed
        to a missing item — they should propagate (not become 404)."""
        # Use a fresh client with raise_server_exceptions=False so unhandled
        # exceptions surface as 500 rather than re-raising in the test.
        local_client = TestClient(api, raise_server_exceptions=False)
        mem = _make_local_mem()
        mem.update_properties.side_effect = ValueError("write_mode must be merge or replace")
        with patch("smartmemory_app.local_api._get_mem", return_value=mem), \
             patch("smartmemory_app.remote_backend.RemoteMemory", new=type("Stub", (), {})):
            r = local_client.patch("/some-id", json={"content": "x"})
        # Whatever the surface code is, it must NOT be 404 (which is reserved
        # for the explicit "not found" path).
        assert r.status_code != 404
        assert r.status_code >= 500


# ---------------------------------------------------------------------------
# Task 6 — /health capability block
#
# This is on the root app (viewer_server.py), not /memory — covered by a
# separate integration test that boots the full app. Here we only assert the
# shape contract via a unit-style construction.
# ---------------------------------------------------------------------------


class TestHealthCapabilityShape:
    """We do not boot viewer_server here — capability block is fixed-shape
    so a contract assertion against the expected dict is enough. The full
    end-to-end /health response is exercised by daemon E2E."""

    def test_capability_keys_present(self):
        # This is the contract the plugin reads; pin it.
        expected_capability_keys = {
            "delete", "patch", "neighbors_direction", "quota", "auth",
        }
        from smartmemory_app import viewer_server
        # Sanity: the source code references all expected keys
        src = open(viewer_server.__file__).read()
        for k in expected_capability_keys:
            assert f'"{k}"' in src, f"viewer_server /health missing capability key: {k}"
        # And the mode field
        assert '"mode"' in src
