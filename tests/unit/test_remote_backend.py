"""Unit tests for smartmemory_app.remote_backend — DIST-LITE-5.

Tests cover the critical behaviors identified in the coverage sweep:
  - ingest() returns "Error: ..." string when _request() fails (not raises)
  - search() returns [{"error": ...}] list when _request() fails
  - get_neighbors() normalizes response to always include "edges" key
  - recall() deduplication handles both "item_id" and "id" field names

All tests mock _request() to avoid network calls.
"""
from unittest.mock import MagicMock, patch

import pytest

from smartmemory_app.remote_backend import RemoteMemory


@pytest.fixture()
def remote(monkeypatch):
    """RemoteMemory instance with keyring and bootstrap suppressed."""
    monkeypatch.setenv("SMARTMEMORY_API_KEY", "sk_test")
    r = RemoteMemory(api_url="https://api.example.com", team_id="t1")
    r._bootstrapped = True  # suppress _bootstrap() network call
    return r


# ── ingest ────────────────────────────────────────────────────────────────


def test_ingest_returns_item_id_on_success(remote):
    with patch.object(remote, "_request", return_value={"item_id": "abc-123"}):
        result = remote.ingest("hello world")
    assert result == "abc-123"


def test_ingest_returns_error_string_on_failure(remote):
    """ingest() must return 'Error: ...' string — not raise — when API fails."""
    with patch.object(remote, "_request", return_value={"error": "upstream timeout"}):
        result = remote.ingest("hello world")
    assert result.startswith("Error:")
    assert "upstream timeout" in result


# ── search ────────────────────────────────────────────────────────────────


def test_search_returns_list_of_dicts_on_success(remote):
    items = [{"item_id": "x", "content": "memory"}, {"item_id": "y", "content": "other"}]
    with patch.object(remote, "_request", return_value=items):
        result = remote.search("query")
    assert result == items


def test_search_returns_error_list_on_failure(remote):
    """search() must return [{"error": ...}] — not raise — when API fails."""
    with patch.object(remote, "_request", return_value={"error": "rate limited"}):
        result = remote.search("query")
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["error"] == "rate limited"


def test_search_returns_empty_list_on_none_response(remote):
    """search() returns [] when _request() returns None (e.g. 204 No Content)."""
    with patch.object(remote, "_request", return_value=None):
        result = remote.search("query")
    assert result == []


# ── get_neighbors ─────────────────────────────────────────────────────────


def test_get_neighbors_injects_edges_key_when_absent(remote):
    """Service (links.py) returns {neighbors, item_id} — no 'edges' key.

    get_neighbors() must normalize so the viewer always receives 'edges'.
    """
    service_response = {
        "neighbors": [{"item_id": "n1", "content": "neighbor"}],
        "item_id": "parent-id",
    }
    with patch.object(remote, "_request", return_value=service_response):
        result = remote.get_neighbors("parent-id")
    assert "edges" in result, "get_neighbors() must inject empty 'edges' key when absent"
    assert result["edges"] == []
    assert result["neighbors"][0]["item_id"] == "n1"


def test_get_neighbors_preserves_edges_when_present(remote):
    """When service does return 'edges', they must be preserved unchanged."""
    service_response = {
        "neighbors": [],
        "edges": [{"source_id": "a", "target_id": "b"}],
    }
    with patch.object(remote, "_request", return_value=service_response):
        result = remote.get_neighbors("a")
    assert len(result["edges"]) == 1


def test_get_neighbors_returns_error_shape_on_failure(remote):
    with patch.object(remote, "_request", return_value={"error": "not found"}):
        result = remote.get_neighbors("missing-id")
    assert result["neighbors"] == []
    assert result["edges"] == []
    assert "error" in result


# ── recall ────────────────────────────────────────────────────────────────


def test_recall_deduplicates_on_item_id_field(remote):
    """recall() must deduplicate items that appear in both recent and semantic results."""
    item = {"item_id": "dup-1", "content": "duplicate", "memory_type": "semantic"}
    with patch.object(remote, "search", return_value=[item]):
        result = remote.recall(cwd="/project", top_k=10)
    # item appears in both calls but must appear only once in output
    assert result.count("duplicate") == 1


def test_recall_deduplicates_on_id_field(remote):
    """recall() must also handle responses where ID field is 'id' (not 'item_id')."""
    item = {"id": "dup-2", "content": "alt id field", "memory_type": "episodic"}
    with patch.object(remote, "search", return_value=[item]):
        result = remote.recall(cwd="/project", top_k=10)
    assert result.count("alt id field") == 1


def test_recall_returns_empty_string_when_no_results(remote):
    with patch.object(remote, "search", return_value=[]):
        result = remote.recall(cwd="/project")
    assert result == ""
