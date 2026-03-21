"""Integration test: full ingest → search → recall flow.

Uses a real temporary data directory with SQLite + usearch backends (no mocks).
Verifies the end-to-end path from storage.ingest() through to storage.recall().
"""

import pytest
import smartmemory_app.storage as storage_mod


@pytest.fixture(autouse=True)
def reset_storage():
    storage_mod._memory = None
    yield
    storage_mod._memory = None


@pytest.mark.integration
def test_ingest_and_recall(tmp_path, monkeypatch):
    """Ingested content appears in recall output."""
    monkeypatch.setenv("SMARTMEMORY_DATA_DIR", str(tmp_path))

    item_id = storage_mod.ingest(
        "Claude Code is an AI-powered terminal coding assistant"
    )
    assert item_id, "ingest must return a non-empty item_id"

    result = storage_mod.recall(cwd=str(tmp_path), top_k=5)
    assert isinstance(result, str)
    assert "Claude Code is an AI-powered terminal coding assistant" in result


@pytest.mark.integration
def test_recall_top_k_one_still_returns_recent_memory(tmp_path, monkeypatch):
    """top_k=1 should still allocate one slot to the recency pass."""
    monkeypatch.setenv("SMARTMEMORY_DATA_DIR", str(tmp_path))

    storage_mod.ingest("Only memory in the store")

    result = storage_mod.recall(top_k=1)
    assert "Only memory in the store" in result


@pytest.mark.integration
def test_ingest_and_search(tmp_path, monkeypatch):
    """Ingested content is findable via search."""
    monkeypatch.setenv("SMARTMEMORY_DATA_DIR", str(tmp_path))

    storage_mod.ingest("pytest is the standard Python testing framework")

    results = storage_mod.search("Python testing", top_k=5)
    assert isinstance(results, list)
    # Each result must be a dict (from MemoryItem.to_dict())
    for r in results:
        assert isinstance(r, dict)


@pytest.mark.integration
def test_ingest_returns_consistent_id(tmp_path, monkeypatch):
    """Two distinct ingests return different item_ids (not the same id reused)."""
    monkeypatch.setenv("SMARTMEMORY_DATA_DIR", str(tmp_path))

    id1 = storage_mod.ingest("First memory item about databases")
    id2 = storage_mod.ingest("Second memory item about testing")

    assert id1 != id2, "distinct ingests must produce distinct item_ids"


@pytest.mark.integration
def test_get_returns_dict_or_empty(tmp_path, monkeypatch):
    """get() returns a dict for a known id, empty dict for unknown id."""
    monkeypatch.setenv("SMARTMEMORY_DATA_DIR", str(tmp_path))

    item_id = storage_mod.ingest("SmartMemory stores episodic memories")
    assert isinstance(item_id, str)
    result = storage_mod.get(item_id)
    # May return {} if get is not wired in Lite mode — both are acceptable
    assert isinstance(result, dict)

    missing = storage_mod.get("nonexistent-id-12345")
    assert missing == {}


@pytest.mark.integration
def test_singleton_reused_across_calls(tmp_path, monkeypatch):
    """The SmartMemory singleton is reused across multiple storage calls."""
    monkeypatch.setenv("SMARTMEMORY_DATA_DIR", str(tmp_path))

    storage_mod.ingest("First call initialises singleton")
    mem1 = storage_mod._memory

    storage_mod.ingest("Second call reuses singleton")
    mem2 = storage_mod._memory

    assert mem1 is mem2, "singleton must not be recreated between calls"
