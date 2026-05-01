"""Integration tests: recall confidence floor and ~ marker.

Uses a real temporary data directory with SQLite + usearch backends (no mocks).
Verifies CORE-PROPS-1 Phase 1b: low-confidence items filtered from recall,
and ~ marker appears on items with confidence < 0.5.
"""

import pytest
import smartmemory_app.storage as storage_mod


@pytest.fixture(autouse=True)
def reset_storage():
    storage_mod._memory = None
    yield
    storage_mod._memory = None


@pytest.mark.integration
def test_recall_excludes_below_confidence_floor(tmp_path, monkeypatch):
    """Items with confidence below the floor are excluded from recall."""
    monkeypatch.setenv("SMARTMEMORY_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SMARTMEMORY_RECALL_FLOOR", "0.5")

    # Ingest with origin that gives low ceiling (unknown → 0.3)
    storage_mod.ingest(
        "low confidence item should be filtered",
        properties={"origin": "unknown"},
    )

    result = storage_mod.recall(top_k=5)
    assert "low confidence item should be filtered" not in result


@pytest.mark.integration
def test_recall_includes_above_confidence_floor(tmp_path, monkeypatch):
    """Items with confidence above the floor appear in recall."""
    monkeypatch.setenv("SMARTMEMORY_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SMARTMEMORY_RECALL_FLOOR", "0.3")

    # Ingest with origin that gives high ceiling (user → 1.0)
    storage_mod.ingest(
        "high confidence item should appear",
        properties={"origin": "user:cli:persist"},
    )

    result = storage_mod.recall(top_k=5)
    assert "high confidence item should appear" in result


@pytest.mark.integration
def test_recall_tilde_marker_on_low_confidence(tmp_path, monkeypatch):
    """Items with confidence < 0.5 get a ~ prefix in recall output.

    HOOK-RECALL-RELEVANCE-1 filters tier-4 origins (hook:*, structured:*) by
    default at recall time. Use origin="unknown" instead — it survives the
    tier filter (legacy untagged-data passthrough per origin_policy) and
    receives the 0.3 confidence ceiling per CORE-PROPS-1.
    """
    monkeypatch.setenv("SMARTMEMORY_DATA_DIR", str(tmp_path))
    # Set floor low enough to include the item
    monkeypatch.setenv("SMARTMEMORY_RECALL_FLOOR", "0.1")

    # unknown → ceiling 0.3, < 0.5 → ~ marker; survives tier filter as legacy
    storage_mod.ingest(
        "low confidence item with tilde marker",
        properties={"origin": "unknown"},
    )

    result = storage_mod.recall(top_k=5)
    assert "~[" in result, f"Low-confidence items should have ~ prefix; got: {result!r}"
    assert "low confidence item with tilde marker" in result


@pytest.mark.integration
def test_recall_no_tilde_on_high_confidence(tmp_path, monkeypatch):
    """Items with confidence >= 0.5 do NOT get a ~ prefix."""
    monkeypatch.setenv("SMARTMEMORY_DATA_DIR", str(tmp_path))

    storage_mod.ingest(
        "user authored high confidence memory",
        properties={"origin": "user:cli:persist"},
    )

    result = storage_mod.recall(top_k=5)
    # Should contain the content but no ~ marker
    assert "user authored high confidence memory" in result
    assert "~[" not in result
