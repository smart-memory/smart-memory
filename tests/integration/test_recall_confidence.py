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
    """Items with confidence < 0.5 get a ~ prefix in recall output."""
    monkeypatch.setenv("SMARTMEMORY_DATA_DIR", str(tmp_path))
    # Set floor low enough to include the item
    monkeypatch.setenv("SMARTMEMORY_RECALL_FLOOR", "0.1")

    # hook:test → ceiling 0.4, which is < 0.5 → should get ~ marker
    storage_mod.ingest(
        "hook generated item with low confidence",
        properties={"origin": "hook:test"},
    )

    result = storage_mod.recall(top_k=5)
    assert "~[" in result, "Low-confidence items should have ~ prefix"
    assert "hook generated item with low confidence" in result


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
