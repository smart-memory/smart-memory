"""Unit tests for CORE-PROPS-1 Phase 2: stale ⚠ markers in recall and search output."""

from unittest.mock import MagicMock, patch

import smartmemory_app.storage as storage_mod


def _reset_singleton():
    storage_mod._memory = None
    storage_mod._remote_memory = None


def _make_item(content, memory_type="semantic", confidence=1.0, stale=False, origin="user:test"):
    """Create a mock MemoryItem for recall output tests.

    HOOK-RECALL-RELEVANCE-1 added origin tier filtering at recall time —
    items must have a real string .origin to survive `filter_by_tiers`.
    Default to a tier-1 origin so the existing tests keep their semantics.
    """
    item = MagicMock(spec=["item_id", "content", "memory_type", "confidence",
                            "stale", "origin", "metadata", "reference"])
    item.item_id = "test-" + content[:8]
    item.content = content
    item.memory_type = memory_type
    item.confidence = confidence
    item.stale = stale
    item.origin = origin
    item.metadata = {}
    item.reference = False
    return item


class TestRecallStaleMarker:
    def setup_method(self):
        _reset_singleton()

    def teardown_method(self):
        _reset_singleton()

    def test_stale_item_has_warning_marker(self, monkeypatch):
        """Stale items in recall output have ⚠ prefix."""
        monkeypatch.setenv("SMARTMEMORY_RECALL_FLOOR", "0.1")
        stale_item = _make_item("stale memory content", stale=True, confidence=0.8)
        mock_mem = MagicMock(spec=[])  # empty spec so isinstance(mock_mem, RemoteMemory) is False
        mock_mem.search = MagicMock(return_value=[stale_item])

        with patch("smartmemory_app.storage.get_memory", return_value=mock_mem):
            result = storage_mod.recall(top_k=5)

        assert "⚠" in result
        assert "stale memory content" in result

    def test_non_stale_item_no_warning(self, monkeypatch):
        """Non-stale items do NOT have ⚠ prefix."""
        monkeypatch.setenv("SMARTMEMORY_RECALL_FLOOR", "0.1")
        fresh_item = _make_item("fresh memory content", stale=False, confidence=0.8)
        mock_mem = MagicMock(spec=[])
        mock_mem.search = MagicMock(return_value=[fresh_item])

        with patch("smartmemory_app.storage.get_memory", return_value=mock_mem):
            result = storage_mod.recall(top_k=5)

        assert "⚠" not in result
        assert "fresh memory content" in result
