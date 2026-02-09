"""Unit tests for MonitoringManager."""

from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.unit

from smartmemory.managers.monitoring import MonitoringManager


@pytest.fixture
def mock_monitoring():
    m = MagicMock()
    m.summary.return_value = {"total_items": 42, "types": {"semantic": 20, "episodic": 22}}
    m.orphaned_notes.return_value = ["note_1", "note_2"]
    m.prune.return_value = {"pruned": 5}
    m.find_old_notes.return_value = ["old_1", "old_2", "old_3"]
    m.self_monitor.return_value = {"health": "ok", "warnings": []}
    m.reflect.return_value = {"insights": ["pattern_1"], "top_items": []}
    m.summarize.return_value = {"summary": "All good", "item_count": 10}
    return m


@pytest.fixture
def manager(mock_monitoring):
    return MonitoringManager(mock_monitoring)


class TestSummary:
    def test_delegates_to_monitoring(self, manager, mock_monitoring):
        result = manager.summary()
        mock_monitoring.summary.assert_called_once()
        assert result["total_items"] == 42

    def test_propagates_error(self, manager, mock_monitoring):
        mock_monitoring.summary.side_effect = RuntimeError("DB unavailable")
        with pytest.raises(RuntimeError, match="DB unavailable"):
            manager.summary()


class TestOrphanedNotes:
    def test_returns_orphan_list(self, manager, mock_monitoring):
        result = manager.orphaned_notes()
        mock_monitoring.orphaned_notes.assert_called_once()
        assert len(result) == 2

    def test_empty_orphans(self, manager, mock_monitoring):
        mock_monitoring.orphaned_notes.return_value = []
        assert manager.orphaned_notes() == []


class TestPrune:
    def test_default_strategy(self, manager, mock_monitoring):
        result = manager.prune()
        mock_monitoring.prune.assert_called_once_with("old", 365)
        assert result["pruned"] == 5

    def test_custom_strategy_and_days(self, manager, mock_monitoring):
        manager.prune(strategy="unused", days=30)
        mock_monitoring.prune.assert_called_once_with("unused", 30)

    def test_extra_kwargs_forwarded(self, manager, mock_monitoring):
        manager.prune(strategy="old", days=90, dry_run=True)
        mock_monitoring.prune.assert_called_once_with("old", 90, dry_run=True)


class TestFindOldNotes:
    def test_default_days(self, manager, mock_monitoring):
        result = manager.find_old_notes()
        mock_monitoring.find_old_notes.assert_called_once_with(365)
        assert len(result) == 3

    def test_custom_days(self, manager, mock_monitoring):
        manager.find_old_notes(days=7)
        mock_monitoring.find_old_notes.assert_called_once_with(7)


class TestSelfMonitor:
    def test_delegates(self, manager, mock_monitoring):
        result = manager.self_monitor()
        mock_monitoring.self_monitor.assert_called_once()
        assert result["health"] == "ok"


class TestReflect:
    def test_default_top_k(self, manager, mock_monitoring):
        result = manager.reflect()
        mock_monitoring.reflect.assert_called_once_with(5)
        assert "insights" in result

    def test_custom_top_k(self, manager, mock_monitoring):
        manager.reflect(top_k=10)
        mock_monitoring.reflect.assert_called_once_with(10)


class TestSummarize:
    def test_default_max_items(self, manager, mock_monitoring):
        result = manager.summarize()
        mock_monitoring.summarize.assert_called_once_with(10)
        assert result["summary"] == "All good"

    def test_custom_max_items(self, manager, mock_monitoring):
        manager.summarize(max_items=50)
        mock_monitoring.summarize.assert_called_once_with(50)
