"""Unit tests for DecisionConfidenceEvolver."""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from smartmemory.plugins.evolvers.decision_confidence import (
    DecisionConfidenceConfig,
    DecisionConfidenceEvolver,
)


@pytest.fixture
def config():
    return DecisionConfidenceConfig()


@pytest.fixture
def evolver(config):
    return DecisionConfidenceEvolver(config=config)


@pytest.fixture
def mock_memory():
    memory = MagicMock()
    memory._graph = MagicMock()
    memory.search.return_value = []
    memory.update_properties.return_value = None
    return memory


def _make_decision_item(
    decision_id,
    content,
    confidence=0.8,
    status="active",
    reinforcement_count=0,
    contradiction_count=0,
    updated_at=None,
    last_reinforced_at=None,
):
    """Helper to create a mock MemoryItem representing a decision."""
    item = MagicMock()
    item.item_id = decision_id
    item.content = content
    item.embedding = None
    item.metadata = {
        "decision_id": decision_id,
        "content": content,
        "confidence": confidence,
        "status": status,
        "decision_type": "inference",
        "reinforcement_count": reinforcement_count,
        "contradiction_count": contradiction_count,
        "evidence_ids": [],
        "contradicting_ids": [],
        "updated_at": (updated_at or datetime.now(timezone.utc)).isoformat(),
        "last_reinforced_at": last_reinforced_at.isoformat() if last_reinforced_at else None,
    }
    return item


def _make_evidence_item(item_id, content, embedding=None):
    """Helper to create a mock evidence MemoryItem."""
    item = MagicMock()
    item.item_id = item_id
    item.content = content
    item.embedding = embedding
    return item


class TestMetadata:
    """Test plugin metadata."""

    def test_name(self):
        meta = DecisionConfidenceEvolver.metadata()
        assert meta.name == "decision_confidence"

    def test_plugin_type(self):
        meta = DecisionConfidenceEvolver.metadata()
        assert meta.plugin_type == "evolver"


class TestConfig:
    """Test configuration defaults."""

    def test_defaults(self):
        cfg = DecisionConfidenceConfig()
        assert cfg.min_confidence_threshold == 0.1
        assert cfg.decay_after_days == 30
        assert cfg.decay_rate == 0.05
        assert cfg.enable_decay is True

    def test_custom(self):
        cfg = DecisionConfidenceConfig(decay_rate=0.1, decay_after_days=14)
        assert cfg.decay_rate == 0.1
        assert cfg.decay_after_days == 14


class TestEvolveDecay:
    """Test confidence decay for stale decisions."""

    def test_decays_stale_decision(self, evolver, mock_memory):
        old_date = datetime.now(timezone.utc) - timedelta(days=60)
        item = _make_decision_item(
            "dec_stale", "Old decision", confidence=0.6,
            updated_at=old_date, last_reinforced_at=old_date,
        )
        mock_memory.search.return_value = [item]

        evolver.evolve(mock_memory)

        mock_memory.update_properties.assert_called()
        call_args = mock_memory.update_properties.call_args_list[0]
        props = call_args[0][1]
        assert props["confidence"] < 0.6

    def test_does_not_decay_recent_decision(self, evolver, mock_memory):
        recent_date = datetime.now(timezone.utc) - timedelta(days=5)
        item = _make_decision_item(
            "dec_recent", "Recent decision", confidence=0.8,
            updated_at=recent_date, last_reinforced_at=recent_date,
        )
        mock_memory.search.return_value = [item]

        evolver.evolve(mock_memory)

        # Should not be updated if no decay needed and no evidence found
        if mock_memory.update_properties.called:
            call_args = mock_memory.update_properties.call_args_list[0]
            props = call_args[0][1]
            assert props["confidence"] == 0.8

    def test_decay_disabled(self, mock_memory):
        cfg = DecisionConfidenceConfig(enable_decay=False)
        evolver = DecisionConfidenceEvolver(config=cfg)
        old_date = datetime.now(timezone.utc) - timedelta(days=60)
        item = _make_decision_item(
            "dec_old", "Old", confidence=0.6,
            updated_at=old_date, last_reinforced_at=old_date,
        )
        mock_memory.search.return_value = [item]

        evolver.evolve(mock_memory)

        # Should not update if decay is disabled and no evidence
        if mock_memory.update_properties.called:
            call_args = mock_memory.update_properties.call_args_list[0]
            props = call_args[0][1]
            assert props["confidence"] == 0.6

    def test_confidence_floors_at_zero(self, mock_memory):
        cfg = DecisionConfidenceConfig(decay_rate=0.5, decay_after_days=1)
        evolver = DecisionConfidenceEvolver(config=cfg)
        old_date = datetime.now(timezone.utc) - timedelta(days=10)
        item = _make_decision_item(
            "dec_low", "Low confidence", confidence=0.05,
            updated_at=old_date, last_reinforced_at=old_date,
        )
        mock_memory.search.return_value = [item]

        evolver.evolve(mock_memory)

        mock_memory.update_properties.assert_called()
        call_args = mock_memory.update_properties.call_args_list[0]
        props = call_args[0][1]
        assert props["confidence"] >= 0.0


class TestEvolveRetract:
    """Test retraction of low-confidence decisions."""

    def test_retracts_below_threshold(self, mock_memory):
        cfg = DecisionConfidenceConfig(min_confidence_threshold=0.2, decay_rate=0.5, decay_after_days=1)
        evolver = DecisionConfidenceEvolver(config=cfg)
        old_date = datetime.now(timezone.utc) - timedelta(days=10)
        item = _make_decision_item(
            "dec_weak", "Weak decision", confidence=0.15,
            updated_at=old_date, last_reinforced_at=old_date,
        )
        mock_memory.search.return_value = [item]

        evolver.evolve(mock_memory)

        mock_memory.update_properties.assert_called()
        # Find the call that sets status to retracted
        retract_calls = [
            c for c in mock_memory.update_properties.call_args_list
            if c[0][1].get("status") == "retracted"
        ]
        assert len(retract_calls) >= 1

    def test_does_not_retract_above_threshold(self, evolver, mock_memory):
        recent = datetime.now(timezone.utc) - timedelta(days=5)
        item = _make_decision_item(
            "dec_ok", "Good decision", confidence=0.8,
            updated_at=recent, last_reinforced_at=recent,
        )
        mock_memory.search.return_value = [item]

        evolver.evolve(mock_memory)

        retract_calls = [
            c for c in mock_memory.update_properties.call_args_list
            if c[0][1].get("status") == "retracted"
        ]
        assert len(retract_calls) == 0


class TestEvolveSkips:
    """Test that evolver skips non-active decisions."""

    def test_skips_superseded(self, evolver, mock_memory):
        item = _make_decision_item("dec_old", "Superseded", status="superseded")
        mock_memory.search.return_value = [item]

        evolver.evolve(mock_memory)

        # Should not update superseded decisions
        assert not mock_memory.update_properties.called

    def test_skips_retracted(self, evolver, mock_memory):
        item = _make_decision_item("dec_ret", "Retracted", status="retracted")
        mock_memory.search.return_value = [item]

        evolver.evolve(mock_memory)

        assert not mock_memory.update_properties.called

    def test_handles_empty_search(self, evolver, mock_memory):
        mock_memory.search.return_value = []
        evolver.evolve(mock_memory)
        assert not mock_memory.update_properties.called

    def test_handles_search_error(self, evolver, mock_memory):
        mock_memory.search.side_effect = Exception("Search failed")
        evolver.evolve(mock_memory)  # Should not raise
        assert not mock_memory.update_properties.called


class TestShouldDecay:
    """Test the staleness check."""

    def test_stale_returns_true(self, evolver):
        old = datetime.now(timezone.utc) - timedelta(days=60)
        assert evolver._should_decay(old, 30) is True

    def test_recent_returns_false(self, evolver):
        recent = datetime.now(timezone.utc) - timedelta(days=5)
        assert evolver._should_decay(recent, 30) is False

    def test_none_returns_true(self, evolver):
        assert evolver._should_decay(None, 30) is True

    def test_exact_boundary(self, evolver):
        boundary = datetime.now(timezone.utc) - timedelta(days=30)
        # At exactly the boundary, should not decay (> not >=)
        assert evolver._should_decay(boundary, 30) is False
