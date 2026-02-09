"""
Unit tests for DecisionConfidenceEvolver.

Tests metadata, config defaults, and evolve() behavior with mocked memory.
The evolve() method is complex (evidence matching, decay, retraction), so
we test the high-level flow and helper methods rather than deep integration.
"""

import pytest

pytestmark = pytest.mark.unit


from datetime import datetime, timedelta, timezone
from unittest.mock import Mock


from smartmemory.plugins.evolvers.decision_confidence import (
    DecisionConfidenceEvolver,
    DecisionConfidenceConfig,
    CONTRADICTION_SIGNALS,
)


class TestDecisionConfidenceMetadata:
    """Tests for plugin metadata."""

    def test_metadata_name(self):
        meta = DecisionConfidenceEvolver.metadata()
        assert meta.name == "decision_confidence"

    def test_metadata_plugin_type(self):
        meta = DecisionConfidenceEvolver.metadata()
        assert meta.plugin_type == "evolver"

    def test_metadata_version(self):
        meta = DecisionConfidenceEvolver.metadata()
        assert meta.version == "2.0.0"

    def test_metadata_tags_include_decision(self):
        meta = DecisionConfidenceEvolver.metadata()
        assert "decision" in meta.tags

    def test_metadata_tags_include_confidence(self):
        meta = DecisionConfidenceEvolver.metadata()
        assert "confidence" in meta.tags

    def test_metadata_tags_include_decay(self):
        meta = DecisionConfidenceEvolver.metadata()
        assert "decay" in meta.tags

    def test_metadata_tags_include_reinforcement(self):
        meta = DecisionConfidenceEvolver.metadata()
        assert "reinforcement" in meta.tags


class TestDecisionConfidenceConfig:
    """Tests for config dataclass defaults and custom values."""

    def test_default_min_confidence_threshold(self):
        config = DecisionConfidenceConfig()
        assert config.min_confidence_threshold == 0.1

    def test_default_decay_after_days(self):
        config = DecisionConfidenceConfig()
        assert config.decay_after_days == 30

    def test_default_decay_rate(self):
        config = DecisionConfidenceConfig()
        assert config.decay_rate == 0.05

    def test_default_enable_decay(self):
        config = DecisionConfidenceConfig()
        assert config.enable_decay is True

    def test_default_lookback_days(self):
        config = DecisionConfidenceConfig()
        assert config.lookback_days == 7

    def test_default_similarity_threshold(self):
        config = DecisionConfidenceConfig()
        assert config.similarity_threshold == 0.7

    def test_default_enable_reinforcement(self):
        config = DecisionConfidenceConfig()
        assert config.enable_reinforcement is True

    def test_custom_values(self):
        config = DecisionConfidenceConfig(
            min_confidence_threshold=0.2,
            decay_after_days=60,
            decay_rate=0.1,
            enable_decay=False,
            lookback_days=14,
            similarity_threshold=0.8,
            enable_reinforcement=False,
        )
        assert config.min_confidence_threshold == 0.2
        assert config.decay_after_days == 60
        assert config.decay_rate == 0.1
        assert config.enable_decay is False
        assert config.lookback_days == 14
        assert config.similarity_threshold == 0.8
        assert config.enable_reinforcement is False


class TestDecisionConfidenceInit:
    """Tests for evolver initialization."""

    def test_default_config_when_none(self):
        evolver = DecisionConfidenceEvolver()
        assert isinstance(evolver.config, DecisionConfidenceConfig)

    def test_custom_config(self):
        config = DecisionConfidenceConfig(decay_rate=0.2)
        evolver = DecisionConfidenceEvolver(config=config)
        assert evolver.config.decay_rate == 0.2


class TestDecisionConfidenceEvolve:
    """Tests for evolve() method with mocked memory."""

    def _make_evolver(self, **kwargs):
        """Create an evolver with optional config overrides."""
        config = DecisionConfidenceConfig(**kwargs)
        return DecisionConfidenceEvolver(config=config)

    def _make_memory(self, search_results=None):
        """Create a mock memory object."""
        memory = Mock()
        memory.search = Mock(return_value=search_results or [])
        memory.update_properties = Mock()
        return memory

    def test_no_active_decisions_does_nothing(self):
        """When no decisions exist, evolve should return early."""
        evolver = self._make_evolver()
        memory = self._make_memory(search_results=[])

        evolver.evolve(memory)

        memory.update_properties.assert_not_called()

    def test_calls_search_for_decisions(self):
        """Evolve should search for decision-type memories."""
        evolver = self._make_evolver()
        memory = self._make_memory(search_results=[])

        evolver.evolve(memory)

        memory.search.assert_called()
        # First call should be searching for decisions
        first_call = memory.search.call_args_list[0]
        assert (
            first_call.kwargs.get("memory_type") == "decision"
            or (len(first_call.args) >= 2 and first_call.args[1] == "decision")
            or first_call[1].get("memory_type") == "decision"
        )


class TestDecisionConfidenceShouldDecay:
    """Tests for _should_decay helper method."""

    def test_no_last_activity_should_decay(self):
        """If there's no last activity, the decision should decay."""
        evolver = DecisionConfidenceEvolver()
        assert evolver._should_decay(None, 30) is True

    def test_recent_activity_should_not_decay(self):
        """If last activity is recent, the decision should not decay."""
        evolver = DecisionConfidenceEvolver()
        recent = datetime.now(timezone.utc) - timedelta(days=5)
        assert evolver._should_decay(recent, 30) is False

    def test_old_activity_should_decay(self):
        """If last activity is old enough, the decision should decay."""
        evolver = DecisionConfidenceEvolver()
        old = datetime.now(timezone.utc) - timedelta(days=60)
        assert evolver._should_decay(old, 30) is True

    def test_exactly_at_threshold_should_not_decay(self):
        """Activity exactly at the threshold boundary should not decay (days > not >=)."""
        evolver = DecisionConfidenceEvolver()
        boundary = datetime.now(timezone.utc) - timedelta(days=30)
        assert evolver._should_decay(boundary, 30) is False


class TestDecisionConfidenceGetLastActivity:
    """Tests for _get_last_activity helper method."""

    def test_empty_metadata_returns_none(self):
        evolver = DecisionConfidenceEvolver()
        assert evolver._get_last_activity({}) is None

    def test_extracts_last_reinforced_at(self):
        evolver = DecisionConfidenceEvolver()
        now = datetime.now(timezone.utc)
        meta = {"last_reinforced_at": now}
        assert evolver._get_last_activity(meta) == now

    def test_extracts_from_iso_string(self):
        evolver = DecisionConfidenceEvolver()
        now = datetime.now(timezone.utc)
        meta = {"updated_at": now.isoformat()}
        result = evolver._get_last_activity(meta)
        assert result is not None

    def test_returns_most_recent_timestamp(self):
        evolver = DecisionConfidenceEvolver()
        old = datetime.now(timezone.utc) - timedelta(days=10)
        recent = datetime.now(timezone.utc) - timedelta(days=1)
        meta = {
            "last_reinforced_at": old,
            "last_contradicted_at": recent,
        }
        result = evolver._get_last_activity(meta)
        assert result == recent


class TestContradictionSignals:
    """Tests for the contradiction signal constants."""

    def test_signals_are_tuple(self):
        assert isinstance(CONTRADICTION_SIGNALS, tuple)

    def test_signals_contain_not(self):
        assert "not" in CONTRADICTION_SIGNALS

    def test_signals_contain_never(self):
        assert "never" in CONTRADICTION_SIGNALS

    def test_signals_contain_changed(self):
        assert "changed" in CONTRADICTION_SIGNALS

    def test_signals_contain_no_longer(self):
        assert "no longer" in CONTRADICTION_SIGNALS
