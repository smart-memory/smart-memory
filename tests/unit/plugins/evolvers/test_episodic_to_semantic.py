"""
Unit tests for EpisodicToSemanticEvolver.

Tests metadata, config defaults, and evolve() behavior with mocked memory.
"""

from unittest.mock import Mock

import pytest


pytestmark = pytest.mark.unit

from smartmemory.plugins.evolvers.episodic_to_semantic import (
    EpisodicToSemanticEvolver,
    EpisodicToSemanticConfig,
)


class TestEpisodicToSemanticMetadata:
    """Tests for plugin metadata."""

    def test_metadata_name(self):
        meta = EpisodicToSemanticEvolver.metadata()
        assert meta.name == "episodic_to_semantic"

    def test_metadata_plugin_type(self):
        meta = EpisodicToSemanticEvolver.metadata()
        assert meta.plugin_type == "evolver"

    def test_metadata_version(self):
        meta = EpisodicToSemanticEvolver.metadata()
        assert meta.version == "1.0.0"


class TestEpisodicToSemanticConfig:
    """Tests for config dataclass defaults and custom values."""

    def test_default_confidence(self):
        config = EpisodicToSemanticConfig()
        assert config.confidence == 0.9

    def test_default_days(self):
        config = EpisodicToSemanticConfig()
        assert config.days == 3

    def test_custom_confidence(self):
        config = EpisodicToSemanticConfig(confidence=0.75)
        assert config.confidence == 0.75

    def test_custom_days(self):
        config = EpisodicToSemanticConfig(days=7)
        assert config.days == 7

    def test_custom_both(self):
        config = EpisodicToSemanticConfig(confidence=0.6, days=14)
        assert config.confidence == 0.6
        assert config.days == 14


class TestEpisodicToSemanticEvolve:
    """Tests for evolve() method with mocked memory."""

    def _make_evolver(self, confidence=0.9, days=3):
        """Create an evolver with typed config."""
        config = EpisodicToSemanticConfig(confidence=confidence, days=days)
        evolver = EpisodicToSemanticEvolver(config=config)
        return evolver

    def _make_memory(self, stable_events=None):
        """Create a mock memory object with episodic and semantic sub-managers."""
        memory = Mock()
        memory.episodic = Mock()
        memory.semantic = Mock()
        memory.episodic.get_stable_events = Mock(return_value=stable_events or [])
        memory.semantic.add = Mock()
        memory.episodic.archive = Mock()
        return memory

    def test_no_stable_events_does_nothing(self):
        """When no stable episodic items exist, nothing is promoted or archived."""
        evolver = self._make_evolver()
        memory = self._make_memory(stable_events=[])

        evolver.evolve(memory)

        memory.episodic.get_stable_events.assert_called_once_with(confidence=0.9, min_days=3)
        memory.semantic.add.assert_not_called()
        memory.episodic.archive.assert_not_called()

    def test_promotes_stable_events_to_semantic(self):
        """Stable episodic events should be added to semantic memory."""
        evolver = self._make_evolver()
        events = [Mock(content="fact A"), Mock(content="fact B")]
        memory = self._make_memory(stable_events=events)

        evolver.evolve(memory)

        assert memory.semantic.add.call_count == 2
        memory.semantic.add.assert_any_call(events[0])
        memory.semantic.add.assert_any_call(events[1])

    def test_archives_promoted_episodic_events(self):
        """After promotion, the original episodic events should be archived."""
        evolver = self._make_evolver()
        events = [Mock(content="fact A"), Mock(content="fact B")]
        memory = self._make_memory(stable_events=events)

        evolver.evolve(memory)

        assert memory.episodic.archive.call_count == 2
        memory.episodic.archive.assert_any_call(events[0])
        memory.episodic.archive.assert_any_call(events[1])

    def test_passes_config_values_to_get_stable_events(self):
        """Custom config values should be forwarded to the memory query."""
        evolver = self._make_evolver(confidence=0.7, days=10)
        memory = self._make_memory(stable_events=[])

        evolver.evolve(memory)

        memory.episodic.get_stable_events.assert_called_once_with(confidence=0.7, min_days=10)

    def test_logger_called_on_promotion(self):
        """If a logger is provided, info() should be called for each promoted event."""
        evolver = self._make_evolver()
        events = [Mock(content="fact A")]
        memory = self._make_memory(stable_events=events)
        logger = Mock()

        evolver.evolve(memory, logger=logger)

        assert logger.info.call_count == 1

    def test_no_logger_does_not_raise(self):
        """Evolve should work without a logger (logger=None)."""
        evolver = self._make_evolver()
        events = [Mock(content="fact A")]
        memory = self._make_memory(stable_events=events)

        # Should not raise
        evolver.evolve(memory, logger=None)

    def test_raises_without_typed_config(self):
        """Evolve should raise TypeError if config lacks required attributes."""
        evolver = EpisodicToSemanticEvolver(config={})
        memory = self._make_memory()

        with pytest.raises(TypeError, match="confidence"):
            evolver.evolve(memory)
