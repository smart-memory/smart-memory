"""
Unit tests for EpisodicDecayEvolver.

Tests metadata, config defaults, and evolve() behavior with mocked memory.
"""

from unittest.mock import Mock

import pytest


pytestmark = pytest.mark.unit

from smartmemory.plugins.evolvers.episodic_decay import (
    EpisodicDecayEvolver,
    EpisodicDecayConfig,
)


class TestEpisodicDecayMetadata:
    """Tests for plugin metadata."""

    def test_metadata_name(self):
        meta = EpisodicDecayEvolver.metadata()
        assert meta.name == "episodic_decay"

    def test_metadata_plugin_type(self):
        meta = EpisodicDecayEvolver.metadata()
        assert meta.plugin_type == "evolver"

    def test_metadata_version(self):
        meta = EpisodicDecayEvolver.metadata()
        assert meta.version == "1.0.0"


class TestEpisodicDecayConfig:
    """Tests for config dataclass defaults and custom values."""

    def test_default_half_life(self):
        config = EpisodicDecayConfig()
        assert config.half_life == 30

    def test_custom_half_life(self):
        config = EpisodicDecayConfig(half_life=60)
        assert config.half_life == 60


class TestEpisodicDecayEvolve:
    """Tests for evolve() method with mocked memory."""

    def _make_evolver(self, half_life=30):
        """Create an evolver with typed config."""
        config = EpisodicDecayConfig(half_life=half_life)
        evolver = EpisodicDecayEvolver(config=config)
        return evolver

    def _make_memory(self, stale_events=None):
        """Create a mock memory with episodic sub-manager."""
        memory = Mock()
        memory.episodic = Mock()
        memory.episodic.get_stale_events = Mock(return_value=stale_events or [])
        memory.episodic.archive = Mock()
        return memory

    def test_no_stale_events_does_nothing(self):
        """When no stale episodic items exist, nothing is archived."""
        evolver = self._make_evolver()
        memory = self._make_memory(stale_events=[])

        evolver.evolve(memory)

        memory.episodic.get_stale_events.assert_called_once_with(half_life=30)
        memory.episodic.archive.assert_not_called()

    def test_archives_stale_events(self):
        """Stale episodic events should be archived."""
        evolver = self._make_evolver()
        stale = [Mock(content="old event A"), Mock(content="old event B")]
        memory = self._make_memory(stale_events=stale)

        evolver.evolve(memory)

        assert memory.episodic.archive.call_count == 2
        memory.episodic.archive.assert_any_call(stale[0])
        memory.episodic.archive.assert_any_call(stale[1])

    def test_does_not_archive_when_no_stale(self):
        """If get_stale_events returns empty, archive should not be called."""
        evolver = self._make_evolver(half_life=90)
        memory = self._make_memory(stale_events=[])

        evolver.evolve(memory)

        memory.episodic.archive.assert_not_called()

    def test_passes_half_life_to_query(self):
        """Custom half_life config should be forwarded to the stale events query."""
        evolver = self._make_evolver(half_life=15)
        memory = self._make_memory(stale_events=[])

        evolver.evolve(memory)

        memory.episodic.get_stale_events.assert_called_once_with(half_life=15)

    def test_logger_called_for_each_archived_event(self):
        """If a logger is provided, info() should be called for each archived event."""
        evolver = self._make_evolver()
        stale = [Mock(content="old event A"), Mock(content="old event B"), Mock(content="old event C")]
        memory = self._make_memory(stale_events=stale)
        logger = Mock()

        evolver.evolve(memory, logger=logger)

        assert logger.info.call_count == 3

    def test_no_logger_does_not_raise(self):
        """Evolve should work without a logger."""
        evolver = self._make_evolver()
        stale = [Mock(content="old event")]
        memory = self._make_memory(stale_events=stale)

        evolver.evolve(memory, logger=None)

    def test_raises_without_typed_config(self):
        """Evolve should raise TypeError if config lacks 'half_life' attribute."""
        evolver = EpisodicDecayEvolver(config={})
        memory = self._make_memory()

        with pytest.raises(TypeError, match="half_life"):
            evolver.evolve(memory)
