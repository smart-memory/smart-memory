"""
Unit tests for Opinion/Observation synthesis evolvers.
"""

from smartmemory.models.memory_item import MemoryItem
from smartmemory.models.opinion import OpinionMetadata
from smartmemory.plugins.evolvers.opinion_synthesis import OpinionSynthesisEvolver, OpinionSynthesisConfig
from smartmemory.plugins.evolvers.observation_synthesis import ObservationSynthesisEvolver
from smartmemory.plugins.evolvers.opinion_reinforcement import OpinionReinforcementEvolver, OpinionReinforcementConfig


class TestOpinionSynthesisEvolver:
    """Tests for OpinionSynthesisEvolver."""

    def test_metadata(self):
        """Test evolver metadata."""
        meta = OpinionSynthesisEvolver.metadata()

        assert meta.name == "opinion_synthesis"
        assert meta.plugin_type == "evolver"
        assert "synthesis" in meta.tags

    def test_config_defaults(self):
        """Test default configuration."""
        config = OpinionSynthesisConfig()

        assert config.min_pattern_occurrences == 3
        assert config.min_confidence == 0.5
        assert config.lookback_days == 30

    def test_pattern_detection_logic(self):
        """Test pattern detection from episodic items."""
        evolver = OpinionSynthesisEvolver()
        config = OpinionSynthesisConfig(min_pattern_occurrences=2)
        evolver.config = config

        # Create mock episodic items with repeated tags
        items = [
            MemoryItem(item_id="e1", content="Used Python", metadata={"tags": ["python", "coding"]}),
            MemoryItem(item_id="e2", content="Wrote Python script", metadata={"tags": ["python"]}),
            MemoryItem(item_id="e3", content="Python debugging", metadata={"tags": ["python", "debugging"]}),
        ]

        patterns = evolver._detect_patterns(items, config)

        # Should detect 'python' as a pattern (appears 3 times)
        python_patterns = [p for p in patterns if p["subject"] == "python"]
        assert len(python_patterns) > 0


class TestObservationSynthesisEvolver:
    """Tests for ObservationSynthesisEvolver."""

    def test_metadata(self):
        """Test evolver metadata."""
        meta = ObservationSynthesisEvolver.metadata()

        assert meta.name == "observation_synthesis"
        assert meta.plugin_type == "evolver"
        assert "entity" in meta.tags

    def test_aspect_detection(self):
        """Test aspect detection from facts."""
        evolver = ObservationSynthesisEvolver()

        facts = [
            MemoryItem(item_id="f1", content="Alice works at Google as an engineer"),
            MemoryItem(item_id="f2", content="Alice graduated from MIT"),
            MemoryItem(item_id="f3", content="Alice lives in San Francisco"),
        ]

        aspects = evolver._detect_aspects(facts)

        assert "career" in aspects  # 'works', 'engineer'
        assert "education" in aspects  # 'graduated'
        assert "location" in aspects  # 'lives'

    def test_simple_synthesis(self):
        """Test simple fact synthesis."""
        evolver = ObservationSynthesisEvolver()

        facts = ["Works at Google", "Graduated from MIT"]
        summary = evolver._simple_synthesis("Alice", facts)

        assert "Alice" in summary
        assert "Google" in summary or "MIT" in summary


class TestOpinionReinforcementEvolver:
    """Tests for OpinionReinforcementEvolver."""

    def test_metadata(self):
        """Test evolver metadata."""
        meta = OpinionReinforcementEvolver.metadata()

        assert meta.name == "opinion_reinforcement"
        assert meta.plugin_type == "evolver"
        assert "confidence" in meta.tags

    def test_config_defaults(self):
        """Test default configuration."""
        config = OpinionReinforcementConfig()

        assert config.min_confidence_threshold == 0.2
        assert config.enable_decay is True
        assert config.decay_rate == 0.05

    def test_decay_logic(self):
        """Test decay timing logic."""
        from datetime import datetime, timezone, timedelta

        evolver = OpinionReinforcementEvolver()
        config = OpinionReinforcementConfig(decay_after_days=30)
        evolver.config = config

        # Recent opinion - should not decay
        recent_meta = OpinionMetadata(confidence=0.7, subject="test")
        recent_meta.formed_at = datetime.now(timezone.utc)
        assert evolver._should_decay(recent_meta, 30) is False

        # Old opinion - should decay
        old_meta = OpinionMetadata(confidence=0.7, subject="test")
        old_meta.formed_at = datetime.now(timezone.utc) - timedelta(days=60)
        assert evolver._should_decay(old_meta, 30) is True
