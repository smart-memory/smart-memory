"""
Unit tests for Opinion/Observation synthesis evolvers.
"""

from smartmemory.models.memory_item import MemoryItem
from smartmemory.models.opinion import OpinionMetadata, ObservationMetadata, Disposition
from smartmemory.plugins.evolvers.opinion_synthesis import OpinionSynthesisEvolver, OpinionSynthesisConfig
from smartmemory.plugins.evolvers.observation_synthesis import ObservationSynthesisEvolver
from smartmemory.plugins.evolvers.opinion_reinforcement import OpinionReinforcementEvolver, OpinionReinforcementConfig


class TestOpinionMetadata:
    """Tests for OpinionMetadata model."""

    def test_initial_state(self):
        """Test initial opinion state."""
        meta = OpinionMetadata(confidence=0.7, subject="Python")

        assert meta.confidence == 0.7
        assert meta.reinforcement_count == 0
        assert meta.contradiction_count == 0
        assert meta.net_reinforcement == 0
        assert meta.stability == 0.5  # Neutral for new opinions

    def test_reinforce(self):
        """Test reinforcement increases confidence."""
        meta = OpinionMetadata(confidence=0.5, subject="Python")
        initial_confidence = meta.confidence

        meta.reinforce("evidence_1")

        assert meta.confidence > initial_confidence
        assert meta.reinforcement_count == 1
        assert "evidence_1" in meta.formed_from
        assert meta.last_reinforced_at is not None

    def test_contradict(self):
        """Test contradiction decreases confidence."""
        meta = OpinionMetadata(confidence=0.8, subject="Python")
        initial_confidence = meta.confidence

        meta.contradict("evidence_1")

        assert meta.confidence < initial_confidence
        assert meta.contradiction_count == 1
        assert "evidence_1" in meta.formed_from
        assert meta.last_contradicted_at is not None

    def test_stability_calculation(self):
        """Test stability reflects reinforcement ratio."""
        meta = OpinionMetadata(confidence=0.5, subject="Python")

        # 3 reinforcements, 1 contradiction = 75% stability
        meta.reinforce("e1")
        meta.reinforce("e2")
        meta.reinforce("e3")
        meta.contradict("e4")

        assert meta.stability == 0.75

    def test_serialization(self):
        """Test to_dict/from_dict roundtrip."""
        meta = OpinionMetadata(
            confidence=0.8,
            subject="functional programming",
            subject_type="preference",
        )
        meta.reinforce("evidence_1")

        data = meta.to_dict()
        restored = OpinionMetadata.from_dict(data)

        assert restored.confidence == meta.confidence
        assert restored.subject == meta.subject
        assert restored.reinforcement_count == meta.reinforcement_count


class TestObservationMetadata:
    """Tests for ObservationMetadata model."""

    def test_initial_state(self):
        """Test initial observation state."""
        meta = ObservationMetadata(
            entity_id="alice_123",
            entity_name="Alice",
            entity_type="person",
        )

        assert meta.entity_id == "alice_123"
        assert meta.completeness == 0.0
        assert len(meta.source_facts) == 0

    def test_add_source(self):
        """Test adding source facts updates completeness."""
        meta = ObservationMetadata(entity_id="alice_123")

        meta.add_source("fact_1", "career")
        meta.add_source("fact_2", "education")

        assert len(meta.source_facts) == 2
        assert "career" in meta.aspects_covered
        assert "education" in meta.aspects_covered
        assert meta.completeness > 0.0
        assert meta.last_updated_at is not None

    def test_serialization(self):
        """Test to_dict/from_dict roundtrip."""
        meta = ObservationMetadata(
            entity_id="alice_123",
            entity_name="Alice",
            entity_type="person",
        )
        meta.add_source("fact_1", "career")

        data = meta.to_dict()
        restored = ObservationMetadata.from_dict(data)

        assert restored.entity_id == meta.entity_id
        assert restored.entity_name == meta.entity_name
        assert len(restored.source_facts) == 1


class TestDisposition:
    """Tests for Disposition model."""

    def test_default_values(self):
        """Test default disposition is neutral."""
        disp = Disposition()

        assert disp.skepticism == 0.5
        assert disp.literalism == 0.5
        assert disp.empathy == 0.5

    def test_custom_values(self):
        """Test custom disposition values."""
        disp = Disposition(skepticism=0.8, literalism=0.3, empathy=0.9)

        assert disp.skepticism == 0.8
        assert disp.literalism == 0.3
        assert disp.empathy == 0.9

    def test_serialization(self):
        """Test to_dict/from_dict roundtrip."""
        disp = Disposition(skepticism=0.7, literalism=0.4, empathy=0.6)

        data = disp.to_dict()
        restored = Disposition.from_dict(data)

        assert restored.skepticism == disp.skepticism
        assert restored.literalism == disp.literalism
        assert restored.empathy == disp.empathy


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
