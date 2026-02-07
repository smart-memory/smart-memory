"""Unit tests for Opinion, Observation, and Disposition models."""

import pytest

from smartmemory.models.opinion import OpinionMetadata, ObservationMetadata, Disposition


class TestOpinionMetadataCreation:
    def test_creation_with_confidence(self):
        op = OpinionMetadata(confidence=0.7)
        assert op.confidence == 0.7
        assert op.formed_from == []
        assert op.reinforcement_count == 0
        assert op.contradiction_count == 0
        assert op.last_reinforced_at is None
        assert op.last_contradicted_at is None
        assert op.disposition is None
        assert op.subject is None
        assert op.subject_type is None


class TestOpinionReinforce:
    def test_increases_confidence(self):
        op = OpinionMetadata(confidence=0.5)
        op.reinforce("ev-1")
        # 0.5 + (1 - 0.5) * 0.1 = 0.55
        assert op.confidence == pytest.approx(0.55)

    def test_increments_reinforcement_count(self):
        op = OpinionMetadata(confidence=0.5)
        op.reinforce("ev-1")
        op.reinforce("ev-2")
        assert op.reinforcement_count == 2

    def test_records_evidence_in_formed_from(self):
        op = OpinionMetadata(confidence=0.5)
        op.reinforce("ev-1")
        assert "ev-1" in op.formed_from

    def test_deduplicates_evidence(self):
        op = OpinionMetadata(confidence=0.5)
        op.reinforce("ev-1")
        op.reinforce("ev-1")
        assert op.formed_from.count("ev-1") == 1
        assert op.reinforcement_count == 2

    def test_diminishing_returns(self):
        op = OpinionMetadata(confidence=0.9)
        op.reinforce("ev-1")
        # 0.9 + (1 - 0.9) * 0.1 = 0.91
        assert op.confidence == pytest.approx(0.91)

    def test_caps_at_one(self):
        op = OpinionMetadata(confidence=0.99)
        op.reinforce("ev-1")
        assert op.confidence <= 1.0

    def test_sets_last_reinforced_at(self):
        op = OpinionMetadata(confidence=0.5)
        assert op.last_reinforced_at is None
        op.reinforce("ev-1")
        assert op.last_reinforced_at is not None


class TestOpinionContradict:
    def test_decreases_confidence(self):
        op = OpinionMetadata(confidence=0.8)
        op.contradict("c-1")
        # 0.8 - 0.8 * 0.15 = 0.68
        assert op.confidence == pytest.approx(0.68)

    def test_increments_contradiction_count(self):
        op = OpinionMetadata(confidence=0.8)
        op.contradict("c-1")
        assert op.contradiction_count == 1

    def test_floors_at_zero(self):
        op = OpinionMetadata(confidence=0.01)
        op.contradict("c-1")
        assert op.confidence >= 0.0

    def test_records_evidence_in_formed_from(self):
        op = OpinionMetadata(confidence=0.5)
        op.contradict("c-1")
        assert "c-1" in op.formed_from

    def test_sets_last_contradicted_at(self):
        op = OpinionMetadata(confidence=0.5)
        assert op.last_contradicted_at is None
        op.contradict("c-1")
        assert op.last_contradicted_at is not None


class TestOpinionProperties:
    def test_net_reinforcement(self):
        op = OpinionMetadata(confidence=0.5)
        op.reinforce("e1")
        op.reinforce("e2")
        op.contradict("c1")
        assert op.net_reinforcement == 1  # 2 - 1

    def test_stability_neutral_for_new(self):
        op = OpinionMetadata(confidence=0.5)
        assert op.stability == 0.5

    def test_stability_computed_when_counts_positive(self):
        op = OpinionMetadata(confidence=0.5)
        op.reinforce("e1")
        op.reinforce("e2")
        op.contradict("c1")
        # 2 / (2 + 1) = 0.6667
        assert op.stability == pytest.approx(2 / 3)

    def test_stability_all_reinforced(self):
        op = OpinionMetadata(confidence=0.5)
        op.reinforce("e1")
        op.reinforce("e2")
        assert op.stability == 1.0


class TestOpinionSerialization:
    def test_to_dict_from_dict_roundtrip(self):
        original = OpinionMetadata(
            confidence=0.75,
            subject="Python",
            subject_type="technology",
        )
        original.reinforce("ev-1")
        original.contradict("c-1")

        data = original.to_dict()
        restored = OpinionMetadata.from_dict(data)

        assert restored.confidence == pytest.approx(original.confidence)
        assert restored.subject == "Python"
        assert restored.subject_type == "technology"
        assert restored.reinforcement_count == 1
        assert restored.contradiction_count == 1
        assert "ev-1" in restored.formed_from
        assert "c-1" in restored.formed_from

    def test_to_dict_includes_computed_properties(self):
        op = OpinionMetadata(confidence=0.5)
        op.reinforce("e1")
        data = op.to_dict()
        assert "net_reinforcement" in data
        assert "stability" in data


class TestObservationMetadataCreation:
    def test_creation_with_entity_id(self):
        obs = ObservationMetadata(entity_id="ent-1")
        assert obs.entity_id == "ent-1"
        assert obs.entity_name is None
        assert obs.entity_type is None
        assert obs.source_facts == []
        assert obs.completeness == 0.0
        assert obs.aspects_covered == []


class TestObservationAddSource:
    def test_adds_fact_and_aspect(self):
        obs = ObservationMetadata(entity_id="ent-1")
        obs.add_source("fact-1", aspect="career")
        assert "fact-1" in obs.source_facts
        assert "career" in obs.aspects_covered

    def test_completeness_formula(self):
        obs = ObservationMetadata(entity_id="ent-1")
        obs.add_source("f1", aspect="career")
        assert obs.completeness == pytest.approx(0.2)
        obs.add_source("f2", aspect="education")
        assert obs.completeness == pytest.approx(0.4)
        obs.add_source("f3", aspect="preferences")
        assert obs.completeness == pytest.approx(0.6)

    def test_completeness_caps_at_one(self):
        obs = ObservationMetadata(entity_id="ent-1")
        for i, aspect in enumerate(["a", "b", "c", "d", "e", "f"]):
            obs.add_source(f"f{i}", aspect=aspect)
        assert obs.completeness <= 1.0

    def test_deduplicates_facts(self):
        obs = ObservationMetadata(entity_id="ent-1")
        obs.add_source("f1", aspect="career")
        obs.add_source("f1", aspect="education")
        assert obs.source_facts.count("f1") == 1

    def test_sets_last_updated_at(self):
        obs = ObservationMetadata(entity_id="ent-1")
        assert obs.last_updated_at is None
        obs.add_source("f1", aspect="career")
        assert obs.last_updated_at is not None


class TestObservationSerialization:
    def test_to_dict_from_dict_roundtrip(self):
        original = ObservationMetadata(
            entity_id="ent-1",
            entity_name="Alice",
            entity_type="person",
        )
        original.add_source("f1", aspect="career")
        original.add_source("f2", aspect="education")

        data = original.to_dict()
        restored = ObservationMetadata.from_dict(data)

        assert restored.entity_id == "ent-1"
        assert restored.entity_name == "Alice"
        assert restored.entity_type == "person"
        assert len(restored.source_facts) == 2
        assert restored.completeness == pytest.approx(0.4)
        assert set(restored.aspects_covered) == {"career", "education"}


class TestDisposition:
    def test_default_values(self):
        d = Disposition()
        assert d.skepticism == 0.5
        assert d.literalism == 0.5
        assert d.empathy == 0.5

    def test_custom_values(self):
        d = Disposition(skepticism=0.8, literalism=0.2, empathy=0.9)
        assert d.skepticism == 0.8
        assert d.literalism == 0.2
        assert d.empathy == 0.9

    def test_to_dict_from_dict_roundtrip(self):
        original = Disposition(skepticism=0.3, literalism=0.7, empathy=0.1)
        data = original.to_dict()
        restored = Disposition.from_dict(data)

        assert restored.skepticism == 0.3
        assert restored.literalism == 0.7
        assert restored.empathy == 0.1

    def test_from_dict_defaults_on_missing_keys(self):
        d = Disposition.from_dict({})
        assert d.skepticism == 0.5
        assert d.literalism == 0.5
        assert d.empathy == 0.5
