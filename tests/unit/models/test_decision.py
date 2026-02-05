"""Unit tests for Decision model."""

from datetime import datetime, timezone

import pytest

from smartmemory.models.decision import Decision
from smartmemory.models.memory_item import MEMORY_TYPES


class TestDecisionModel:
    """Test Decision dataclass creation and properties."""

    def test_default_values(self):
        d = Decision()
        assert d.decision_id == ""
        assert d.content == ""
        assert d.decision_type == "inference"
        assert d.confidence == 0.8
        assert d.status == "active"
        assert d.source_type == "inferred"
        assert d.evidence_ids == []
        assert d.contradicting_ids == []
        assert d.reinforcement_count == 0
        assert d.contradiction_count == 0

    def test_custom_values(self):
        d = Decision(
            decision_id="dec_abc123",
            content="User prefers dark mode",
            decision_type="preference",
            confidence=0.95,
            source_type="explicit",
            domain="preferences",
            tags=["ui", "theme"],
        )
        assert d.decision_id == "dec_abc123"
        assert d.content == "User prefers dark mode"
        assert d.decision_type == "preference"
        assert d.confidence == 0.95
        assert d.domain == "preferences"
        assert d.tags == ["ui", "theme"]

    def test_is_active(self):
        d = Decision(status="active")
        assert d.is_active is True

        d.status = "superseded"
        assert d.is_active is False

        d.status = "retracted"
        assert d.is_active is False

    def test_has_provenance_with_trace(self):
        d = Decision(source_trace_id="trace_abc")
        assert d.has_provenance is True

    def test_has_provenance_with_evidence(self):
        d = Decision(evidence_ids=["mem_1", "mem_2"])
        assert d.has_provenance is True

    def test_has_provenance_without(self):
        d = Decision()
        assert d.has_provenance is False

    def test_generate_id(self):
        id1 = Decision.generate_id()
        id2 = Decision.generate_id()
        assert id1.startswith("dec_")
        assert len(id1) == 16  # "dec_" + 12 hex chars
        assert id1 != id2


class TestDecisionConfidence:
    """Test reinforce/contradict with diminishing returns (same as OpinionMetadata)."""

    def test_reinforce_increases_confidence(self):
        d = Decision(confidence=0.5)
        d.reinforce("evidence_1")
        assert d.confidence == pytest.approx(0.55)  # 0.5 + (1 - 0.5) * 0.1
        assert d.reinforcement_count == 1
        assert "evidence_1" in d.evidence_ids

    def test_reinforce_diminishing_returns(self):
        d = Decision(confidence=0.9)
        d.reinforce("evidence_1")
        # 0.9 + (1 - 0.9) * 0.1 = 0.91
        assert d.confidence == pytest.approx(0.91)

    def test_reinforce_caps_at_1(self):
        d = Decision(confidence=0.99)
        d.reinforce("evidence_1")
        assert d.confidence <= 1.0

    def test_reinforce_tracks_timestamp(self):
        d = Decision(confidence=0.5)
        assert d.last_reinforced_at is None
        d.reinforce("evidence_1")
        assert d.last_reinforced_at is not None
        assert d.updated_at is not None

    def test_reinforce_deduplicates_evidence(self):
        d = Decision(confidence=0.5)
        d.reinforce("evidence_1")
        d.reinforce("evidence_1")
        assert d.evidence_ids.count("evidence_1") == 1
        assert d.reinforcement_count == 2  # Count still increments

    def test_contradict_decreases_confidence(self):
        d = Decision(confidence=0.8)
        d.contradict("counter_1")
        # 0.8 - 0.8 * 0.15 = 0.68
        assert d.confidence == pytest.approx(0.68)
        assert d.contradiction_count == 1
        assert "counter_1" in d.contradicting_ids

    def test_contradict_floors_at_0(self):
        d = Decision(confidence=0.01)
        d.contradict("counter_1")
        assert d.confidence >= 0.0

    def test_contradict_tracks_timestamp(self):
        d = Decision(confidence=0.8)
        assert d.last_contradicted_at is None
        d.contradict("counter_1")
        assert d.last_contradicted_at is not None

    def test_net_reinforcement(self):
        d = Decision(confidence=0.5)
        d.reinforce("e1")
        d.reinforce("e2")
        d.contradict("c1")
        assert d.net_reinforcement == 1  # 2 - 1

    def test_stability_neutral_for_new(self):
        d = Decision()
        assert d.stability == 0.5

    def test_stability_high_when_reinforced(self):
        d = Decision(confidence=0.5)
        d.reinforce("e1")
        d.reinforce("e2")
        d.reinforce("e3")
        # 3 reinforcements, 0 contradictions → stability = 1.0
        assert d.stability == 1.0

    def test_stability_mixed(self):
        d = Decision(confidence=0.5)
        d.reinforce("e1")
        d.reinforce("e2")
        d.contradict("c1")
        # 2 reinforcements, 1 contradiction → stability = 2/3
        assert d.stability == pytest.approx(2 / 3)


class TestDecisionSerialization:
    """Test to_dict/from_dict roundtrip."""

    def test_roundtrip_basic(self):
        original = Decision(
            decision_id="dec_abc123",
            content="User prefers TypeScript",
            decision_type="preference",
            confidence=0.9,
            source_type="explicit",
            domain="preferences",
            tags=["language"],
        )
        data = original.to_dict()
        restored = Decision.from_dict(data)

        assert restored.decision_id == original.decision_id
        assert restored.content == original.content
        assert restored.decision_type == original.decision_type
        assert restored.confidence == original.confidence
        assert restored.source_type == original.source_type
        assert restored.domain == original.domain
        assert restored.tags == original.tags

    def test_roundtrip_with_evidence(self):
        original = Decision(
            decision_id="dec_xyz",
            content="Test decision",
            evidence_ids=["mem_1", "mem_2"],
            contradicting_ids=["mem_3"],
            source_trace_id="trace_abc",
        )
        data = original.to_dict()
        restored = Decision.from_dict(data)

        assert restored.evidence_ids == ["mem_1", "mem_2"]
        assert restored.contradicting_ids == ["mem_3"]
        assert restored.source_trace_id == "trace_abc"

    def test_roundtrip_with_timestamps(self):
        original = Decision(decision_id="dec_ts", content="Test")
        original.reinforce("e1")
        original.contradict("c1")

        data = original.to_dict()
        restored = Decision.from_dict(data)

        assert restored.last_reinforced_at is not None
        assert restored.last_contradicted_at is not None
        assert restored.updated_at is not None
        assert restored.reinforcement_count == 1
        assert restored.contradiction_count == 1

    def test_roundtrip_preserves_confidence_after_operations(self):
        original = Decision(decision_id="dec_ops", content="Test", confidence=0.5)
        original.reinforce("e1")
        original.reinforce("e2")
        original.contradict("c1")

        data = original.to_dict()
        restored = Decision.from_dict(data)

        assert restored.confidence == pytest.approx(original.confidence)
        assert restored.reinforcement_count == 2
        assert restored.contradiction_count == 1

    def test_to_dict_includes_computed_properties(self):
        d = Decision(confidence=0.5)
        d.reinforce("e1")
        data = d.to_dict()
        assert "net_reinforcement" in data
        assert "stability" in data
        assert data["net_reinforcement"] == 1
        assert data["stability"] == 1.0

    def test_from_dict_with_missing_fields(self):
        """from_dict should handle partial data gracefully."""
        data = {"content": "Minimal decision"}
        d = Decision.from_dict(data)
        assert d.content == "Minimal decision"
        assert d.decision_type == "inference"
        assert d.confidence == 0.8
        assert d.status == "active"

    def test_from_dict_with_empty_dict(self):
        d = Decision.from_dict({})
        assert d.decision_id == ""
        assert d.confidence == 0.8

    def test_roundtrip_superseded(self):
        original = Decision(
            decision_id="dec_old",
            content="Old decision",
            status="superseded",
            superseded_by="dec_new",
        )
        data = original.to_dict()
        restored = Decision.from_dict(data)
        assert restored.status == "superseded"
        assert restored.superseded_by == "dec_new"
        assert restored.is_active is False

    def test_roundtrip_context_snapshot(self):
        original = Decision(
            decision_id="dec_ctx",
            content="Test",
            context_snapshot={"query": "test", "matches": 5},
        )
        data = original.to_dict()
        restored = Decision.from_dict(data)
        assert restored.context_snapshot == {"query": "test", "matches": 5}


class TestDecisionTypes:
    """Test all valid type values."""

    @pytest.mark.parametrize("decision_type", [
        "inference", "preference", "classification", "choice", "belief", "policy",
    ])
    def test_valid_decision_types(self, decision_type):
        d = Decision(decision_type=decision_type)
        assert d.decision_type == decision_type
        data = d.to_dict()
        assert data["decision_type"] == decision_type

    @pytest.mark.parametrize("source_type", [
        "reasoning", "explicit", "imported", "inferred",
    ])
    def test_valid_source_types(self, source_type):
        d = Decision(source_type=source_type)
        assert d.source_type == source_type

    @pytest.mark.parametrize("status", ["active", "superseded", "retracted"])
    def test_valid_statuses(self, status):
        d = Decision(status=status)
        assert d.status == status


class TestDecisionMemoryType:
    """Test that 'decision' is registered as a valid memory type."""

    def test_decision_in_memory_types(self):
        assert "decision" in MEMORY_TYPES

    def test_all_expected_types_present(self):
        expected = {"semantic", "episodic", "procedural", "working", "zettel",
                    "reasoning", "opinion", "observation", "decision"}
        assert expected.issubset(MEMORY_TYPES)
