"""Unit tests for DecisionConfidenceEvolver."""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, PropertyMock

import pytest


pytestmark = pytest.mark.unit

from smartmemory.models.decision import Decision
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
    # Prevent hasattr(memory, "episodic") from returning True.
    # MagicMock auto-creates attributes on access; PropertyMock raising
    # AttributeError makes hasattr() return False.
    type(memory).episodic = PropertyMock(side_effect=AttributeError)
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
    last_contradicted_at=None,
    domain=None,
    tags=None,
    embedding=None,
    evidence_ids=None,
    contradicting_ids=None,
):
    """Helper to create a mock MemoryItem representing a decision."""
    item = MagicMock()
    item.item_id = decision_id
    item.content = content
    item.embedding = embedding
    item.metadata = {
        "decision_id": decision_id,
        "content": content,
        "confidence": confidence,
        "status": status,
        "decision_type": "inference",
        "reinforcement_count": reinforcement_count,
        "contradiction_count": contradiction_count,
        "evidence_ids": evidence_ids or [],
        "contradicting_ids": contradicting_ids or [],
        "updated_at": (updated_at or datetime.now(timezone.utc)).isoformat(),
        "last_reinforced_at": last_reinforced_at.isoformat() if last_reinforced_at else None,
        "last_contradicted_at": last_contradicted_at.isoformat() if last_contradicted_at else None,
    }
    if domain:
        item.metadata["domain"] = domain
    if tags:
        item.metadata["tags"] = tags
    return item


def _make_evidence_item(item_id, content, embedding=None):
    """Helper to create a mock evidence MemoryItem."""
    item = MagicMock()
    item.item_id = item_id
    item.content = content
    item.embedding = embedding
    return item


# search() is called: 1x for decisions + 3x for evidence types (episodic, semantic, opinion).
# Helper to build a side_effect list for the standard 4-call pattern.
def _search_side_effect(decisions, episodic=None, semantic=None, opinion=None):
    """Build a side_effect list for mock_memory.search covering the standard 4-call pattern."""
    return [decisions, episodic or [], semantic or [], opinion or []]


class TestMetadata:
    """Test plugin metadata."""

    def test_name(self):
        meta = DecisionConfidenceEvolver.metadata()
        assert meta.name == "decision_confidence"

    def test_plugin_type(self):
        meta = DecisionConfidenceEvolver.metadata()
        assert meta.plugin_type == "evolver"

    def test_version(self):
        meta = DecisionConfidenceEvolver.metadata()
        assert meta.version == "2.0.0"

    def test_tags_include_reinforcement(self):
        meta = DecisionConfidenceEvolver.metadata()
        assert "reinforcement" in meta.tags


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

    def test_new_fields_defaults(self):
        cfg = DecisionConfidenceConfig()
        assert cfg.lookback_days == 7
        assert cfg.similarity_threshold == 0.7
        assert cfg.enable_reinforcement is True

    def test_new_fields_custom(self):
        cfg = DecisionConfidenceConfig(
            lookback_days=14, similarity_threshold=0.5, enable_reinforcement=False
        )
        assert cfg.lookback_days == 14
        assert cfg.similarity_threshold == 0.5
        assert cfg.enable_reinforcement is False


class TestEvolveDecay:
    """Test confidence decay for stale decisions."""

    def test_decays_stale_decision(self, mock_memory):
        cfg = DecisionConfidenceConfig(enable_reinforcement=False)
        evolver = DecisionConfidenceEvolver(config=cfg)
        old_date = datetime.now(timezone.utc) - timedelta(days=60)
        item = _make_decision_item(
            "dec_stale", "Old decision", confidence=0.6,
            updated_at=old_date, last_reinforced_at=old_date,
        )
        mock_memory.search.return_value = [item]

        evolver.evolve(mock_memory)

        mock_memory.update_properties.assert_called()
        props = mock_memory.update_properties.call_args_list[0][0][1]
        # Linear decay: 0.6 - 0.05 = 0.55
        assert props["confidence"] == pytest.approx(0.55)
        assert props["updated_at"] is not None

    def test_does_not_decay_recent_decision(self, mock_memory):
        cfg = DecisionConfidenceConfig(enable_reinforcement=False)
        evolver = DecisionConfidenceEvolver(config=cfg)
        recent_date = datetime.now(timezone.utc) - timedelta(days=5)
        item = _make_decision_item(
            "dec_recent", "Recent decision", confidence=0.8,
            updated_at=recent_date, last_reinforced_at=recent_date,
        )
        mock_memory.search.return_value = [item]

        evolver.evolve(mock_memory)

        assert not mock_memory.update_properties.called

    def test_decay_disabled(self, mock_memory):
        cfg = DecisionConfidenceConfig(enable_decay=False, enable_reinforcement=False)
        evolver = DecisionConfidenceEvolver(config=cfg)
        old_date = datetime.now(timezone.utc) - timedelta(days=60)
        item = _make_decision_item(
            "dec_old", "Old", confidence=0.6,
            updated_at=old_date, last_reinforced_at=old_date,
        )
        mock_memory.search.return_value = [item]

        evolver.evolve(mock_memory)

        assert not mock_memory.update_properties.called

    def test_confidence_floors_at_zero(self, mock_memory):
        cfg = DecisionConfidenceConfig(
            decay_rate=0.5, decay_after_days=1, enable_reinforcement=False
        )
        evolver = DecisionConfidenceEvolver(config=cfg)
        old_date = datetime.now(timezone.utc) - timedelta(days=10)
        item = _make_decision_item(
            "dec_low", "Low confidence", confidence=0.05,
            updated_at=old_date, last_reinforced_at=old_date,
        )
        mock_memory.search.return_value = [item]

        evolver.evolve(mock_memory)

        mock_memory.update_properties.assert_called()
        props = mock_memory.update_properties.call_args_list[0][0][1]
        assert props["confidence"] == 0.0


class TestEvolveRetract:
    """Test retraction of low-confidence decisions."""

    def test_retracts_below_threshold(self, mock_memory):
        cfg = DecisionConfidenceConfig(
            min_confidence_threshold=0.2, decay_rate=0.5,
            decay_after_days=1, enable_reinforcement=False,
        )
        evolver = DecisionConfidenceEvolver(config=cfg)
        old_date = datetime.now(timezone.utc) - timedelta(days=10)
        item = _make_decision_item(
            "dec_weak", "Weak decision", confidence=0.15,
            updated_at=old_date, last_reinforced_at=old_date,
        )
        mock_memory.search.return_value = [item]

        evolver.evolve(mock_memory)

        mock_memory.update_properties.assert_called()
        retract_calls = [
            c for c in mock_memory.update_properties.call_args_list
            if c[0][1].get("status") == "retracted"
        ]
        assert len(retract_calls) == 1

    def test_does_not_retract_above_threshold(self, mock_memory):
        cfg = DecisionConfidenceConfig(enable_reinforcement=False)
        evolver = DecisionConfidenceEvolver(config=cfg)
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


class TestEvolveReinforce:
    """Test evidence-based reinforcement."""

    def test_reinforces_with_supporting_evidence(self, mock_memory):
        cfg = DecisionConfidenceConfig(enable_reinforcement=True, enable_decay=False)
        evolver = DecisionConfidenceEvolver(config=cfg)

        decision = _make_decision_item("dec_1", "User prefers TypeScript", confidence=0.7)
        evidence = _make_evidence_item("ev_1", "User prefers TypeScript and uses it for all projects")

        mock_memory.search.side_effect = _search_side_effect([decision], episodic=[evidence])

        evolver.evolve(mock_memory)

        mock_memory.update_properties.assert_called_once()
        props = mock_memory.update_properties.call_args[0][1]
        # reinforce formula: min(1.0, 0.7 + (1 - 0.7) * 0.1) = 0.73
        assert props["confidence"] == pytest.approx(0.73)
        assert props["reinforcement_count"] == 1
        assert "ev_1" in props["evidence_ids"]
        assert props["last_reinforced_at"] is not None

    def test_contradicts_with_opposing_evidence(self, mock_memory):
        cfg = DecisionConfidenceConfig(enable_reinforcement=True, enable_decay=False)
        evolver = DecisionConfidenceEvolver(config=cfg)

        decision = _make_decision_item("dec_2", "User prefers Python", confidence=0.8)
        evidence = _make_evidence_item("ev_2", "User no longer says User prefers Python as before")

        mock_memory.search.side_effect = _search_side_effect([decision], episodic=[evidence])

        evolver.evolve(mock_memory)

        mock_memory.update_properties.assert_called_once()
        props = mock_memory.update_properties.call_args[0][1]
        # contradict formula: max(0.0, 0.8 - 0.8 * 0.15) = 0.68
        assert props["confidence"] == pytest.approx(0.68)
        assert props["contradiction_count"] == 1
        assert "ev_2" in props["contradicting_ids"]
        assert props["last_contradicted_at"] is not None

    def test_reinforcement_disabled_skips_evidence(self, mock_memory):
        cfg = DecisionConfidenceConfig(enable_reinforcement=False, enable_decay=False)
        evolver = DecisionConfidenceEvolver(config=cfg)

        recent = datetime.now(timezone.utc) - timedelta(days=1)
        decision = _make_decision_item(
            "dec_3", "Some decision", confidence=0.8,
            updated_at=recent, last_reinforced_at=recent,
        )
        mock_memory.search.return_value = [decision]

        evolver.evolve(mock_memory)

        assert not mock_memory.update_properties.called

    def test_multiple_evidence_items_reinforce(self, mock_memory):
        """Multiple supporting items should each call reinforce()."""
        cfg = DecisionConfidenceConfig(enable_reinforcement=True, enable_decay=False)
        evolver = DecisionConfidenceEvolver(config=cfg)

        decision = _make_decision_item("dec_m", "python development", confidence=0.7)
        ev1 = _make_evidence_item("ev_m1", "Did python development work today")
        ev2 = _make_evidence_item("ev_m2", "More python development in the afternoon")
        ev3 = _make_evidence_item("ev_m3", "Finished python development sprint")

        mock_memory.search.side_effect = _search_side_effect([decision], episodic=[ev1, ev2, ev3])

        evolver.evolve(mock_memory)

        props = mock_memory.update_properties.call_args[0][1]
        assert props["reinforcement_count"] == 3
        assert set(props["evidence_ids"]) == {"ev_m1", "ev_m2", "ev_m3"}
        # 3 reinforce calls: 0.7 -> 0.73 -> 0.757 -> 0.7813
        assert props["confidence"] == pytest.approx(0.7813, abs=0.001)

    def test_both_supporting_and_contradicting_same_decision(self, mock_memory):
        """A decision can receive both supporting and contradicting evidence in one cycle."""
        cfg = DecisionConfidenceConfig(enable_reinforcement=True, enable_decay=False)
        evolver = DecisionConfidenceEvolver(config=cfg)

        decision = _make_decision_item("dec_both", "using kubernetes", confidence=0.8)
        supporting = _make_evidence_item("ev_sup", "Team is using kubernetes for deployment")
        contradicting = _make_evidence_item("ev_con", "using kubernetes has stopped for new projects")

        mock_memory.search.side_effect = _search_side_effect(
            [decision], episodic=[supporting, contradicting]
        )

        evolver.evolve(mock_memory)

        props = mock_memory.update_properties.call_args[0][1]
        assert props["reinforcement_count"] == 1
        assert props["contradiction_count"] == 1
        assert "ev_sup" in props["evidence_ids"]
        assert "ev_con" in props["contradicting_ids"]
        # reinforce then contradict: 0.8 -> 0.82 -> 0.82 - 0.82*0.15 = 0.697
        assert props["confidence"] == pytest.approx(0.697)

    def test_pre_existing_evidence_ids_preserved(self, mock_memory):
        """Decisions with existing evidence should append, not overwrite."""
        cfg = DecisionConfidenceConfig(enable_reinforcement=True, enable_decay=False)
        evolver = DecisionConfidenceEvolver(config=cfg)

        decision = _make_decision_item(
            "dec_pre", "python development", confidence=0.7,
            reinforcement_count=3, evidence_ids=["old_1", "old_2"],
        )
        new_ev = _make_evidence_item("ev_new", "More python development today")

        mock_memory.search.side_effect = _search_side_effect([decision], episodic=[new_ev])

        evolver.evolve(mock_memory)

        props = mock_memory.update_properties.call_args[0][1]
        assert props["reinforcement_count"] == 4
        assert "old_1" in props["evidence_ids"]
        assert "old_2" in props["evidence_ids"]
        assert "ev_new" in props["evidence_ids"]


class TestEvolveContradict:
    """Test contradiction signals in evidence matching."""

    def test_contradiction_signal_stopped(self, mock_memory):
        cfg = DecisionConfidenceConfig(enable_reinforcement=True, enable_decay=False)
        evolver = DecisionConfidenceEvolver(config=cfg)

        decision = _make_decision_item("dec_c1", "User uses Docker", confidence=0.8)
        evidence = _make_evidence_item("ev_c1", "User uses Docker has stopped being true")

        mock_memory.search.side_effect = _search_side_effect([decision], episodic=[evidence])

        evolver.evolve(mock_memory)

        mock_memory.update_properties.assert_called_once()
        props = mock_memory.update_properties.call_args[0][1]
        assert props["contradiction_count"] == 1
        assert props["confidence"] == pytest.approx(0.68)

    def test_contradiction_signal_changed(self, mock_memory):
        cfg = DecisionConfidenceConfig(enable_reinforcement=True, enable_decay=False)
        evolver = DecisionConfidenceEvolver(config=cfg)

        decision = _make_decision_item("dec_c2", "User uses Vim", confidence=0.8)
        evidence = _make_evidence_item("ev_c2", "User uses Vim is no longer true, changed to VS Code")

        mock_memory.search.side_effect = _search_side_effect([decision], episodic=[evidence])

        evolver.evolve(mock_memory)

        mock_memory.update_properties.assert_called_once()
        props = mock_memory.update_properties.call_args[0][1]
        assert props["contradiction_count"] == 1
        assert props["confidence"] == pytest.approx(0.68)


class TestFindMatchingEvidence:
    """Test the evidence matching logic directly."""

    def test_keyword_content_match(self, evolver):
        item = _make_decision_item("dec_k1", "python development")
        dec = Decision.from_dict(item.metadata)
        evidence = [_make_evidence_item("ev_1", "Working on python development tasks today")]

        supporting, contradicting = evolver._find_matching_evidence(dec, item, evidence, 0.7)

        assert len(supporting) == 1
        assert len(contradicting) == 0

    def test_domain_match(self, evolver):
        item = _make_decision_item("dec_k2", "some decision", domain="machine learning")
        dec = Decision.from_dict(item.metadata)
        evidence = [_make_evidence_item("ev_2", "Started a new machine learning course")]

        supporting, _ = evolver._find_matching_evidence(dec, item, evidence, 0.7)

        assert len(supporting) == 1

    def test_tag_match(self, evolver):
        item = _make_decision_item("dec_k3", "some decision", tags=["react", "frontend"])
        dec = Decision.from_dict(item.metadata)
        evidence = [_make_evidence_item("ev_3", "Built a new react component today")]

        supporting, _ = evolver._find_matching_evidence(dec, item, evidence, 0.7)

        assert len(supporting) == 1

    def test_semantic_match(self, evolver):
        emb1 = [1.0, 0.0, 0.0, 0.0]
        emb2 = [0.9, 0.1, 0.0, 0.0]
        item = _make_decision_item("dec_s1", "unique content xyz", embedding=emb1)
        dec = Decision.from_dict(item.metadata)
        evidence = [_make_evidence_item("ev_s1", "completely different text abc", embedding=emb2)]

        supporting, _ = evolver._find_matching_evidence(dec, item, evidence, 0.7)

        assert len(supporting) == 1

    def test_no_match_returns_empty(self, evolver):
        item = _make_decision_item("dec_nm", "apples and oranges")
        dec = Decision.from_dict(item.metadata)
        evidence = [_make_evidence_item("ev_nm", "the weather is nice today")]

        supporting, contradicting = evolver._find_matching_evidence(dec, item, evidence, 0.7)

        assert len(supporting) == 0
        assert len(contradicting) == 0

    def test_contradiction_detected(self, evolver):
        item = _make_decision_item("dec_cd", "user preference")
        dec = Decision.from_dict(item.metadata)
        evidence = [_make_evidence_item("ev_cd", "the user preference has changed dramatically")]

        supporting, contradicting = evolver._find_matching_evidence(dec, item, evidence, 0.7)

        assert len(contradicting) == 1
        assert len(supporting) == 0

    def test_empty_evidence_content_skipped(self, evolver):
        item = _make_decision_item("dec_e", "some content")
        dec = Decision.from_dict(item.metadata)
        evidence = [_make_evidence_item("ev_e", "")]

        supporting, contradicting = evolver._find_matching_evidence(dec, item, evidence, 0.7)

        assert len(supporting) == 0
        assert len(contradicting) == 0

    def test_none_evidence_content_skipped(self, evolver):
        item = _make_decision_item("dec_n", "some content")
        dec = Decision.from_dict(item.metadata)
        evidence = [_make_evidence_item("ev_n", None)]

        supporting, contradicting = evolver._find_matching_evidence(dec, item, evidence, 0.7)

        assert len(supporting) == 0
        assert len(contradicting) == 0


class TestEvolveFullCycle:
    """Test a full evolution cycle: reinforce + decay + retract."""

    def test_mixed_decisions(self, mock_memory):
        cfg = DecisionConfidenceConfig(
            enable_reinforcement=True, enable_decay=True,
            decay_after_days=10, decay_rate=0.5,
            min_confidence_threshold=0.1,
        )
        evolver = DecisionConfidenceEvolver(config=cfg)

        old = datetime.now(timezone.utc) - timedelta(days=60)
        # Decision with matching evidence (will be reinforced)
        reinforced_dec = _make_decision_item("dec_r", "TypeScript usage", confidence=0.7)
        # Stale decision with no matching evidence (will decay)
        stale_dec = _make_decision_item(
            "dec_s", "unrelated topic xyz", confidence=0.6,
            updated_at=old, last_reinforced_at=old,
        )
        # Very low confidence stale decision (will be retracted after decay)
        retract_dec = _make_decision_item(
            "dec_ret", "another unrelated abc", confidence=0.08,
            updated_at=old, last_reinforced_at=old,
        )

        evidence = _make_evidence_item("ev_r", "Been using TypeScript usage extensively")

        mock_memory.search.side_effect = _search_side_effect(
            [reinforced_dec, stale_dec, retract_dec], episodic=[evidence]
        )

        evolver.evolve(mock_memory)

        assert mock_memory.update_properties.call_count == 3

        # Verify each decision's outcome by ID
        calls_by_id = {c[0][0]: c[0][1] for c in mock_memory.update_properties.call_args_list}

        # Reinforced: confidence should increase from 0.7
        assert calls_by_id["dec_r"]["confidence"] == pytest.approx(0.73)
        assert calls_by_id["dec_r"]["reinforcement_count"] == 1

        # Decayed: 0.6 - 0.5 = 0.1, which is at threshold so persisted (not retracted)
        assert calls_by_id["dec_s"]["confidence"] == pytest.approx(0.1)

        # Retracted: 0.08 is already below 0.1, decays further, gets retracted
        assert calls_by_id["dec_ret"]["status"] == "retracted"


class TestEvolveSkips:
    """Test that evolver skips non-active decisions."""

    def test_skips_superseded(self, mock_memory):
        cfg = DecisionConfidenceConfig(enable_reinforcement=False)
        evolver = DecisionConfidenceEvolver(config=cfg)
        item = _make_decision_item("dec_old", "Superseded", status="superseded")
        mock_memory.search.return_value = [item]

        evolver.evolve(mock_memory)

        assert not mock_memory.update_properties.called

    def test_skips_retracted(self, mock_memory):
        cfg = DecisionConfidenceConfig(enable_reinforcement=False)
        evolver = DecisionConfidenceEvolver(config=cfg)
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


class TestGetLastActivity:
    """Test _get_last_activity returns the most recent timestamp."""

    def test_returns_most_recent_across_fields(self, evolver):
        old = datetime(2025, 1, 1, tzinfo=timezone.utc)
        recent = datetime(2025, 6, 15, tzinfo=timezone.utc)
        meta = {
            "last_reinforced_at": old.isoformat(),
            "last_contradicted_at": recent.isoformat(),
            "updated_at": old.isoformat(),
        }
        assert evolver._get_last_activity(meta) == recent

    def test_returns_only_available_timestamp(self, evolver):
        ts = datetime(2025, 3, 1, tzinfo=timezone.utc)
        meta = {"updated_at": ts.isoformat()}
        assert evolver._get_last_activity(meta) == ts

    def test_returns_none_when_all_missing(self, evolver):
        assert evolver._get_last_activity({}) is None

    def test_handles_datetime_objects(self, evolver):
        old = datetime(2025, 1, 1, tzinfo=timezone.utc)
        recent = datetime(2025, 9, 1, tzinfo=timezone.utc)
        meta = {"last_reinforced_at": old, "updated_at": recent}
        assert evolver._get_last_activity(meta) == recent

    def test_skips_invalid_date_strings(self, evolver):
        valid = datetime(2025, 5, 1, tzinfo=timezone.utc)
        meta = {
            "last_reinforced_at": "not-a-date",
            "updated_at": valid.isoformat(),
        }
        assert evolver._get_last_activity(meta) == valid


class TestRegistration:
    """Test that the evolver is registered in the evolution registry."""

    def test_registered_in_registry(self):
        from smartmemory.evolution.registry import EVOLVER_REGISTRY
        cls = EVOLVER_REGISTRY.try_resolve("decision_confidence")
        assert cls is not None

    def test_resolves_to_correct_class(self):
        from smartmemory.evolution.registry import EVOLVER_REGISTRY
        cls = EVOLVER_REGISTRY.get("decision_confidence")
        assert cls is DecisionConfidenceEvolver

    def test_registry_spec_tags(self):
        from smartmemory.evolution.registry import EVOLVER_REGISTRY
        specs = EVOLVER_REGISTRY.list_specs()
        spec = specs["decision_confidence"]
        assert "decision" in spec.tags
        assert "reinforcement" in spec.tags
        assert "builtin" in spec.tags
