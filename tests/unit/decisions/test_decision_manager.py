"""Unit tests for DecisionManager."""

from unittest.mock import MagicMock

import pytest


pytestmark = pytest.mark.unit

from smartmemory.decisions.manager import DecisionManager
from smartmemory.models.decision import Decision
from smartmemory.models.memory_item import MemoryItem


@pytest.fixture
def mock_memory():
    """Create a mock SmartMemory instance."""
    memory = MagicMock()
    memory._graph = MagicMock()
    memory.add.return_value = "dec_test123456"
    memory.get.return_value = None
    memory.update_properties.return_value = None
    memory.search.return_value = []
    return memory


@pytest.fixture
def mock_graph(mock_memory):
    """Get the mock graph from mock memory."""
    return mock_memory._graph


@pytest.fixture
def manager(mock_memory):
    """Create a DecisionManager with mocked dependencies."""
    return DecisionManager(mock_memory)


class TestCreate:
    """Test decision creation."""

    def test_create_basic(self, manager, mock_memory):
        d = manager.create("User prefers dark mode")
        assert d.content == "User prefers dark mode"
        assert d.decision_type == "inference"
        assert d.confidence == 0.8
        assert d.decision_id.startswith("dec_")
        mock_memory.add.assert_called_once()

    def test_create_with_all_params(self, manager, mock_memory):
        d = manager.create(
            content="Python over JavaScript",
            decision_type="preference",
            confidence=0.95,
            source_type="explicit",
            source_trace_id="trace_abc",
            source_session_id="sess_123",
            evidence_ids=["mem_1", "mem_2"],
            domain="preferences",
            tags=["language"],
            context_snapshot={"query": "test"},
        )
        assert d.decision_type == "preference"
        assert d.confidence == 0.95
        assert d.source_trace_id == "trace_abc"
        assert d.domain == "preferences"
        assert d.tags == ["language"]

    def test_create_stores_as_memory_item(self, manager, mock_memory):
        manager.create("Test decision")
        args = mock_memory.add.call_args
        item = args[0][0]
        assert isinstance(item, MemoryItem)
        assert item.memory_type == "decision"
        assert item.content == "Test decision"

    def test_create_with_trace_creates_produced_edge(self, manager, mock_memory, mock_graph):
        d = manager.create("Decision from reasoning", source_trace_id="trace_abc")
        produced_calls = [
            c for c in mock_graph.add_edge.call_args_list
            if c[1].get("edge_type") == "PRODUCED"
        ]
        assert len(produced_calls) == 1
        assert produced_calls[0][1]["source_id"] == "trace_abc"
        assert produced_calls[0][1]["target_id"] == d.decision_id

    def test_create_with_evidence_creates_derived_from_edges(self, manager, mock_memory, mock_graph):
        manager.create("Test", evidence_ids=["mem_1", "mem_2"])
        # Should create DERIVED_FROM edges for each evidence
        derived_calls = [
            c for c in mock_graph.add_edge.call_args_list
            if c[1].get("edge_type") == "DERIVED_FROM"
        ]
        assert len(derived_calls) == 2

    def test_create_without_trace_no_produced_edge(self, manager, mock_graph):
        manager.create("No trace")
        produced_calls = [
            c for c in mock_graph.add_edge.call_args_list
            if c[1].get("edge_type") == "PRODUCED"
        ]
        assert len(produced_calls) == 0


class TestGetDecision:
    """Test decision retrieval."""

    def test_get_found(self, manager, mock_memory):
        mock_item = MagicMock()
        mock_item.content = "Test decision"
        mock_item.metadata = {
            "decision_id": "dec_test123456",
            "content": "Test decision",
            "decision_type": "inference",
            "confidence": 0.8,
            "status": "active",
        }
        mock_memory.get.return_value = mock_item
        d = manager.get_decision("dec_test123456")
        assert d is not None
        assert d.content == "Test decision"

    def test_get_not_found(self, manager, mock_memory):
        mock_memory.get.return_value = None
        d = manager.get_decision("dec_nonexistent")
        assert d is None


class TestSupersede:
    """Test decision supersession."""

    def test_supersede_marks_old_as_superseded(self, manager, mock_memory):
        # Set up old decision
        old_item = MagicMock()
        old_item.content = "Old decision"
        old_item.metadata = {
            "decision_id": "dec_old",
            "content": "Old decision",
            "decision_type": "inference",
            "confidence": 0.8,
            "status": "active",
        }
        mock_memory.get.return_value = old_item

        new_d = Decision(decision_id="dec_new", content="New decision")
        manager.supersede("dec_old", new_d, reason="Updated info")

        # Check old decision was updated
        mock_memory.update_properties.assert_called()
        update_args = mock_memory.update_properties.call_args_list[0]
        assert update_args[0][0] == "dec_old"
        props = update_args[0][1]
        assert props["status"] == "superseded"
        assert props["superseded_by"] == "dec_new"

    def test_supersede_stores_new_decision(self, manager, mock_memory):
        old_item = MagicMock()
        old_item.content = "Old"
        old_item.metadata = {"decision_id": "dec_old", "content": "Old", "status": "active"}
        mock_memory.get.return_value = old_item

        new_d = Decision(decision_id="dec_new", content="New")
        manager.supersede("dec_old", new_d, reason="test")
        assert mock_memory.add.called

    def test_supersede_creates_edge(self, manager, mock_memory, mock_graph):
        old_item = MagicMock()
        old_item.content = "Old"
        old_item.metadata = {"decision_id": "dec_old", "content": "Old", "status": "active"}
        mock_memory.get.return_value = old_item

        new_d = Decision(decision_id="dec_new", content="New")
        manager.supersede("dec_old", new_d, reason="test")

        supersede_calls = [
            c for c in mock_graph.add_edge.call_args_list
            if c[1].get("edge_type") == "SUPERSEDES"
        ]
        assert len(supersede_calls) == 1
        assert supersede_calls[0][1]["source_id"] == "dec_new"
        assert supersede_calls[0][1]["target_id"] == "dec_old"

    def test_supersede_not_found_raises(self, manager, mock_memory):
        mock_memory.get.return_value = None
        with pytest.raises(ValueError, match="Decision not found"):
            manager.supersede("dec_missing", Decision(content="New"), reason="test")

    def test_supersede_generates_id_if_empty(self, manager, mock_memory):
        old_item = MagicMock()
        old_item.content = "Old"
        old_item.metadata = {"decision_id": "dec_old", "content": "Old", "status": "active"}
        mock_memory.get.return_value = old_item

        new_d = Decision(content="New without ID")
        result = manager.supersede("dec_old", new_d, reason="test")
        assert result.decision_id.startswith("dec_")
        assert len(result.decision_id) == 16


class TestRetract:
    """Test decision retraction."""

    def test_retract_marks_as_retracted(self, manager, mock_memory):
        item = MagicMock()
        item.content = "To retract"
        item.metadata = {"decision_id": "dec_ret", "content": "To retract", "status": "active"}
        mock_memory.get.return_value = item

        manager.retract("dec_ret", reason="No longer valid")

        update_args = mock_memory.update_properties.call_args_list[0]
        props = update_args[0][1]
        assert props["status"] == "retracted"
        assert props["context_snapshot"]["retraction_reason"] == "No longer valid"

    def test_retract_not_found_raises(self, manager, mock_memory):
        mock_memory.get.return_value = None
        with pytest.raises(ValueError, match="Decision not found"):
            manager.retract("dec_missing", reason="test")


class TestReinforce:
    """Test decision reinforcement."""

    def test_reinforce_updates_confidence(self, manager, mock_memory):
        item = MagicMock()
        item.content = "Test"
        item.metadata = {
            "decision_id": "dec_r", "content": "Test", "confidence": 0.5,
            "evidence_ids": [], "reinforcement_count": 0, "contradiction_count": 0,
            "status": "active",
        }
        mock_memory.get.return_value = item

        d = manager.reinforce("dec_r", "evidence_1")
        assert d.confidence == pytest.approx(0.55)  # 0.5 + (1-0.5)*0.1
        assert d.reinforcement_count == 1
        assert "evidence_1" in d.evidence_ids

    def test_reinforce_creates_derived_from_edge(self, manager, mock_memory, mock_graph):
        item = MagicMock()
        item.content = "Test"
        item.metadata = {
            "decision_id": "dec_r", "content": "Test", "confidence": 0.5,
            "evidence_ids": [], "reinforcement_count": 0, "contradiction_count": 0,
            "status": "active",
        }
        mock_memory.get.return_value = item

        manager.reinforce("dec_r", "evidence_1")

        derived_calls = [
            c for c in mock_graph.add_edge.call_args_list
            if c[1].get("edge_type") == "DERIVED_FROM"
        ]
        assert len(derived_calls) == 1

    def test_reinforce_not_found_raises(self, manager, mock_memory):
        mock_memory.get.return_value = None
        with pytest.raises(ValueError, match="Decision not found"):
            manager.reinforce("dec_missing", "evidence_1")


class TestContradict:
    """Test decision contradiction."""

    def test_contradict_decreases_confidence(self, manager, mock_memory):
        item = MagicMock()
        item.content = "Test"
        item.metadata = {
            "decision_id": "dec_c", "content": "Test", "confidence": 0.8,
            "contradicting_ids": [], "reinforcement_count": 0, "contradiction_count": 0,
            "status": "active",
        }
        mock_memory.get.return_value = item

        d = manager.contradict("dec_c", "counter_1")
        assert d.confidence == pytest.approx(0.68)  # 0.8 - 0.8*0.15
        assert d.contradiction_count == 1
        assert "counter_1" in d.contradicting_ids

    def test_contradict_creates_contradicts_edge(self, manager, mock_memory, mock_graph):
        item = MagicMock()
        item.content = "Test"
        item.metadata = {
            "decision_id": "dec_c", "content": "Test", "confidence": 0.8,
            "contradicting_ids": [], "reinforcement_count": 0, "contradiction_count": 0,
            "status": "active",
        }
        mock_memory.get.return_value = item

        manager.contradict("dec_c", "counter_1")

        contradict_calls = [
            c for c in mock_graph.add_edge.call_args_list
            if c[1].get("edge_type") == "CONTRADICTS"
        ]
        assert len(contradict_calls) == 1
        assert contradict_calls[0][1]["source_id"] == "counter_1"
        assert contradict_calls[0][1]["target_id"] == "dec_c"


class TestFindConflicts:
    """Test conflict detection."""

    def test_find_conflicts_returns_same_domain(self, manager, mock_memory):
        existing = MagicMock()
        existing.content = "User likes Python"
        existing.metadata = {
            "decision_id": "dec_existing",
            "content": "User likes Python",
            "domain": "preferences",
            "status": "active",
            "decision_type": "preference",
            "confidence": 0.9,
        }
        mock_memory.search.return_value = [existing]

        d = Decision(decision_id="dec_new", content="User dislikes Python", domain="preferences")
        conflicts = manager.find_conflicts(d)
        assert len(conflicts) == 1
        assert conflicts[0].decision_id == "dec_existing"

    def test_find_conflicts_skips_inactive(self, manager, mock_memory):
        inactive = MagicMock()
        inactive.content = "Old decision"
        inactive.metadata = {
            "decision_id": "dec_old",
            "content": "Old decision",
            "domain": "preferences",
            "status": "superseded",
        }
        mock_memory.search.return_value = [inactive]

        d = Decision(decision_id="dec_new", content="New decision", domain="preferences")
        conflicts = manager.find_conflicts(d)
        assert len(conflicts) == 0

    def test_find_conflicts_skips_self(self, manager, mock_memory):
        self_item = MagicMock()
        self_item.content = "Same decision"
        self_item.metadata = {
            "decision_id": "dec_same",
            "content": "Same decision",
            "domain": "preferences",
            "status": "active",
        }
        mock_memory.search.return_value = [self_item]

        d = Decision(decision_id="dec_same", content="Same decision", domain="preferences")
        conflicts = manager.find_conflicts(d)
        assert len(conflicts) == 0

    def test_find_conflicts_content_overlap(self, manager, mock_memory):
        similar = MagicMock()
        similar.content = "User prefers dark mode on all devices"
        similar.metadata = {
            "decision_id": "dec_similar",
            "content": "User prefers dark mode on all devices",
            "status": "active",
        }
        mock_memory.search.return_value = [similar]

        # Different domain but high content overlap
        d = Decision(decision_id="dec_new", content="User prefers dark mode on mobile devices")
        conflicts = manager.find_conflicts(d)
        assert len(conflicts) == 1

    def test_find_conflicts_handles_search_error(self, manager, mock_memory):
        mock_memory.search.side_effect = Exception("Search failed")
        d = Decision(content="Test")
        conflicts = manager.find_conflicts(d)
        assert conflicts == []


class TestContentOverlap:
    """Test the content overlap heuristic."""

    def test_high_overlap(self):
        assert DecisionManager._content_overlap(
            "User prefers dark mode on desktop",
            "User prefers dark mode on mobile",
        ) is True

    def test_low_overlap(self):
        assert DecisionManager._content_overlap(
            "User prefers dark mode",
            "The weather is sunny today",
        ) is False

    def test_empty_strings(self):
        assert DecisionManager._content_overlap("", "") is False
        assert DecisionManager._content_overlap("hello", "") is False


class TestNoGraph:
    """Test manager behavior when graph is not available."""

    def test_create_without_graph(self, mock_memory):
        mock_memory._graph = None
        manager = DecisionManager(mock_memory, graph=None)
        d = manager.create("Test", source_trace_id="trace_abc", evidence_ids=["mem_1"])
        assert d.content == "Test"
        # Should not raise even though graph operations are skipped
        mock_memory.add.assert_called_once()
