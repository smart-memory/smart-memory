"""Unit tests for ResiduationManager."""

from unittest.mock import MagicMock

import pytest


pytestmark = pytest.mark.unit

from smartmemory.models.decision import Decision, PendingRequirement
from smartmemory.reasoning.residuation import ResiduationManager


@pytest.fixture
def mock_memory():
    memory = MagicMock()
    memory._graph = MagicMock()
    memory.search.return_value = []
    return memory


@pytest.fixture
def mock_decision_manager():
    dm = MagicMock()
    dm.create.side_effect = lambda **kwargs: Decision(
        decision_id=Decision.generate_id(), **kwargs
    )
    dm.get_decision.return_value = None
    return dm


@pytest.fixture
def residuation(mock_memory, mock_decision_manager):
    return ResiduationManager(mock_memory, decision_manager=mock_decision_manager)


class TestCreatePending:
    def test_creates_pending_decision(self, residuation, mock_decision_manager):
        reqs = [{"description": "Need language preference", "requirement_type": "evidence"}]
        residuation.create_pending("User might prefer Python", reqs)
        mock_decision_manager.create.assert_called_once()
        call_kwargs = mock_decision_manager.create.call_args[1]
        assert call_kwargs.get("status") == "pending"

    def test_pending_has_requirements(self, residuation):
        reqs = [
            {"description": "Need evidence A", "requirement_type": "evidence"},
            {"description": "Need confirmation", "requirement_type": "confirmation"},
        ]
        decision = residuation.create_pending("Test decision", reqs)
        assert len(decision.pending_requirements) == 2


class TestResolveRequirement:
    def test_resolve_marks_requirement(self, residuation, mock_decision_manager):
        req = PendingRequirement(requirement_id="req_001", description="Need data", requirement_type="data")
        pending_decision = Decision(
            decision_id="dec_test",
            content="Test",
            status="pending",
            pending_requirements=[req],
        )
        mock_decision_manager.get_decision.return_value = pending_decision

        result = residuation.resolve_requirement("dec_test", "req_001", "mem_evidence")
        assert result is True

    def test_resolve_nonexistent_requirement(self, residuation, mock_decision_manager):
        pending_decision = Decision(
            decision_id="dec_test",
            content="Test",
            status="pending",
            pending_requirements=[],
        )
        mock_decision_manager.get_decision.return_value = pending_decision

        result = residuation.resolve_requirement("dec_test", "nonexistent", "mem_evidence")
        assert result is False


class TestTryActivate:
    def test_activates_when_all_resolved(self, residuation, mock_decision_manager):
        req = PendingRequirement(
            requirement_id="req_001", description="Done", requirement_type="evidence",
            resolved=True, resolved_by="mem_123",
        )
        decision = Decision(
            decision_id="dec_test", content="Test", status="pending",
            pending_requirements=[req],
        )
        mock_decision_manager.get_decision.return_value = decision

        activated = residuation.try_activate("dec_test")
        assert activated is True

    def test_does_not_activate_with_unresolved(self, residuation, mock_decision_manager):
        req = PendingRequirement(
            requirement_id="req_001", description="Still needed", requirement_type="evidence",
        )
        decision = Decision(
            decision_id="dec_test", content="Test", status="pending",
            pending_requirements=[req],
        )
        mock_decision_manager.get_decision.return_value = decision

        activated = residuation.try_activate("dec_test")
        assert activated is False
