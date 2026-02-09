"""Integration tests for decisions module against real SmartMemory backends.

Requires running FalkorDB (port 9010) and Redis (port 9012).
Tests are skipped gracefully if backends are unavailable.
"""

import os

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def memory():
    """Create a real SmartMemory instance for decision integration tests."""
    os.environ.setdefault("FALKORDB_PORT", "9010")
    os.environ.setdefault("REDIS_PORT", "9012")
    os.environ.setdefault("VECTOR_BACKEND", "falkordb")

    try:
        from smartmemory.smart_memory import SmartMemory
        sm = SmartMemory()
    except Exception as e:
        pytest.skip(f"Integration environment not ready: {e}")

    yield sm

    try:
        sm.clear()
    except Exception:
        pass
    for key in ("FALKORDB_PORT", "REDIS_PORT", "VECTOR_BACKEND"):
        os.environ.pop(key, None)


@pytest.fixture
def dm(memory):
    """Create a DecisionManager from the real SmartMemory instance."""
    from smartmemory.decisions.manager import DecisionManager
    return DecisionManager(memory)


class TestDecisionLifecycle:
    """Test full decision lifecycle against real graph."""

    def test_create_and_get(self, dm):
        decision = dm.create(content="User prefers Python over Java", domain="preferences")
        assert decision is not None
        assert decision.decision_id is not None
        assert decision.content == "User prefers Python over Java"

        retrieved = dm.get_decision(decision.decision_id)
        assert retrieved is not None
        assert retrieved.content == decision.content

    def test_reinforce(self, dm, memory):
        from smartmemory.models.memory_item import MemoryItem

        decision = dm.create(content="User likes dark mode", domain="ui")
        original_confidence = decision.confidence

        # Create an evidence memory item
        evidence_id = memory.add(MemoryItem(content="User set dark mode again", memory_type="semantic"))
        reinforced = dm.reinforce(decision.decision_id, evidence_id)
        assert reinforced is not None
        assert reinforced.confidence >= original_confidence

    def test_supersede(self, dm):
        from smartmemory.decisions.manager import Decision

        old = dm.create(content="User prefers tabs", domain="editor")
        new_decision = Decision(
            decision_id=Decision.generate_id(),
            content="User prefers spaces",
            decision_type="preference",
        )
        result = dm.supersede(old.decision_id, new_decision, reason="Changed preference")
        assert result is not None
        assert result.content == "User prefers spaces"

        old_refreshed = dm.get_decision(old.decision_id)
        assert old_refreshed.status == "superseded"

    def test_retract(self, dm):
        decision = dm.create(content="User dislikes TypeScript", domain="languages")
        # retract returns None
        dm.retract(decision.decision_id, reason="User changed mind")

        retracted = dm.get_decision(decision.decision_id)
        assert retracted is not None
        assert retracted.status == "retracted"

    def test_get_nonexistent(self, dm):
        result = dm.get_decision("nonexistent_decision_xyz")
        assert result is None


class TestResiduationIntegration:
    """Test ResiduationManager with real backends."""

    def test_create_pending_and_activate(self, memory, dm):
        from smartmemory.reasoning.residuation import ResiduationManager

        rm = ResiduationManager(memory, decision_manager=dm)

        decision = rm.create_pending(
            content="User might prefer Rust",
            requirements=[
                {"description": "Need language usage data", "requirement_type": "evidence"},
            ],
        )
        assert decision is not None
        assert decision.status == "pending"
        assert len(decision.pending_requirements) == 1

        req_id = decision.pending_requirements[0].requirement_id

        resolved = rm.resolve_requirement(decision.decision_id, req_id, "mem_evidence_123")
        assert resolved is True

        activated = rm.try_activate(decision.decision_id)
        assert activated is True

        refreshed = dm.get_decision(decision.decision_id)
        assert refreshed.status == "active"

    def test_try_activate_with_unresolved(self, memory, dm):
        from smartmemory.reasoning.residuation import ResiduationManager

        rm = ResiduationManager(memory, decision_manager=dm)
        decision = rm.create_pending(
            content="Tentative decision",
            requirements=[
                {"description": "Need more data", "requirement_type": "evidence"},
            ],
        )
        activated = rm.try_activate(decision.decision_id)
        assert activated is False
