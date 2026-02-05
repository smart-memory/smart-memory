"""Unit tests for pending decision (residuation) support."""

import pytest

from smartmemory.models.decision import Decision, PendingRequirement


class TestPendingRequirement:
    def test_create_requirement(self):
        req = PendingRequirement(
            requirement_id="req_001",
            description="Need user's preferred language",
            requirement_type="evidence",
        )
        assert req.requirement_id == "req_001"
        assert req.resolved is False
        assert req.resolved_by is None

    def test_requirement_to_dict(self):
        req = PendingRequirement(
            requirement_id="req_001",
            description="Need evidence",
            requirement_type="data",
            query_hint="search for language preference",
        )
        d = req.to_dict()
        assert d["requirement_id"] == "req_001"
        assert d["resolved"] is False
        assert d["query_hint"] == "search for language preference"

    def test_requirement_from_dict(self):
        d = {
            "requirement_id": "req_002",
            "description": "Need confirmation",
            "requirement_type": "confirmation",
            "resolved": True,
            "resolved_by": "mem_abc",
        }
        req = PendingRequirement.from_dict(d)
        assert req.resolved is True
        assert req.resolved_by == "mem_abc"


class TestDecisionPendingStatus:
    def test_pending_status(self):
        d = Decision(
            decision_id="dec_test",
            content="User might prefer Python",
            status="pending",
        )
        assert d.status == "pending"
        assert d.is_pending is True
        assert d.is_active is False

    def test_pending_with_requirements(self):
        req = PendingRequirement(
            requirement_id="req_001",
            description="Need more evidence",
            requirement_type="evidence",
        )
        d = Decision(
            decision_id="dec_test",
            content="Test",
            status="pending",
            pending_requirements=[req],
        )
        assert len(d.pending_requirements) == 1
        assert d.pending_requirements[0].resolved is False

    def test_all_requirements_resolved(self):
        req = PendingRequirement(
            requirement_id="req_001",
            description="Need evidence",
            requirement_type="evidence",
            resolved=True,
            resolved_by="mem_123",
        )
        d = Decision(
            decision_id="dec_test",
            content="Test",
            status="pending",
            pending_requirements=[req],
        )
        assert d.has_unresolved_requirements is False

    def test_to_dict_includes_pending(self):
        req = PendingRequirement(
            requirement_id="req_001",
            description="Need evidence",
            requirement_type="evidence",
        )
        d = Decision(
            decision_id="dec_test",
            content="Test",
            status="pending",
            pending_requirements=[req],
        )
        result = d.to_dict()
        assert result["status"] == "pending"
        assert len(result["pending_requirements"]) == 1
        assert result["pending_requirements"][0]["requirement_id"] == "req_001"

    def test_from_dict_with_pending(self):
        data = {
            "decision_id": "dec_test",
            "content": "Test",
            "status": "pending",
            "pending_requirements": [
                {
                    "requirement_id": "req_001",
                    "description": "Need evidence",
                    "requirement_type": "evidence",
                    "resolved": False,
                }
            ],
        }
        d = Decision.from_dict(data)
        assert d.status == "pending"
        assert len(d.pending_requirements) == 1

    def test_active_decision_has_no_pending_requirements(self):
        d = Decision(decision_id="dec_test", content="Active decision")
        assert d.is_active is True
        assert d.is_pending is False
        assert len(d.pending_requirements) == 0
