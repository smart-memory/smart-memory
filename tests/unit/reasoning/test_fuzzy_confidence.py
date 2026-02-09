"""Unit tests for FuzzyConfidenceCalculator."""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest


pytestmark = pytest.mark.unit

from smartmemory.models.decision import Decision
from smartmemory.reasoning.fuzzy_confidence import ConfidenceScore, FuzzyConfidenceCalculator


class TestConfidenceScore:
    def test_combined_score(self):
        score = ConfidenceScore(evidence=0.9, recency=0.8, consensus=0.7, directness=1.0)
        combined = score.combined()
        assert 0.0 <= combined <= 1.0

    def test_to_dict(self):
        score = ConfidenceScore(evidence=0.9, recency=0.8, consensus=0.7, directness=1.0)
        d = score.to_dict()
        assert "evidence" in d
        assert "combined" in d


@pytest.fixture
def mock_graph():
    return MagicMock()


@pytest.fixture
def calc(mock_graph):
    return FuzzyConfidenceCalculator(mock_graph)


class TestCalculate:
    def test_high_confidence_decision(self, calc):
        d = Decision(
            decision_id="dec_1", content="Test",
            confidence=0.9, reinforcement_count=5, contradiction_count=0,
            evidence_ids=["e1", "e2", "e3"],
            created_at=datetime.now(timezone.utc),
        )
        score = calc.calculate(d)
        assert score.combined() > 0.7

    def test_low_confidence_old_decision(self, calc):
        d = Decision(
            decision_id="dec_1", content="Test",
            confidence=0.3, reinforcement_count=1, contradiction_count=3,
            evidence_ids=["e1"],
            created_at=datetime.now(timezone.utc) - timedelta(days=365),
        )
        score = calc.calculate(d)
        assert score.combined() < 0.5

    def test_pending_decision_zero(self, calc):
        d = Decision(decision_id="dec_1", content="Test", status="pending", confidence=0.0)
        score = calc.calculate(d)
        assert score.combined() == 0.0

    def test_custom_weights(self, mock_graph):
        calc = FuzzyConfidenceCalculator(
            mock_graph, weights={"evidence": 1.0, "recency": 0.0, "consensus": 0.0, "directness": 0.0},
        )
        d = Decision(
            decision_id="dec_1", content="Test",
            confidence=0.5, evidence_ids=["e1", "e2", "e3", "e4", "e5"],
        )
        score = calc.calculate(d)
        assert score.evidence > 0.5
