"""Unit tests for InferenceEngine."""

from unittest.mock import MagicMock

import pytest


pytestmark = pytest.mark.unit

from smartmemory.inference.engine import InferenceEngine, InferenceResult
from smartmemory.inference.rules import InferenceRule, get_default_rules


class TestInferenceRule:
    def test_to_dict(self):
        rule = InferenceRule(name="test_rule", edge_type="CAUSES", confidence=0.8)
        d = rule.to_dict()
        assert d["name"] == "test_rule"
        assert d["confidence"] == 0.8

    def test_from_dict(self):
        rule = InferenceRule.from_dict({"name": "r1", "edge_type": "CAUSES", "confidence": 0.5})
        assert rule.name == "r1"
        assert rule.confidence == 0.5

    def test_default_rules(self):
        rules = get_default_rules()
        assert len(rules) == 3
        names = {r.name for r in rules}
        assert "causal_transitivity" in names
        assert "contradiction_symmetry" in names
        assert "topic_inheritance" in names


@pytest.fixture
def mock_graph():
    return MagicMock()


@pytest.fixture
def engine(mock_graph):
    memory = MagicMock()
    memory._graph = mock_graph
    return InferenceEngine(memory)


class TestRun:
    def test_no_matches_no_edges(self, engine, mock_graph):
        mock_graph.backend.execute_cypher.return_value = []
        result = engine.run()
        assert result.edges_created == 0
        assert result.rules_applied == []

    def test_creates_edges_on_match(self, engine, mock_graph):
        # First call returns matches, subsequent calls are create operations
        mock_graph.backend.execute_cypher.side_effect = [
            [["node_a", "node_c", 0.9, 0.8]],  # causal transitivity match
            None,  # create edge
            [],  # contradiction symmetry - no matches
            [],  # topic inheritance - no matches
        ]
        result = engine.run()
        assert result.edges_created == 1
        assert "causal_transitivity" in result.rules_applied

    def test_dry_run_no_creation(self, engine, mock_graph):
        mock_graph.backend.execute_cypher.side_effect = [
            [["a", "c", 0.9, 0.8]],  # match
            [],  # no match
            [],  # no match
        ]
        result = engine.run(dry_run=True)
        assert result.edges_created == 1
        # Should only have 3 calls (pattern queries, no creates)
        assert mock_graph.backend.execute_cypher.call_count == 3

    def test_disabled_rule_skipped(self, mock_graph):
        memory = MagicMock()
        memory._graph = mock_graph
        rule = InferenceRule(name="disabled", enabled=False)
        engine = InferenceEngine(memory, rules=[rule])
        mock_graph.backend.execute_cypher.return_value = [["a", "b"]]
        result = engine.run()
        assert result.edges_created == 0
        mock_graph.backend.execute_cypher.assert_not_called()

    def test_handles_cypher_error_gracefully(self, engine, mock_graph):
        mock_graph.backend.execute_cypher.side_effect = RuntimeError("cypher error")
        result = engine.run()
        # Engine catches cypher errors at the query level - no crash, no edges
        assert result.edges_created == 0

    def test_custom_rules(self, mock_graph):
        memory = MagicMock()
        memory._graph = mock_graph
        custom_rule = InferenceRule(
            name="custom",
            pattern_cypher="MATCH (a)-[:FOO]->(b) RETURN a.item_id, b.item_id",
            edge_type="BAR",
            confidence=0.5,
        )
        engine = InferenceEngine(memory, rules=[custom_rule])
        mock_graph.backend.execute_cypher.side_effect = [
            [["x", "y"]],  # match
            None,  # create
        ]
        result = engine.run()
        assert result.edges_created == 1
        assert "custom" in result.rules_applied


class TestInferenceResult:
    def test_to_dict(self):
        result = InferenceResult(edges_created=3, rules_applied=["r1", "r2"])
        d = result.to_dict()
        assert d["edges_created"] == 3
        assert len(d["rules_applied"]) == 2
