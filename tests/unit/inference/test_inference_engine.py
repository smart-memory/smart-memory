"""Unit tests for InferenceEngine and InferenceRule."""

from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit

from smartmemory.inference.rules import InferenceRule, get_default_rules
from smartmemory.inference.engine import InferenceEngine, InferenceResult


# ---------------------------------------------------------------------------
# InferenceRule
# ---------------------------------------------------------------------------
class TestInferenceRule:
    def test_defaults(self):
        rule = InferenceRule()
        assert rule.name == ""
        assert rule.confidence == 0.7
        assert rule.enabled is True

    def test_to_dict(self):
        rule = InferenceRule(name="test", edge_type="CAUSES", confidence=0.9)
        d = rule.to_dict()
        assert d["name"] == "test"
        assert d["edge_type"] == "CAUSES"
        assert d["confidence"] == 0.9
        assert d["enabled"] is True

    def test_from_dict(self):
        data = {"name": "r1", "pattern_cypher": "MATCH ...", "edge_type": "X", "confidence": 0.5, "enabled": False}
        rule = InferenceRule.from_dict(data)
        assert rule.name == "r1"
        assert rule.confidence == 0.5
        assert rule.enabled is False

    def test_from_dict_ignores_unknown_keys(self):
        data = {"name": "r1", "unknown_key": "ignored"}
        rule = InferenceRule.from_dict(data)
        assert rule.name == "r1"

    def test_round_trip(self):
        original = InferenceRule(name="test", description="desc", edge_type="E", confidence=0.8, enabled=True)
        restored = InferenceRule.from_dict(original.to_dict())
        assert restored.name == original.name
        assert restored.confidence == original.confidence


class TestGetDefaultRules:
    def test_returns_three_rules(self):
        rules = get_default_rules()
        assert len(rules) == 3

    def test_all_enabled(self):
        for rule in get_default_rules():
            assert rule.enabled is True

    def test_rule_names(self):
        names = {r.name for r in get_default_rules()}
        assert names == {"causal_transitivity", "contradiction_symmetry", "topic_inheritance"}

    def test_all_have_cypher(self):
        for rule in get_default_rules():
            assert rule.pattern_cypher != ""
            assert rule.edge_type != ""


# ---------------------------------------------------------------------------
# InferenceResult
# ---------------------------------------------------------------------------
class TestInferenceResult:
    def test_defaults(self):
        r = InferenceResult()
        assert r.edges_created == 0
        assert r.rules_applied == []
        assert r.errors == []

    def test_to_dict(self):
        r = InferenceResult(edges_created=3, rules_applied=["r1"], errors=["e1"])
        d = r.to_dict()
        assert d["edges_created"] == 3
        assert d["rules_applied"] == ["r1"]
        assert d["errors"] == ["e1"]


# ---------------------------------------------------------------------------
# InferenceEngine
# ---------------------------------------------------------------------------
class TestInferenceEngine:
    @pytest.fixture
    def mock_graph(self):
        g = MagicMock()
        g.backend = MagicMock()
        return g

    @pytest.fixture
    def engine(self, mock_graph):
        memory = MagicMock()
        memory._graph = mock_graph
        return InferenceEngine(memory, graph=mock_graph)

    def test_run_no_matches(self, engine, mock_graph):
        mock_graph.backend.execute_cypher.return_value = []
        result = engine.run()
        assert result.edges_created == 0
        assert result.rules_applied == []
        assert result.errors == []

    def test_run_with_matches(self, engine, mock_graph):
        # First rule matches 2 rows, others match 0
        mock_graph.backend.execute_cypher.side_effect = [
            [("src1", "tgt1"), ("src2", "tgt2")],  # causal_transitivity pattern
            [],  # causal_transitivity edge creation 1
            [],  # causal_transitivity edge creation 2
            [],  # contradiction_symmetry pattern
            [],  # topic_inheritance pattern
        ]
        result = engine.run()
        assert result.edges_created == 2
        assert "causal_transitivity" in result.rules_applied

    def test_run_dry_run(self, engine, mock_graph):
        mock_graph.backend.execute_cypher.side_effect = [
            [("src1", "tgt1")],  # causal_transitivity pattern
            [],  # contradiction_symmetry pattern
            [],  # topic_inheritance pattern
        ]
        result = engine.run(dry_run=True)
        assert result.edges_created == 1
        assert "causal_transitivity" in result.rules_applied
        # dry_run queries all 3 rules' patterns but creates no edges
        assert mock_graph.backend.execute_cypher.call_count == 3

    def test_run_disabled_rule_skipped(self, mock_graph):
        memory = MagicMock()
        memory._graph = mock_graph
        rules = [InferenceRule(name="disabled", pattern_cypher="MATCH ...", edge_type="X", enabled=False)]
        engine = InferenceEngine(memory, graph=mock_graph, rules=rules)
        mock_graph.backend.execute_cypher.return_value = []
        result = engine.run()
        assert result.edges_created == 0
        mock_graph.backend.execute_cypher.assert_not_called()

    def test_run_rule_failure_logged(self, mock_graph):
        memory = MagicMock()
        memory._graph = mock_graph
        rules = [InferenceRule(name="broken", pattern_cypher="BAD CYPHER", edge_type="X")]
        engine = InferenceEngine(memory, graph=mock_graph, rules=rules)
        # _cypher swallows exceptions, so we need to make _apply_rule raise
        # by patching it to simulate a failure at the rule level
        with patch.object(engine, "_apply_rule", side_effect=RuntimeError("Rule failed")):
            result = engine.run()
        assert result.edges_created == 0
        assert len(result.errors) == 1
        assert "broken" in result.errors[0]

    def test_no_graph_returns_zero(self):
        memory = MagicMock(spec=[])  # no _graph attribute
        engine = InferenceEngine(memory, graph=None)
        result = engine.run()
        assert result.edges_created == 0

    def test_confidence_decay_from_row(self, engine, mock_graph):
        # Row with 4 elements: source, target, conf1, conf2
        mock_graph.backend.execute_cypher.side_effect = [
            [("src", "tgt", 0.8, 0.6)],  # pattern with confidence values
            [],  # edge creation
            [],  # next rules
            [],
        ]
        result = engine.run()
        assert result.edges_created == 1
        # Verify the edge creation call used decayed confidence
        create_call = mock_graph.backend.execute_cypher.call_args_list[1]
        props = create_call[0][1]["props"]
        assert props["confidence"] == min(0.7, 0.8 * 0.6)  # min(rule.confidence, conf1*conf2)

    def test_custom_rules(self, mock_graph):
        memory = MagicMock()
        memory._graph = mock_graph
        custom = [InferenceRule(name="custom", pattern_cypher="MATCH (a)-[:X]->(b) RETURN a.item_id, b.item_id", edge_type="Y", confidence=0.5)]
        engine = InferenceEngine(memory, graph=mock_graph, rules=custom)
        mock_graph.backend.execute_cypher.side_effect = [
            [("a1", "b1")],
            [],
        ]
        result = engine.run()
        assert result.edges_created == 1
        assert result.rules_applied == ["custom"]
