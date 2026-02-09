"""Unit tests for clustering.graph_aggregator — multi-graph aggregation."""

from unittest.mock import patch, MagicMock

import pytest

pytestmark = pytest.mark.unit

from smartmemory.clustering.graph_aggregator import GraphAggregator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _graph(entities=None, relations=None):
    return {"entities": entities or [], "relations": relations or []}


# ---------------------------------------------------------------------------
# aggregate() basics
# ---------------------------------------------------------------------------
class TestGraphAggregatorBasics:
    def test_empty_list(self):
        agg = GraphAggregator()
        result = agg.aggregate([])
        assert result["entities"] == []
        assert result["relations"] == []
        assert result["source_count"] == 0

    def test_single_graph_passthrough(self):
        g = _graph(
            entities=[{"name": "Python", "entity_type": "language"}],
            relations=[{"subject": "Python", "predicate": "is_a", "object": "language"}],
        )
        result = GraphAggregator().aggregate([g])
        assert result["source_count"] == 1
        assert len(result["entities"]) == 1
        assert len(result["relations"]) == 1

    def test_merge_strategy_default_union(self):
        agg = GraphAggregator()
        assert agg.merge_strategy == "union"


# ---------------------------------------------------------------------------
# Entity merging
# ---------------------------------------------------------------------------
class TestEntityMerging:
    def test_deduplicates_same_entity(self):
        g1 = _graph(entities=[{"name": "Python", "entity_type": "language"}])
        g2 = _graph(entities=[{"name": "Python", "entity_type": "language"}])
        result = GraphAggregator().aggregate([g1, g2])
        assert len(result["entities"]) == 1
        assert 0 in result["entities"][0]["_source_graphs"]
        assert 1 in result["entities"][0]["_source_graphs"]

    def test_keeps_different_entities(self):
        g1 = _graph(entities=[{"name": "Python", "entity_type": "language"}])
        g2 = _graph(entities=[{"name": "Java", "entity_type": "language"}])
        result = GraphAggregator().aggregate([g1, g2])
        assert len(result["entities"]) == 2

    def test_skips_nameless_entities(self):
        g = _graph(entities=[{"entity_type": "unknown"}])
        result = GraphAggregator().aggregate([g, _graph()])
        assert len(result["entities"]) == 0

    def test_latest_strategy_overwrites(self):
        g1 = _graph(entities=[{"name": "Python", "entity_type": "language", "confidence": 0.5}])
        g2 = _graph(entities=[{"name": "Python", "entity_type": "language", "confidence": 0.9}])
        result = GraphAggregator(merge_strategy="latest").aggregate([g1, g2])
        assert len(result["entities"]) == 1
        assert result["entities"][0]["confidence"] == 0.9

    def test_highest_confidence_strategy(self):
        g1 = _graph(entities=[{"name": "Python", "entity_type": "language", "confidence": 0.9}])
        g2 = _graph(entities=[{"name": "Python", "entity_type": "language", "confidence": 0.5}])
        result = GraphAggregator(merge_strategy="highest_confidence").aggregate([g1, g2])
        assert len(result["entities"]) == 1
        assert result["entities"][0]["confidence"] == 0.9


# ---------------------------------------------------------------------------
# Relation merging
# ---------------------------------------------------------------------------
class TestRelationMerging:
    def test_deduplicates_same_relation(self):
        rel = {"subject": "Python", "predicate": "is_a", "object": "language"}
        g1 = _graph(relations=[rel])
        g2 = _graph(relations=[dict(rel)])
        result = GraphAggregator().aggregate([g1, g2])
        assert len(result["relations"]) == 1

    def test_keeps_different_relations(self):
        r1 = {"subject": "Python", "predicate": "is_a", "object": "language"}
        r2 = {"subject": "Java", "predicate": "is_a", "object": "language"}
        result = GraphAggregator().aggregate([_graph(relations=[r1]), _graph(relations=[r2])])
        assert len(result["relations"]) == 2

    def test_skips_incomplete_relations(self):
        r = {"subject": "Python", "predicate": "", "object": ""}
        result = GraphAggregator().aggregate([_graph(relations=[r]), _graph()])
        assert len(result["relations"]) == 0

    def test_source_tracking(self):
        rel = {"subject": "A", "predicate": "rel", "object": "B"}
        result = GraphAggregator().aggregate([_graph(relations=[rel]), _graph(relations=[dict(rel)])])
        assert len(result["relations"]) == 1
        assert 0 in result["relations"][0]["_source_graphs"]
        assert 1 in result["relations"][0]["_source_graphs"]
