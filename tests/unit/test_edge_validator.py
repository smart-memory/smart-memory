"""Unit tests for EdgeValidator."""

from unittest.mock import MagicMock

import pytest

from smartmemory.validation.edge_validator import EdgeValidator


@pytest.fixture
def mock_graph():
    graph = MagicMock()
    graph.get_edges_for_node.return_value = [
        {"source": "dec_1", "target": "sem_1", "type": "DERIVED_FROM"},
    ]
    graph.get_node.side_effect = lambda item_id: MagicMock(
        memory_type="decision" if "dec" in item_id else "semantic"
    )
    return graph


@pytest.fixture
def validator(mock_graph):
    return EdgeValidator(graph=mock_graph)


class TestValidateEdge:
    def test_valid_edge(self, validator):
        result = validator.validate_edge("decision", "semantic", "DERIVED_FROM", {})
        assert result.is_valid

    def test_unknown_edge_type_warns(self, validator):
        result = validator.validate_edge("decision", "semantic", "NONEXISTENT_EDGE", {})
        assert any(i.severity == "warning" for i in result.issues)

    def test_unknown_edge_type_is_still_valid(self, validator):
        """Unknown edge type produces a warning, not an error."""
        result = validator.validate_edge("decision", "semantic", "NONEXISTENT_EDGE", {})
        assert result.is_valid

    def test_invalid_source_type_produces_error(self, validator):
        result = validator.validate_edge("invalid_type", "semantic", "DERIVED_FROM", {})
        assert not result.is_valid
        assert any(i.field == "source_type" for i in result.errors)

    def test_invalid_target_type_produces_error(self, validator):
        result = validator.validate_edge("decision", "invalid_type", "DERIVED_FROM", {})
        assert not result.is_valid
        assert any(i.field == "target_type" for i in result.errors)


class TestFindOrphanNodes:
    def test_no_orphans(self, validator, mock_graph):
        mock_graph.get_all_nodes.return_value = [
            MagicMock(item_id="n1"),
            MagicMock(item_id="n2"),
        ]
        mock_graph.get_edges_for_node.return_value = [
            {"source": "n1", "target": "n2", "type": "RELATED"}
        ]
        orphans = validator.find_orphan_nodes()
        assert len(orphans) == 0

    def test_finds_orphans(self, validator, mock_graph):
        mock_graph.get_all_nodes.return_value = [
            MagicMock(item_id="n1"),
            MagicMock(item_id="orphan"),
        ]
        mock_graph.get_edges_for_node.side_effect = lambda item_id: (
            [{"source": "n1", "target": "n2", "type": "RELATED"}]
            if item_id == "n1"
            else []
        )
        orphans = validator.find_orphan_nodes()
        assert "orphan" in orphans
        assert "n1" not in orphans

    def test_handles_dict_nodes(self, validator, mock_graph):
        mock_graph.get_all_nodes.return_value = [
            {"item_id": "dict_node"},
        ]
        mock_graph.get_edges_for_node.return_value = []
        orphans = validator.find_orphan_nodes()
        assert "dict_node" in orphans

    def test_graceful_on_graph_error(self, validator, mock_graph):
        mock_graph.get_all_nodes.side_effect = RuntimeError("connection lost")
        orphans = validator.find_orphan_nodes()
        assert orphans == []


class TestFindProvenanceGaps:
    def test_node_with_provenance(self, validator, mock_graph):
        mock_graph.get_edges_for_node.return_value = [
            {"source": "dec_1", "target": "sem_1", "type": "DERIVED_FROM"},
        ]
        gaps = validator.find_provenance_gaps("dec_1")
        assert len(gaps) == 0

    def test_node_without_provenance(self, validator, mock_graph):
        mock_graph.get_edges_for_node.return_value = []
        gaps = validator.find_provenance_gaps("dec_1")
        assert "no_provenance_edge" in gaps

    def test_non_provenance_edge_counts_as_gap(self, validator, mock_graph):
        mock_graph.get_edges_for_node.return_value = [
            {"source": "dec_1", "target": "sem_1", "type": "RELATED"},
        ]
        gaps = validator.find_provenance_gaps("dec_1")
        assert "no_provenance_edge" in gaps

    def test_caused_by_counts_as_provenance(self, validator, mock_graph):
        mock_graph.get_edges_for_node.return_value = [
            {"source": "n1", "target": "n2", "type": "CAUSED_BY"},
        ]
        gaps = validator.find_provenance_gaps("n1")
        assert len(gaps) == 0

    def test_produced_counts_as_provenance(self, validator, mock_graph):
        mock_graph.get_edges_for_node.return_value = [
            {"source": "n1", "target": "n2", "type": "PRODUCED"},
        ]
        gaps = validator.find_provenance_gaps("n1")
        assert len(gaps) == 0

    def test_graceful_on_graph_error(self, validator, mock_graph):
        mock_graph.get_edges_for_node.side_effect = RuntimeError("gone")
        gaps = validator.find_provenance_gaps("dec_1")
        assert gaps == []
