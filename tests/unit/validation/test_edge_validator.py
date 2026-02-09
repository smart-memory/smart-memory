"""Unit tests for EdgeValidator."""

from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.unit

from smartmemory.validation.edge_validator import EdgeValidator


class TestEdgeValidator:
    @pytest.fixture
    def mock_graph(self):
        return MagicMock()

    @pytest.fixture
    def validator(self, mock_graph):
        return EdgeValidator(graph=mock_graph)

    def test_unregistered_edge_type_is_warning(self, validator):
        result = validator.validate_edge("semantic", "semantic", "TOTALLY_FAKE_EDGE_TYPE_XYZ")
        assert result.is_valid is True  # warnings only
        assert any(i.severity == "warning" and "No schema" in i.message for i in result.issues)

    def test_registered_edge_valid(self, validator):
        # SIMILAR_TO is a commonly registered edge type
        if "SIMILAR_TO" in validator._schema_validator.edge_schemas:
            schema = validator._schema_validator.edge_schemas["SIMILAR_TO"]
            src = list(schema.source_node_types)[0] if schema.source_node_types else "semantic"
            tgt = list(schema.target_node_types)[0] if schema.target_node_types else "semantic"
            result = validator.validate_edge(src, tgt, "SIMILAR_TO", {})
            # May have warnings for missing optional props, but no errors from type mismatch
            type_errors = [i for i in result.errors if i.field in ("source_type", "target_type")]
            assert len(type_errors) == 0

    def test_invalid_source_type_is_error(self, validator):
        # Find any registered edge to test with
        schemas = validator._schema_validator.edge_schemas
        if not schemas:
            pytest.skip("No edge schemas registered")
        edge_type = next(iter(schemas))
        schema = schemas[edge_type]
        # Use a source type that's definitely not in the schema
        result = validator.validate_edge("INVALID_TYPE_XYZ", "semantic", edge_type, {})
        if "INVALID_TYPE_XYZ" not in schema.source_node_types:
            assert any(i.field == "source_type" and i.severity == "error" for i in result.issues)

    def test_missing_required_property_is_error(self, validator):
        schemas = validator._schema_validator.edge_schemas
        # Find an edge type with required properties
        for edge_type, schema in schemas.items():
            if schema.required_properties:
                src = list(schema.source_node_types)[0] if schema.source_node_types else "semantic"
                tgt = list(schema.target_node_types)[0] if schema.target_node_types else "semantic"
                result = validator.validate_edge(src, tgt, edge_type, {})
                assert any(i.severity == "error" and "Missing required" in i.message for i in result.issues)
                return
        pytest.skip("No edge schemas with required properties found")


class TestFindOrphanNodes:
    def test_finds_orphans(self):
        graph = MagicMock()
        node1 = MagicMock(item_id="n1")
        node2 = MagicMock(item_id="n2")
        graph.get_all_nodes.return_value = [node1, node2]
        graph.get_edges_for_node.side_effect = lambda nid: [] if nid == "n1" else [{"type": "X"}]

        validator = EdgeValidator(graph=graph)
        orphans = validator.find_orphan_nodes()
        assert "n1" in orphans
        assert "n2" not in orphans

    def test_no_orphans(self):
        graph = MagicMock()
        node = MagicMock(item_id="n1")
        graph.get_all_nodes.return_value = [node]
        graph.get_edges_for_node.return_value = [{"type": "SIMILAR_TO"}]

        validator = EdgeValidator(graph=graph)
        assert validator.find_orphan_nodes() == []

    def test_graph_error_returns_empty(self):
        graph = MagicMock()
        graph.get_all_nodes.side_effect = RuntimeError("DB down")

        validator = EdgeValidator(graph=graph)
        assert validator.find_orphan_nodes() == []


class TestFindProvenanceGaps:
    def test_has_provenance(self):
        graph = MagicMock()
        graph.get_edges_for_node.return_value = [{"type": "DERIVED_FROM"}]
        validator = EdgeValidator(graph=graph)
        assert validator.find_provenance_gaps("n1") == []

    def test_no_provenance(self):
        graph = MagicMock()
        graph.get_edges_for_node.return_value = [{"type": "SIMILAR_TO"}]
        validator = EdgeValidator(graph=graph)
        gaps = validator.find_provenance_gaps("n1")
        assert "no_provenance_edge" in gaps

    def test_no_edges_at_all(self):
        graph = MagicMock()
        graph.get_edges_for_node.return_value = []
        validator = EdgeValidator(graph=graph)
        gaps = validator.find_provenance_gaps("n1")
        assert "no_provenance_edge" in gaps

    def test_graph_error_returns_empty(self):
        graph = MagicMock()
        graph.get_edges_for_node.side_effect = RuntimeError("DB down")
        validator = EdgeValidator(graph=graph)
        assert validator.find_provenance_gaps("n1") == []
