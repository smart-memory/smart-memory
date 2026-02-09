"""Tests for Wave 2 edge and node schema registrations."""

import pytest

pytestmark = pytest.mark.unit


from smartmemory.graph.models.schema_validator import get_validator


class TestWave2Schemas:
    def test_inferred_from_edge_registered(self):
        v = get_validator()
        assert "INFERRED_FROM" in v.edge_schemas

    def test_inferred_from_allows_any_to_any(self):
        v = get_validator()
        schema = v.edge_schemas["INFERRED_FROM"]
        assert "semantic" in schema.source_node_types
        assert "decision" in schema.source_node_types
        assert "semantic" in schema.target_node_types
        assert "decision" in schema.target_node_types

    def test_inferred_from_properties(self):
        v = get_validator()
        schema = v.edge_schemas["INFERRED_FROM"]
        assert "rule_name" in schema.optional_properties
        assert "confidence" in schema.optional_properties
        assert "inferred_at" in schema.optional_properties

    def test_requires_edge_registered(self):
        v = get_validator()
        assert "REQUIRES" in v.edge_schemas

    def test_requires_edge_types(self):
        v = get_validator()
        schema = v.edge_schemas["REQUIRES"]
        assert "decision" in schema.source_node_types

    def test_decision_schema_has_pending_fields(self):
        v = get_validator()
        schema = v.node_schemas["decision"]
        assert "pending_requirements" in schema.optional_fields

    def test_validate_inferred_edge(self):
        v = get_validator()
        assert v.validate_edge("semantic", "semantic", "INFERRED_FROM", {"rule_name": "transitivity"})
