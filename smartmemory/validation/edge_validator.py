"""Edge validation and structural integrity checks.

Validates edges against registered schemas and checks for structural issues
like orphan nodes and provenance gaps.
"""

import logging
from typing import Any

from smartmemory.graph.models.schema_validator import get_validator
from smartmemory.validation.memory_validator import ValidationIssue, ValidationResult

logger = logging.getLogger(__name__)

PROVENANCE_EDGE_TYPES = {"DERIVED_FROM", "CAUSED_BY", "PRODUCED"}


class EdgeValidator:
    """Validates graph edges and checks structural integrity.

    Usage:
        validator = EdgeValidator(graph=smart_memory._graph)
        result = validator.validate_edge("decision", "semantic", "DERIVED_FROM", {})
        orphans = validator.find_orphan_nodes()
    """

    def __init__(self, graph: Any):
        self.graph = graph
        self._schema_validator = get_validator()

    def validate_edge(
        self,
        source_type: str,
        target_type: str,
        edge_type: str,
        properties: dict[str, Any] | None = None,
    ) -> ValidationResult:
        """Validate an edge against registered schemas.

        Args:
            source_type: Memory type of source node.
            target_type: Memory type of target node.
            edge_type: Edge type string.
            properties: Edge properties dict.

        Returns:
            ValidationResult with any issues found.
        """
        issues: list[ValidationIssue] = []
        properties = properties or {}

        if edge_type not in self._schema_validator.edge_schemas:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    field="edge_type",
                    message=f"No schema registered for edge type: {edge_type}",
                )
            )
            return ValidationResult(issues=issues)

        schema = self._schema_validator.edge_schemas[edge_type]

        if source_type not in schema.source_node_types:
            issues.append(
                ValidationIssue(
                    severity="error",
                    field="source_type",
                    message=(
                        f"Invalid source type {source_type} for {edge_type}. Expected: {schema.source_node_types}"
                    ),
                )
            )

        if target_type not in schema.target_node_types:
            issues.append(
                ValidationIssue(
                    severity="error",
                    field="target_type",
                    message=(
                        f"Invalid target type {target_type} for {edge_type}. Expected: {schema.target_node_types}"
                    ),
                )
            )

        for prop in schema.required_properties:
            if prop not in properties:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        field=prop,
                        message=f"Missing required edge property: {prop}",
                    )
                )

        return ValidationResult(issues=issues)

    def find_orphan_nodes(self) -> list[str]:
        """Find nodes with no edges (orphans).

        Returns:
            List of orphan node item_ids.
        """
        orphans = []
        try:
            all_nodes = self.graph.get_all_nodes()
            for node in all_nodes:
                item_id = getattr(node, "item_id", None) or (node.get("item_id") if isinstance(node, dict) else None)
                if not item_id:
                    continue
                edges = self.graph.get_edges_for_node(item_id)
                if not edges:
                    orphans.append(item_id)
        except Exception as e:
            logger.warning(f"Failed to check orphan nodes: {e}")
        return orphans

    def find_provenance_gaps(self, item_id: str) -> list[str]:
        """Check if a node has provenance edges (DERIVED_FROM, CAUSED_BY, or PRODUCED).

        Args:
            item_id: Node to check.

        Returns:
            List of missing provenance types (empty = has provenance).
        """
        gaps = []
        try:
            edges = self.graph.get_edges_for_node(item_id)
            has_provenance = any(e.get("type") in PROVENANCE_EDGE_TYPES for e in (edges or []))
            if not has_provenance:
                gaps.append("no_provenance_edge")
        except Exception as e:
            logger.warning(f"Failed to check provenance for {item_id}: {e}")
        return gaps
