"""Graph health metrics collection.

Measures graph quality: orphan nodes, edge distribution, provenance coverage.
All queries use property-based Cypher (not label-based) for FalkorDB compatibility.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from smartmemory.models.base import MemoryBaseModel

logger = logging.getLogger(__name__)

ORPHAN_THRESHOLD = 0.2
PROVENANCE_THRESHOLD = 0.5


@dataclass
class HealthReport(MemoryBaseModel):
    """Graph health metrics snapshot."""

    total_nodes: int = 0
    total_edges: int = 0
    orphan_count: int = 0
    type_distribution: dict[str, int] = field(default_factory=dict)
    edge_distribution: dict[str, int] = field(default_factory=dict)
    provenance_coverage: float = 0.0

    @property
    def orphan_ratio(self) -> float:
        """Fraction of nodes with no connections."""
        if self.total_nodes == 0:
            return 0.0
        return self.orphan_count / self.total_nodes

    @property
    def is_healthy(self) -> bool:
        """Whether the graph meets health thresholds."""
        return self.orphan_ratio < ORPHAN_THRESHOLD and self.provenance_coverage > PROVENANCE_THRESHOLD

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_nodes": self.total_nodes,
            "total_edges": self.total_edges,
            "orphan_count": self.orphan_count,
            "orphan_ratio": round(self.orphan_ratio, 4),
            "type_distribution": self.type_distribution,
            "edge_distribution": self.edge_distribution,
            "provenance_coverage": round(self.provenance_coverage, 4),
            "is_healthy": self.is_healthy,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HealthReport":
        return cls(
            total_nodes=data.get("total_nodes", 0),
            total_edges=data.get("total_edges", 0),
            orphan_count=data.get("orphan_count", 0),
            type_distribution=data.get("type_distribution", {}),
            edge_distribution=data.get("edge_distribution", {}),
            provenance_coverage=data.get("provenance_coverage", 0.0),
        )


class GraphHealthChecker:
    """Collects graph health metrics via Cypher queries.

    Usage:
        checker = GraphHealthChecker(smart_memory)
        report = checker.collect_health()
        if not report.is_healthy:
            logger.warning("Graph unhealthy: %s", report.to_dict())
    """

    def __init__(self, memory: Any, graph: Any = None):
        self.memory = memory
        self.graph = graph or getattr(memory, "_graph", None)

    def collect_health(self) -> HealthReport:
        """Collect all health metrics into a single report."""
        report = HealthReport()
        try:
            report.total_nodes = self._count_nodes()
            report.total_edges = self._count_edges()
            report.type_distribution = self._type_distribution()
            report.edge_distribution = self._edge_distribution()
            report.orphan_count = self._count_orphans()
            report.provenance_coverage = self._provenance_coverage()
        except Exception as e:
            logger.warning("Health check failed: %s", e)
        return report

    def _count_nodes(self) -> int:
        rows = self._cypher("MATCH (n) RETURN count(n)")
        return rows[0][0] if rows else 0

    def _count_edges(self) -> int:
        rows = self._cypher("MATCH ()-[r]->() RETURN count(r)")
        return rows[0][0] if rows else 0

    def _type_distribution(self) -> dict[str, int]:
        rows = self._cypher("MATCH (n) WHERE n.memory_type IS NOT NULL RETURN n.memory_type AS t, count(n) AS c")
        return {row[0]: row[1] for row in rows} if rows else {}

    def _edge_distribution(self) -> dict[str, int]:
        rows = self._cypher("MATCH ()-[r]->() RETURN type(r) AS t, count(r) AS c")
        return {row[0]: row[1] for row in rows} if rows else {}

    def _count_orphans(self) -> int:
        rows = self._cypher("MATCH (n) WHERE NOT (n)-[]-() RETURN count(n)")
        return rows[0][0] if rows else 0

    def _provenance_coverage(self) -> float:
        """Fraction of decision nodes that have provenance edges."""
        total_rows = self._cypher("MATCH (n {memory_type: 'decision'}) RETURN count(n)")
        total_count = total_rows[0][0] if total_rows else 0
        if total_count == 0:
            return 1.0

        with_prov_rows = self._cypher(
            "MATCH (n {memory_type: 'decision'})-[:DERIVED_FROM|:CAUSED_BY|:PRODUCED]-() RETURN count(DISTINCT n)"
        )
        with_count = with_prov_rows[0][0] if with_prov_rows else 0
        return with_count / total_count

    def _cypher(self, query: str, params: dict | None = None) -> list:
        if not self.graph:
            return []
        try:
            return self.graph.backend.execute_cypher(query, params)
        except Exception as e:
            logger.debug("Cypher query failed: %s", e)
            return []
