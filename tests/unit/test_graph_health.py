"""Unit tests for GraphHealthChecker."""

from unittest.mock import MagicMock

import pytest

from smartmemory.metrics.graph_health import GraphHealthChecker, HealthReport


@pytest.fixture
def mock_graph():
    return MagicMock()


@pytest.fixture
def checker(mock_graph):
    memory = MagicMock()
    memory._graph = mock_graph
    return GraphHealthChecker(memory)


class TestHealthReport:
    def test_orphan_ratio_zero_nodes(self):
        report = HealthReport(total_nodes=0, orphan_count=0)
        assert report.orphan_ratio == 0.0

    def test_orphan_ratio(self):
        report = HealthReport(total_nodes=100, orphan_count=10)
        assert report.orphan_ratio == 0.1

    def test_is_healthy(self):
        report = HealthReport(
            total_nodes=100, orphan_count=5, provenance_coverage=0.8,
        )
        assert report.is_healthy

    def test_unhealthy_high_orphans(self):
        report = HealthReport(
            total_nodes=100, orphan_count=30, provenance_coverage=0.8,
        )
        assert not report.is_healthy

    def test_unhealthy_low_provenance(self):
        report = HealthReport(
            total_nodes=100, orphan_count=5, provenance_coverage=0.3,
        )
        assert not report.is_healthy

    def test_to_dict(self):
        report = HealthReport(total_nodes=10, total_edges=5, orphan_count=2, provenance_coverage=0.8)
        d = report.to_dict()
        assert d["total_nodes"] == 10
        assert "orphan_ratio" in d
        assert "is_healthy" in d

    def test_from_dict(self):
        original = HealthReport(total_nodes=50, provenance_coverage=0.9)
        restored = HealthReport.from_dict(original.to_dict())
        assert restored.total_nodes == 50


class TestCollectHealth:
    def test_collects_basic_metrics(self, checker, mock_graph):
        mock_graph.backend.execute_cypher.side_effect = [
            [[100]],   # count nodes
            [[50]],    # count edges
            [["semantic", 60], ["episodic", 40]],  # type dist
            [["RELATED", 30], ["DERIVED_FROM", 20]],  # edge dist
            [[5]],     # orphans
            [[10]],    # total decisions
            [[8]],     # decisions with provenance
        ]
        report = checker.collect_health()
        assert report.total_nodes == 100
        assert report.total_edges == 50
        assert report.orphan_count == 5
        assert report.provenance_coverage == 0.8

    def test_handles_empty_graph(self, checker, mock_graph):
        mock_graph.backend.execute_cypher.return_value = []
        report = checker.collect_health()
        assert report.total_nodes == 0

    def test_handles_no_decisions(self, checker, mock_graph):
        mock_graph.backend.execute_cypher.side_effect = [
            [[0]], [[0]], [], [], [[0]], [[0]],
        ]
        report = checker.collect_health()
        assert report.provenance_coverage == 1.0  # no decisions = no gap

    def test_handles_graph_error(self, checker, mock_graph):
        mock_graph.backend.execute_cypher.side_effect = RuntimeError("connection lost")
        report = checker.collect_health()
        assert report.total_nodes == 0  # returns zeroed report
