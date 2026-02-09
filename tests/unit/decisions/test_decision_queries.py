"""Unit tests for DecisionQueries."""

from unittest.mock import MagicMock

import pytest


pytestmark = pytest.mark.unit

from smartmemory.decisions.queries import DecisionQueries


@pytest.fixture
def mock_memory():
    """Create a mock SmartMemory instance."""
    memory = MagicMock()
    memory._graph = MagicMock()
    memory.get.return_value = None
    memory.search.return_value = []
    return memory


@pytest.fixture
def mock_graph(mock_memory):
    return mock_memory._graph


@pytest.fixture
def queries(mock_memory):
    return DecisionQueries(mock_memory)


def _make_decision_item(decision_id, content, status="active", domain=None,
                         decision_type="inference", confidence=0.8, **extra):
    """Helper to create a mock MemoryItem representing a decision."""
    item = MagicMock()
    item.content = content
    metadata = {
        "decision_id": decision_id,
        "content": content,
        "status": status,
        "domain": domain,
        "decision_type": decision_type,
        "confidence": confidence,
    }
    metadata.update(extra)
    item.metadata = metadata
    return item


class TestGetActiveDecisions:
    """Test filtered decision retrieval."""

    def test_returns_active_only(self, queries, mock_memory):
        active = _make_decision_item("dec_1", "Active decision", status="active")
        superseded = _make_decision_item("dec_2", "Old decision", status="superseded")
        mock_memory.search.return_value = [active, superseded]

        results = queries.get_active_decisions()
        assert len(results) == 1
        assert results[0].decision_id == "dec_1"

    def test_filter_by_domain(self, queries, mock_memory):
        prefs = _make_decision_item("dec_1", "Pref", domain="preferences")
        facts = _make_decision_item("dec_2", "Fact", domain="facts")
        mock_memory.search.return_value = [prefs, facts]

        results = queries.get_active_decisions(domain="preferences")
        assert len(results) == 1
        assert results[0].domain == "preferences"

    def test_filter_by_type(self, queries, mock_memory):
        pref = _make_decision_item("dec_1", "Pref", decision_type="preference")
        inf = _make_decision_item("dec_2", "Inf", decision_type="inference")
        mock_memory.search.return_value = [pref, inf]

        results = queries.get_active_decisions(decision_type="preference")
        assert len(results) == 1
        assert results[0].decision_type == "preference"

    def test_filter_by_min_confidence(self, queries, mock_memory):
        high = _make_decision_item("dec_1", "High", confidence=0.9)
        low = _make_decision_item("dec_2", "Low", confidence=0.3)
        mock_memory.search.return_value = [high, low]

        results = queries.get_active_decisions(min_confidence=0.5)
        assert len(results) == 1
        assert results[0].confidence == 0.9

    def test_respects_limit(self, queries, mock_memory):
        items = [_make_decision_item(f"dec_{i}", f"Decision {i}") for i in range(10)]
        mock_memory.search.return_value = items

        results = queries.get_active_decisions(limit=3)
        assert len(results) == 3

    def test_handles_search_failure(self, queries, mock_memory):
        mock_memory.search.side_effect = Exception("Search error")
        results = queries.get_active_decisions()
        assert results == []


class TestGetDecisionsAbout:
    """Test semantic search for decisions."""

    def test_returns_active_matches(self, queries, mock_memory):
        match = _make_decision_item("dec_1", "User likes dark mode")
        mock_memory.search.return_value = [match]

        results = queries.get_decisions_about("dark mode")
        assert len(results) == 1
        mock_memory.search.assert_called_once_with(
            query="dark mode", memory_type="decision", top_k=20,
        )

    def test_filters_inactive(self, queries, mock_memory):
        active = _make_decision_item("dec_1", "Active", status="active")
        retracted = _make_decision_item("dec_2", "Retracted", status="retracted")
        mock_memory.search.return_value = [active, retracted]

        results = queries.get_decisions_about("test")
        assert len(results) == 1

    def test_handles_empty_results(self, queries, mock_memory):
        mock_memory.search.return_value = []
        results = queries.get_decisions_about("nonexistent topic")
        assert results == []


class TestGetDecisionProvenance:
    """Test provenance chain retrieval."""

    def test_returns_decision(self, queries, mock_memory):
        item = _make_decision_item("dec_1", "Test decision")
        mock_memory.get.return_value = item
        mock_memory._graph.get_incoming_neighbors.return_value = []
        mock_memory._graph.get_neighbors.return_value = []

        result = queries.get_decision_provenance("dec_1")
        assert result["decision"] is not None
        assert result["decision"].decision_id == "dec_1"

    def test_returns_reasoning_trace(self, queries, mock_memory, mock_graph):
        item = _make_decision_item("dec_1", "Test")
        mock_memory.get.return_value = item

        trace_node = {"item_id": "trace_abc", "content": "Reasoning trace"}
        mock_graph.get_incoming_neighbors.return_value = [trace_node]
        mock_graph.get_neighbors.return_value = []

        result = queries.get_decision_provenance("dec_1")
        assert result["reasoning_trace"] == trace_node

    def test_returns_evidence(self, queries, mock_memory, mock_graph):
        item = _make_decision_item("dec_1", "Test")
        mock_memory.get.return_value = item
        mock_graph.get_incoming_neighbors.return_value = []

        evidence_node = {"item_id": "mem_1", "content": "Evidence"}

        # get_neighbors called for DERIVED_FROM and SUPERSEDES
        def neighbors_side_effect(item_id, edge_type=None):
            if edge_type == "DERIVED_FROM":
                return [evidence_node]
            return []

        mock_graph.get_neighbors.side_effect = neighbors_side_effect

        result = queries.get_decision_provenance("dec_1")
        assert len(result["evidence"]) == 1
        assert result["evidence"][0]["memory_id"] == "mem_1"

    def test_returns_superseded(self, queries, mock_memory, mock_graph):
        item = _make_decision_item("dec_new", "New")
        mock_memory.get.return_value = item
        mock_graph.get_incoming_neighbors.return_value = []

        old_node = MagicMock()
        old_node.content = "Old decision"
        old_node.metadata = {"decision_id": "dec_old", "content": "Old decision", "status": "superseded"}

        def neighbors_side_effect(item_id, edge_type=None):
            if edge_type == "SUPERSEDES":
                return [old_node]
            return []

        mock_graph.get_neighbors.side_effect = neighbors_side_effect

        result = queries.get_decision_provenance("dec_new")
        assert len(result["superseded"]) == 1

    def test_not_found(self, queries, mock_memory):
        mock_memory.get.return_value = None
        result = queries.get_decision_provenance("dec_missing")
        assert result["decision"] is None
        assert result["evidence"] == []

    def test_no_graph(self, mock_memory):
        mock_memory._graph = None
        queries = DecisionQueries(mock_memory, graph=None)
        item = _make_decision_item("dec_1", "Test")
        mock_memory.get.return_value = item

        result = queries.get_decision_provenance("dec_1")
        assert result["decision"] is not None
        # No graph means no provenance edges
        assert result["reasoning_trace"] is None
        assert result["evidence"] == []


class TestGetCausalChain:
    """Test causal chain traversal."""

    def test_causes_direction(self, queries, mock_memory, mock_graph):
        item = _make_decision_item("dec_1", "Decision")
        mock_memory.get.return_value = item

        cause_node = {"item_id": "mem_cause", "content": "Cause"}
        mock_graph.get_neighbors.return_value = [cause_node]

        result = queries.get_causal_chain("dec_1", direction="causes", max_depth=1)
        assert result["decision"] is not None
        assert len(result["causes"]) > 0
        assert result["effects"] == []

    def test_effects_direction(self, queries, mock_memory, mock_graph):
        item = _make_decision_item("dec_1", "Decision")
        mock_memory.get.return_value = item

        effect_node = {"item_id": "mem_effect", "content": "Effect"}
        mock_graph.get_neighbors.return_value = [effect_node]

        result = queries.get_causal_chain("dec_1", direction="effects", max_depth=1)
        assert result["decision"] is not None
        assert result["causes"] == []
        assert len(result["effects"]) > 0

    def test_both_direction(self, queries, mock_memory, mock_graph):
        item = _make_decision_item("dec_1", "Decision")
        mock_memory.get.return_value = item

        neighbor = {"item_id": "mem_related", "content": "Related"}
        mock_graph.get_neighbors.return_value = [neighbor]

        result = queries.get_causal_chain("dec_1", direction="both", max_depth=1)
        assert len(result["causes"]) > 0
        assert len(result["effects"]) > 0

    def test_max_depth_respected(self, queries, mock_memory, mock_graph):
        item = _make_decision_item("dec_1", "Decision")
        mock_memory.get.return_value = item

        # Return a neighbor that has no further neighbors (depth 1 only)
        depth1_node = {"item_id": "mem_d1", "content": "Depth 1"}

        call_count = [0]

        def neighbors_side_effect(item_id, edge_type=None):
            call_count[0] += 1
            if item_id == "dec_1":
                return [depth1_node]
            return []  # No further neighbors

        mock_graph.get_neighbors.side_effect = neighbors_side_effect

        result = queries.get_causal_chain("dec_1", direction="causes", max_depth=2)
        assert len(result["causes"]) > 0
        # The depth-1 node should have empty causes (no further depth)
        assert result["causes"][0]["causes"] == []

    def test_not_found(self, queries, mock_memory):
        mock_memory.get.return_value = None
        result = queries.get_causal_chain("dec_missing")
        assert result["decision"] is None
        assert result["causes"] == []
        assert result["effects"] == []

    def test_no_graph(self, mock_memory):
        mock_memory._graph = None
        queries = DecisionQueries(mock_memory, graph=None)
        item = _make_decision_item("dec_1", "Test")
        mock_memory.get.return_value = item

        result = queries.get_causal_chain("dec_1")
        assert result["decision"] is not None
        assert result["causes"] == []
        assert result["effects"] == []


class TestCausalChainDepth:
    """Test recursive causal chain traversal."""

    def test_multi_level_causes(self, queries, mock_memory, mock_graph):
        item = _make_decision_item("dec_root", "Root")
        mock_memory.get.return_value = item

        level1 = {"item_id": "mem_l1", "content": "Level 1"}
        level2 = {"item_id": "mem_l2", "content": "Level 2"}

        def neighbors_side_effect(item_id, edge_type=None):
            # Only return for DERIVED_FROM to avoid double-counting from CAUSED_BY
            if edge_type != "DERIVED_FROM":
                return []
            if item_id == "dec_root":
                return [level1]
            elif item_id == "mem_l1":
                return [level2]
            return []

        mock_graph.get_neighbors.side_effect = neighbors_side_effect

        result = queries.get_causal_chain("dec_root", direction="causes", max_depth=3)
        assert len(result["causes"]) == 1
        assert result["causes"][0]["node_id"] == "mem_l1"
        # Level 2 should be nested
        nested = result["causes"][0]["causes"]
        assert len(nested) == 1
        assert nested[0]["node_id"] == "mem_l2"
