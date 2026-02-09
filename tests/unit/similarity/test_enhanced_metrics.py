"""Unit tests for enhanced similarity metrics."""

from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.unit

from smartmemory.models.memory_item import MemoryItem
from smartmemory.similarity.enhanced_metrics import (
    SimilarityResult,
    ContentSimilarityMetric,
    SemanticSimilarityMetric,
    TemporalSimilarityMetric,
    GraphSimilarityMetric,
    MetadataSimilarityMetric,
    AgentWorkflowSimilarityMetric,
)


def _make_item(content="test", item_id=None, metadata=None, transaction_time=None):
    kwargs = {"content": content}
    if item_id:
        kwargs["item_id"] = item_id
    if metadata:
        kwargs["metadata"] = metadata
    if transaction_time:
        kwargs["transaction_time"] = transaction_time
    return MemoryItem(**kwargs)


# ---------------------------------------------------------------------------
# SimilarityResult
# ---------------------------------------------------------------------------
class TestSimilarityResult:
    def test_defaults(self):
        r = SimilarityResult(overall_score=0.5)
        assert r.overall_score == 0.5
        assert r.semantic_score == 0.0
        assert r.confidence == 1.0
        assert r.explanation == ""

    def test_all_fields(self):
        r = SimilarityResult(
            overall_score=0.8, semantic_score=0.9, graph_score=0.7,
            temporal_score=0.6, metadata_score=0.5, content_score=0.4,
            agent_workflow_score=0.3, explanation="test", confidence=0.95,
        )
        assert r.agent_workflow_score == 0.3


# ---------------------------------------------------------------------------
# ContentSimilarityMetric
# ---------------------------------------------------------------------------
class TestContentSimilarityMetric:
    @pytest.fixture
    def metric(self):
        return ContentSimilarityMetric()

    def test_identical_content(self, metric):
        item1 = _make_item("the quick brown fox jumps over the lazy dog")
        item2 = _make_item("the quick brown fox jumps over the lazy dog")
        score = metric.calculate(item1, item2)
        assert score > 0.8

    def test_completely_different(self, metric):
        item1 = _make_item("alpha beta gamma delta")
        item2 = _make_item("one two three four")
        score = metric.calculate(item1, item2)
        assert score < 0.2

    def test_partial_overlap(self, metric):
        item1 = _make_item("python is a great programming language")
        item2 = _make_item("python is used for data science")
        score = metric.calculate(item1, item2)
        assert 0.1 < score < 0.9

    def test_empty_content(self, metric):
        item1 = _make_item("")
        item2 = _make_item("something")
        assert metric.calculate(item1, item2) == 0.0

    def test_both_empty(self, metric):
        item1 = _make_item("")
        item2 = _make_item("")
        assert metric.calculate(item1, item2) == 0.0

    def test_name_property(self, metric):
        assert metric.name == "content_similarity"

    def test_jaccard_similarity(self, metric):
        # Direct test of internal method
        score = metric._jaccard_similarity("a b c", "b c d")
        # intersection={b,c}, union={a,b,c,d} => 2/4 = 0.5
        assert score == pytest.approx(0.5)

    def test_length_similarity_bonus(self, metric):
        bonus = metric._length_similarity_bonus("short", "short text here")
        assert 0.0 < bonus < 1.0

    def test_length_similarity_identical_length(self, metric):
        bonus = metric._length_similarity_bonus("abcd", "efgh")
        assert bonus == pytest.approx(1.0)

    def test_fuzzy_disabled(self):
        metric = ContentSimilarityMetric(use_fuzzy_matching=False)
        item1 = _make_item("programming language")
        item2 = _make_item("programming languages")
        score = metric.calculate(item1, item2)
        # Without fuzzy, "language" != "languages" so lower score
        assert score < 0.8

    def test_edit_distance_similarity(self, metric):
        assert metric._edit_distance_similarity("test", "test") == pytest.approx(1.0)
        assert metric._edit_distance_similarity("test", "tast") == pytest.approx(0.75)
        assert metric._edit_distance_similarity("", "") == pytest.approx(0.0)
        assert metric._edit_distance_similarity("", "abc") == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# SemanticSimilarityMetric
# ---------------------------------------------------------------------------
class TestSemanticSimilarityMetric:
    @pytest.fixture
    def metric(self):
        return SemanticSimilarityMetric()

    def test_name_property(self, metric):
        assert metric.name == "semantic_similarity"

    def test_fallback_to_conceptual_similarity(self, metric):
        # Without sentence-transformers installed, falls back to conceptual
        item1 = _make_item("machine learning algorithms for classification")
        item2 = _make_item("machine learning models for prediction")
        score = metric.calculate(item1, item2)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_conceptual_similarity_overlap(self, metric):
        score = metric._conceptual_similarity(
            "python programming language features",
            "python programming data structures"
        )
        assert score > 0.0

    def test_conceptual_similarity_no_overlap(self, metric):
        score = metric._conceptual_similarity("abc def", "xyz uvw")
        assert score == 0.0

    def test_extract_concepts_filters_stopwords(self, metric):
        concepts = metric._extract_concepts("the quick brown fox and the lazy dog")
        assert "the" not in concepts
        assert "and" not in concepts
        assert "quick" in concepts
        assert "brown" in concepts

    def test_extract_concepts_filters_short_words(self, metric):
        concepts = metric._extract_concepts("a is to be or not")
        assert len(concepts) == 0

    def test_cosine_similarity_identical(self, metric):
        vec = [1.0, 0.0, 1.0]
        assert metric._cosine_similarity(vec, vec) == pytest.approx(1.0)

    def test_cosine_similarity_orthogonal(self, metric):
        assert metric._cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_cosine_similarity_different_lengths(self, metric):
        assert metric._cosine_similarity([1.0], [1.0, 2.0]) == 0.0

    def test_cosine_similarity_zero_vector(self, metric):
        assert metric._cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


# ---------------------------------------------------------------------------
# TemporalSimilarityMetric
# ---------------------------------------------------------------------------
class TestTemporalSimilarityMetric:
    @pytest.fixture
    def metric(self):
        return TemporalSimilarityMetric()

    def test_name_property(self, metric):
        assert metric.name == "temporal_similarity"

    def test_identical_timestamps(self, metric):
        now = datetime.now(timezone.utc)
        item1 = _make_item("a", transaction_time=now)
        item2 = _make_item("b", transaction_time=now)
        score = metric.calculate(item1, item2)
        assert score > 0.8

    def test_very_close_timestamps(self, metric):
        now = datetime.now(timezone.utc)
        item1 = _make_item("a", transaction_time=now)
        item2 = _make_item("b", transaction_time=now + timedelta(seconds=30))
        score = metric.calculate(item1, item2)
        assert score > 0.8

    def test_same_hour(self, metric):
        now = datetime.now(timezone.utc)
        item1 = _make_item("a", transaction_time=now)
        item2 = _make_item("b", transaction_time=now + timedelta(minutes=30))
        score = metric.calculate(item1, item2)
        assert 0.2 < score < 0.9

    def test_same_day(self, metric):
        now = datetime.now(timezone.utc)
        item1 = _make_item("a", transaction_time=now)
        item2 = _make_item("b", transaction_time=now + timedelta(hours=12))
        score = metric.calculate(item1, item2)
        assert 0.0 < score < 0.7

    def test_different_weeks(self, metric):
        now = datetime.now(timezone.utc)
        item1 = _make_item("a", transaction_time=now)
        item2 = _make_item("b", transaction_time=now + timedelta(days=30))
        score = metric.calculate(item1, item2)
        assert score < 0.2

    def test_no_timestamps(self, metric):
        item1 = _make_item("a")
        item2 = _make_item("b")
        # Both have transaction_time set by default (very close), so score should be high
        score = metric.calculate(item1, item2)
        assert score >= 0.0

    def test_temporal_pattern_extraction(self, metric):
        patterns = metric._extract_temporal_patterns("I work every monday morning")
        assert "monday" in patterns
        assert "morning" in patterns

    def test_temporal_pattern_no_match(self, metric):
        patterns = metric._extract_temporal_patterns("python is great")
        assert len(patterns) == 0

    def test_shared_temporal_patterns_boost(self, metric):
        # Items with shared temporal keywords get a pattern boost
        now = datetime.now(timezone.utc)
        item1 = _make_item("meeting every monday morning", transaction_time=now)
        item2 = _make_item("standup on monday morning", transaction_time=now + timedelta(days=30))
        score = metric.calculate(item1, item2)
        # Even with distant timestamps, shared patterns provide some score
        assert score > 0.0


# ---------------------------------------------------------------------------
# GraphSimilarityMetric
# ---------------------------------------------------------------------------
class TestGraphSimilarityMetric:
    @pytest.fixture
    def metric(self):
        return GraphSimilarityMetric()

    def test_name_property(self, metric):
        assert metric.name == "graph_similarity"

    def test_entity_overlap(self, metric):
        item1 = _make_item("Einstein developed relativity", metadata={"entities": ["Einstein", "relativity"]})
        item2 = _make_item("Einstein won Nobel Prize", metadata={"entities": ["Einstein", "Nobel"]})
        score = metric._entity_overlap_score(item1, item2)
        assert score > 0.0

    def test_no_entity_overlap(self, metric):
        item1 = _make_item("cats are cute", metadata={"entities": ["cats"]})
        item2 = _make_item("dogs are loyal", metadata={"entities": ["dogs"]})
        score = metric._entity_overlap_score(item1, item2)
        # Entities from metadata don't overlap, but simple extraction from content
        # may find capitalized words
        assert isinstance(score, float)

    def test_relationship_patterns(self, metric):
        item1 = _make_item("rain causes flooding")
        item2 = _make_item("drought leads to famine")
        score = metric._relationship_similarity_score(item1, item2)
        # Both have causal patterns
        assert score > 0.0

    def test_no_relationship_patterns(self, metric):
        item1 = _make_item("hello world")
        item2 = _make_item("goodbye world")
        score = metric._relationship_similarity_score(item1, item2)
        assert score == 0.0

    def test_graph_connectivity_no_store(self, metric):
        item1 = _make_item("a", item_id="id1")
        item2 = _make_item("b", item_id="id2")
        score = metric._graph_connectivity_score(item1, item2)
        assert score == 0.0

    def test_graph_connectivity_with_store(self):
        store = MagicMock()
        neighbor = MagicMock()
        neighbor.item_id = "shared_neighbor"
        store.get_neighbors.return_value = [neighbor]

        metric = GraphSimilarityMetric(graph_store=store)
        item1 = _make_item("a", item_id="id1")
        item2 = _make_item("b", item_id="id2")
        score = metric._graph_connectivity_score(item1, item2)
        assert score > 0.0

    def test_simple_entity_extraction(self, metric):
        entities = metric._simple_entity_extraction("Albert Einstein was born in Germany")
        assert "Albert" in entities
        assert "Einstein" in entities
        assert "Germany" in entities
        assert "was" not in entities

    def test_extract_relationship_patterns(self, metric):
        patterns = metric._extract_relationship_patterns("A causes B and contains C")
        assert "causal" in patterns
        assert "containment" in patterns


# ---------------------------------------------------------------------------
# MetadataSimilarityMetric
# ---------------------------------------------------------------------------
class TestMetadataSimilarityMetric:
    @pytest.fixture
    def metric(self):
        return MetadataSimilarityMetric()

    def test_name_property(self, metric):
        assert metric.name == "metadata_similarity"

    def test_same_type(self, metric):
        item1 = _make_item("a", metadata={"type": "semantic"})
        item2 = _make_item("b", metadata={"type": "semantic"})
        score = metric._type_similarity(item1, item2)
        assert score == 1.0

    def test_different_type(self, metric):
        item1 = _make_item("a", metadata={"type": "semantic"})
        item2 = _make_item("b", metadata={"type": "episodic"})
        score = metric._type_similarity(item1, item2)
        assert score == 0.0

    def test_tag_overlap(self, metric):
        item1 = _make_item("a", metadata={"tags": ["python", "ml"]})
        item2 = _make_item("b", metadata={"tags": ["python", "web"]})
        score = metric._tag_similarity(item1, item2)
        # intersection=1, union=3 => 1/3
        assert score == pytest.approx(1.0 / 3.0)

    def test_no_tags(self, metric):
        item1 = _make_item("a", metadata={})
        item2 = _make_item("b", metadata={})
        assert metric._tag_similarity(item1, item2) == 0.0

    def test_same_category(self, metric):
        item1 = _make_item("a", metadata={"category": "tech"})
        item2 = _make_item("b", metadata={"category": "tech"})
        assert metric._category_similarity(item1, item2) == 1.0

    def test_different_category(self, metric):
        item1 = _make_item("a", metadata={"category": "tech"})
        item2 = _make_item("b", metadata={"category": "health"})
        assert metric._category_similarity(item1, item2) == 0.0

    def test_confidence_similarity(self, metric):
        item1 = _make_item("a", metadata={"confidence": 0.9})
        item2 = _make_item("b", metadata={"confidence": 0.8})
        score = metric._confidence_similarity(item1, item2)
        assert score == pytest.approx(0.9)

    def test_confidence_identical(self, metric):
        item1 = _make_item("a", metadata={"confidence": 0.7})
        item2 = _make_item("b", metadata={"confidence": 0.7})
        assert metric._confidence_similarity(item1, item2) == pytest.approx(1.0)

    def test_full_calculate(self, metric):
        item1 = _make_item("a", metadata={"type": "semantic", "tags": ["ml"], "category": "tech", "confidence": 0.9})
        item2 = _make_item("b", metadata={"type": "semantic", "tags": ["ml"], "category": "tech", "confidence": 0.9})
        score = metric.calculate(item1, item2)
        assert score > 0.8


# ---------------------------------------------------------------------------
# AgentWorkflowSimilarityMetric
# ---------------------------------------------------------------------------
class TestAgentWorkflowSimilarityMetric:
    @pytest.fixture
    def metric(self):
        return AgentWorkflowSimilarityMetric()

    def test_name_property(self, metric):
        assert metric.name == "agent_workflow_similarity"

    def test_empty_content(self, metric):
        item1 = _make_item("")
        item2 = _make_item("something")
        assert metric.calculate(item1, item2) == 0.0

    def test_sequential_step_numbers(self, metric):
        item1 = _make_item("Step 1: Set up the database connection")
        item2 = _make_item("Step 2: Create the schema and indexes")
        score = metric.calculate(item1, item2)
        assert score > 0.7

    def test_problem_solution_pair(self, metric):
        item1 = _make_item("Issue: Database queries are running slowly")
        item2 = _make_item("Solution: Added indexes on frequently queried columns")
        score = metric.calculate(item1, item2)
        assert score > 0.7

    def test_learning_progression(self, metric):
        item1 = _make_item("Initial attempt failed to implement the authentication system")
        item2 = _make_item("Successfully implemented OAuth authentication with JWT tokens")
        score = metric.calculate(item1, item2)
        assert score > 0.5

    def test_unrelated_content(self, metric):
        item1 = _make_item("The weather is sunny today")
        item2 = _make_item("I had pasta for lunch")
        score = metric.calculate(item1, item2)
        assert score < 0.3

    def test_extract_workflow_patterns(self, metric):
        patterns = metric._extract_workflow_patterns("Step 1: first configure the database service")
        assert patterns["has_step_number"] is True
        assert "first" in patterns["sequential_indicators"]
        assert "database" in patterns["technical_terms"]
        assert "service" in patterns["technical_terms"]

    def test_is_sequential_workflow(self, metric):
        p1 = metric._extract_workflow_patterns("Step 1: Setup database")
        p2 = metric._extract_workflow_patterns("Step 2: Create indexes")
        assert metric._is_sequential_workflow(p1, p2) is True

    def test_is_problem_solution_pair(self, metric):
        p1 = metric._extract_workflow_patterns("Issue: Authentication error with OAuth")
        p2 = metric._extract_workflow_patterns("Solution: Updated JWT token configuration")
        assert metric._is_problem_solution_pair(p1, p2) is True

    def test_is_not_problem_solution_pair(self, metric):
        p1 = metric._extract_workflow_patterns("The sky is blue")
        p2 = metric._extract_workflow_patterns("Water is wet")
        assert metric._is_problem_solution_pair(p1, p2) is False

    def test_technical_context_similarity(self, metric):
        p1 = metric._extract_workflow_patterns("Configure the database queries and indexes")
        p2 = metric._extract_workflow_patterns("Optimize database indexes for performance")
        score = metric._technical_context_similarity(p1, p2)
        assert score > 0.5
