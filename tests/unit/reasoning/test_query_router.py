"""Unit tests for QueryRouter."""

from unittest.mock import MagicMock

import pytest

from smartmemory.reasoning.query_router import QueryRouter, QueryType


@pytest.fixture
def mock_memory():
    memory = MagicMock()
    memory._graph = MagicMock()
    memory._graph.backend = MagicMock()
    memory.search.return_value = []
    return memory


@pytest.fixture
def router(mock_memory):
    return QueryRouter(mock_memory)


class TestClassify:
    def test_who_is_symbolic(self, router):
        assert router.classify("Who created Python?") == QueryType.SYMBOLIC

    def test_what_is_symbolic(self, router):
        assert router.classify("What is the capital of France?") == QueryType.SYMBOLIC

    def test_similar_to_is_semantic(self, router):
        assert router.classify("Find memories similar to machine learning") == QueryType.SEMANTIC

    def test_related_is_semantic(self, router):
        assert router.classify("What's related to Python programming?") == QueryType.SEMANTIC

    def test_why_is_hybrid(self, router):
        assert router.classify("Why did we choose FastAPI?") == QueryType.HYBRID

    def test_explain_is_hybrid(self, router):
        assert router.classify("Explain the authentication decision") == QueryType.HYBRID

    def test_default_is_semantic(self, router):
        assert router.classify("machine learning best practices") == QueryType.SEMANTIC


class TestRoute:
    def test_symbolic_uses_cypher(self, router, mock_memory):
        mock_memory._graph.backend.execute_cypher.return_value = [["result"]]
        result = router.route("Who is the CEO?")
        assert result["query_type"] == "symbolic"
        mock_memory._graph.backend.execute_cypher.assert_called()

    def test_semantic_uses_search(self, router, mock_memory):
        mock_memory.search.return_value = [MagicMock(content="test")]
        result = router.route("Find similar to deep learning")
        assert result["query_type"] == "semantic"
        mock_memory.search.assert_called()

    def test_hybrid_uses_both(self, router, mock_memory):
        mock_memory._graph.backend.execute_cypher.return_value = []
        mock_memory.search.return_value = []
        result = router.route("Why did we pick React?")
        assert result["query_type"] == "hybrid"
