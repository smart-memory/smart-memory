"""Unit tests for reasoning detection cascade and individual detectors."""

from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit

from smartmemory.models.memory_item import MemoryItem
from smartmemory.reasoning.models import Conflict, ConflictType, ResolutionStrategy
from smartmemory.reasoning.detection.base import ContradictionDetector, DetectionContext
from smartmemory.reasoning.detection.cascade import DetectionCascade
from smartmemory.reasoning.detection.heuristic import (
    HeuristicDetector,
    _extract_common_context,
    NEGATION_PATTERNS,
)
from smartmemory.reasoning.detection.graph import GraphDetector, FUNCTIONAL_PATTERNS


def _make_ctx(new_assertion: str, existing_content: str, metadata=None) -> DetectionContext:
    item = MemoryItem(content=existing_content, memory_type="semantic", metadata=metadata or {})
    return DetectionContext(
        new_assertion=new_assertion,
        existing_item=item,
        existing_fact=existing_content,
    )


# ---------------------------------------------------------------------------
# DetectionCascade
# ---------------------------------------------------------------------------
class TestDetectionCascade:
    def test_returns_first_hit(self):
        d1 = MagicMock(spec=ContradictionDetector)
        d1.name = "first"
        d1.detect.return_value = Conflict(
            existing_item=MagicMock(), existing_fact="a", new_fact="b",
            conflict_type=ConflictType.DIRECT_CONTRADICTION, confidence=0.9,
            explanation="found by first",
            suggested_resolution=ResolutionStrategy.DEFER,
        )
        d2 = MagicMock(spec=ContradictionDetector)
        d2.name = "second"

        cascade = DetectionCascade([d1, d2])
        ctx = MagicMock()
        result = cascade.detect(ctx)

        assert result is not None
        assert "[FIRST]" in result.explanation
        d2.detect.assert_not_called()

    def test_returns_none_when_no_hit(self):
        d1 = MagicMock(spec=ContradictionDetector)
        d1.name = "d1"
        d1.detect.return_value = None
        d2 = MagicMock(spec=ContradictionDetector)
        d2.name = "d2"
        d2.detect.return_value = None

        cascade = DetectionCascade([d1, d2])
        assert cascade.detect(MagicMock()) is None

    def test_empty_cascade(self):
        cascade = DetectionCascade([])
        assert cascade.detect(MagicMock()) is None

    def test_skips_none_results(self):
        d1 = MagicMock(spec=ContradictionDetector)
        d1.name = "skip"
        d1.detect.return_value = None
        d2 = MagicMock(spec=ContradictionDetector)
        d2.name = "hit"
        d2.detect.return_value = Conflict(
            existing_item=MagicMock(), existing_fact="a", new_fact="b",
            conflict_type=ConflictType.DIRECT_CONTRADICTION, confidence=0.8,
            explanation="found",
            suggested_resolution=ResolutionStrategy.DEFER,
        )

        cascade = DetectionCascade([d1, d2])
        result = cascade.detect(MagicMock())
        assert result is not None
        assert "[HIT]" in result.explanation


# ---------------------------------------------------------------------------
# HeuristicDetector
# ---------------------------------------------------------------------------
class TestHeuristicDetector:
    def test_negation_is_not(self):
        ctx = _make_ctx("Python is not a compiled language", "Python is a compiled language")
        result = HeuristicDetector().detect(ctx)
        assert result is not None
        assert result.conflict_type == ConflictType.DIRECT_CONTRADICTION

    def test_negation_reverse(self):
        ctx = _make_ctx("Python is a compiled language", "Python is not a compiled language")
        result = HeuristicDetector().detect(ctx)
        assert result is not None

    def test_no_negation_no_conflict(self):
        ctx = _make_ctx("Python is a language", "Java is a language")
        result = HeuristicDetector().detect(ctx)
        assert result is None

    def test_numeric_mismatch(self):
        ctx = _make_ctx("The population of Paris is 3 million", "The population of Paris is 2 million")
        result = HeuristicDetector().detect(ctx)
        assert result is not None
        assert result.conflict_type == ConflictType.NUMERIC_MISMATCH

    def test_same_numbers_no_conflict(self):
        ctx = _make_ctx("The population is 2 million", "The population is 2 million")
        result = HeuristicDetector().detect(ctx)
        assert result is None

    def test_numbers_without_common_context(self):
        ctx = _make_ctx("There are 5 cats", "The price is 10 dollars")
        result = HeuristicDetector().detect(ctx)
        assert result is None

    def test_negation_patterns_list_not_empty(self):
        assert len(NEGATION_PATTERNS) > 0
        for neg, pos in NEGATION_PATTERNS:
            assert isinstance(neg, str)
            assert isinstance(pos, str)


class TestExtractCommonContext:
    def test_returns_common_words(self):
        result = _extract_common_context("the population of Paris is large", "the population of Paris is small")
        assert result is not None
        assert "population" in result.lower() or "paris" in result.lower()

    def test_returns_none_for_no_overlap(self):
        result = _extract_common_context("cats are great", "dogs are terrible")
        assert result is None

    def test_ignores_stopwords(self):
        result = _extract_common_context("the is a", "the is a")
        assert result is None


# ---------------------------------------------------------------------------
# GraphDetector
# ---------------------------------------------------------------------------
class TestGraphDetector:
    def test_functional_property_conflict(self):
        # 'president of' pattern: extracts next 1-2 words after pattern
        # "president of France" -> "France" vs "Germany" -> conflict
        ctx = _make_ctx(
            "president of France visited Berlin today",
            "president of Germany visited Berlin today",
        )
        result = GraphDetector().detect(ctx)
        assert result is not None
        assert result.conflict_type == ConflictType.DIRECT_CONTRADICTION
        assert result.confidence == 0.85

    def test_no_conflict_same_value(self):
        ctx = _make_ctx(
            "president of France visited Berlin",
            "president of France visited Berlin",
        )
        result = GraphDetector().detect(ctx)
        assert result is None

    def test_no_conflict_insufficient_overlap(self):
        ctx = _make_ctx("Cats are cute", "Dogs are loyal")
        result = GraphDetector().detect(ctx)
        assert result is None

    def test_no_functional_pattern(self):
        ctx = _make_ctx(
            "Python was created by Guido van Rossum",
            "Python is a programming language used widely",
        )
        result = GraphDetector().detect(ctx)
        assert result is None

    def test_functional_patterns_list(self):
        assert "capital of" in FUNCTIONAL_PATTERNS
        assert "president of" in FUNCTIONAL_PATTERNS

    def test_entity_metadata_used(self):
        # 'founder of' pattern extracts next 1-2 words
        # 'founder of Apple' -> 'Apple was' vs 'Apple was' (same) — need different values
        # Use 'born in' pattern: 'born in Paris' vs 'born in London'
        ctx = _make_ctx(
            "Einstein born in Germany studied physics deeply",
            "Einstein born in Switzerland studied physics deeply",
            metadata={"entities": [{"name": "Einstein", "type": "person"}]},
        )
        result = GraphDetector().detect(ctx)
        # 'born in' extracts 'germany studied' vs 'switzerland studied' -> conflict
        assert result is not None


# ---------------------------------------------------------------------------
# EmbeddingDetector (mocked — requires external deps)
# ---------------------------------------------------------------------------
class TestEmbeddingDetector:
    def test_returns_none_on_import_error(self):
        from smartmemory.reasoning.detection.embedding import EmbeddingDetector

        ctx = _make_ctx("test new", "test existing")
        # Will fail to import create_embeddings in test env
        result = EmbeddingDetector().detect(ctx)
        assert result is None

    def test_detects_contradiction_with_mocked_embeddings(self):
        from smartmemory.reasoning.detection.embedding import EmbeddingDetector
        import numpy as np

        # High similarity, opposite polarity
        emb = np.array([1.0, 0.0, 0.0])
        with patch("smartmemory.plugins.embedding.create_embeddings", return_value=emb):
            ctx = _make_ctx("Python is not a compiled language", "Python is a compiled language")
            result = EmbeddingDetector().detect(ctx)
            assert result is not None
            assert result.conflict_type == ConflictType.DIRECT_CONTRADICTION

    def test_no_conflict_same_polarity(self):
        from smartmemory.reasoning.detection.embedding import EmbeddingDetector
        import numpy as np

        emb = np.array([1.0, 0.0, 0.0])
        with patch("smartmemory.plugins.embedding.create_embeddings", return_value=emb):
            ctx = _make_ctx("Python is a great language", "Python is a popular language")
            result = EmbeddingDetector().detect(ctx)
            assert result is None

    def test_low_similarity_no_conflict(self):
        from smartmemory.reasoning.detection.embedding import EmbeddingDetector
        import numpy as np

        with patch("smartmemory.plugins.embedding.create_embeddings") as mock_emb:
            mock_emb.side_effect = [np.array([1, 0, 0]), np.array([0, 1, 0])]
            ctx = _make_ctx("Python is not good", "Java is great")
            result = EmbeddingDetector().detect(ctx)
            assert result is None


# ---------------------------------------------------------------------------
# LLMDetector (mocked)
# ---------------------------------------------------------------------------
class TestLLMDetector:
    def test_returns_none_on_import_error(self):
        from smartmemory.reasoning.detection.llm import LLMDetector

        ctx = _make_ctx("new fact", "existing fact")
        result = LLMDetector().detect(ctx)
        assert result is None

    def test_detects_contradiction_with_mocked_llm(self):
        from smartmemory.reasoning.detection.llm import LLMDetector

        llm_response = {
            "contradicts": True,
            "conflict_type": "direct_contradiction",
            "confidence": 0.9,
            "explanation": "These statements directly contradict",
            "resolution": "defer",
        }
        with patch("smartmemory.utils.llm.call_llm", return_value=(llm_response, "")):
            ctx = _make_ctx("Earth is flat", "Earth is round")
            result = LLMDetector().detect(ctx)
            assert result is not None
            assert result.confidence == 0.9

    def test_no_contradiction_from_llm(self):
        from smartmemory.reasoning.detection.llm import LLMDetector

        llm_response = {"contradicts": False, "conflict_type": "none", "confidence": 0.1, "explanation": "No conflict"}
        with patch("smartmemory.utils.llm.call_llm", return_value=(llm_response, "")):
            ctx = _make_ctx("Python is popular", "Python is widely used")
            result = LLMDetector().detect(ctx)
            assert result is None

    def test_fallback_json_parsing(self):
        from smartmemory.reasoning.detection.llm import LLMDetector

        raw = '{"contradicts": true, "conflict_type": "direct_contradiction", "confidence": 0.8, "explanation": "test", "resolution": "defer"}'
        with patch("smartmemory.utils.llm.call_llm", return_value=(None, raw)):
            ctx = _make_ctx("A", "B")
            result = LLMDetector().detect(ctx)
            assert result is not None
            assert result.confidence == 0.8
