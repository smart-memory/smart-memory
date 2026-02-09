"""Unit tests for reasoning resolution cascade and individual resolvers."""

from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit

from smartmemory.models.memory_item import MemoryItem
from smartmemory.reasoning.models import Conflict, ConflictType, ResolutionStrategy
from smartmemory.reasoning.resolution.base import ConflictResolver
from smartmemory.reasoning.resolution.cascade import ResolutionCascade
from smartmemory.reasoning.resolution.recency import RecencyResolver
from smartmemory.reasoning.resolution.grounding import GroundingResolver
from smartmemory.reasoning.resolution.wikipedia import (
    WikipediaResolver,
    _extract_entities_for_lookup,
    _text_supports_claim,
)


def _make_conflict(existing_content="Earth is round", new_fact="Earth is flat", metadata=None):
    item = MemoryItem(content=existing_content, memory_type="semantic", metadata=metadata or {})
    return Conflict(
        existing_item=item,
        existing_fact=existing_content,
        new_fact=new_fact,
        conflict_type=ConflictType.DIRECT_CONTRADICTION,
        confidence=0.8,
        explanation="test conflict",
        suggested_resolution=ResolutionStrategy.DEFER,
    )


# ---------------------------------------------------------------------------
# ResolutionCascade
# ---------------------------------------------------------------------------
class TestResolutionCascade:
    def test_returns_first_resolved(self):
        r1 = MagicMock(spec=ConflictResolver)
        r1.resolve.return_value = {"auto_resolved": True, "method": "first"}
        r2 = MagicMock(spec=ConflictResolver)

        cascade = ResolutionCascade([r1, r2])
        result = cascade.resolve(_make_conflict())
        assert result is not None
        assert result["method"] == "first"
        r2.resolve.assert_not_called()

    def test_skips_unresolved(self):
        r1 = MagicMock(spec=ConflictResolver)
        r1.resolve.return_value = None
        r2 = MagicMock(spec=ConflictResolver)
        r2.resolve.return_value = {"auto_resolved": True, "method": "second"}

        cascade = ResolutionCascade([r1, r2])
        result = cascade.resolve(_make_conflict())
        assert result["method"] == "second"

    def test_returns_none_when_all_fail(self):
        r1 = MagicMock(spec=ConflictResolver)
        r1.resolve.return_value = None

        cascade = ResolutionCascade([r1])
        assert cascade.resolve(_make_conflict()) is None

    def test_empty_cascade(self):
        cascade = ResolutionCascade([])
        assert cascade.resolve(_make_conflict()) is None

    def test_skips_non_auto_resolved(self):
        r1 = MagicMock(spec=ConflictResolver)
        r1.resolve.return_value = {"auto_resolved": False, "method": "partial"}
        r2 = MagicMock(spec=ConflictResolver)
        r2.resolve.return_value = {"auto_resolved": True, "method": "full"}

        cascade = ResolutionCascade([r1, r2])
        result = cascade.resolve(_make_conflict())
        assert result["method"] == "full"


# ---------------------------------------------------------------------------
# RecencyResolver
# ---------------------------------------------------------------------------
class TestRecencyResolver:
    def test_resolves_with_recency_indicator(self):
        conflict = _make_conflict(
            existing_content="The CEO is John",
            new_fact="The CEO is currently Jane",
            metadata={"valid_start_time": "2024-01-01"},
        )
        result = RecencyResolver().resolve(conflict)
        assert result is not None
        assert result["auto_resolved"] is True
        assert result["method"] == "recency"
        assert result["resolution"] == ResolutionStrategy.ACCEPT_NEW

    def test_no_resolve_without_recency_indicator(self):
        conflict = _make_conflict(
            existing_content="The CEO is John",
            new_fact="The CEO is Jane",
            metadata={"valid_start_time": "2024-01-01"},
        )
        result = RecencyResolver().resolve(conflict)
        assert result is None

    def test_no_resolve_without_timestamp(self):
        conflict = _make_conflict(
            existing_content="The CEO is John",
            new_fact="The CEO is currently Jane",
        )
        result = RecencyResolver().resolve(conflict)
        assert result is None

    def test_various_recency_indicators(self):
        for indicator in ["now", "today", "as of", "recent"]:
            conflict = _make_conflict(
                existing_content="Old fact",
                new_fact=f"The answer is {indicator} different",
                metadata={"timestamp": "2024-06-01"},
            )
            result = RecencyResolver().resolve(conflict)
            assert result is not None, f"Should resolve with indicator '{indicator}'"


# ---------------------------------------------------------------------------
# GroundingResolver
# ---------------------------------------------------------------------------
class TestGroundingResolver:
    def test_resolves_with_grounded_to(self):
        conflict = _make_conflict(metadata={"grounded_to": "https://example.com"})
        result = GroundingResolver().resolve(conflict)
        assert result is not None
        assert result["auto_resolved"] is True
        assert result["method"] == "grounding"
        assert result["resolution"] == ResolutionStrategy.KEEP_EXISTING

    def test_resolves_with_wikipedia_provenance(self):
        conflict = _make_conflict(metadata={"provenance": {"wikipedia_id": "Q123"}})
        result = GroundingResolver().resolve(conflict)
        assert result is not None
        assert result["auto_resolved"] is True

    def test_resolves_with_trusted_source(self):
        conflict = _make_conflict(metadata={"provenance": {"source": "Wikipedia article"}})
        result = GroundingResolver().resolve(conflict)
        assert result is not None
        assert result["confidence"] == 0.7

    def test_no_resolve_without_provenance(self):
        conflict = _make_conflict(metadata={})
        result = GroundingResolver().resolve(conflict)
        assert result is None

    def test_no_resolve_untrusted_source(self):
        conflict = _make_conflict(metadata={"provenance": {"source": "random blog"}})
        result = GroundingResolver().resolve(conflict)
        assert result is None


# ---------------------------------------------------------------------------
# Wikipedia helpers
# ---------------------------------------------------------------------------
class TestExtractEntitiesForLookup:
    def test_extracts_capitalized_words(self):
        entities = _extract_entities_for_lookup("The capital of France is Paris")
        assert "France" in entities
        assert "Paris" in entities

    def test_deduplicates(self):
        # First word of each text is skipped, so "Paris" at position 0 is skipped
        # Use texts where the entity appears after position 0
        entities = _extract_entities_for_lookup("The Paris skyline", "Visit Paris today")
        assert entities.count("Paris") == 1

    def test_skips_first_word(self):
        entities = _extract_entities_for_lookup("The cat sat")
        # "The" is first word, skipped; "cat" and "sat" are lowercase
        assert len(entities) == 0

    def test_strips_punctuation(self):
        entities = _extract_entities_for_lookup("I visited Paris, London, and Berlin.")
        assert "Paris" in entities
        assert "London" in entities
        assert "Berlin" in entities


class TestTextSupportsClaim:
    def test_supports_matching_claim(self):
        ref = "Paris is the capital city of France, located in Europe"
        claim = "Paris is the capital of France"
        assert _text_supports_claim(ref, claim) is True

    def test_does_not_support_unrelated(self):
        ref = "Tokyo is the capital of Japan"
        claim = "Berlin is the capital of Germany"
        assert _text_supports_claim(ref, claim) is False

    def test_empty_claim_words(self):
        assert _text_supports_claim("some text", "the a an") is False


# ---------------------------------------------------------------------------
# WikipediaResolver (mocked)
# ---------------------------------------------------------------------------
class TestWikipediaResolver:
    def test_returns_none_on_import_error(self):
        conflict = _make_conflict()
        result = WikipediaResolver().resolve(conflict)
        # Will fail to import WikipediaGrounder
        assert result is None

    def test_returns_none_for_no_entities(self):
        conflict = _make_conflict(existing_content="yes", new_fact="no")
        result = WikipediaResolver().resolve(conflict)
        assert result is None


# ---------------------------------------------------------------------------
# LLMResolver (mocked)
# ---------------------------------------------------------------------------
class TestLLMResolver:
    def test_returns_none_on_import_error(self):
        from smartmemory.reasoning.resolution.llm import LLMResolver

        conflict = _make_conflict()
        with patch("smartmemory.utils.llm.call_llm", side_effect=ImportError("mocked")):
            result = LLMResolver().resolve(conflict)
        assert result is None

    def test_resolves_with_mocked_llm_keep_existing(self):
        from smartmemory.reasoning.resolution.llm import LLMResolver

        llm_response = {
            "correct_statement": "A",
            "confidence": 0.9,
            "reasoning": "Existing is correct",
            "source": "general knowledge",
        }
        with patch("smartmemory.utils.llm.call_llm", return_value=(llm_response, "")):
            result = LLMResolver().resolve(_make_conflict())
            assert result is not None
            assert result["auto_resolved"] is True
            assert result["resolution"] == ResolutionStrategy.KEEP_EXISTING

    def test_resolves_with_mocked_llm_accept_new(self):
        from smartmemory.reasoning.resolution.llm import LLMResolver

        llm_response = {
            "correct_statement": "B",
            "confidence": 0.85,
            "reasoning": "New is correct",
            "source": "verified",
        }
        with patch("smartmemory.utils.llm.call_llm", return_value=(llm_response, "")):
            result = LLMResolver().resolve(_make_conflict())
            assert result is not None
            assert result["resolution"] == ResolutionStrategy.ACCEPT_NEW

    def test_no_resolve_low_confidence(self):
        from smartmemory.reasoning.resolution.llm import LLMResolver

        llm_response = {"correct_statement": "A", "confidence": 0.5, "reasoning": "Unsure"}
        with patch("smartmemory.utils.llm.call_llm", return_value=(llm_response, "")):
            result = LLMResolver().resolve(_make_conflict())
            assert result is None

    def test_no_resolve_unknown(self):
        from smartmemory.reasoning.resolution.llm import LLMResolver

        llm_response = {"correct_statement": "unknown", "confidence": 0.9, "reasoning": "Cannot determine"}
        with patch("smartmemory.utils.llm.call_llm", return_value=(llm_response, "")):
            result = LLMResolver().resolve(_make_conflict())
            assert result is None
