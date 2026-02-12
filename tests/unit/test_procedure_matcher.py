"""Unit tests for CFS-2: ProcedureMatcher."""

from unittest.mock import MagicMock

import pytest

from smartmemory.procedure_matcher import (
    ProcedureMatcher,
    ProcedureMatcherConfig,
    ProcedureMatchResult,
)


@pytest.fixture
def mock_smart_memory():
    sm = MagicMock()
    sm.search.return_value = []
    return sm


class TestProcedureMatchResult:
    def test_serialization(self):
        result = ProcedureMatchResult(
            matched=True,
            procedure_id="proc-1",
            procedure_name="Test Procedure",
            confidence=0.92,
            recommended_profile="quick_extract",
            threshold=0.85,
            match_id="test-match-id",
        )
        d = result.to_dict()
        assert d["matched"] is True
        assert d["procedure_id"] == "proc-1"
        assert d["procedure_name"] == "Test Procedure"
        assert d["confidence"] == 0.92
        assert d["recommended_profile"] == "quick_extract"
        assert d["threshold"] == 0.85
        assert d["match_id"] == "test-match-id"

    def test_default_values(self):
        result = ProcedureMatchResult()
        d = result.to_dict()
        assert d["matched"] is False
        assert d["procedure_id"] is None
        assert d["confidence"] == 0.0
        assert d["recommended_profile"] is None
        assert d["threshold"] == 0.85
        assert d["match_id"]  # auto-generated uuid

    def test_to_dict_has_exact_keys(self):
        result = ProcedureMatchResult()
        d = result.to_dict()
        assert set(d.keys()) == {
            "matched",
            "procedure_id",
            "procedure_name",
            "confidence",
            "recommended_profile",
            "threshold",
            "match_id",
            "drift_detected",
            "drift_event_id",
            "effective_confidence",
        }


class TestProcedureMatcherConfig:
    def test_disabled_by_default(self):
        config = ProcedureMatcherConfig()
        assert config.enabled is False
        assert config.confidence_threshold == 0.85
        assert config.max_candidates == 3

    def test_custom_threshold(self):
        config = ProcedureMatcherConfig(enabled=True, confidence_threshold=0.70)
        assert config.enabled is True
        assert config.confidence_threshold == 0.70

    def test_profile_mapping_defaults(self):
        config = ProcedureMatcherConfig()
        assert "extraction" in config.profile_mapping
        assert config.profile_mapping["extraction"] == "quick_extract"


class TestProcedureMatcher:
    def test_disabled_returns_no_match(self, mock_smart_memory):
        matcher = ProcedureMatcher(mock_smart_memory)
        result = matcher.match("some content")
        assert result.matched is False
        mock_smart_memory.search.assert_not_called()

    def test_enabled_no_procedures_returns_no_match(self, mock_smart_memory):
        config = ProcedureMatcherConfig(enabled=True)
        matcher = ProcedureMatcher(mock_smart_memory, config=config)
        result = matcher.match("some content")
        assert result.matched is False

    def test_empty_content_returns_no_match(self, mock_smart_memory):
        config = ProcedureMatcherConfig(enabled=True)
        matcher = ProcedureMatcher(mock_smart_memory, config=config)
        result = matcher.match("")
        assert result.matched is False
        mock_smart_memory.search.assert_not_called()

    def test_below_threshold_returns_no_match(self, mock_smart_memory):
        config = ProcedureMatcherConfig(enabled=True, confidence_threshold=0.85)
        mock_smart_memory.search.return_value = [{"item_id": "proc-1", "content": "test procedure", "score": 0.70}]
        matcher = ProcedureMatcher(mock_smart_memory, config=config)
        result = matcher.match("some content")
        assert result.matched is False
        assert result.confidence == 0.70

    def test_above_threshold_returns_match(self, mock_smart_memory):
        config = ProcedureMatcherConfig(enabled=True, confidence_threshold=0.85)
        mock_smart_memory.search.return_value = [
            {"item_id": "proc-1", "content": "test procedure", "score": 0.92, "metadata": {}}
        ]
        matcher = ProcedureMatcher(mock_smart_memory, config=config)
        result = matcher.match("some content")
        assert result.matched is True
        assert result.procedure_id == "proc-1"
        assert result.confidence == 0.92
        assert result.recommended_profile == "quick_extract"

    def test_preferred_profile_from_metadata(self, mock_smart_memory):
        config = ProcedureMatcherConfig(enabled=True, confidence_threshold=0.85)
        mock_smart_memory.search.return_value = [
            {
                "item_id": "proc-1",
                "content": "test procedure",
                "score": 0.95,
                "metadata": {"preferred_profile": "custom_profile"},
            }
        ]
        matcher = ProcedureMatcher(mock_smart_memory, config=config)
        result = matcher.match("some content")
        assert result.matched is True
        assert result.recommended_profile == "custom_profile"

    def test_profile_mapping_by_procedure_type(self, mock_smart_memory):
        config = ProcedureMatcherConfig(
            enabled=True,
            confidence_threshold=0.85,
            profile_mapping={"summarization": "summary_only"},
        )
        mock_smart_memory.search.return_value = [
            {
                "item_id": "proc-1",
                "content": "summarize notes",
                "score": 0.90,
                "metadata": {"procedure_type": "summarization"},
            }
        ]
        matcher = ProcedureMatcher(mock_smart_memory, config=config)
        result = matcher.match("summarize my meeting notes")
        assert result.matched is True
        assert result.recommended_profile == "summary_only"

    def test_explicit_pipeline_config_skips_matching(self, mock_smart_memory):
        """When pipeline_config is already set, matching should be skipped.

        This is tested at the SmartMemory.ingest() level — the ProcedureMatcher
        itself always runs when enabled. The guard is in ingest().
        """
        config = ProcedureMatcherConfig(enabled=True)
        matcher = ProcedureMatcher(mock_smart_memory, config=config)
        # Even if we call match(), it runs (the guard is in ingest, not matcher)
        result = matcher.match("content")
        assert result.matched is False  # No procedures available

    def test_search_failure_is_non_fatal(self, mock_smart_memory):
        config = ProcedureMatcherConfig(enabled=True)
        mock_smart_memory.search.side_effect = RuntimeError("DB down")
        matcher = ProcedureMatcher(mock_smart_memory, config=config)
        result = matcher.match("some content")
        assert result.matched is False

    def test_picks_best_candidate(self, mock_smart_memory):
        config = ProcedureMatcherConfig(enabled=True, confidence_threshold=0.85)
        mock_smart_memory.search.return_value = [
            {"item_id": "proc-1", "content": "low match", "score": 0.60, "metadata": {}},
            {"item_id": "proc-2", "content": "best match", "score": 0.95, "metadata": {}},
            {"item_id": "proc-3", "content": "mid match", "score": 0.80, "metadata": {}},
        ]
        matcher = ProcedureMatcher(mock_smart_memory, config=config)
        result = matcher.match("test query")
        assert result.matched is True
        assert result.procedure_id == "proc-2"
        assert result.confidence == 0.95

    def test_at_threshold_returns_match(self, mock_smart_memory):
        config = ProcedureMatcherConfig(enabled=True, confidence_threshold=0.85)
        mock_smart_memory.search.return_value = [
            {"item_id": "proc-1", "content": "exact threshold", "score": 0.85, "metadata": {}}
        ]
        matcher = ProcedureMatcher(mock_smart_memory, config=config)
        result = matcher.match("some content")
        assert result.matched is True
        assert result.confidence == 0.85

    def test_non_dict_candidate_preserves_metadata(self, mock_smart_memory):
        """When search returns objects (not dicts), metadata should be preserved."""
        config = ProcedureMatcherConfig(enabled=True, confidence_threshold=0.85)
        candidate = MagicMock()
        candidate.item_id = "proc-1"
        candidate.content = "test procedure"
        candidate.score = 0.95
        candidate.metadata = {"preferred_profile": "custom_profile"}
        mock_smart_memory.search.return_value = [candidate]
        matcher = ProcedureMatcher(mock_smart_memory, config=config)
        result = matcher.match("test query")
        assert result.matched is True
        assert result.recommended_profile == "custom_profile"

    def test_candidate_without_id_returns_no_match(self, mock_smart_memory):
        """When best candidate has neither item_id nor id, matched should be False."""
        config = ProcedureMatcherConfig(enabled=True, confidence_threshold=0.85)
        mock_smart_memory.search.return_value = [{"content": "orphan procedure", "score": 0.95, "metadata": {}}]
        matcher = ProcedureMatcher(mock_smart_memory, config=config)
        result = matcher.match("some content")
        assert result.matched is False
        assert result.confidence == 0.95

    def test_truncates_content_for_search_query(self, mock_smart_memory):
        config = ProcedureMatcherConfig(enabled=True)
        matcher = ProcedureMatcher(mock_smart_memory, config=config)
        long_content = "x" * 1000
        matcher.match(long_content)
        call_args = mock_smart_memory.search.call_args
        assert len(call_args.kwargs.get("query", call_args[1].get("query", ""))) == 500
