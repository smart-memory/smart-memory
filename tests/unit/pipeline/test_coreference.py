"""Tests for coreference resolution stage."""
import pytest


pytestmark = pytest.mark.unit
from unittest.mock import MagicMock, patch

from smartmemory.memory.pipeline.stages.coreference import (
    CoreferenceStage,
    CoreferenceResult,
)
from smartmemory.memory.pipeline.config import CoreferenceConfig


class TestCoreferenceConfig:
    """Test CoreferenceConfig validation."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CoreferenceConfig()
        assert config.enabled is True
        assert config.resolver == "fastcoref"
        assert config.device == "auto"
        assert config.min_text_length == 50

    def test_invalid_resolver(self):
        """Test that invalid resolver raises error."""
        with pytest.raises(ValueError, match="resolver must be one of"):
            CoreferenceConfig(resolver="invalid")

    def test_invalid_device(self):
        """Test that invalid device raises error."""
        with pytest.raises(ValueError, match="device must be one of"):
            CoreferenceConfig(device="tpu")


class TestCoreferenceStage:
    """Test CoreferenceStage functionality."""

    def test_skip_short_text(self):
        """Test that short text is skipped."""
        stage = CoreferenceStage(min_text_length=50)
        result = stage.run("Short text")

        assert result.skipped is True
        assert "too short" in result.skip_reason
        assert result.resolved_text == "Short text"

    def test_skip_empty_text(self):
        """Test that empty text is skipped."""
        stage = CoreferenceStage()
        result = stage.run("")

        assert result.skipped is True
        assert result.resolved_text == ""

    def test_skip_when_disabled(self):
        """Test that coreference is skipped when disabled."""
        stage = CoreferenceStage()
        config = CoreferenceConfig(enabled=False)
        result = stage.run("Some text that is long enough to process", config=config)

        assert result.skipped is True
        assert "disabled" in result.skip_reason

    @patch("smartmemory.memory.pipeline.stages.coreference.get_coref_model")
    def test_model_not_available(self, mock_get_model):
        """Test handling when model is not available."""
        mock_get_model.return_value = None

        stage = CoreferenceStage()
        result = stage.run("This is a long enough text to be processed by the system.")

        assert result.skipped is True
        assert "not available" in result.skip_reason

    @patch("smartmemory.memory.pipeline.stages.coreference.get_coref_model")
    def test_coreference_resolution(self, mock_get_model):
        """Test successful coreference resolution."""
        # Mock the fastcoref model
        mock_model = MagicMock()
        mock_result = MagicMock()
        mock_result.get_clusters.return_value = [
            ["Apple Inc.", "The company", "it"],
            ["Tim Cook", "He", "the CEO"],
        ]
        mock_model.predict.return_value = [mock_result]
        mock_get_model.return_value = mock_model

        stage = CoreferenceStage()
        text = "Apple Inc. announced earnings. The company exceeded expectations. It grew revenue."
        result = stage.run(text)

        assert result.skipped is False
        assert len(result.chains) > 0
        # Check that "The company" was replaced with "Apple Inc."
        assert "Apple Inc." in result.chains[0]["head"]

    def test_score_mention_prefers_proper_nouns(self):
        """Test that mention scoring prefers proper nouns."""
        stage = CoreferenceStage()

        # Proper noun should score higher
        assert stage._score_mention("Apple Inc.") > stage._score_mention("it")
        assert stage._score_mention("Tim Cook") > stage._score_mention("he")

    def test_score_mention_penalizes_pronouns(self):
        """Test that pronouns are penalized."""
        stage = CoreferenceStage()

        assert stage._score_mention("he") < 0
        assert stage._score_mention("she") < 0
        assert stage._score_mention("it") < 0
        assert stage._score_mention("they") < 0

    def test_find_head_mention(self):
        """Test finding the most informative mention."""
        stage = CoreferenceStage()

        mentions = ["Apple Inc.", "The company", "it"]
        head = stage._find_head_mention(mentions)
        assert head == "Apple Inc."

        mentions = ["Tim Cook", "He", "the CEO"]
        head = stage._find_head_mention(mentions)
        assert head == "Tim Cook"


class TestCoreferenceResult:
    """Test CoreferenceResult dataclass."""

    def test_result_fields(self):
        """Test result dataclass fields."""
        result = CoreferenceResult(
            original_text="original",
            resolved_text="resolved",
            chains=[{"mentions": ["Apple", "it"], "head": "Apple"}],
            replacements_made=1,
        )

        assert result.original_text == "original"
        assert result.resolved_text == "resolved"
        assert len(result.chains) == 1
        assert result.replacements_made == 1
        assert result.skipped is False
