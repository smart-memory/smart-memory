"""Unit tests for SimplifyStage."""

from unittest.mock import MagicMock, patch

from smartmemory.pipeline.config import PipelineConfig, SimplifyConfig
from smartmemory.pipeline.state import PipelineState
from smartmemory.pipeline.stages.simplify import SimplifyStage


def _make_mock_nlp(sentences_text=None):
    """Build a mock spaCy Language that returns configurable sentences."""
    nlp = MagicMock()

    def process(text):
        doc = MagicMock()
        doc.text = text

        if sentences_text is None:
            # Single sentence with no special deps
            sent = MagicMock()
            sent.text = text
            sent.start_char = 0
            sent.__iter__ = lambda self: iter([])
            doc.sents = [sent]
        else:
            sents = []
            for s_text in sentences_text:
                sent = MagicMock()
                sent.text = s_text
                sent.start_char = text.find(s_text) if s_text in text else 0
                sent.__iter__ = lambda self: iter([])
                sents.append(sent)
            doc.sents = sents

        return doc

    nlp.side_effect = process
    return nlp


class TestSimplifyStage:
    """Tests for the simplify pipeline stage."""

    def test_disabled_mode_returns_text_as_single_element(self):
        """When disabled, returns text as single-element list."""
        stage = SimplifyStage()
        state = PipelineState(text="Hello world, this is a test.")
        config = PipelineConfig()
        config.simplify = SimplifyConfig(enabled=False)

        result = stage.execute(state, config)

        assert result.simplified_sentences == ["Hello world, this is a test."]

    def test_short_text_returned_as_single_sentence(self):
        """Text shorter than min_token_count is returned as-is."""
        nlp = _make_mock_nlp()
        stage = SimplifyStage(nlp=nlp)
        state = PipelineState(text="Hi there")
        config = PipelineConfig()
        config.simplify = SimplifyConfig(min_token_count=10)

        result = stage.execute(state, config)

        assert result.simplified_sentences == ["Hi there"]

    def test_empty_text_returns_empty_list(self):
        """Empty text produces empty list."""
        stage = SimplifyStage()
        state = PipelineState(text="")
        config = PipelineConfig()

        result = stage.execute(state, config)

        assert result.simplified_sentences == []

    def test_prefers_resolved_text(self):
        """Stage uses resolved_text over text when available."""
        nlp = _make_mock_nlp(["John Smith is incredibly smart and talented."])
        stage = SimplifyStage(nlp=nlp)
        state = PipelineState(
            text="He is incredibly smart and talented.",
            resolved_text="John Smith is incredibly smart and talented.",
        )
        config = PipelineConfig()

        result = stage.execute(state, config)

        # Verify the nlp was called with resolved_text (not text)
        assert nlp.call_count == 1
        assert nlp.call_args[0][0] == "John Smith is incredibly smart and talented."

    def test_basic_text_passes_through(self):
        """Simple text with no special deps passes through."""
        nlp = _make_mock_nlp(["Claude is an AI assistant."])
        stage = SimplifyStage(nlp=nlp)
        state = PipelineState(text="Claude is an AI assistant.")
        config = PipelineConfig()

        result = stage.execute(state, config)

        assert "Claude is an AI assistant." in result.simplified_sentences

    def test_spacy_not_available_returns_text(self):
        """When spaCy is not available, returns text as single sentence."""
        stage = SimplifyStage(nlp=None)
        state = PipelineState(text="This is a long enough sentence for testing purposes.")
        config = PipelineConfig()

        with patch("smartmemory.pipeline.stages.simplify._get_nlp", return_value=None):
            result = stage.execute(state, config)

        assert result.simplified_sentences == ["This is a long enough sentence for testing purposes."]

    def test_exception_falls_back_to_text(self):
        """When spaCy processing raises, returns text as fallback."""
        nlp = MagicMock(side_effect=RuntimeError("spaCy failed"))
        stage = SimplifyStage(nlp=nlp)
        state = PipelineState(text="This is a test sentence for the stage.")
        config = PipelineConfig()

        result = stage.execute(state, config)

        assert result.simplified_sentences == ["This is a test sentence for the stage."]

    def test_flag_disabled_skips_transform(self):
        """When all flags disabled, text passes through without transforms."""
        nlp = _make_mock_nlp(["Claude is an AI and it works well."])
        stage = SimplifyStage(nlp=nlp)
        state = PipelineState(text="Claude is an AI and it works well.")
        config = PipelineConfig()
        config.simplify = SimplifyConfig(
            split_clauses=False,
            extract_relative=False,
            passive_to_active=False,
            extract_appositives=False,
        )

        result = stage.execute(state, config)

        assert "Claude is an AI and it works well." in result.simplified_sentences

    def test_undo_clears_simplified_sentences(self):
        """Undo resets simplified_sentences to empty."""
        stage = SimplifyStage()
        state = PipelineState(
            text="Some text.",
            simplified_sentences=["Some text."],
        )

        result = stage.undo(state)

        assert result.simplified_sentences == []
