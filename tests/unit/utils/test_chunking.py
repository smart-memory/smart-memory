"""Unit tests for utils.chunking — text chunking strategies."""

import pytest

pytestmark = pytest.mark.unit

from smartmemory.utils.chunking import (
    chunk_text,
    _chunk_by_characters,
    _chunk_by_sentences,
    _chunk_by_paragraphs,
    _chunk_recursive,
    _chunk_by_markdown,
)


# ---------------------------------------------------------------------------
# chunk_text dispatcher
# ---------------------------------------------------------------------------
class TestChunkText:
    def test_empty_text(self):
        assert chunk_text("") == []

    def test_short_text_returns_single(self):
        assert chunk_text("Hello world", chunk_size=100) == ["Hello world"]

    def test_character_strategy(self):
        text = "a" * 200
        chunks = chunk_text(text, chunk_size=100, overlap=0, strategy="character")
        assert len(chunks) == 2

    def test_sentence_strategy_default(self):
        text = "First sentence. Second sentence. Third sentence. " * 20
        chunks = chunk_text(text, chunk_size=100, overlap=0, strategy="sentence")
        assert len(chunks) > 1

    def test_paragraph_strategy(self):
        text = "Para one content.\n\nPara two content.\n\nPara three content." * 10
        chunks = chunk_text(text, chunk_size=100, overlap=0, strategy="paragraph")
        assert len(chunks) > 1

    def test_recursive_strategy(self):
        text = "Para one.\n\nPara two.\n\nPara three." * 20
        chunks = chunk_text(text, chunk_size=100, overlap=0, strategy="recursive")
        assert len(chunks) > 1


# ---------------------------------------------------------------------------
# Character chunking
# ---------------------------------------------------------------------------
class TestChunkByCharacters:
    def test_exact_fit(self):
        text = "a" * 100
        chunks = _chunk_by_characters(text, 100, 0)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_split_with_no_overlap(self):
        text = "a" * 250
        chunks = _chunk_by_characters(text, 100, 0)
        assert len(chunks) == 3
        assert len(chunks[0]) == 100
        assert len(chunks[1]) == 100
        assert len(chunks[2]) == 50

    def test_split_with_overlap(self):
        text = "a" * 200
        chunks = _chunk_by_characters(text, 100, 20)
        # With overlap of 20, second chunk starts at 80
        assert len(chunks) >= 2
        assert len(chunks[0]) == 100


# ---------------------------------------------------------------------------
# Sentence chunking
# ---------------------------------------------------------------------------
class TestChunkBySentences:
    def test_preserves_sentences(self):
        text = "First sentence. Second sentence. Third sentence."
        chunks = _chunk_by_sentences(text, 50, 0)
        # Each chunk should contain complete sentences
        for chunk in chunks:
            assert chunk.strip()

    def test_long_sentence_falls_back_to_chars(self):
        text = "A" * 200 + ". Short."
        chunks = _chunk_by_sentences(text, 100, 0)
        assert len(chunks) >= 2

    def test_single_sentence_fits(self):
        text = "Just one sentence."
        chunks = _chunk_by_sentences(text, 100, 0)
        assert len(chunks) == 1


# ---------------------------------------------------------------------------
# Paragraph chunking
# ---------------------------------------------------------------------------
class TestChunkByParagraphs:
    def test_preserves_paragraphs(self):
        text = "Para one.\n\nPara two.\n\nPara three."
        chunks = _chunk_by_paragraphs(text, 30, 0)
        assert len(chunks) >= 2

    def test_long_paragraph_falls_back(self):
        text = "A" * 200 + "\n\nShort para."
        chunks = _chunk_by_paragraphs(text, 100, 0)
        assert len(chunks) >= 2


# ---------------------------------------------------------------------------
# Recursive chunking
# ---------------------------------------------------------------------------
class TestChunkRecursive:
    def test_single_large_paragraph(self):
        text = "Word. " * 100  # No paragraph breaks
        chunks = _chunk_recursive(text, 100, 0)
        assert len(chunks) >= 2

    def test_multiple_paragraphs(self):
        text = "Para one content here.\n\nPara two content here.\n\nPara three content here."
        chunks = _chunk_recursive(text, 40, 0)
        assert len(chunks) >= 2


# ---------------------------------------------------------------------------
# Markdown chunking
# ---------------------------------------------------------------------------
class TestChunkByMarkdown:
    def test_splits_on_headers(self):
        text = "# Header 1\nContent one.\n# Header 2\nContent two." * 5
        chunks = _chunk_by_markdown(text, 80, 0)
        assert len(chunks) >= 2

    def test_preserves_code_blocks(self):
        text = "# Intro\nSome text.\n```python\nprint('hello')\n```\n# Next\nMore text." * 5
        chunks = _chunk_by_markdown(text, 100, 0)
        # Code blocks should not be split
        for chunk in chunks:
            if "```" in chunk:
                assert chunk.count("```") % 2 == 0 or "```" not in chunk

    def test_no_markdown_falls_back(self):
        text = "Plain text without any markdown headers at all. " * 20
        chunks = _chunk_by_markdown(text, 100, 0)
        assert len(chunks) >= 1
