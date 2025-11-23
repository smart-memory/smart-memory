"""
Tests for deduplication utility.
"""
import pytest
from smartmemory.models.memory_item import MemoryItem
from smartmemory.utils.deduplication import deduplicate_entities, normalize_text, lemmatize_text

def test_normalize_text():
    assert normalize_text("Apple") == "apple"
    assert normalize_text("  SPACES  ") == "spaces"
    assert normalize_text("Café") == "café"

def test_lemmatize_text():
    # Note: Spacy model behavior depends on version, but generally:
    assert "apple" in lemmatize_text("Apples")
    assert "run" in lemmatize_text("running")

def test_deduplicate_entities_simple():
    e1 = MemoryItem(content="Apple", item_id="1", memory_type="concept", metadata={"name": "Apple", "entity_type": "fruit", "confidence": 0.9})
    e2 = MemoryItem(content="apple", item_id="2", memory_type="concept", metadata={"name": "apple", "entity_type": "fruit", "confidence": 0.8})
    
    deduped = deduplicate_entities([e1, e2])
    assert len(deduped) == 1
    assert deduped[0].content == "Apple"  # Should pick higher confidence one
    assert deduped[0].metadata["confidence"] == 0.9

def test_deduplicate_entities_plural():
    e1 = MemoryItem(content="Cat", item_id="1", memory_type="concept", metadata={"name": "Cat", "entity_type": "animal", "confidence": 0.9})
    e2 = MemoryItem(content="Cats", item_id="2", memory_type="concept", metadata={"name": "Cats", "entity_type": "animal", "confidence": 0.8})
    
    deduped = deduplicate_entities([e1, e2])
    assert len(deduped) == 1
    # Lemma of Cats is Cat, so they should merge
    assert deduped[0].metadata["name"] in ["Cat", "Cats"]

def test_deduplicate_entities_different_types():
    e1 = MemoryItem(content="Apple", item_id="1", memory_type="concept", metadata={"name": "Apple", "entity_type": "fruit", "confidence": 0.9})
    e2 = MemoryItem(content="Apple", item_id="2", memory_type="concept", metadata={"name": "Apple", "entity_type": "company", "confidence": 0.9})
    
    deduped = deduplicate_entities([e1, e2])
    # Should NOT merge because types are different (fruit vs company)
    assert len(deduped) == 2
