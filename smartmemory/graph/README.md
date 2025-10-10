# Entity/Relation Extraction - Usage and Extensibility

This module provides a pluggable interface for extracting entities and relations from text, powering graph-based memory and agentic workflows.

## Interface

- `EntityRelationExtractor`: Protocol for any extractor implementation.
- `LLMEntityRelationExtractor`: Default, LLM-based implementation (can use OpenAI, Anthropic, etc.).
- `get_entity_relation_extractor(name, **kwargs)`: Factory for selecting an extractor.

## How to Use

- Ingest a note or memory:
    1. Call `get_entity_relation_extractor()` to get the default extractor.
    2. Call `extractor.extract(note_text)` to get entities and relations.
    3. Store or use these in your graph/memory system.

## Extending

- Add new extractors by subclassing `EntityRelationExtractor` and registering in `EXTRACTORS`.
- Examples: `SpacyEntityRelationExtractor`, `RelikEntityRelationExtractor`, etc.

## Example

```python
from smartmemory.graph.entity_relation_extraction import get_entity_relation_extractor

extractor = get_entity_relation_extractor()
result = extractor.extract("Marie Curie discovered radium in Paris in 1898.")
print(result)
```

## Roadmap

- Replace print with real graph DB integration.
- Add more extractors as needed.
