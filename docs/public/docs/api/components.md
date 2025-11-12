# Components API

This page documents the component-level APIs available in SmartMemory. Components provide modular, reusable building blocks for memory operations, enrichment, and integration.

## Core Components

### 1. Memory Adapter
Adapters allow SmartMemory to interface with different storage backends and data sources.

**Key Methods:**
- `add(item)`
- `get(item_id)`
- `search(query)`

### 2. Enricher
Enrichers add metadata, semantic tags, or perform transformations on memory items during ingestion.

**Key Methods:**
- `enrich(item)`

### 3. Extractor
Extractors identify entities, relationships, and key information from memory content.

**Key Methods:**
- `extract(item)`

### 4. Converter
Converters transform memory items between different formats or schemas.

**Key Methods:**
- `convert(item)`

## Example Usage

```python
from smartmemory.adapters import MemoryAdapter
from smartmemory.enrichers import MetadataEnricher
from smartmemory.extractors import EntityExtractor
from smartmemory.converters import JsonConverter

adapter = MemoryAdapter()
enricher = MetadataEnricher()
extractor = EntityExtractor()
converter = JsonConverter()

item = {"content": "Alice met Bob at OpenAI."}

# Enrich and extract
item = enricher.enrich(item)
entities = extractor.extract(item)
json_item = converter.convert(item)

adapter.add(json_item)
```

## See Also
- [SmartMemory API](smart-memory)
- [Tools API](tools)
- [Factories API](factories)
