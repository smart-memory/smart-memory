# Factories API

This page documents the Factory Pattern APIs in SmartMemory. Factories provide standardized ways to create and configure memory components, adapters, and tools for various use cases.

## Overview

Factories are responsible for instantiating and configuring:
- Memory adapters (for different storage backends)
- Enrichers and extractors
- Tools for integration (e.g., MCP, LangChain)

## Core Factory Methods

### 1. Adapter Factory
Creates adapters for supported storage backends.

**Example:**
```python
from smartmemory.factories import AdapterFactory
adapter = AdapterFactory.create("FalkorDBBackend")
```

### 2. Enricher Factory
Creates enrichers for metadata, semantic tags, etc.

**Example:**
```python
from smartmemory.factories import EnricherFactory
enricher = EnricherFactory.create("MetadataEnricher")
```

### 3. Extractor Factory
Creates extractors for entity/relationship extraction.

**Example:**
```python
from smartmemory.factories import ExtractorFactory
extractor = ExtractorFactory.create("EntityExtractor")
```

### 4. Tool Factory
Creates integration tools for external systems.

**Example:**
```python
from smartmemory.factories import ToolFactory
tool = ToolFactory.create("MCPTool")
```

## See Also
- [SmartMemory API](smart-memory)
- [Components API](components)
- [Tools API](tools)
