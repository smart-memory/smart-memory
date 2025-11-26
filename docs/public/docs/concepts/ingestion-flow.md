# Ingestion Flow

SmartMemory's ingestion pipeline transforms raw input into enriched, interconnected memories through a sophisticated multi-stage process.

## Overview

The ingestion flow consists of **11 stages** that process memories from initial input to final storage:

1. **Input Adaptation** - Parse and validate input (str, dict, MemoryItem)
2. **Classification** - Determine memory type (semantic, episodic, procedural, working)
3. **Extraction** - Extract entities and relationships (LLM → SpaCy → GLiNER → Relik fallback)
4. **Storage** - Create memory node + entity nodes in FalkorDB
5. **Linking** - Connect to related existing memories
6. **Vector Storage** - Generate embeddings, store in HNSW index
7. **Enrichment** - Add Wikipedia summaries, categories, metadata
8. **Grounding** - Create GROUNDED_IN edges to Wikipedia nodes
9. **Evolution** - Promote working → episodic/procedural if thresholds met
10. **Clustering** - SemHash + embedding deduplication of entities
11. **Versioning** - Create bi-temporal version record

## Stage Details

### 1. Input Processing

```python
# Raw input can be various formats
memory.ingest("I learned Python programming in 2020")
memory.ingest({
    "content": "Meeting notes from today",
    "metadata": {"participants": ["Alice", "Bob"]},
    "memory_type": "episodic"
})
```

**Note:** Use `ingest()` for full pipeline processing. Use `add()` for simple storage without extraction/linking/evolution.

**Processing:**
- Content validation and sanitization
- Metadata extraction and normalization
- Format standardization

### 2. Content Analysis

SmartMemory supports multiple extractors for entity and relationship extraction, each with different capabilities and performance characteristics.

#### Extraction Types

**Entity Extraction:**
- People, places, organizations
- Dates, times, durations
- Technical concepts and terms
- Actions and events

**Triple/Relationship Extraction:**
- Subject-predicate-object triples
- Temporal relationships (BEFORE, AFTER, DURING)
- Causal connections (CAUSES, RESULTS_IN)
- Hierarchical structures (PART_OF, CONTAINS)
- Conversational flow (RESPONDS_TO, FOLLOWS)

**Ontology Integration (Optional):**
- Formal ontology creation and management
- Entity type hierarchies
- Relationship type definitions
- Knowledge graph schema enforcement

> **Note**: Triple extraction and ontology management are separate concerns. Triple extraction can be enabled without full ontology management for lightweight relationship modeling.

#### Available Extractors

**1. SpaCy Extractor (`spacy`)**
- **Purpose**: Basic NLP processing and entity recognition
- **Capabilities**: Named entities (PERSON, ORG, GPE, etc.), POS tagging
- **Relationships**: None (entities only)
- **Performance**: Fast, lightweight
- **Use Case**: Simple entity extraction without relationships

```python
# Configuration
"extractor": {
  "default": "spacy",
  "spacy": {
    "model_name": "en_core_web_sm"
  }
}
```

**2. REBEL Extractor (`rebel`)**
- **Purpose**: Relationship extraction using transformer models
- **Capabilities**: Subject-relation-object triples, entities
- **Relationships**: Full triple extraction (subject, predicate, object)
- **Performance**: Medium speed, good accuracy
- **Use Case**: Balanced relationship extraction for most applications

```python
# Configuration
"extractor": {
  "default": "rebel",
  "rebel": {
    "model_name": "Babelscape/rebel-large"
  }
}
```

**3. Relik Extractor (`relik`)**
- **Purpose**: Advanced relation extraction with Wikipedia knowledge
- **Capabilities**: Entity-relation-entity triples, knowledge grounding
- **Relationships**: Contextual relationship extraction
- **Performance**: Medium speed, high accuracy
- **Use Case**: Knowledge-intensive applications requiring grounded relationships

```python
# Configuration
"extractor": {
  "default": "relik",
  "relik": {
    "model_name": "relik-ie/relik-relation-extraction-small-wikipedia-ner"
  }
}
```

**4. LLM Extractor (`llm`)** - **Default**
- **Purpose**: AI-powered extraction using large language models
- **Capabilities**: Entities, relationships, contextual understanding
- **Relationships**: Sophisticated triple extraction with reasoning
- **Performance**: Slower, highest accuracy and flexibility
- **Use Case**: Complex content requiring deep understanding

```python
# Configuration
"extractor": {
  "default": "llm",
  "fallback_order": ["llm", "relik", "gliner", "spacy"],
  "llm": {
    "model_name": "gpt-5-mini",
    "openai_api_key": "your-api-key"
  }
}
```

**5. GLiNER Extractor (`gliner`)**
- **Purpose**: Fast local entity extraction using GLiNER2
- **Capabilities**: Schema-driven extraction with entity type descriptions
- **Relationships**: Co-occurrence based relations
- **Performance**: Fast, CPU-optimized, privacy-preserving (no API calls)
- **Use Case**: Privacy-sensitive applications, offline processing

```python
# Configuration
"extractor": {
  "default": "gliner",
  "gliner": {
    "model_name": "gliner-multitask-large-v0.5"
  }
}
```

#### Fallback Chain

Extractors are tried in order until one succeeds:

```
LLM → Relik → GLiNER → SpaCy
```

Configure via:
```python
"extractor": {
  "fallback_order": ["llm", "relik", "gliner", "spacy"]
}
```

#### Large Text Handling

Texts > 8000 characters are automatically chunked:
- Split by sentence boundaries
- Processed in parallel (ThreadPoolExecutor)
- Results aggregated with entity deduplication

### 3. Memory Classification

**Automatic Classification:**
- Temporal markers → Episodic
- Factual statements → Semantic
- Process descriptions → Procedural
- Context-dependent → Working

**Manual Override:**
```python
memory.add(content, memory_type="semantic", force=True)
```

### 4. Enrichment

**Semantic Enhancement:**
- Concept expansion and synonyms
- Related topic identification
- Contextual information addition
- Knowledge graph integration

**Temporal Processing:**
- Time normalization
- Event sequencing
- Duration calculation
- Temporal relationship mapping

### 5. Linking

**Similarity-Based Linking:**
- Semantic similarity using embeddings
- Temporal proximity for episodic memories
- Conceptual overlap detection
- Entity co-occurrence analysis

**Explicit Relationship Creation:**
- Causal relationships
- Part-whole relationships
- Temporal sequences
- Conceptual hierarchies

### 6. Vector Storage

**FalkorDB HNSW Index:**
- Native vector indexing with `vecf32` type
- Configurable HNSW parameters (M, efConstruction, efRuntime)
- Cosine similarity search
- Automatic tenant isolation via ScopeProvider

### 7. Enrichment

**Wikipedia Integration:**
- Lookup entities in Wikipedia
- Add summaries, categories, URLs
- Create enrichment metadata

### 8. Grounding

**Provenance Linking:**
- Create Wikipedia nodes (shared globally)
- Create GROUNDED_IN edges from entities to Wikipedia
- Track source attribution

### 9. Evolution

**Memory Promotion:**
- Working → Episodic (threshold: 3+ items)
- Working → Procedural (threshold: 5+ items)
- Episodic → Semantic (stable facts)
- Episodic decay and archival

### 10. Clustering

**Entity Deduplication:**
- SemHash pre-deduplication (0.95 threshold)
- KMeans embedding clustering (~128 items per cluster)
- LLM semantic clustering (Joe ↔ Joseph)
- Graph node merging

### 11. Versioning

**Bi-Temporal Tracking:**
- Valid time (when fact was true)
- Transaction time (when recorded)
- Version history with HAS_VERSION edges
- Time-travel queries

## Configuration

```python
memory = SmartMemory(
    enable_background_processing=True,
    enrichment_level="full",  # minimal, standard, full
    linking_strategy="aggressive",  # conservative, standard, aggressive
    auto_classify=True
)
```

## Performance Characteristics

- **Fast Ingestion**: Immediate storage with background enrichment
- **Scalable Processing**: Parallel pipeline stages
- **Fault Tolerance**: Graceful degradation and retry mechanisms
- **Memory Efficiency**: Streaming processing for large inputs

## Monitoring and Debugging

```python
# Enable detailed logging
memory.set_log_level("DEBUG")

# Access ingestion metrics
stats = memory.get_ingestion_stats()
print(f"Processed: {stats.total_memories}")
print(f"Average processing time: {stats.avg_processing_time}ms")
```

The ingestion flow is designed to balance speed, accuracy, and resource efficiency while providing rich, interconnected memories for intelligent retrieval and reasoning.
