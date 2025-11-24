# Ingestion Flow

SmartMemory's ingestion pipeline transforms raw input into enriched, interconnected memories through a sophisticated multi-stage process.

## Overview

The ingestion flow consists of several stages that process memories from initial input to final storage:

1. **Input Processing** - Parse and validate input
2. **Content Analysis** - Extract entities and relationships
3. **Memory Classification** - Determine memory type
4. **Enrichment** - Add semantic and contextual information
5. **Linking** - Connect to existing memories
6. **Storage** - Persist to appropriate backends
7. **Background Processing** - Asynchronous optimization

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

**4. LLM Extractor (`llm`)**
- **Purpose**: AI-powered extraction using large language models
- **Capabilities**: Entities, relationships, contextual understanding
- **Relationships**: Sophisticated triple extraction with reasoning
- **Performance**: Slower, highest accuracy and flexibility
- **Use Case**: Complex content requiring deep understanding

```python
# Configuration
"extractor": {
  "default": "llm",
  "llm": {
    "model_name": "gpt-5.1-mini",
    "openai_api_key": "your-api-key"
  }
}
```

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

### 6. Storage

**Multi-Backend Persistence:**
- Graph database for relationships
- Vector store for semantic search
- Metadata stores for structured data
- Full-text search indices

**Atomic Operations:**
- Transactional consistency
- Rollback on failure
- Duplicate detection
- Version management

### 7. Background Processing

**Asynchronous Optimization:**
- Memory consolidation
- Relationship refinement
- Index optimization
- Evolution algorithm application

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
