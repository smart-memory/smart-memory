# Memory Types

SmartMemory supports five distinct memory types, each optimized for different kinds of information and use cases.

## Semantic Memory

Stores facts, concepts, and general knowledge that is context-independent.

**Characteristics:**
- Timeless information
- Factual knowledge
- Concepts and definitions
- General principles

**Examples:**
- "Python is a programming language"
- "The capital of France is Paris"
- "Machine learning requires training data"

## Episodic Memory

Stores personal experiences and events with temporal and contextual information.

**Characteristics:**
- Time-bound experiences
- Personal context
- Situational details
- Autobiographical events

**Examples:**
- "I learned Python in 2020 during the pandemic"
- "Yesterday's team meeting discussed the new API"
- "The debugging session on Friday solved the memory leak"

## Procedural Memory

Stores skills, procedures, and how-to knowledge for performing tasks.

**Characteristics:**
- Step-by-step processes
- Skills and techniques
- Operational knowledge
- Action sequences

**Examples:**
- "How to deploy a Docker container"
- "Steps for debugging memory issues"
- "Process for code review workflow"

## Working Memory

Temporary storage for information currently being processed or actively used.

**Characteristics:**
- Short-term storage
- Active processing
- Current context
- Session-specific

**Examples:**
- Current conversation context
- Active problem-solving steps
- Immediate tasks and goals

## Zettelkasten Memory

Atomic, interconnected knowledge management inspired by the traditional Zettelkasten method. Stores individual ideas as notes with automatic cross-linking and knowledge graph formation.

**Characteristics:**
- Atomic notes (one idea per note)
- Automatic entity extraction
- Bidirectional linking
- Emergent knowledge structure
- Visual graph navigation

**Examples:**
- "Attention mechanisms allow models to focus on relevant input parts"
- "Neural networks learn through backpropagation" (auto-linked to optimization notes)
- Research insights with cross-references to related concepts

**Unique Features:**
- Interactive graph visualization
- Background enrichment and auto-linking
- Integration with external knowledge bases
- Support for `[[note-id]]` cross-reference syntax

> **Learn More**: See [Zettelkasten Memory](./zettelkasten-memory) for comprehensive documentation.


SmartMemory automatically classifies memories based on content analysis, but you can also specify the type explicitly:

```python
memory.add("Paris is the capital of France", memory_type="semantic")
memory.add("I visited paris last summer", memory_type="episodic")
memory.add("How to book a flight to paris", memory_type="procedural")
memory.add("Attention mechanisms allow models to focus on relevant input parts", memory_type="zettel")
```

## Bitemporal Capabilities

SmartMemory implements full bitemporal support, tracking two distinct time dimensions for each memory item:
### Valid Time vs Transaction Time

**Valid Time (`valid_time`)**
- When the fact was true in the real world
- Represents the actual time period of validity
- Can be in the past, present, or future
- Example: "John worked at Company X from 2020-2023"

**Transaction Time (`transaction_time`)**
- When the fact was recorded in the system
- Always in the past (when we learned about it)
- Immutable once recorded
- Used for audit trails and data provenance

### Temporal Enrichment

SmartMemory automatically enriches memories with temporal metadata:

```python
# Example temporal enrichment result
temporal_data = {
    "entities": {
        "John Smith": {
            "valid_start": "2020-01-15T00:00:00Z",
            "valid_end": "2023-12-31T23:59:59Z",
            "transaction_time": "2024-01-15T10:30:00Z"
        }
    }
}
```

### Use Cases for Bitemporal Data

- **Historical Analysis**: Query what we knew at any point in time
- **Fact Correction**: Update information without losing historical context
- **Audit Trails**: Track when facts were recorded vs when they were valid
- **Time-Travel Queries**: "What did we know about X on date Y?"
- **Data Provenance**: Understand the evolution of knowledge over time

### Comparison with Competitors

- **Mem0**: No bitemporal support (basic timestamps only)
- **Zep**: Single temporal dimension (no valid/transaction time separation)
- **SmartMemory**: Full bitemporal model with automated temporal enrichment

## Cross-Type Relationships

SmartMemory automatically discovers relationships between different memory types:

- **Semantic ↔ Episodic**: Facts connected to personal experiences
- **Procedural ↔ Episodic**: Skills learned through specific events
- **Working ↔ All Types**: Temporary connections to long-term memories

This creates a rich, interconnected memory network that mirrors human cognitive patterns.

## Memory Evolution

Memory types can evolve and transform:

- **Working → Episodic**: Recent interactions become experiences
- **Episodic → Semantic**: Repeated patterns become general knowledge
- **Episodic → Zettelkasten**: Important events become structured notes
- **Semantic ↔ Zettelkasten**: Facts and notes cross-reference each other
