# Zettelkasten Memory

The SmartMemory system includes a powerful **Zettelkasten memory type** that implements atomic, interconnected knowledge management inspired by the traditional Zettelkasten method used by researchers and knowledge workers.

## What is Zettelkasten Memory?

Zettelkasten (German for "slip box") is a method of knowledge management that stores atomic ideas as individual notes with unique identifiers and creates dense networks of cross-references between related concepts.

**Key Characteristics:**
- **Atomic Notes**: Each memory item contains a single, focused idea
- **Unique Identifiers**: Every note has a permanent, linkable ID
- **Bidirectional Linking**: Notes automatically link to related content
- **Emergent Structure**: Knowledge organization emerges from connections
- **Contextual Discovery**: Find information through associative pathways

## How It Works in SmartMemory

### Automatic Entity Extraction

```python
from smartmemory.memory.types.zettel_memory import ZettelMemory

zettel = ZettelMemory()

# Add a note - entities are automatically extracted
note = zettel.add_note(
    content="Machine learning models require large datasets for training effectiveness",
    tags=["#ml", "#data", "#training"]
)
```

The system automatically:
- Extracts entities ("machine learning", "datasets", "training")
- Creates relationships between concepts
- Links to existing related notes
- Builds a knowledge graph

### Bidirectional Auto-Linking

When you add new notes, the system automatically:

1. **Semantic Linking**: Uses vector embeddings to find related content
2. **Entity Linking**: Connects notes sharing extracted entities
3. **Keyword Linking**: Links notes with common tags or keywords
4. **Manual Linking**: Supports explicit cross-references using `[[note-id]]` syntax

### Example Knowledge Evolution

```python
# Week 1: Add foundational concepts
zettel.add_note("Neural networks learn through backpropagation", tags=["#neural-networks"])
zettel.add_note("Gradient descent optimizes models parameters", tags=["#optimization"])

# Week 2: Add related concept - automatically links to existing notes
zettel.add_note("Deep learning uses multi-layer neural networks with gradient-based optimization")
# ^ This automatically links to both previous notes due to shared entities
```

## Interactive Graph Visualization

The system includes a **React-based graph viewer** that lets you:

- **Visualize connections** between all your notes
- **Navigate associatively** by clicking through related concepts  
- **Discover knowledge gaps** by seeing sparse areas
- **Track knowledge growth** over time

Access the viewer at `/zettel-graph-viewer/` in your local installation.

## Background Enrichment

The Zettelkasten memory runs continuous background processes:

### Auto-Linking Routines
- **Semantic similarity**: Links notes above 0.75 cosine similarity
- **Entity overlap**: Connects notes sharing extracted entities
- **Temporal clustering**: Groups notes from similar time periods

### Knowledge Graph Maintenance
- **Orphan detection**: Identifies unconnected notes
- **Link quality scoring**: Ranks connection strength
- **Duplicate detection**: Finds similar or redundant content

### Analytics & Statistics
- **Network metrics**: Clustering coefficients, centrality measures
- **Growth tracking**: Note creation and linking patterns
- **Knowledge density**: Connectivity and coverage analysis

## Integration with Other Memory Types

Zettelkasten memory works seamlessly with other SmartMemory types:

### From Working Memory
```python
# Working memory items can evolve into Zettel notes
working_item = MemoryItem(content="Learning about transformers", memory_type="working")
memory.add(working_item)

# After multiple related items, system suggests Zettel conversion
# "Convert these 3 transformer-related working memories into a Zettel?"
```

### To Semantic/Episodic Memory
```python
# Zettel notes can be elevated to other memory types
zettel_note = zettel.get_note("transformer-attention-mechanism")
semantic_memory.promote_from_zettel(zettel_note)  # Becomes semantic knowledge
```

## Best Practices

### Atomic Note Principle
✅ **Good**: "Attention mechanisms allow models to focus on relevant input parts"
❌ **Bad**: "Transformers use attention and are good for NLP and have encoder-decoder..."

### Effective Tagging
```python
# Use hierarchical tags
tags=["#ml/attention", "#architecture/transformer", "#nlp/language-modeling"]

# Include domain and granularity
tags=["#concept", "#implementation", "#theory"]
```

### Cross-Reference Patterns
```markdown
# In your note content
This builds on [[gradient-descent.md]] and relates to [[neural-networks.md]].
See also: [[backpropagation.md]], [[optimization-theory.md]]
```

## Advanced Features

### Domain-Specific Extraction
```python
# Configure extraction for specific domains
zettel.configure_extraction(
    domain="software_engineering",
    extract_entities=["functions", "classes", "patterns"],
    extract_relations=["implements", "depends_on", "extends"]
)
```

### Custom Enrichment Rules
```python
# Add custom linking rules
zettel.add_linking_rule(
    condition="both_contain_code_snippets",
    action="create_implementation_link",
    strength=0.8
)
```

### Export and Integration
```python
# Export knowledge graph
graph_data = zettel.export_graph(format="cytoscape")

# Import from external sources
zettel.import_notes(source="obsidian", path="/vault/notes/")
```

## Use Cases

### Research & Learning
- **Literature review**: Connect papers, concepts, and insights
- **Course notes**: Build interconnected understanding across subjects
- **Project documentation**: Link decisions, implementations, and learnings

### Software Development
- **Architecture documentation**: Connect components, patterns, and decisions
- **Code knowledge**: Link algorithms, implementations, and examples
- **Bug tracking**: Connect issues, solutions, and root causes

### Creative Work
- **Idea development**: Connect inspirations, concepts, and iterations
- **Project planning**: Link requirements, resources, and constraints
- **Knowledge synthesis**: Combine insights from multiple domains

## API Reference

```python
from smartmemory.memory.types.zettel_memory import ZettelMemory

# Initialize
zettel = ZettelMemory()

# Add notes
note_id = zettel.add_note(content, tags=[], metadata={})

# Search and retrieve
results = zettel.search(query, max_results=10)
note = zettel.get_note(note_id)

# Navigate connections
related = zettel.get_related_notes(note_id, max_depth=2)
path = zettel.find_connection_path(note_id_1, note_id_2)

# Analytics
stats = zettel.get_statistics()
orphans = zettel.find_orphan_notes()
clusters = zettel.find_knowledge_clusters()
```

## Configuration

```json
{
  "zettelkasten": {
    "auto_linking": {
      "semantic_threshold": 0.75,
      "entity_threshold": 0.5,
      "temporal_window_days": 30
    },
    "extraction": {
      "enabled": true,
      "model": "gpt-4",
      "extract_entities": true,
      "extract_relations": true
    },
    "enrichment": {
      "background_enabled": true,
      "frequency_hours": 24,
      "batch_size": 100
    }
  }
}
```

The Zettelkasten memory type transforms SmartMemory into a powerful knowledge management system that grows smarter and more connected over time, supporting both personal learning and collaborative knowledge building.
