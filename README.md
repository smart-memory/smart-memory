# SmartMemory - Multi-Layered AI Memory System

[![PyPI version](https://badge.fury.io/py/smartmemory.svg)](https://pypi.org/project/smartmemory/)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

SmartMemory is a comprehensive AI memory system that provides persistent, multi-layered memory storage and retrieval for AI applications. It combines graph databases, vector stores, and intelligent processing pipelines to create a unified memory architecture.

## 🚀 Quick Install

```bash
pip install smartmemory
```

## Architecture Overview

SmartMemory implements a multi-layered memory architecture with the following components:

### Core Components

- **SmartMemory**: Main unified memory interface (`smartmemory.smart_memory.SmartMemory`)
- **SmartGraph**: Graph database backend using FalkorDB for relationship storage
- **Memory Types**: Specialized memory stores for different data types
- **Pipeline Stages**: Processing stages for ingestion, enrichment, and evolution
- **Plugin System**: Extensible architecture for custom evolvers and enrichers

### Memory Types

- **Working Memory**: Short-term context buffer (in-memory, capacity=10)
- **Semantic Memory**: Facts and concepts with vector embeddings
- **Episodic Memory**: Personal experiences and learning history
- **Procedural Memory**: Skills, strategies, and learned patterns
- **Zettelkasten Memory**: Bidirectional note-taking system with AI-powered knowledge discovery

### Storage Backend

- **FalkorDB**: Graph database for relationships and vector storage
- **Redis**: Caching layer for performance optimization

### Processing Pipeline

The memory ingestion flow processes data through several stages:

1. **Input Adaptation**: Convert input data to MemoryItem format
2. **Classification**: Determine appropriate memory type
3. **Extraction**: Extract entities and relationships
4. **Storage**: Persist to appropriate memory stores
5. **Linking**: Create connections between related memories
6. **Enrichment**: Enhance memories with additional context
7. **Evolution**: Transform memories based on configured rules

## Key Features

- **Multi-Type Memory System**: Working, Semantic, Episodic, and Procedural memory types
- **Graph-Based Storage**: FalkorDB backend for complex relationship modeling
- **Vector Similarity**: FalkorDB vector storage for semantic search
- **Extensible Pipeline**: Modular processing stages for ingestion and evolution
- **Plugin Architecture**: 18 built-in plugins with external plugin support
- **Plugin Security**: Sandboxing, permissions, and resource limits for safe plugin execution
- **Multi-User Support**: User and group isolation for enterprise applications
- **Caching Layer**: Redis-based performance optimization
- **Configuration Management**: Flexible configuration with environment variable support

## 📦 Installation

### From PyPI (Recommended)

```bash
# Core package
pip install smartmemory

# With optional features
pip install smartmemory[cli]           # CLI tools
pip install smartmemory[rebel]         # REBEL relation extractor
pip install smartmemory[relik]         # ReliK relation extractor
pip install smartmemory[wikipedia]     # Wikipedia enrichment
pip install smartmemory[all]           # All optional features
```

### From Source (Development)

```bash
git clone https://github.com/smart-memory/smart-memory.git
cd smart-memory
pip install -e ".[dev]"

# Install spaCy model for entity extraction
python -m spacy download en_core_web_sm
```

### Docker Deployment

See the [smart-memory-service](https://github.com/smart-memory/smart-memory-service) repository for production-ready Docker deployment with FastAPI, authentication, and multi-tenancy support.

## 🎯 Quick Start

### Basic Usage

```python
from smartmemory import SmartMemory, MemoryItem

# Initialize SmartMemory
memory = SmartMemory()

# Add a memory
item = MemoryItem(
    content="User prefers Python for data analysis tasks",
    memory_type="semantic",
    user_id="user123",
    metadata={'topic': 'preferences', 'domain': 'programming'}
)
memory.add(item)

# Search memories
results = memory.search("Python programming", top_k=5)
for result in results:
    print(f"Content: {result.content}")
    print(f"Type: {result.memory_type}")

# Get memory summary
summary = memory.summary()
print(f"Total memories: {summary}")
```

### Using Different Memory Types

```python
from smartmemory import SmartMemory, MemoryItem

# Initialize SmartMemory
memory = SmartMemory()

# Add working memory (short-term context)
working_item = MemoryItem(
    content="Current conversation context",
    memory_type="working"
)
memory.add(working_item)

# Add semantic memory (facts and concepts)
semantic_item = MemoryItem(
    content="Python is a high-level programming language",
    memory_type="semantic"
)
memory.add(semantic_item)

# Add episodic memory (experiences)
episodic_item = MemoryItem(
    content="User completed Python tutorial on 2024-01-15",
    memory_type="episodic"
)
memory.add(episodic_item)

# Add procedural memory (skills and procedures)
procedural_item = MemoryItem(
    content="To sort a list in Python: use list.sort() or sorted(list)",
    memory_type="procedural"
)
memory.add(procedural_item)

# Add Zettelkasten note (interconnected knowledge)
zettel_item = MemoryItem(
    content="# Machine Learning\n\nML learns from data using algorithms.",
    memory_type="zettel",
    metadata={'title': 'ML Basics', 'tags': ['ai', 'ml']}
)
memory.add(zettel_item)
```

### CLI Usage (Optional)

```bash
# Install CLI tools
pip install smartmemory[cli]

# Show system info
smartmemory info

# List all plugins
smartmemory plugins list

# Get plugin details
smartmemory plugins info llm

# Add a memory
smartmemory add "Python is great for AI" --memory-type semantic

# Search memories
smartmemory search "Python programming" --top-k 5

# Show summary
smartmemory summary
```

## Use Cases

### Conversational AI Systems
- Maintain context across multiple conversation sessions
- Learn user preferences and adapt responses
- Build comprehensive user profiles over time

### Educational Applications
- Track learning progress and adapt teaching strategies
- Remember previous topics and build upon them
- Personalize content based on individual learning patterns

### Knowledge Management
- Store and retrieve complex information relationships
- Connect related concepts across different domains
- Evolve understanding through continuous learning
- Build a personal knowledge base with Zettelkasten method

### Personal AI Assistants
- Remember user preferences and past interactions
- Provide contextually relevant recommendations
- Learn from user feedback to improve responses

## Examples

The `examples/` directory contains several demonstration scripts:

- `memory_system_usage_example.py`: Basic memory operations (add, search, delete)
- `zettelkasten_example.py`: Complete Zettelkasten system demonstration
- `conversational_assistant_example.py`: Conversational AI with memory
- `advanced_programming_tutor.py`: Educational application example
- `working_holistic_example.py`: Comprehensive multi-session demo
- `background_processing_demo.py`: Asynchronous processing example

## Configuration

SmartMemory uses environment variables for configuration:

### Environment Variables

Key environment variables:
- `FALKORDB_HOST`: FalkorDB server host (default: localhost)
- `FALKORDB_PORT`: FalkorDB server port (default: 6379)
- `REDIS_HOST`: Redis server host (default: localhost)
- `REDIS_PORT`: Redis server port (default: 6379)
- `OPENAI_API_KEY`: OpenAI API key for embeddings

```bash
# Example .env file
export FALKORDB_HOST=localhost
export FALKORDB_PORT=6379
export REDIS_HOST=localhost
export REDIS_PORT=6379
export OPENAI_API_KEY=your-api-key-here
```

## Memory Evolution

SmartMemory includes built-in evolvers that automatically transform memories:

### Available Evolvers

- **WorkingToEpisodicEvolver**: Converts working memory to episodic when buffer is full
- **WorkingToProceduralEvolver**: Extracts repeated patterns as procedures
- **EpisodicToSemanticEvolver**: Promotes stable facts to semantic memory
- **EpisodicToZettelEvolver**: Converts episodic events to Zettelkasten notes
- **EpisodicDecayEvolver**: Archives old episodic memories
- **SemanticDecayEvolver**: Prunes low-relevance semantic facts
- **ZettelPruneEvolver**: Merges duplicate or low-quality notes

Evolvers run automatically as part of the memory lifecycle. See the [examples](examples/) directory for evolution demonstrations.

## Plugin System

SmartMemory features a **unified, extensible plugin architecture** that allows you to customize and extend functionality. All plugins follow a consistent class-based pattern.

### Built-in Plugins

SmartMemory includes **18 built-in plugins** across 4 types:

- **4 Extractors**: Extract entities and relationships
  - `SpacyExtractor`, `LLMExtractor`, `RebelExtractor`, `RelikExtractor`
- **6 Enrichers**: Add context and metadata to memories
  - `BasicEnricher`, `SentimentEnricher`, `TemporalEnricher`, `TopicEnricher`, `SkillsToolsEnricher`, `WikipediaEnricher`
- **1 Grounder**: Connect to external knowledge
  - `WikipediaGrounder`
- **7 Evolvers**: Transform memories based on rules
  - `WorkingToEpisodicEvolver`, `EpisodicToSemanticEvolver`, `SemanticDecayEvolver`, etc.

### Creating Custom Plugins

Create your own plugins by extending the base classes:

```python
from smartmemory.plugins.base import EnricherPlugin, PluginMetadata

class MyCustomEnricher(EnricherPlugin):
    @classmethod
    def metadata(cls):
        return PluginMetadata(
            name="my_enricher",
            version="1.0.0",
            author="Your Name",
            description="My custom enricher",
            plugin_type="enricher",
            dependencies=["some-lib>=1.0.0"],
            security_profile="standard",
            requires_network=False,
            requires_llm=False
        )
    
    def enrich(self, item, node_ids=None):
        # Your enrichment logic
        item.metadata["custom_field"] = "value"
        return item.metadata
```

See the [examples](examples/) directory for complete plugin examples.

### Publishing Plugins

Publish your plugin as a Python package with entry points:

```toml
# pyproject.toml
[project.entry-points."smartmemory.plugins.enrichers"]
my_enricher = "my_package:MyCustomEnricher"
```

Install and use:
```bash
pip install my-smartmemory-plugin
# Automatically discovered and loaded!
```

### Plugin Types

- **ExtractorPlugin**: Extract entities and relationships from text
- **EnricherPlugin**: Add metadata and context to memories
- **GrounderPlugin**: Link memories to external knowledge sources
- **EvolverPlugin**: Transform memories based on conditions

All plugins are automatically discovered and registered at startup.

### Plugin Security

SmartMemory includes a comprehensive security system for plugins:

- **4 Security Profiles**: `trusted`, `standard` (default), `restricted`, `untrusted`
- **Permission System**: Control memory, network, file, and LLM access
- **Resource Limits**: Automatic timeout (30s), memory limits, network request limits
- **Sandboxing**: Isolated execution with security enforcement
- **Static Validation**: Detects security issues before execution

```python
# Plugins are secure by default
PluginMetadata(
    security_profile="standard",  # Balanced security
    requires_network=True,        # Explicitly declare requirements
    requires_llm=False
)
```

See `docs/PLUGIN_SECURITY.md` for complete security documentation.

### Examples

See the `examples/` directory for complete plugin examples:
- `custom_enricher_example.py` - Sentiment analysis and keyword extraction
- `custom_evolver_example.py` - Memory promotion and archival
- `custom_extractor_example.py` - Regex and domain-specific NER
- `custom_grounder_example.py` - DBpedia and custom API grounding

## Testing

Run the test suite:

```bash
# Run all tests
PYTHONPATH=. pytest -v tests/

# Run specific test categories
PYTHONPATH=. pytest tests/unit/
PYTHONPATH=. pytest tests/integration/
PYTHONPATH=. pytest tests/e2e/

# Run examples
PYTHONPATH=. python examples/memory_system_usage_example.py
PYTHONPATH=. python examples/conversational_assistant_example.py
```

## API Reference

### SmartMemory Class

Main interface for memory operations:

```python
class SmartMemory:
    def add(self, item: MemoryItem) -> Optional[MemoryItem]
    def get(self, item_id: str) -> Optional[MemoryItem]
    def search(self, query: str, top_k: int = 10) -> List[MemoryItem]
    def delete(self, item_id: str) -> bool
    def clear(self) -> None
    def summary(self) -> Dict[str, Any]
    def ingest(self, content: str, **kwargs) -> MemoryItem
```

### MemoryItem Class

Core data structure for memory storage:

```python
@dataclass
class MemoryItem:
    content: str
    memory_type: str = 'semantic'
    item_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    group_id: Optional[str] = None
    valid_start_time: Optional[datetime] = None
    valid_end_time: Optional[datetime] = None
    transaction_time: datetime = field(default_factory=datetime.now)
    embedding: Optional[List[float]] = None
    entities: Optional[list] = None
    relations: Optional[list] = None
    metadata: dict = field(default_factory=dict)
```

## Dependencies

### Core Dependencies

SmartMemory requires the following key dependencies:

- `falkordb`: Graph database and vector storage backend
- `spacy`: Natural language processing and entity extraction
- `litellm`: LLM integration layer
- `openai`: OpenAI API client (for embeddings)
- `redis`: Caching layer
- `scikit-learn`: Machine learning utilities
- `pydantic`: Data validation
- `python-dateutil`: Date/time handling
- `vaderSentiment`: Sentiment analysis
- `jinja2`: Template rendering

**Note:** SmartMemory uses FalkorDB for both graph and vector storage. While the codebase contains legacy ChromaDB integration code, FalkorDB is the primary and recommended backend.

### Optional Dependencies

Install additional features as needed:

```bash
# Specific extractors
pip install smartmemory[rebel]     # REBEL relation extraction
pip install smartmemory[relik]     # ReliK relation extraction

# Integrations
pip install smartmemory[slack]     # Slack integration
pip install smartmemory[aws]       # AWS integration
pip install smartmemory[wikipedia] # Wikipedia enrichment

# Tools
pip install smartmemory[cli]       # Command-line interface

# Everything
pip install smartmemory[all]       # All optional features
```

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

For major changes, please open an issue first to discuss the proposed changes.

## 📄 License

SmartMemory is dual-licensed to provide flexibility for both open-source and commercial use. See [LICENSE](LICENSE) for details.

## Security

SmartMemory takes plugin security seriously. All plugins run in a sandboxed environment with:

- ✅ **Permission checks** - Plugins must declare what they access
- ✅ **Resource limits** - Automatic timeouts and memory limits
- ✅ **Execution isolation** - Sandboxed plugin execution
- ✅ **Static analysis** - Security validation before execution

External plugins use the `standard` security profile by default. See `docs/PLUGIN_SECURITY.md` for details.

## 🔗 Links

- **📦 PyPI Package**: https://pypi.org/project/smartmemory/
- **📚 Documentation**: https://docs.smartmemory.ai
- **🐙 GitHub Repository**: https://github.com/smart-memory/smart-memory
- **🐛 Issue Tracker**: https://github.com/smart-memory/smart-memory/issues
- **🔒 Security Policy**: See `docs/PLUGIN_SECURITY.md`
- **🚀 Production Service**: https://github.com/smart-memory/smart-memory-service

---

**Get started with SmartMemory today!**

```bash
pip install smartmemory
```

Explore the [examples](examples/) directory for complete demonstrations and use cases.

---

## 🚧 In Progress

The following features are currently under active development:

### Ontology System
- **Current**: Basic concept and relation extraction
- **In Progress**: 
  - Ontology governance and validation
  - External ontology integration (DBpedia, Schema.org)
  - Semantic clustering and concept hierarchies
  - Ontology-driven query expansion
- **Planned**: Full ontology management with Maya integration

### Temporal Queries
- **Current**: Basic bi-temporal support (valid_time, transaction_time)
- **In Progress**:
  - Time-travel queries
  - Version history tracking
  - Audit trail generation
- **Planned**: Advanced temporal analytics

### Multi-Tenancy (Service Layer)
- **Current**: Single-user mode
- **In Progress**: Full multi-tenancy support in smart-memory-service
- **Planned**: Team collaboration features

These features are functional but not yet production-ready. Check the [GitHub repository](https://github.com/smart-memory/smart-memory) for the latest updates.
