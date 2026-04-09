# SmartMemory - Multi-Layered AI Memory System

[![Docs](https://img.shields.io/badge/docs-smartmemory.ai-blue)](https://docs.smartmemory.ai/smartmemory/intro)
[![PyPI version](https://badge.fury.io/py/smartmemory.svg)](https://pypi.org/project/smartmemory/)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

**[Read the docs](https://docs.smartmemory.ai/smartmemory/intro)** | **[Maya sample app](https://docs.smartmemory.ai/maya)**

SmartMemory is a comprehensive AI memory system that provides persistent, multi-layered memory storage and retrieval for AI applications. It combines graph databases, vector stores, and intelligent processing pipelines to create a unified memory architecture.

## Install

```bash
pip install smartmemory[local]           # Local memory + MCP server + viewer + CLI (recommended)
pip install smartmemory                  # Remote client only (connects to a SmartMemory service)
pip install smartmemory-core[lite]       # Core library only, local mode (for developers)
pip install smartmemory-core[server]     # Core library only, server mode (FalkorDB + Redis)
```

> **`smartmemory`** is the distribution package — MCP server, graph viewer, and CLI.
> **`smartmemory-core`** is the core library for developers building on top of SmartMemory.
> `smartmemory[local]` bundles `smartmemory-core[lite]` for local SQLite storage. Without `[local]`, it's a remote client only.

## First Run

```bash
smartmemory setup
```

Launches a Textual TUI with arrow-key selection for LLM provider, live model discovery from ollama/lmstudio, embedding provider, and a summary screen. Falls back to text prompts in non-interactive environments (Docker, CI).

**Local mode** wires Claude Code hooks, downloads the spaCy language model (~15MB), and starts a persistent daemon.

**Remote mode** validates your API key and stores it in the OS keychain.

## Quick Start — Local Python API

```python
from smartmemory.tools.factory import create_lite_memory, lite_context

# Simple usage — full LLM extraction runs if OPENAI_API_KEY is set
memory = create_lite_memory()
item_id = memory.ingest("Alice leads Project Atlas")
results = memory.search("who leads Atlas", top_k=5)

# Preferred in scripts — cleans up globals and closes SQLite on exit
with lite_context() as memory:
    item_id = memory.ingest("Alice leads Project Atlas")
    results = memory.search("who leads Atlas")

# Force no LLM calls (even if OPENAI_API_KEY is set)
from smartmemory.pipeline.config import PipelineConfig
memory = create_lite_memory(pipeline_profile=PipelineConfig.lite(llm_enabled=False))
```

## Daemon

SmartMemory runs a persistent background daemon so CLI commands respond in <200ms instead of cold-starting Python every time (~22s).

```bash
smartmemory start       # Start daemon (auto-started by setup)
smartmemory stop        # Stop daemon
smartmemory restart     # Restart daemon
smartmemory status      # Show status, memory count, enrichment queue
```

On macOS, `smartmemory setup` installs a launchd plist — the daemon auto-starts on login and restarts on crash.

### Two-tier ingest

When an LLM API key is available, the daemon runs **two-tier ingestion**:

- **Tier 1 (sync, ~4ms):** spaCy + EntityRuler extracts entities immediately, returns item_id
- **Tier 2 (async, ~740ms):** Background drain thread runs LLM extraction, adds net-new entities and relations

This means `smartmemory add` returns instantly while quality improves in the background.

## Commands

### Core

```bash
smartmemory add "text"                # Add a memory (default type: episodic)
smartmemory add --type semantic "..."  # Add with specific memory type
smartmemory add - < notes.txt          # Add from stdin, one memory per line
smartmemory add --all - < doc.txt      # Add entire stdin as one memory
smartmemory add --project atlas "..."  # Add with arbitrary property flags

smartmemory search "query"             # Semantic search
smartmemory search "*"                 # List all memories
smartmemory search --top-k 20 "query"  # Control result count (default: 5)
smartmemory search --project atlas "q" # Filter by property

smartmemory get <item_id>              # Fetch a single memory by ID
smartmemory recall                     # Session context for Claude Code hooks
smartmemory recall --cwd /path         # Recall with working directory context
smartmemory viewer                     # Open knowledge graph viewer in browser
smartmemory viewer --port 8080         # Custom port for viewer
smartmemory models                     # List available LLM models
smartmemory config                     # View all settings
smartmemory config llm_provider        # View one setting
smartmemory config llm_provider groq   # Change a setting
smartmemory clear                      # Delete all memories and reset vectors
```

### Daemon

```bash
smartmemory start                      # Start daemon + enrichment workers
smartmemory stop                       # Stop daemon
smartmemory restart                    # Restart daemon
smartmemory status                     # Daemon health + enrichment stats
smartmemory worker                     # Run enrichment worker (drain and exit)
smartmemory worker --loop              # Run enrichment worker continuously
```

### Setup & Lifecycle

```bash
smartmemory setup                      # Interactive first-run questionnaire
smartmemory setup --mode local         # Skip questionnaire, set local mode
smartmemory setup --mode remote --api-key sk_...  # Non-interactive remote setup
smartmemory setup --for cursor         # Configure for Cursor instead of Claude Code
smartmemory server                     # Start MCP server (called by MCP clients)
smartmemory uninstall                  # Remove hooks, skills, plist, and data
smartmemory uninstall --keep-data      # Remove hooks/plist but keep memories
```

### Admin

```bash
smartmemory admin export out.jsonl     # Export memories to corpus JSONL
smartmemory admin import data.jsonl    # Import corpus JSONL into SmartMemory
smartmemory admin reindex              # Re-embed all memories with current model
smartmemory admin reextract            # Re-run entity extraction on all memories
smartmemory admin list-packs           # List available seed packs
smartmemory admin install-pack NAME    # Install a seed pack
smartmemory admin mine                 # Mine Wikidata entities via SPARQL
smartmemory admin convert-rebel        # Convert REBEL dataset to corpus JSONL
```

## Architecture Overview

SmartMemory implements a multi-layered memory architecture:

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
- **Reasoning Memory**: Chain-of-thought traces capturing "why" decisions were made (System 2)
- **Opinion Memory**: Beliefs with confidence scores, reinforced or contradicted over time
- **Observation Memory**: Synthesized entity summaries from scattered facts
- **Decision Memory**: First-class decisions with confidence tracking, provenance chains, and lifecycle management

### Storage Backends

- **Lite mode**: SQLite graph + usearch vectors — no Docker, no external services
- **Server mode**: FalkorDB (graph + vectors) + Redis (caching) — full-featured, requires Docker

### Processing Pipeline

`ingest()` runs an 11-stage pipeline:

```
classify -> coreference -> simplify -> entity_ruler -> llm_extract -> ontology_constrain -> store -> link -> enrich -> ground -> evolve
```

Each stage implements the `StageCommand` protocol (`execute(state, config) -> state`, `undo(state) -> state`). The pipeline supports breakpoint execution (`run_to()`, `run_from()`, `undo_to()`) for debugging and resumption.

`add()` is simple storage: normalize -> store -> embed (use for internal/derived items).

## Key Features

- **9 Memory Types**: Working, Semantic, Episodic, Procedural, Zettelkasten, Reasoning, Opinion, Observation, Decision
- **11-Stage NLP Pipeline**: classify -> coreference -> simplify -> entity_ruler -> llm_extract -> ontology_constrain -> store -> link -> enrich -> ground -> evolve
- **Self-Learning EntityRuler**: Pattern-matching NER that improves with use — LLM discoveries feed back into rules (96.9% entity F1 at 4ms)
- **Evolver Framework**: Core auto-registered evolvers plus specialist lifecycle evolvers for decay, consolidation, opinion synthesis, retrieval-based strengthening, Hebbian co-retrieval, and stale memory detection
- **Code Indexer**: AST-based Python + TypeScript parser with cross-file call resolution, semantic code search, and memory-to-code graph bridging
- **Zero-Infra Lite Mode**: SQLite + usearch backend — `pip install smartmemory[local]` and go
- **Server Mode**: FalkorDB graph + Redis caching for production-scale deployments
- **Hybrid Search**: Graph-structured search + BM25/embedding RRF fusion with query decomposition for compound queries
- **20 Auto-Registered Plugins**: 4 extractors, 5 enrichers, 10 evolvers, and 1 grounder loaded by default
- **Plugin Security**: Sandboxing, permissions, and resource limits for safe plugin execution
- **Flexible Scoping**: Optional `ScopeProvider` for multi-tenancy or unrestricted OSS usage
- **Persistent Daemon**: Background process for <200ms CLI response times
- **Two-Tier Ingestion**: Instant spaCy extraction + async LLM enrichment
- **MCP Server**: Works with Claude Code, Cursor, and other MCP-compatible tools
- **Knowledge Graph Viewer**: Interactive browser-based graph visualization

## Memory Evolution

SmartMemory includes built-in evolvers that automatically transform memories. In lite mode, evolution runs incrementally in the background.

### Available Evolvers

**Core evolvers** — memory type transitions and lifecycle:
- **WorkingToEpisodicEvolver**: Converts working memory to episodic when buffer is full
- **WorkingToProceduralEvolver**: Extracts repeated patterns as procedures
- **EpisodicToSemanticEvolver**: Promotes stable facts to semantic memory
- **EpisodicToZettelEvolver**: Converts episodic events to Zettelkasten notes
- **EpisodicDecayEvolver**: Archives old episodic memories
- **SemanticDecayEvolver**: Prunes low-relevance semantic facts
- **ZettelPruneEvolver**: Merges duplicate or low-quality notes
- **DecisionConfidenceEvolver**: Decays confidence on stale decisions, auto-retracts below threshold
- **OpinionSynthesisEvolver**: Synthesizes opinions from accumulated observations
- **ObservationSynthesisEvolver**: Creates entity summaries from scattered facts
- **OpinionReinforcementEvolver**: Adjusts opinion confidence based on new evidence
- **StaleMemoryEvolver**: Flags memories as stale when referenced source code changes

**Enhanced evolvers** — neuroscience-inspired dynamics:
- **ExponentialDecayEvolver**: Time-based activation decay with configurable half-life
- **RetrievalBasedStrengtheningEvolver**: Memories accessed more frequently become harder to forget
- **HebbianCoRetrievalEvolver**: Reinforces edges between memories retrieved together ("neurons that fire together wire together")
- **InterferenceBasedConsolidationEvolver**: Similar competing memories interfere, strengthening the dominant one
- **EnhancedWorkingToEpisodicEvolver**: Context-aware working-to-episodic transition with richer metadata

## Plugin System

SmartMemory features a unified, extensible plugin architecture. All plugins follow a consistent class-based pattern.

### Built-in Plugins

**Auto-registered by default** (loaded by `PluginManager`):
- **4 Extractors**: `LLMExtractor`, `LLMSingleExtractor`, `ConversationAwareLLMExtractor`, `SpacyExtractor`
- **5 Enrichers**: `BasicEnricher`, `SentimentEnricher`, `TemporalEnricher`, `ExtractSkillsToolsEnricher`, `TopicEnricher`
- **10 Evolvers**: `WorkingToEpisodicEvolver`, `WorkingToProceduralEvolver`, `EpisodicToSemanticEvolver`, `EpisodicToZettelEvolver`, `EpisodicDecayEvolver`, `SemanticDecayEvolver`, `ZettelPruneEvolver`, `ExponentialDecayEvolver`, `InterferenceBasedConsolidationEvolver`, `RetrievalBasedStrengtheningEvolver`
- **1 Grounder**: `WikipediaGrounder`

**Specialist plugins** (used by specific pipeline stages or opt-in features):
- **Extractors**: `GroqExtractor`, `DecisionExtractor`, `ReasoningExtractor`
- **Enrichers**: `LinkExpansionEnricher`
- **Evolvers**: `DecisionConfidenceEvolver`, `OpinionSynthesisEvolver`, `ObservationSynthesisEvolver`, `OpinionReinforcementEvolver`, `StaleMemoryEvolver`, `HebbianCoRetrievalEvolver`, `EnhancedWorkingToEpisodicEvolver`
- **Grounders**: `PublicKnowledgeGrounder` (Wikidata QIDs)

### Creating Custom Plugins

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
        item.metadata["custom_field"] = "value"
        return item.metadata
```

### Publishing Plugins

```toml
# pyproject.toml
[project.entry-points."smartmemory.plugins.enrichers"]
my_enricher = "my_package:MyCustomEnricher"
```

```bash
pip install my-smartmemory-plugin
# Automatically discovered and loaded!
```

## Non-interactive / CI

```bash
# Local
SMARTMEMORY_MODE=local smartmemory server

# Remote
SMARTMEMORY_MODE=remote SMARTMEMORY_API_KEY=sk_... smartmemory server
```

Env vars always override config file — the correct path for Docker and CI. The TUI is automatically disabled in non-interactive environments.

## Configuration

Config file: `~/.config/smartmemory/config.toml` (XDG on Linux/macOS, `%APPDATA%\smartmemory\config.toml` on Windows).

API keys are stored in the OS keychain, never in the config file. Set `SMARTMEMORY_API_KEY` as an env var on headless systems where the keychain is unavailable.

### Environment Variables

| Variable | Description |
|----------|-------------|
| `SMARTMEMORY_MODE` | `local` or `remote` — overrides config file |
| `SMARTMEMORY_API_KEY` | API key for remote mode — bypasses keychain |
| `SMARTMEMORY_API_URL` | Remote API URL (default: `https://api.smartmemory.ai`) |
| `SMARTMEMORY_TEAM_ID` | Team/workspace ID for remote mode |
| `SMARTMEMORY_DATA_DIR` | Local data directory (default: `~/.smartmemory`) |
| `SMARTMEMORY_LLM_PROVIDER` | LLM provider for local enrichment |
| `SMARTMEMORY_EMBEDDING_PROVIDER` | Embedding provider (`local`, `openai`, `ollama`) |
| `SMARTMEMORY_DAEMON_PORT` | Daemon port (default: `9014`) |
| `SMARTMEMORY_ASYNC_ENRICHMENT` | Enable/disable background enrichment |
| `OPENAI_API_KEY` | OpenAI API key for embeddings and LLM extraction |
| `GROQ_API_KEY` | Groq API key — alternative to OpenAI for LLM extraction |

## API Reference

### SmartMemory Class

```python
class SmartMemory:
    def __init__(
        self,
        scope_provider=None,
        vector_backend=None,
        cache=None,
        observability: bool = True,
        pipeline_profile=None,
        entity_ruler_patterns=None,
    )

    # Primary API
    def ingest(self, item, sync=True, **kwargs) -> str  # Full pipeline
    def add(self, item, **kwargs) -> str                # Simple storage
    def get(self, item_id: str) -> Optional[MemoryItem]
    def search(self, query: str, top_k: int = 5, memory_type: str = None) -> List[MemoryItem]
    def delete(self, item_id: str) -> bool

    # Graph Integrity
    def delete_run(self, run_id: str) -> int
    def rename_entity_type(self, old: str, new: str) -> int
    def merge_entity_types(self, sources: List[str], target: str) -> int

    # Advanced
    def run_clustering(self) -> dict
    def run_evolution_cycle(self) -> None
    def personalize(self, traits: dict = None, preferences: dict = None) -> None
    def get_all_items_debug(self) -> Dict[str, Any]
    def close(self) -> None
```

### MemoryItem Class

```python
@dataclass
class MemoryItem:
    content: str
    memory_type: str = 'semantic'
    item_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    valid_start_time: Optional[datetime] = None
    valid_end_time: Optional[datetime] = None
    transaction_time: datetime = field(default_factory=datetime.now)
    embedding: Optional[List[float]] = None
    entities: Optional[list] = None
    relations: Optional[list] = None
    metadata: dict = field(default_factory=dict)
```

## Testing

```bash
# Run all tests
PYTHONPATH=. pytest -v tests/

# Run specific test categories
PYTHONPATH=. pytest tests/unit/
PYTHONPATH=. pytest tests/integration/
PYTHONPATH=. pytest tests/e2e/
```

## Use Cases

### Conversational AI Systems
- Maintain context across multiple conversation sessions
- Learn user preferences and adapt responses
- Build comprehensive user profiles over time

### Knowledge Management
- Store and retrieve complex information relationships
- Connect related concepts across different domains
- Build a personal knowledge base with Zettelkasten method

### Personal AI Assistants
- Remember user preferences and past interactions
- Provide contextually relevant recommendations
- Learn from user feedback to improve responses

### Educational Applications
- Track learning progress and adapt teaching strategies
- Personalize content based on individual learning patterns

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

SmartMemory is dual-licensed to provide flexibility for both open-source and commercial use. See [LICENSE](LICENSE) for details.

## Links

- **PyPI Package**: https://pypi.org/project/smartmemory/
- **Core Library**: https://pypi.org/project/smartmemory-core/
- **Documentation**: https://docs.smartmemory.ai
- **GitHub**: https://github.com/smart-memory
- **Issue Tracker**: https://github.com/smart-memory/smart-memory-core/issues

---

```bash
pip install smartmemory[local]
smartmemory setup
```
