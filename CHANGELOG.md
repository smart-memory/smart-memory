# Changelog

All notable changes to SmartMemory will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added

#### System 2 Memory: Reasoning Traces
- **New memory type**: `reasoning` - Captures chain-of-thought reasoning traces
- **New extractor**: `ReasoningExtractor` - Extracts reasoning from Thought:/Action:/Observation: markers or via LLM detection
- **New models**: `ReasoningStep`, `ReasoningTrace`, `ReasoningEvaluation`, `TaskContext`
- **New edge types**: `CAUSES` and `CAUSED_BY` for linking reasoning to artifacts
- **Use case**: Query "why" decisions were made, not just the outcomes

#### Synthesis Memory: Opinions & Observations
- **New memory type**: `opinion` - Beliefs with confidence scores that can be reinforced or contradicted
- **New memory type**: `observation` - Synthesized entity summaries from scattered facts
- **New models**: `OpinionMetadata`, `ObservationMetadata`, `Disposition`
- **New evolvers**:
  - `OpinionSynthesisEvolver` - Forms opinions from episodic patterns
  - `ObservationSynthesisEvolver` - Creates entity summaries from facts
  - `OpinionReinforcementEvolver` - Updates confidence based on new evidence

#### New Exports
- `smartmemory.models`: Added `ReasoningStep`, `ReasoningTrace`, `ReasoningEvaluation`, `TaskContext`, `OpinionMetadata`, `ObservationMetadata`, `Disposition`
- `smartmemory.plugins.extractors`: Added `ReasoningExtractor`
- `smartmemory.plugins.evolvers`: Added `OpinionSynthesisEvolver`, `ObservationSynthesisEvolver`, `OpinionReinforcementEvolver`

---

## [0.2.2] - 2025-11-24

### 🎯 Major: API Clarification - `ingest()` vs `add()`

**Breaking Changes**: Renamed methods to align behavior with intent

#### Changed

- **`add()` renamed to `ingest()`**: Full agentic pipeline (extract → store → link → enrich → evolve)
- **`_add_basic()` renamed to `add()`**: Simple storage (normalize → store → embed)
- **Removed `ingest_old()`**: Consolidated async queueing into `ingest(sync=False)`

#### API Design

```python
# Full pipeline - use for user-facing ingestion
item_id = memory.ingest("content")  # sync=True by default
result = memory.ingest("content", sync=False)  # async: {"item_id": str, "queued": bool}

# Simple storage - use for internal operations, derived items
item_id = memory.add(item)
```

#### Migration Guide

**Before (v0.2.2)**:
```python
memory.add(item)  # Ran full pipeline (confusing!)
memory._add_basic(item)  # Simple storage (private method)
```

**After (v0.2.2)**:
```python
memory.ingest(item)  # Full pipeline (clear intent)
memory.add(item)     # Simple storage (public, clear intent)
```

#### Internal Callers Updated

- `cli.py`: Now uses `ingest()` for CLI add command
- `mcp_handler.py`: Now uses `ingest()` for external MCP calls
- `evolution.py`: Uses `add()` for evolved items (no re-evolution)
- `enrichment.py`: Uses `add()` for derived items (no re-pipeline)

---

## [0.2.2] - 2025-11-23

### 🎯 Major: Complete Scoping Architecture Refactor

**Breaking Changes**: Method signatures changed - removed all hardcoded scoping parameters

#### Changed

- **`SmartMemory.__init__()`**: Now accepts optional `scope_provider` parameter (defaults to `DefaultScopeProvider()` for OSS usage)
- **`SmartMemory.search()`**: Removed `user_id` parameter - uses `ScopeProvider` exclusively
- **`SmartMemory.personalize()`**: Removed `user_id` parameter - operates on current user via `ScopeProvider`
- **`SmartMemory.run_clustering()`**: Removed `workspace_id`, `tenant_id`, `user_id` parameters
- **`SmartMemory.run_evolution_cycle()`**: Removed `workspace_id`, `tenant_id`, `user_id` parameters
- **`SmartMemory.get_all_items_debug()`**: Removed `tenant_id`, `user_id` parameters - uses `ScopeProvider`
- **`VectorStore.add()`**: Removed `workspace_id` parameter
- **`VectorStore.upsert()`**: Removed `workspace_id` parameter
- **`VectorStore.search()`**: Removed `workspace_id` parameter override
- **Evolution methods**: All evolution-related methods now use `ScopeProvider` automatically
- **Clustering methods**: All clustering methods now use `ScopeProvider` automatically

#### Added

- **`DefaultScopeProvider`**: Returns empty filters for unrestricted OSS usage
- **Lazy import pattern**: Avoids circular dependencies when importing `DefaultScopeProvider`
- **Complete documentation**: `SCOPING_ARCHITECTURE.md` with line-by-line trace of scoping flow
- **OSS simplicity**: `SmartMemory()` works out-of-the-box without configuration
- **Service security**: Service layer always provides secure `ScopeProvider` with tenant isolation

#### Benefits

- ✅ **Zero configuration** for OSS single-user applications
- ✅ **Automatic tenant isolation** in service layer
- ✅ **No hardcoded parameters** - clean method signatures
- ✅ **Single source of truth** - all scoping through `ScopeProvider`
- ✅ **Core library agnostic** - no knowledge of multi-tenancy concepts
- ✅ **Backward compatible** - service layer unchanged (always provided `ScopeProvider`)

#### Migration Guide

**Before (v0.1.16)**:
```python
# Had to pass user_id everywhere
memory.search("query", user_id="user123")
memory.run_clustering(tenant_id="tenant456", workspace_id="ws789")
```

**After (v0.2.2)**:
```python
# OSS: No parameters needed
memory = SmartMemory()
memory.search("query")
memory.run_clustering()

# Service: ScopeProvider injected automatically
memory = SmartMemory(scope_provider=my_scope_provider)
memory.search("query")  # Automatically filtered
```

### Added

#### Similarity Graph Traversal (SSG) Retrieval
- Novel graph-based semantic search algorithms for enhanced multi-hop reasoning
- Two high-performance algorithms implemented:
  - `query_traversal`: Best for general queries (100% test pass, 0.91 precision/recall)
  - `triangulation_fulldim`: Best for high-precision factual queries
- Features:
  - Hybrid neighbor discovery (graph relationships + vector similarity)
  - Early stopping to prevent over-retrieval
  - Similarity caching for performance
  - Multi-tenant workspace filtering
  - Configurable via `config.json`
- New module: `smartmemory/retrieval/ssg_traversal.py`
- Configuration section added to `config.json`
- Integration with `SmartGraphSearch` via `_search_with_ssg_traversal()`
- Comprehensive test suite (18 unit tests, all passing)

#### API Examples

```python
from smartmemory import SmartMemory
from smartmemory.retrieval.ssg_traversal import SimilarityGraphTraversal

# Initialize
sm = SmartMemory()
ssg = SimilarityGraphTraversal(sm)

# Query traversal (best for general queries)
results = ssg.query_traversal(
    query="How do neural networks work?",
    max_results=15,
    workspace_id="workspace_123"
)

# Triangulation (best for precision)
results = ssg.triangulation_fulldim(
    query="What is the capital of France?",
    max_results=10,
    workspace_id="workspace_123"
)

# Integrated search with automatic fallback
results = sm.search("neural networks", use_ssg=True)
```

#### Reference
Eric Lester. (2025). Novel Semantic Similarity Graph Traversal Algorithms for Semantic Retrieval Augmented Generation Systems. https://github.com/glacier-creative-git/semantic-similarity-graph-traversal-semantic-rag-research

---

## [0.1.5] - 2025-10-04

### Added

#### Version Tracking System
- Bi-temporal version tracking with valid time and transaction time
- Automatic version numbering for memory updates
- Version comparison and diff functionality
- Graph-backed persistence
- Version caching

#### Temporal Search
- Time-range search functionality
- Version-aware search results
- Temporal metadata in results
- Integration with version tracker

#### Temporal Relationship Queries
- Query relationships at specific points in time
- Relationship history tracking
- Temporal pattern detection
- Co-occurring relationship detection

#### Bi-Temporal Joins
- Temporal overlap joins
- Concurrent event detection
- Temporal correlation queries
- Multiple join strategies

#### Performance Optimizations
- Temporal indexes for time-based lookups
- Query optimization and planning
- Batch operations
- Result caching
- Performance statistics

### API Examples

```python
# Version tracking
version = memory.temporal.version_tracker.create_version(
    item_id="memory123",
    content="Updated content",
    changed_by="user_id",
    change_reason="Correction"
)

# Temporal search
results = memory.temporal.search_temporal(
    "Python programming",
    start_time="2024-09-01",
    end_time="2024-09-30"
)

# Relationship queries
rels = memory.temporal.relationships.get_relationships_at_time(
    "memory123",
    datetime(2024, 9, 15)
)

# Bi-temporal joins
overlaps = memory.temporal.relationships.temporal_join(
    ["memory1", "memory2", "memory3"],
    start_time=datetime(2024, 9, 1),
    end_time=datetime(2024, 9, 30),
    join_type='overlap'
)

# Performance optimization
index = TemporalIndex()
optimizer = TemporalQueryOptimizer(index)
plan = optimizer.optimize_time_range_query(start, end)
```

### Tests
- 115 tests covering temporal features
- 25 unit tests for version tracking and relationships
- 90 integration tests for temporal search and joins

---

## [0.1.4] - 2025-10-04

### Added

- Temporal query system with 8 methods for time-travel and audit trails
- TemporalQueries API (`memory.temporal`):
  - `get_history()` - Version history of memories
  - `at_time()` - Query memories at specific points in time
  - `get_changes()` - Track changes to memory items
  - `compare_versions()` - Compare memory versions
  - `rollback()` - Rollback with dry-run preview
  - `get_audit_trail()` - Audit logs for compliance
  - `find_memories_changed_since()` - Find recent changes
  - `get_timeline()` - Timeline visualization

- Time-travel context manager (`memory.time_travel()`):
  - Execute queries in temporal context
  - Nested time-travel support

- Documentation:
  - `docs/BITEMPORAL_GUIDE.md` - Complete guide
  - `docs/IMPLEMENTATION_GUIDE.md` - Developer reference
  - `.cascade/smartmemory_implementation_memory.md` - Implementation patterns

- Examples:
  - `examples/temporal_queries_basic.py` - Feature demonstrations
  - `examples/temporal_audit_trail.py` - Compliance scenario
  - `examples/temporal_debugging.py` - Debugging use case

- Test suite (84 tests):
  - 39 unit tests for TemporalQueries
  - 24 unit tests for time-travel context
  - 21 integration tests for compliance

---

## [0.1.3] - 2025-10-04

### Added

- Zettelkasten implementation with bidirectional linking, emergent structure detection, and discovery
- Wikilink parser with automatic bidirectional linking
  - `[[Note Title]]` - Wikilinks with automatic link creation
  - `[[Note|Alias]]` - Wikilink aliases
  - `((Concept))` - Concept mentions
  - `#hashtag` - Hashtag extraction
- Documentation (`docs/ZETTELKASTEN.md`)
- Example scripts:
  - `examples/zettelkasten_example.py` - System demonstration
  - `examples/wikilink_demo.py` - Wikilink showcase
- CLI commands for Zettelkasten operations:
  - `smartmemory zettel add` - Add notes with wikilinks
  - `smartmemory zettel overview` - System overview
  - `smartmemory zettel backlinks` - Show backlinks
  - `smartmemory zettel connections` - Show all connections
  - `smartmemory zettel suggest` - Get AI suggestions
  - `smartmemory zettel clusters` - Detect knowledge clusters
  - `smartmemory zettel parse` - Parse wikilinks from content
- Unit tests for wikilink parser (18 tests)

### Fixed
- Fixed `FalkorDB.get_neighbors()` to handle Node objects properly
- Fixed `ZettelBacklinkSystem` to handle MemoryItem returns
- Fixed `ZettelEmergentStructure._get_all_notes()` to accept label='Note'
- Fixed `ZettelEmergentStructure._get_note_links()` for cluster detection
- Fixed wikilink resolution timing (now creates links after note is added)

### Documentation
- Zettelkasten user guide with API reference
- Updated README with Zettelkasten sections

---

## [0.1.2] - 2025-10-04

### Changed
- ChromaDB is now optional - Moved to optional dependency `[chromadb]`
- FalkorDB is default vector backend - Handles both graph and vector storage
- Python requirement updated to >=3.10
- Version externalized to `__version__.py`
- Removed hardcoded paths from tests

### Fixed
- Fixed Python 3.10+ syntax compatibility (`str | None` → `Optional[str]`)
- Fixed vector backend registry to handle missing ChromaDB gracefully
- Fixed all evolver tests with proper typed configs
- Fixed all enricher tests with correct API expectations
- Fixed test isolation issues

### Added
- Zettelkasten Memory System
  - Bidirectional linking with automatic backlinks
  - Knowledge cluster detection
  - Discovery engine for connections
  - Knowledge path finding
  - Missing connection suggestions
  - Random walk exploration
  - System health analytics
- `smartmemory/memory/types/zettel_memory.py` - Zettelkasten memory type
- `smartmemory/memory/types/zettel_extensions.py` - Discovery and structure detection
- `smartmemory/graph/types/zettel.py` - Zettelkasten graph backend
- `smartmemory/stores/converters/zettel_converter.py` - Zettel data conversion
- `smartmemory/plugins/evolvers/episodic_to_zettel.py` - Episodic → Zettel evolver
- `smartmemory/plugins/evolvers/zettel_prune.py` - Zettel pruning evolver
- `docs/ZETTELKASTEN.md` - Zettelkasten documentation
- `examples/zettelkasten_example.py` - Zettelkasten demo
- `tests/integration/zettelkasten/` - Test suite
- `smartmemory/__version__.py` - Single source of truth for version
- `docs/BACKEND_PLUGIN_DESIGN.md` - Design for future vectorstore plugins
- `VERSION_AND_COMPATIBILITY.md` - Version and compatibility documentation
- Better error messages for missing backends

### Documentation
- Updated README with Zettelkasten memory type
- Zettelkasten guide with API reference
- Updated dependency documentation

---

## [0.1.1] - 2025-10-03

### Added
- Plugin system with 19 built-in plugins
- Security system with 4 security profiles
- Plugin security features: Sandboxing, permissions, resource limits
- External plugin support via entry points
- CLI tools (optional install with `[cli]`)
- `docs/PLUGIN_SECURITY.md` - Security documentation
- 4 plugin examples

### Changed
- Plugins converted to class-based architecture
- Plugin discovery and registration system
- Security profiles: trusted, standard, restricted, untrusted

### Fixed
- Plugin loading and initialization
- Security validation and enforcement

---

## [0.1.0] - 2025-10-01

### Added
- Initial release
- Multi-type memory system (working, semantic, episodic, procedural)
- FalkorDB graph backend
- ChromaDB vector backend
- Basic plugin system
- Redis caching
- LLM integration via LiteLLM

---

## Version Numbering

SmartMemory follows [Semantic Versioning](https://semver.org/):
- **MAJOR.MINOR.PATCH** (e.g., 0.1.5)
- **MAJOR**: Breaking changes
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes, backward compatible

**Current Status:** Pre-1.0 (Beta)
- API may change between minor versions
- Production-ready but evolving
