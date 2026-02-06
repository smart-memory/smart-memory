# Changelog

All notable changes to SmartMemory will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added

#### Pipeline v2 Foundation (Phase 1)
- **New unified pipeline architecture** (`smartmemory/pipeline/`): Replaces three separate orchestrators with a single composable pipeline built on the StageCommand protocol
  - `StageCommand` protocol (structural subtyping) with `execute()` and `undo()` methods
  - `PipelineState` immutable-by-convention dataclass with full serialization (`to_dict()`/`from_dict()`)
  - `PipelineConfig` nested dataclass hierarchy with named factories (`default()`, `preview()`)
  - `Transport` protocol with `InProcessTransport` for in-process execution
  - `PipelineRunner` with `run()`, `run_to()`, `run_from()`, `undo_to()` — supports breakpoints, resumption, rollback, and per-stage retry with exponential backoff
- **8 stage wrappers** (`smartmemory/pipeline/stages/`): classify, coreference, extract, store, link, enrich, ground, evolve — each wraps existing pipeline components as StageCommands
- **OntologyGraph** (`smartmemory/graph/ontology_graph.py`): Dedicated FalkorDB graph for entity type definitions with three-tier status (seed → provisional → confirmed) and 14 seed types

### Changed
- `SmartMemory.ingest()` now delegates to `PipelineRunner.run()` instead of `MemoryIngestionFlow.run()` — identical output, new internal architecture

### Removed
- `FastIngestionFlow` (`smartmemory/memory/fast_ingestion_flow.py`, 502 LOC) — async ingestion is now a config flag (`mode="async"`) on the unified pipeline

---

## [0.3.2] - 2026-02-05

### Changed

#### Extraction Pipeline
- **Default extractor changed to Groq** (`GroqExtractor`): Llama-3.3-70b-versatile via Groq API — 100% E-F1, 89.3% R-F1, 878ms. Requires `GROQ_API_KEY` env var.
- **New `GroqExtractor` class** in `llm_single.py`: Zero-arg constructor wrapper for registry lazy loading.
- **Fallback chain updated**: groq → llm → llm_single → conversation_aware_llm → spaCy. SpaCy extractor (no API keys needed) re-registered as last-resort fallback.
- **Direct module imports** in registry: Extractors now imported from specific module files (not `__init__.py`) to avoid transitive dependency failures (e.g., GLiNER not installed).
- **Robust fallback iteration**: Extraction pipeline now tries ALL fallbacks (not just the first) when the primary extractor fails to instantiate.
- **`select_default_extractor()` never returns `None`**: Raises `ValueError` with actionable message if no extractors are available.

### Added

#### Graph Validation & Health (Wave 2)
- **New package**: `smartmemory.validation` - Runtime schema validation for memories and edges
  - `MemoryValidator` - Validates memory items against schema constraints (required fields, content length, type validity, metadata types)
  - `EdgeValidator` - Validates edges against registered schemas (allowed source/target types, required metadata, cardinality limits)
- **New package**: `smartmemory.metrics` - Graph health metrics collection
  - `GraphHealthChecker` - Collects orphan ratio, type distribution, edge distribution, provenance coverage via Cypher queries
  - `HealthReport` dataclass with `is_healthy` property (thresholds: orphan < 20%, provenance > 50%)
- **New package**: `smartmemory.inference` - Automatic graph inference engine
  - `InferenceEngine` - Runs pattern-matching rules to create inferred edges with provenance metadata
  - `InferenceRule` dataclass with Cypher pattern, edge type, and confidence
  - 3 built-in rules: causal transitivity, contradiction symmetry, topic inheritance

#### Symbolic Reasoning Layer (Wave 2)
- **New module**: `smartmemory.reasoning.residuation` - Pause reasoning when data is incomplete
  - `ResiduationManager` - Manages pending requirements on decisions; auto-resumes when data arrives via `check_and_resume()`
  - `PendingRequirement` model added to `Decision` with description, created_at, resolved flag
- **New module**: `smartmemory.reasoning.query_router` - Route queries to cheapest effective retrieval
  - `QueryRouter` - Classifies queries as SYMBOLIC (graph Cypher), SEMANTIC (vector search), or HYBRID (both)
  - Pattern-based classification with priority: hybrid > semantic > symbolic
- **New module**: `smartmemory.reasoning.proof_tree` - Auditable reasoning chains
  - `ProofTreeBuilder` - Builds proof trees from graph traversal tracing evidence back to sources
  - `ProofTree` and `ProofNode` with `render_text()` for human-readable proof output
- **New module**: `smartmemory.reasoning.fuzzy_confidence` - Multi-dimensional confidence scoring
  - `FuzzyConfidenceCalculator` - Scores decisions on 4 dimensions: evidence, recency, consensus, directness
  - `ConfidenceScore` with per-dimension breakdown and weighted composite

#### Extended Decision Model (Wave 2)
- `Decision.status` now supports `"pending"` for residuation (decisions awaiting data)
- `Decision.pending_requirements` list for tracking what data is needed
- `PendingRequirement` dataclass with description, created_at, resolved fields
- New edge types: `INFERRED_FROM` (inference provenance), `REQUIRES` (residuation dependencies)

### Changed

#### Evolver Inheritance Cleanup
- Eliminate dual-inheritance in all 12 evolvers: inherit only from `EvolverPlugin` (not `Evolver` + `EvolverPlugin`)
- `Evolver` base class in `plugins/evolvers/base.py` is now a backwards-compatible alias for `EvolverPlugin`

#### AssertionChallenger Strategy Pattern Extraction
- Extract 4 contradiction detection strategies into `reasoning/detection/` package: `LLMDetector`, `GraphDetector`, `EmbeddingDetector`, `HeuristicDetector`
- Extract 4 conflict resolution strategies into `reasoning/resolution/` package: `WikipediaResolver`, `LLMResolver`, `GroundingResolver`, `RecencyResolver`
- Add `DetectionCascade` and `ResolutionCascade` orchestrators for ordered strategy execution
- Extract confidence operations into `ConfidenceManager` class (`reasoning/confidence.py`)
- Slim `AssertionChallenger` from 1,249 to ~200 lines while preserving all public API methods

#### SmartMemory Decomposition
- Extract monitoring operations into `MonitoringManager` (7 methods: summary, orphaned_notes, prune, find_old_notes, self_monitor, reflect, summarize)
- Extract evolution operations into `EvolutionManager` (4 methods: run_evolution_cycle, commit_working_to_episodic, commit_working_to_procedural, run_clustering)
- Extract debug operations into `DebugManager` (4 methods: debug_search, get_all_items_debug, fix_search_if_broken, clear)
- Extract enrichment operations into `EnrichmentManager` (4 methods: enrich, ground, ground_context, resolve_external)
- SmartMemory reduced from 961 to ~750 lines; all public methods preserved as one-line delegates

#### Constructor Injection
- Add optional dependency injection parameters to `SmartMemory.__init__` for all 13 dependencies (graph, crud, search, linking, enrichment, grounding, personalization, monitoring, evolution, clustering, external_resolver, version_tracker, temporal)
- All parameters default to `None` (existing behavior preserved); when provided, injected instances are used instead of creating defaults
- Enables unit testing with mocks without requiring FalkorDB/Redis infrastructure

---

## [0.3.1] - 2026-02-05

### Fixed
- Replace all bare `except:` clauses with `except Exception:` across 6 core files to avoid catching SystemExit/KeyboardInterrupt
- Fix undefined `setFilteredCategories`/`setFilteredNodeTypes` in KnowledgeGraph.jsx (delegate to prop callbacks)
- Fix rules-of-hooks violation in TripleResultsViewer.jsx (conditional `useMemoryStore` call moved to top level)
- Fix undefined `predicateTypes` variable in TripleResultsViewer.jsx (fall back to store value)
- Fix undefined `properties` variable in insights database.py (6 references)
- Add missing `SAMPLE_USER_PROFILE` fixture in maya test_proactive_scheduler.py
- Replace bare `except:` in zettelkasten.py (6 instances), mcp_memory_manager.py (2), background_tasks.py (1)

### Changed
- Standardize ruff config across all Python projects: select `E`, `F`, `B`; ignore `B008`, `E501`
- Fix studio ruff config: move `select` under `[tool.ruff.lint]`, line-length 100 -> 120
- Create ruff configs for maya and insights (previously unconfigured)
- Add `react/prop-types: "off"` to all ESLint configs (web, studio, insights)
- Add `process: "readonly"` global to all ESLint configs

### Version
- Bump all repos to 0.3.1

---

## [0.3.0] - 2026-02-05

### Added

#### Decision Memory System
- **New memory type**: `decision` - First-class decisions with confidence tracking and lifecycle management
- **New model**: `Decision` - Dataclass with provenance, confidence (reinforce/contradict with diminishing returns), and lifecycle (active/superseded/retracted)
- **New module**: `smartmemory.decisions` with:
  - `DecisionManager` - Create, supersede, retract, reinforce, contradict decisions with graph edge management
  - `DecisionQueries` - Filtered retrieval, provenance chains, recursive causal chain traversal
- **New extractor**: `DecisionExtractor` - Regex-based extraction from text + `extract_from_trace()` for ReasoningTrace integration
- **New edge types**: `PRODUCED`, `DERIVED_FROM`, `SUPERSEDES`, `CONTRADICTS`, `INFLUENCES` registered in schema validator
- **Decision types**: inference, preference, classification, choice, belief, policy
- **Conflict detection**: Semantic search + content overlap heuristic for finding contradicting decisions
- **Keyword classification**: Automatic decision type classification from content (no LLM required)
- **Provenance tracking**: Full chain from evidence → reasoning trace → decision → superseded decisions
- **Causal chains**: Recursive traversal of DERIVED_FROM, CAUSED_BY, CAUSES, INFLUENCES, PRODUCED edges with configurable depth
- **New evolver**: `DecisionConfidenceEvolver` - Confidence decay for stale decisions with automatic retraction below threshold
- **Graceful degradation**: All components work without graph (skip edge operations, return empty lists)
- **Service API**: 11 REST endpoints at `/memory/decisions/*` for full decision lifecycle (create, get, list, search, supersede, retract, reinforce, contradict, provenance, causal-chain, conflicts)
- **Maya commands**: `/decide`, `/beliefs`, `/why` with aliases (`/decision`, `/decisions`, `/provenance`, `/explain`)
- **Tests**: 149 unit tests covering model, manager, queries, extractor, and evolver

---

## [Unreleased]

### Added

#### Link Expansion Enricher
- **New enricher**: `LinkExpansionEnricher` - Expands URLs in memory items into rich graph structures
- **URL detection**: Regex-first extraction with fallback to extraction stage URLs
- **Metadata extraction**: Title, description, OG tags, author, published date via BeautifulSoup
- **Entity extraction**: Heuristic extraction from JSON-LD (Schema.org) structured data
- **LLM analysis (optional)**: Summary and deeper entity extraction when `enable_llm=True`
- **Graph integration**: Creates `WebResource` nodes linked to `Entity` nodes via `MENTIONS` edges
- **Error handling**: Failed fetches create nodes with `status='failed'` for retry capability
- **Config**: `LinkExpansionEnricherConfig` with timeout, max URLs, user agent settings

#### Per-Enricher Configuration Support
- **New field**: `EnrichmentConfig.enricher_configs` - Dict mapping enricher names to config dicts
- **Config passthrough**: Enrichment stage now passes per-enricher config to enricher instances
- **Auto config class discovery**: Automatically finds `*Config` class for typed config instantiation
- **Backward compatible**: Enrichers without config or with mismatched config continue to work

#### Claude CLI Integration
- **External package**: `claude-cli` extracted to `regression-io/claude-cli` (private)
- **Optional dependency**: `pip install smartmemory[claude-cli]`
- **No API key required**: Uses Claude subscription authentication via subprocess
- **Simple API**: `claude = Claude(); answer = claude("prompt")`
- **Structured output**: `claude.structured(prompt, schema=MyModel)` with Pydantic
- **Framework adapters**: LangChain and DSPy adapters included
- **Integration**: Available via `LLMClient(provider='claude-cli')`
- **Note**: Experimental, for internal testing

#### Security & Authentication Documentation
- **New documentation**: `docs/SECURITY_AND_AUTH.md` - Comprehensive guide to SmartMemory's security architecture
- **Updated**: `README.md` - Added security/auth examples and references
- **Updated**: `docs/ARCHITECTURE.md` - Enhanced multi-tenancy section with isolation levels

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

#### Coreference Resolution Stage (from feed integration)
- **New pipeline stage**: `CoreferenceStage` - Resolves pronouns and vague references to explicit entity names
- **Example**: "Apple announced... The company exceeded..." → "Apple announced... Apple exceeded..."
- **Enabled by default**: Runs automatically in `ingest()` pipeline before entity extraction
- **Uses fastcoref**: High-quality neural coreference resolution
- **Optional dependency**: `pip install smartmemory[coreference]`

#### Conversation-Aware Extraction with Coreference
- **Enhanced extractor**: `ConversationAwareLLMExtractor` now uses fastcoref chains for entity resolution
- **Coreference chains in context**: `ConversationContext` now includes `coreference_chains` field
- **Auto-selection**: Pipeline auto-selects `conversation_aware_llm` extractor when conversation context is present
- **Resolution priority**: Uses fastcoref chains (high quality) before falling back to heuristic resolution
- **LLM context enhancement**: Coreference mappings included in extraction prompts for better entity recognition
- **Configuration**: `CoreferenceConfig` with resolver, device, enabled settings
- **Location**: `smartmemory.memory.pipeline.stages.coreference`
- **Metadata stored**: Original content and coreference chains preserved in item metadata
- **Use case**: Improves entity extraction quality by making implicit references explicit

#### New Exports
- `smartmemory.memory.pipeline.stages`: Added `CoreferenceStage`, `CoreferenceResult`
- `smartmemory.memory.pipeline.config`: Added `CoreferenceConfig`
- `smartmemory.models`: Added `ReasoningStep`, `ReasoningTrace`, `ReasoningEvaluation`, `TaskContext`, `OpinionMetadata`, `ObservationMetadata`, `Disposition`
- `smartmemory.plugins.extractors`: Added `ReasoningExtractor`
- `smartmemory.plugins.evolvers`: Added `OpinionSynthesisEvolver`, `ObservationSynthesisEvolver`, `OpinionReinforcementEvolver`

#### Dependencies
- Added optional dependency group `coreference` with `fastcoref>=2.1.0`

---

## [0.2.6] - 2025-11-24

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

**Before (v0.2.6)**:
```python
memory.add(item)  # Ran full pipeline (confusing!)
memory._add_basic(item)  # Simple storage (private method)
```

**After (v0.2.6)**:
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

## [0.2.6] - 2025-11-23

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

**After (v0.2.6)**:
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
