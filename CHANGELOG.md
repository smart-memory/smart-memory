# Changelog

All notable changes to SmartMemory will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.3.9] - 2026-02-11

### Security

#### Temporal Query Tenant Isolation (SEC-TEMPORAL-1)

- **Fixed cross-tenant data leak in `TemporalQueries`** (`smartmemory/temporal/queries.py`): Added `_get_node_scoped()` helper method that enforces workspace isolation when retrieving nodes
- **`compare_versions()` now respects tenant boundaries**: Uses scoped node retrieval to prevent cross-workspace version comparison
- **`rollback()` now respects tenant boundaries**: Uses scoped node retrieval to prevent cross-workspace item access and modification
- **OSS mode backward compatibility**: When no scope provider is configured, temporal queries work as before (no restrictions)

### Added

#### Token Cost Instrumentation (CFS-1)

- **`PipelineTokenTracker`** (`smartmemory/pipeline/token_tracker.py`): Per-request tracker that rides on `PipelineState`, recording tokens spent and avoided per pipeline stage with model attribution and cost estimation
- **`StageTokenRecord`** dataclass for individual token events with prompt/completion breakdown, model, and reason fields
- **`token_tracker` field on `PipelineState`**: Automatically created by `PipelineRunner` at pipeline start, populated by stages during execution
- **LLM extract stage tracking**: Records spent tokens after LLM calls, avoided tokens on cache hits and stage disabled
- **Ground stage tracking**: Records avoided tokens when Wikipedia entities resolve from graph instead of API
- **Ontology constrain tracking**: Records avoided tokens on entity-pair cache hits
- **`get_last_usage()` in DSPy adapter**: Thread-local consume-once accessor for LLM token usage from DSPy calls
- **`SmartMemory.last_token_summary` property**: Access the most recent pipeline run's token summary
- **Extended `COST_PER_1K_TOKENS`** in `utils/token_tracking.py`: Added Groq Llama, Gemini Flash, Claude Haiku pricing

---

## [0.3.8] - 2026-02-11

### Added

#### Graph Integrity — Run Tracking & Entity Type Migration (CORE-GRAPH-INT-1)

- **`run_id` injection in StoreStage** (`smartmemory/pipeline/stages/store.py`): Pipeline now injects `run_id` from raw_metadata into stored item metadata, enabling run-based entity cleanup
- **`SmartGraph.delete_by_run_id(run_id)`** (`smartmemory/graph/smartgraph.py`): New method to delete all nodes created by a specific pipeline run, with workspace scope filtering
- **`SmartGraph.rename_entity_type(old, new)`** (`smartmemory/graph/smartgraph.py`): Bulk-rename entity types across all matching nodes in the graph
- **`SmartGraph.merge_entity_types(sources, target)`** (`smartmemory/graph/smartgraph.py`): Merge multiple entity types into a single target type
- **`SmartMemory.delete_run(run_id)`** (`smartmemory/smart_memory.py`): Public API to delete all entities from a specific pipeline run
- **`SmartMemory.rename_entity_type(old, new)`** (`smartmemory/smart_memory.py`): Public API for ontology evolution — rename entity types in-place
- **`SmartMemory.merge_entity_types(sources, target)`** (`smartmemory/smart_memory.py`): Public API to consolidate fragmented entity types

---

## [0.3.7] - 2026-02-10

### Fixed

- **Evolution typed config**: Fixed `EvolutionOrchestrator.commit_working_to_episodic()` and `commit_working_to_procedural()` to pass typed config objects (`WorkingToEpisodicConfig`, `WorkingToProceduralConfig`) instead of raw dicts — eliminates TypeError from evolvers expecting typed configs

---

## [0.3.6] - 2026-02-08

### Changed

#### Corpus → Library Rename (STU-UX-13 V1)
- **Renamed `Corpus` model to `Library`** (`smartmemory/models/library.py`): New primary model with `retention_policy`, `retention_days` fields
- **Renamed `corpus_id` to `library_id`** in `Document` model, added `content_hash`, `storage_phase`, `evict_at` fields
- **Backward-compat shim** (`smartmemory/models/corpus.py`): Re-exports `Library` as `Corpus` for existing imports
- **Updated `CorpusStore`** (`smartmemory/stores/corpus/store.py`): Now operates on `Library` objects, supports both `library_id` and `corpus_id` filters
- **Updated `models/__init__.py`**: Exports `Library` and `Document`

---

## [0.3.5] - 2026-02-07

### Added

#### Batch Ontology Update — Core Metadata (WS-8 Phase 1)
- **`extraction_status` pipeline field** (`smartmemory/pipeline/state.py`): New `extraction_status` field on `PipelineState` tracking whether LLM extraction ran (`ruler_only`, `llm_enriched`, `llm_failed`)
- **`LLMExtractStage` sets extraction_status** (`smartmemory/pipeline/stages/llm_extract.py`): Status set on all code paths — disabled, empty text, success, and failure
- **`StoreStage` injects extraction_status into metadata** (`smartmemory/pipeline/stages/store.py`): `extraction_status` flows through to FalkorDB node properties for batch worker queries
- **`ensure_extraction_indexes()`** (`smartmemory/graph/indexes.py`): Standalone index utility for creating FalkorDB index on `extraction_status`, callable from service worker or tests

#### Auto-assign Team and Workspace on Signup
- **`TeamModel`** (`service_common/models/auth.py`): New dataclass for persisting teams in MongoDB `teams` collection
- **`UserModel.default_team_id`** (`service_common/models/auth.py`): New optional field storing the user's default team assignment
- **Team persistence on signup** (`service_common/services/auth_service.py`): Default team is now created and persisted in MongoDB during user registration
- **Team context in JWT auth** (`service_common/auth/jwt_provider.py`): `ServiceUser` is now populated with `team_memberships`, `current_team_id`, and `current_team_role` at token authentication time (previously only patched in by `get_service_user()`)
- **Workspace-team linking** (`service_common/repositories/workspace_repo.py`): `ensure_personal_workspace()` now accepts `default_team_id` and grants team admin access to the personal workspace
- **API key auth includes team context** (`service_common/auth/jwt_provider.py`): `_authenticate_api_key` now populates team memberships on ServiceUser (consistent with JWT flow)
- **`/auth/me` returns `default_team_id`** (`memory_service/api/routes/auth.py`): UserResponse now always includes the user's default team ID

#### Team/Workspace Roadmap (WS-9)
- **WS-9 workstream** (`docs/plans/2026-02-04-reasoning-system-roadmap.md`): Added Team & Workspace Management roadmap entry — auto-assignment done, remaining CRUD APIs, invitation flow, multi-team, admin UI

### Removed
- **Backward compatibility fallbacks**: Removed all deterministic `team_{tenant_suffix}` fallback computations from `jwt_provider.py`, `auth_service.py`, `middleware.py`, `auth.py` (service + maya), and `core.py` — no users exist, clean slate
- **Workspace backfill logic**: Removed `elif` branch in `ensure_personal_workspace()` that backfilled team access on existing workspaces
- **In-memory team patching in `get_service_user()`**: Removed fallback that computed team membership when `ServiceUser.team_memberships` was empty — now handled at authentication time by `JWTAuthProvider`

### Changed

#### Test Suite Sweep
- **Removed dead code**: Deleted 4 debug/benchmark scripts from `tests/` (zero test functions), removed empty `tests/e2e/` scaffold (11 dirs with only `__init__.py`)
- **Trimmed conftest**: Removed ~150 lines of unused fixtures from `tests/conftest.py` (sample_memory_items, sample_entities, sample_relations, sample_embeddings, conversation_context, conversation_manager, mock_smartmemory_dependencies, clean_memory, TestDataFactory)
- **Relocated orphaned tests**: Moved `tests/plugins/evolvers/test_opinion_synthesis.py` and `tests/plugins/extractors/test_reasoning.py` to proper `tests/unit/plugins/` locations
- **Marked deprecated extractors**: SpacyExtractor and RelikExtractor test classes marked `@pytest.mark.skip(reason="deprecated extractor")`
- **Coverage config**: Added `--cov=smartmemory --cov-report=term-missing`, marker registration, and `pytest-cov` dev dependency to `pyproject.toml`
- **Removed duplicate tests**: Removed TestOpinionMetadata, TestObservationMetadata, TestDisposition from `test_opinion_synthesis.py` (covered by `tests/unit/models/test_opinion.py`)
- **Removed tautology tests**: Deleted TestExtractorComparison class (3 no-op self-equality assertions)
- **Removed redundant conftest markers**: Marker assignments now handled by `pyproject.toml`

### Added

#### Ontology Layers: Public Base + Private Overlay
- **`OntologySubscription` model** (`smartmemory/ontology/models.py`): Tracks base registry subscription with pinned version and hidden types; serializes to/from dict with backward compatibility
- **`LayerDiff` model** (`smartmemory/ontology/models.py`): Dataclass representing diff between base and overlay layers (base_only, overlay_only, overridden, hidden)
- **`LayeredOntology` class** (`smartmemory/ontology/layered.py`): Merged read view of base + overlay ontologies — entity-level override (overlay wins), rules not inherited from base, hidden types excluded, provenance tracking (local/base/override/hidden), detach to flatten
- **`LayeredOntologyService`** (`memory_service/services/layer_service.py`): Subscription lifecycle orchestration — subscribe, unsubscribe (flatten), pin/unpin version, hide/unhide types, diff computation (service layer, not core)
- **8 layer API endpoints** (`memory_service/api/routes/ontology_layers.py`): `POST/DELETE /ontology/registry/{id}/subscribe`, `PUT/DELETE /ontology/registry/{id}/subscribe/pin`, `POST /ontology/registry/{id}/subscribe/hidden`, `DELETE /ontology/registry/{id}/subscribe/hidden/{type_name}`, `GET /ontology/registry/{id}/layers`, `GET /ontology/registry/{id}/layer-diff`
- **75 tests**: 59 unit tests (models, LayeredOntology, LayeredOntologyService) + 16 API endpoint tests

- **Archive API endpoints**: `POST /api/archive/store` and `GET /api/archive/{uri}` for durable conversation artifact storage
- **Graph path endpoint**: `GET /api/graph/path` for shortest-path queries between knowledge graph nodes
- **`SmartMemory.find_shortest_path()`**: Public method for graph path traversal using FalkorDB `shortestPath()` Cypher
- **`SecureSmartMemory` proxy methods**: `archive_put`, `archive_get`, `find_shortest_path` delegate to core with tenant scoping

- **Model tests**: `test_memory_item.py`, `test_entity.py`, `test_opinion.py`, `test_reasoning.py` — 82 tests covering creation, validation, serialization, and key behaviors
- **Observability tests**: `test_events.py`, `test_json_formatter.py`, `test_logging_filter.py` — 27 tests covering EventSpooler, JsonFormatter, LogContextFilter (all mocked, no Docker)
- **Evolver tests**: `test_episodic_to_semantic.py`, `test_working_to_episodic.py`, `test_episodic_decay.py`, `test_decision_confidence.py` — 71 tests covering metadata, config, and evolution logic

### Security

- **Workspace access fail-closed** (`service_common/auth/scope.py`): `_can_access_workspace` now returns `False` on exception instead of `True` — prevents tenant isolation bypass when MongoDB is unreachable
- **Team ID exact match validation** (`service_common/auth/scope.py`): Changed substring `in` check to exact `==` for team ID validation — prevents cross-tenant access via crafted team IDs
- **OAuth random password** (`memory_service/api/routes/auth.py`, `maya/auth/routes.py`): OAuth users now get `secrets.token_urlsafe(32)` instead of static `"oauth-no-password"` — prevents password-based login to OAuth accounts
- **Open redirect prevention** (`memory_service/api/routes/auth.py`): `frontend_callback` parameter validated against allowed origins (FRONTEND_URL, MAYA_FRONTEND_URL, CORS_ORIGINS, localhost) — prevents token theft via crafted OAuth login link
- **Password reset method fix** (`memory_service/api/routes/auth.py`): Fixed `invalidate_all_refresh_tokens` → `revoke_all_user_tokens` (was calling non-existent method)
- **Ontology tenant isolation**: Added `validate_registry_ownership()` to 5 registry endpoints that were missing tenant checks (`apply_changeset`, `list_registry_snapshots`, `get_registry_changelog`, `rollback_registry`, `import_registry_snapshot`). Fixed `export_registry_snapshot` which had no authentication at all.
- **WebSocket authentication**: Added JWT auth to `/ws/feedback` endpoint via `?token=<jwt>` query param. Invalid/missing tokens rejected with close code 1008 (Policy Violation).
- **Ontology access model**: Formalized workspace-shared access model — `created_by` is audit metadata, not an access gate. Removed 6 TODO comments and converted warning logs to informational audit logs.

### Added

#### Decision Confidence Evolver v2.0
- **Evidence-based reinforcement/contradiction** (`smartmemory/plugins/evolvers/decision_confidence.py`): Enhanced from decay-only to full evidence matching — keyword (content, domain, tags), semantic (cosine similarity), and contradiction signal detection against recent episodic/semantic/opinion memories
- **Decision model integration**: Uses `Decision.reinforce()` and `Decision.contradict()` for confidence math with diminishing returns and proportional penalties, persists full `Decision.to_dict()` state
- **Registry registration** (`smartmemory/evolution/registry.py`): Registered as `decision_confidence` so it runs in the evolution pipeline
- **Duplicate episodic prevention**: `fetched_episodic` flag skips episodic in standard search loop when the dedicated episodic API succeeds
- **47 unit tests** (`tests/unit/decisions/test_decision_evolver.py`): Covering reinforcement, contradiction, decay, retraction, evidence matching (keyword/domain/tag/semantic), full cycle, staleness, and registry integration

#### Ontology Templates
- **Template catalog** (`smartmemory/ontology/template_service.py`): `TemplateService` — browse, preview, clone, save-as-template, delete custom templates. Built-in templates cached in memory, custom templates stored via `OntologyStorage` with `is_template` flag
- **Built-in templates** (`smartmemory/ontology/templates/`): 3 curated ontology templates — General Purpose (12 entity types, 8 relationships), Software Engineering (15 entity types, 10 relationships), Business & Finance (14 entity types, 9 relationships)
- **Template data models** (`smartmemory/ontology/models.py`): `TemplateInfo` and `TemplatePreview` dataclasses for catalog browsing and preview
- **Template API endpoints** (`memory_service/api/routes/ontology.py`): 5 endpoints — `GET /templates` (list), `GET /templates/{id}` (preview), `POST /templates/clone` (clone into workspace), `POST /templates` (save as template), `DELETE /templates/{id}` (delete custom)
- **Ontology model extensions**: `is_template` and `source_template` fields on `Ontology`, serialized via `to_dict`/`from_dict`, included in `FileSystemOntologyStorage.list_ontologies`

### Fixed

- **Ontology deserialization bug**: Fixed operator precedence in `Ontology.from_dict()` — `data.get('x') or {}.items()` was iterating raw dict keys instead of key-value pairs. Added parentheses: `(data.get('x') or {}).items()`

#### Studio Pipeline UI (Phase 7)
- **Learning Page backend** (`smart-memory-studio/server/memory_studio/api/routes/learning.py`): 8 API endpoints — stats, convergence, promotions (with approve/reject), patterns, types, activity feed — all proxying to core OntologyGraph with tenant isolation
- **Learning Page frontend** (`smart-memory-studio/web/src/pages/Learning.jsx`): Full dashboard with stats cards, promotion queue (approve/reject actions), type registry (searchable/sortable), pattern browser (layer filtering), activity feed (auto-refresh 30s)
- **Learning hooks** (`smart-memory-studio/web/src/hooks/learning/`): 5 data hooks — useLearningStats, useLearningPromotions, useLearningPatterns, useLearningTypes, useLearningActivity
- **LearningService** (`smart-memory-studio/web/src/services/LearningService.js`): API client for all learning endpoints
- **PipelineConfigEditor** (`smart-memory-studio/web/src/components/pipeline/PipelineConfigEditor.jsx`): Accordion-based config editor matching PipelineConfig dataclass, save/load named profiles via Profiles API
- **ConfigField** + **StageConfigSection**: Smart form field renderer (switch/number/text/select by type) and collapsible per-stage sections with reset-to-default
- **BreakpointRunner** (`smart-memory-studio/web/src/components/pipeline/BreakpointRunner.jsx`): Debug pipeline runner — set breakpoints on any of 11 stages, run-to/resume/undo with intermediate state inspection
- **PipelineStepper** + **StateInspector**: Visual step indicator with status icons and tabbed state viewer (entities, relations, text, timings)
- **PromptsPanel polish**: Character count, copy-to-clipboard button, last-modified timestamp, improved empty state messaging
- **Navigation**: Learning page with TrendingUp icon added to Studio nav

#### Insights + Observability & Decision Memory UI (Phase 6)
- **`MetricsConsumer`** (`smartmemory/pipeline/metrics_consumer.py`): Reads pipeline metric events from Redis Streams, aggregates into 5-min time buckets, writes pre-aggregated metrics to Redis Hashes for fast dashboard reads
- **Pipeline Metrics API** (`smart-memory-insights/server/observability/api.py`): `GET /api/pipeline/metrics`, `/api/pipeline/bottlenecks` — per-stage latency, throughput, error rates from MetricsConsumer aggregated data
- **Ontology Metrics API**: `GET /api/ontology/status`, `/api/ontology/growth`, `/api/ontology/promotion-rates` — direct FalkorDB Cypher queries for type/pattern counts, convergence estimates, promotion rates
- **Extraction Quality API**: `GET /api/extraction/quality`, `/api/extraction/attribution` — confidence distribution histograms, provisional ratio, ruler vs LLM attribution from graph entity data
- **Pipeline Performance dashboard** (`smart-memory-insights/web/src/pages/Pipeline.jsx`): Summary cards, per-stage latency bar chart, throughput area chart, error rate chart with 1H/6H/24H period selector
- **Ontology Health dashboard** (`smart-memory-insights/web/src/pages/Ontology.jsx`): Summary cards, type status donut, pattern layers donut, convergence curve, pattern growth area chart, promotion rates bar chart
- **Extraction Quality dashboard update** (`smart-memory-insights/web/src/pages/Extraction.jsx`): ConfidenceDistribution histogram, AttributionChart pie, TypeRatioChart bar replacing previous stub data
- **Decision Memory client SDK** (`smart-memory-client`): `create_decision()`, `get_decision()`, `list_decisions()`, `supersede_decision()`, `retract_decision()`, `reinforce_decision()`, `get_provenance_chain()`, `get_causal_chain()` methods
- **Decision Memory web UI** (`smart-memory-web`): DecisionCard, DecisionList, ProvenanceChain components; standalone `/Decisions` route with Scale icon + amber accent in navigation
- **Insights navigation**: Pipeline and Ontology pages added to sidebar and router

#### Pipeline v2 Service + API (Phase 5)
- **Pipeline Config CRUD** (`smart-memory-service/memory_service/api/routes/pipeline.py`): Named pipeline config management — `GET/POST/PUT/DELETE /memory/pipeline/configs` with FalkorDB storage as `PipelineConfig` nodes in the ontology graph
- **`OntologyGraph` pipeline config storage**: `save_pipeline_config()`, `get_pipeline_config()`, `list_pipeline_configs()`, `delete_pipeline_config()` for persistent named configs per workspace
- **Pattern Admin routes** (`smart-memory-service/memory_service/api/routes/ontology.py`): `GET/POST/DELETE /memory/ontology/patterns` for entity pattern management with stats, Redis pub/sub reload notifications
- **`OntologyGraph.delete_entity_pattern()`**: Delete EntityPattern nodes with workspace scoping
- **Ontology Status endpoint** (`GET /memory/ontology/status`): Convergence metrics — pattern counts by source layer, entity type status distribution (seed/provisional/confirmed)
- **Ontology Import/Export** (`POST /memory/ontology/import`, `GET /memory/ontology/export`): Portable ontology data for backup/migration with entity type and pattern support
- **`EventBusTransport`** (`smartmemory/pipeline/transport/event_bus.py`): Redis Streams transport for async pipeline execution with per-stage consumer groups, status polling, and result retrieval
- **`StageConsumer`**: Worker class for consuming and executing pipeline stages from Redis Streams
- **Async ingest mode**: `POST /memory/ingest?mode=async` submits via `EventBusTransport`, returns `run_id` for polling; `GET /memory/ingest/async/{run_id}` for status
- **`StageRetryPolicy`** (`smartmemory/pipeline/config.py`): Per-stage retry policy with configurable `backoff_multiplier`, overrides global `RetryConfig`
- **`PipelineConfig.stage_retry_policies`**: Dict mapping stage names to `StageRetryPolicy` for fine-grained retry control

### Changed
- **`PipelineRunner._execute_stage()`** now checks `config.stage_retry_policies` for per-stage overrides, calls `stage.undo()` on retry exhaustion before raising or skipping
- **`transport.py` → `transport/` package**: Converted single file to package to accommodate `event_bus.py`; backward-compatible imports preserved

#### Pipeline v2 Self-Learning Loop (Phase 4)
- **`PromotionEvaluator`** (`smartmemory/ontology/promotion.py`): Six-gate evaluation pipeline for entity type promotion — min name length, common word blocklist (~200 words), min confidence, min frequency, type consistency, optional LLM reasoning validation
- **`PromotionWorker`** (`smartmemory/ontology/promotion_worker.py`): Background Redis Stream consumer for async promotion candidate processing with DLQ error handling
- **`PatternManager`** (`smartmemory/ontology/pattern_manager.py`): Hot-reloadable dictionary of learned entity patterns with Redis pub/sub reload notifications and common word filtering
- **`EntityPairCache`** (`smartmemory/ontology/entity_pair_cache.py`): Redis read-through cache for known entity-pair relations with 30-min TTL, graph fallback, and invalidation
- **`ReasoningValidator`** (`smartmemory/ontology/reasoning_validator.py`): LLM-based validation for promotion candidates with ReasoningTrace storage as `reasoning` memory type
- **`OntologyGraph` frequency tracking**: `increment_frequency()`, `get_frequency()` for atomic type frequency and running average confidence tracking
- **`OntologyGraph` entity patterns**: `add_entity_pattern()`, `get_entity_patterns()`, `get_type_assignments()` for three-layer pattern storage (seed / promoted / tenant)
- **`OntologyGraph` pattern layers**: `seed_entity_patterns()` for bootstrap, `get_pattern_stats()` for layer counts
- **`RedisStreamQueue.for_promote()`**: Factory method for promotion job queue (follows `for_enrich()` pattern)
- **`_ngram_scan()`**: N-gram dictionary scan (up to 4-grams) for entity pattern matching in `EntityRulerStage`

### Changed
- **`OntologyConstrainStage`** now tracks entity type frequency on every accepted entity, enqueues promotion candidates to Redis stream (with inline fallback), and injects cached entity-pair relations
- **`EntityRulerStage`** now accepts optional `PatternManager` for learned pattern dictionary scan after spaCy NER
- **`PromotionConfig`** extended with `min_type_consistency` (0.8) and `min_name_length` (3) fields
- **`SmartMemory._create_pipeline_runner()`** wires `PatternManager`, `EntityPairCache`, and promotion queue into the pipeline

#### Pipeline v2 Storage & Post-Processing (Phase 3)
- **`PipelineMetricsEmitter`** (`smartmemory/pipeline/metrics.py`): Fire-and-forget pipeline metrics emission via Redis Streams. Emits `stage_complete` events per stage and `pipeline_complete` summary with timing breakdown, entity/relation counts, and workspace context. Stream: `smartmemory:metrics:pipeline`
- **`SmartMemory.create_pipeline_runner()`**: Public factory method exposing the v2 `PipelineRunner` for Studio and other consumers that need `run_to()`/`run_from()` breakpoint execution
- **Studio `get_v2_runner()`** (`pipeline_registry.py`): Cached v2 `PipelineRunner` per tenant for Studio backend, backed by `SecureSmartMemory`
- **Studio `run_extraction_v2()`** (`extraction.py`): Extraction preview via `PipelineRunner.run_to("ontology_constrain")` — replaces old pipeline path for extraction previews

### Changed
- **`PipelineRunner`** accepts optional `metrics_emitter` parameter — calls `on_stage_complete()` after each stage and `on_pipeline_complete()` after full run
- **Extraction preview** in Studio (`transaction.py`) now uses v2 `PipelineRunner` with fallback to v1 `ExtractorPipeline`
- **Normalization deduplication**: `EnrichmentPipeline` and `StorageEngine` now use canonical `sanitize_relation_type()` from `memory.ingestion.utils` instead of weak local `_sanitize_relation_type()` (space→underscore only). Relation types now get regex normalization, uppercase, FalkorDB-safe prefix, and 50-char limit

### Deprecated
- **`ExtractorPipeline`** (`smartmemory/memory/pipeline/extractor.py`): Emits `DeprecationWarning` on instantiation. Use `smartmemory.pipeline.stages` via `PipelineRunner` instead. Full removal deferred to Phase 7

#### Pipeline v2 Native Extraction (Phase 2)
- **4 native extraction stages** replace the single `ExtractStage` wrapper — pipeline grows from 8 to 11 stages:
  - `SimplifyStage` — splits complex sentences into atomic statements using spaCy dependency parsing with configurable transforms (clause splitting, relative clause extraction, appositive extraction)
  - `EntityRulerStage` — rule-based entity extraction using spaCy NER with label mapping, deduplication, and confidence filtering
  - `LLMExtractStage` — LLM-based entity/relation extraction via `LLMSingleExtractor` with configurable truncation limits
  - `OntologyConstrainStage` — merges ruler + LLM entities, validates types against `OntologyGraph`, filters relations by accepted endpoints, auto-promotes provisional types
- **Extended `PipelineConfig`** with Phase 2 fields:
  - `SimplifyConfig` — 4 boolean transform flags, `min_token_count`
  - `EntityRulerConfig` — `pattern_sources`, `min_confidence`, `spacy_model`
  - `LLMExtractConfig` — `max_relations`
  - `ConstrainConfig` — `domain_range_validation`
  - `PromotionConfig` — `reasoning_validation`, `min_frequency`, `min_confidence`
- **`PipelineState.simplified_sentences`** — renamed from `simplified_text` (Optional[str]) to `List[str]` for multi-sentence output
- **Studio backend models** — `SimplifyRequest`, `EntityRulerRequest`, `OntologyConstrainRequest` dataclasses added; pipeline component descriptions updated

#### Pipeline v2 Foundation (Phase 1)
- **New unified pipeline architecture** (`smartmemory/pipeline/`): Replaces three separate orchestrators with a single composable pipeline built on the StageCommand protocol
  - `StageCommand` protocol (structural subtyping) with `execute()` and `undo()` methods
  - `PipelineState` immutable-by-convention dataclass with full serialization (`to_dict()`/`from_dict()`)
  - `PipelineConfig` nested dataclass hierarchy with named factories (`default()`, `preview()`)
  - `Transport` protocol with `InProcessTransport` for in-process execution
  - `PipelineRunner` with `run()`, `run_to()`, `run_from()`, `undo_to()` — supports breakpoints, resumption, rollback, and per-stage retry with exponential backoff
- **11 stage wrappers** (`smartmemory/pipeline/stages/`): classify, coreference, simplify, entity_ruler, llm_extract, ontology_constrain, store, link, enrich, ground, evolve
- **OntologyGraph** (`smartmemory/graph/ontology_graph.py`): Dedicated FalkorDB graph for entity type definitions with three-tier status (seed → provisional → confirmed) and 14 seed types

### Changed
- `SmartMemory.ingest()` now delegates to `PipelineRunner.run()` instead of `MemoryIngestionFlow.run()` — identical output, new internal architecture
- Pipeline stage count increased from 8 to 11 with native extraction stages

### Removed
- `ExtractStage` (`smartmemory/pipeline/stages/extract.py`) — replaced by 4 native extraction stages (simplify, entity_ruler, llm_extract, ontology_constrain)
- `ExtractionPipeline` and `IngestionRegistry` no longer needed for pipeline v2 extraction
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
