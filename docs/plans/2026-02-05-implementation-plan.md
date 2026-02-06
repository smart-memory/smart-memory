# Ontology-Grounded Extraction: Implementation Plan

**Date:** 2026-02-05
**Version:** 1.0 (restructured from strategic plan v5)
**Status:** IN PROGRESS — Phase 1 COMPLETE, Phase 2 COMPLETE, Phase 3 COMPLETE, Phase 4 COMPLETE, Phase 5 COMPLETE, Phase 6 COMPLETE, Phase 7 COMPLETE
**Predecessor docs:** See [Evidence Base](design/evidence-base.md) for benchmark data and research findings.

---

## Governance

This plan was produced through iterative design review (Q0-Q14). Implementation of each phase requires explicit sign-off before beginning. The decision log in Section 7 records all architectural choices and their rationale.

**Design documents** (in `docs/plans/design/`):
- [Pipeline Architecture](design/pipeline-architecture.md) — StageCommand, PipelineState, PipelineConfig, Runner
- [Ontology Model](design/ontology-model.md) — Separate graphs, type system, promotion, TBox/ABox
- [Extraction Stages](design/extraction-stages.md) — Per-stage specifications
- [Self-Learning](design/self-learning.md) — Promotion flow, EntityRuler growth, reasoning validation
- [Service & API](design/service-api.md) — Routes, event-bus, prompt management
- [Metrics & Observability](design/metrics-observability.md) — Redis Streams, Insights dashboard
- [Evidence Base](design/evidence-base.md) — Benchmark results, research findings, anti-patterns

---

## 1. Principles

Five principles govern all implementation decisions. Violating any of these is a design error.

1. **Ontology is foundational** — Always on, no toggle. "Ontology cold" (minimal patterns) not "ontology off." Remove `ontology_enabled` booleans.

2. **One pipeline** — Unified architecture with breakpoint execution, command pattern with undo, serializable state. Three current orchestrators collapse to one. Studio calls core directly.

3. **Parameters govern process** — Use cases are parameter combinations, not categories. `PipelineConfig` is the bridge between Studio and core. 25 use cases analyzed as 10-parameter space (see `2026-02-05-use-cases-reference.md`).

4. **Reasoning validates** — Self-learning uses reasoning (not just statistics) to validate ontology decisions. Level 1 (audit trail) at launch, Level 2 (reasoning validates promotion) in Phase 4.

5. **Discovery over declaration** — System infers its own configuration from usage patterns. Users don't pick a use case; parameters are auto-tuned from observed data.

---

## 2. Architecture

### 2.1 Pipeline

```
                        PipelineConfig (per-workspace, tunable via Studio)
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                     Pipeline (linear, StageCommands)                      │
│                                                                          │
│  text ──► classify ──► coreference ──► simplify ──► entity_ruler         │
│               │             │               │               │            │
│               ▼             ▼               ▼               ▼            │
│          PipelineState flows through, accumulating at each stage          │
│               │             │               │               │            │
│               ▼             ▼               ▼               ▼            │
│  ──► llm_extract ──► ontology_constrain ──► store ──► link ──► enrich   │
│                                                                    │     │
│                                                              ──► evolve  │
│                                                                          │
│  Each stage: StageCommand.execute(state, config) -> state               │
│              StageCommand.undo(state) -> state                          │
└──────────────────────────────────────────────────────────────────────────┘
```

**Key abstractions:**
- **StageCommand** — Protocol with `execute(state, config) → state` and `undo(state) → state`
- **PipelineState** — Serializable dataclass that accumulates as pipeline progresses. Checkpointable.
- **PipelineConfig** — Nested Pydantic dataclasses, per-workspace, tunable via Studio
- **PipelineRunner** — Two orthogonal axes: breakpoints (where to stop) × transport (how state moves)

**Execution modes** (same runner, different config):

| Use Case | Breakpoints | Transport | Effect |
|----------|-------------|-----------|--------|
| Dev/CI | none | in-process | Full pipeline, function calls |
| Production sync | none | in-process | Full pipeline with retry |
| Production async | none | event-bus | Each stage is a Redis Stream consumer |
| Studio/tuning | user-set | in-process | `run_to()`, `run_from()`, undo, iterate |
| Grid search | set + replay | in-process | `run_to()` once, `run_from()` N times |

See [Pipeline Architecture](design/pipeline-architecture.md) for full specifications.

### 2.2 Ontology Model

```
┌─────────────────────────────────────────────────────────┐
│              Global TBox (FalkorDB)                      │
│  Seed entity types (14): Person, Organization,           │
│  Technology, Concept, Event, Location, Document, Tool,   │
│  Skill, Decision, Claim, Action, Metric, Process         │
│  + type-pair priors, mutual exclusion constraints         │
│  + 100+ discoverable types in vocabulary                 │
│  Read-only for tenants. Service-level ops.               │
├─────────────────────────────────────────────────────────┤
│           Tenant Soft-TBox (FalkorDB)                    │
│  Learned patterns promoted via quality gate:             │
│    Phase 1-3: statistical pre-filter + audit trail       │
│    Phase 4+: reasoning validation (Level 2)              │
│  Feeds EntityRuler + LLM prompt schema                   │
│  AssertionChallenger wired for type contradictions        │
├─────────────────────────────────────────────────────────┤
│             Tenant ABox (FalkorDB)                       │
│  The actual graph. Entities, relations, memories.        │
│  Soft-TBox is a materialized view derived from this.     │
└─────────────────────────────────────────────────────────┘
```

**Separate FalkorDB graphs** — Ontology lives in `ws_{id}_ontology`, data lives in `ws_{id}_data`. No label filtering needed, no bleed risk, no existing queries to audit.

**All data is ontology-governed** — When extraction discovers an unknown type, `ontology_constrain` creates it as `provisional` in the ontology graph BEFORE data stores. Every entity in the data graph links to a type that exists in the ontology. No ungoverned data.

**Three-tier type status:**

| Status | Meaning | How It Gets There |
|--------|---------|-------------------|
| `seed` | Curated starting types (14) | Shipped with system |
| `provisional` | Seen but not yet validated | Auto-created on first encounter |
| `confirmed` | Validated through promotion gates | Passes configurable PromotionConfig |

**PromotionConfig** (parameterized gates, per-workspace via Studio):

```python
@dataclass
class PromotionConfig:
    reasoning_validation: bool = True     # Default: reason about promotion
    min_frequency: int = 1                # 1 = confirm on first high-confidence extraction
    min_confidence: float = 0.8
    human_review: bool = False            # HITL gate (deferred — no review UI yet)
```

**Dual-axis type system:**
- **Memory types** (classifies the container): working, semantic, episodic, procedural, zettel, reasoning, opinion, observation, decision
- **Entity types** (classifies content): Person, Organization, Technology, Concept, etc. — extractable + linkable
- **Bridge:** Decision, Opinion, ReasoningTrace, Observation are BOTH memory types AND entity types. A Decision entity links to a Decision model with evidence chain.

See [Ontology Model](design/ontology-model.md) for full specifications.

### 2.3 Self-Learning Loop

```
Ingestion (real-time):
  text → Pipeline.run(text, config) → stored memory + entities (4ms)

Background (event bus):
  promotion_candidates (from PipelineState)
    → statistical pre-filter (confidence > 0.8, frequency > 2)
    → reasoning validation (Phase 4+, LLM reasons about promotion)
    → quality gate passes → write pattern to Tenant Soft-TBox (FalkorDB)
    → EntityRuler reloaded with new pattern
    → next ingestion benefits from learned pattern

Convergence:
  ~1,000 memories: most common domain entities covered
  ~5,000 memories: ruler covers 95%+ of domain vocabulary
  ~10,000 memories: ruler covers 99%+, LLM rarely finds new entities
```

See [Self-Learning](design/self-learning.md) for full specifications.

### 2.4 Multi-Tenancy & Privacy

| Knowledge Type | Scope | Why |
|---------------|-------|-----|
| Entity vocabulary (name → type) | **Global** | Vocabulary, no information content |
| Type-pair priors | **Global** | Pre-built from Wikidata stats |
| Wikidata facts | **Global** | Public knowledge |
| Entity-pair relations from user text | **Always tenant-scoped** | Reveals proprietary processes |
| Memory content | **Always tenant-scoped** | User's private knowledge |
| Graph structure / topology | **Always tenant-scoped** | Process shapes are proprietary |

**Pattern promotion rules:**
1. Wikidata-linkable entity → promote to global immediately (public vocabulary)
2. Non-Wikidata, single tenant → stays tenant-scoped (could be internal codename)
3. Non-Wikidata, multiple tenants independently discover → promote to global after N tenants

### 2.5 PipelineConfig Hierarchy

```
PipelineConfig
├── name, workspace_id, mode (sync|async|preview), retry: RetryConfig
├── parameters (cross-cutting, auto-inferred + manually tunable)
│   └── domain_vocabulary, relation_depth, temporal_sensitivity,
│       contradiction_tolerance, confidence_requirement, scope
├── classify: ClassifyConfig
│   └── model, fallback_type, confidence_threshold
├── coreference: CoreferenceConfig
│   └── enabled, model
├── simplify: SimplifyConfig
│   └── clause_splitting, relative_clause_extraction, passive_to_active,
│       appositive_extraction (all bool, shared spaCy dep parse)
├── extraction: ExtractionConfig (composite stage config)
│   ├── entity_ruler: EntityRulerConfig
│   │   └── enabled, pattern_sources, min_confidence
│   ├── llm_extract: LLMExtractConfig
│   │   └── model, prompt (resolved via PromptProvider), temperature, max_entities
│   ├── ontology_constrain: ConstrainConfig
│   │   └── promotion: PromotionConfig, domain_range_validation
│   ├── enrichment_tier: str | None     ("groq" | "gemma" | None)
│   └── self_learning_enabled: bool
├── store: StoreConfig
│   └── embed_model, vector_dims, dedup_threshold
├── link: LinkConfig
│   └── similarity_threshold, max_links, cross_type_linking
├── enrich: EnrichConfig
│   ├── wikidata: WikidataConfig
│   ├── sentiment: SentimentConfig
│   ├── temporal: TemporalConfig
│   └── topic: TopicConfig
└── evolve: EvolveConfig
    └── enabled_evolvers, decay_rates, synthesis_threshold
```

**Storage:** Saved per workspace in FalkorDB. Named configs ("default", "high-precision", "bulk-import").

**Prompt management:** Three layers — `prompts.json` (defaults) → MongoDB (per-tenant overrides via Studio) → PipelineConfig (runtime values). PromptProvider resolves at pipeline start.

**Future abstraction note:** Nested Pydantic dataclasses for now. If Studio implementation reveals need for generic config traversal (render any subtree as a form, diff from defaults), add config registry/visitor pattern then. Trigger: when hand-coding Studio UI per config type becomes repetitive.

---

## 3. Existing Code Inventory

### What Gets Absorbed

| Existing Code | Lines | Becomes |
|---------------|-------|---------|
| `MemoryIngestionFlow` | 473 | `Pipeline.run(text, config)` |
| `FastIngestionFlow` | 502 | Deleted — async mode is a config flag |
| `ExtractorPipeline` (Studio) | 491 | Deleted — Studio calls core directly |
| Ontology CRUD routes (776 LOC) | 776 | PipelineConfig management routes |
| `OntologyManager` (198 LOC) | 198 | Splits: config loading → PipelineConfig loader; inference → pipeline stages |
| `OntologyStorage` (FileSystem) | — | Deprecated — ontology in separate FalkorDB graph |
| Studio `OntologyConfigSection.jsx` toggle | — | Removed. Ontology is foundational. |
| 3 Studio integration endpoints | — | Replaced by pipeline breakpoint execution |
| ~130 LOC duplicated normalization | 130 | Consolidated into pipeline stages |

**Total deleted:** ~1,100+ LOC of duplicate/obsolete code.

### What Stays Useful

| Existing Code | Role in New Architecture |
|---------------|--------------------------|
| `SmartMemory.add()` | Recursion guard for internal stages (correct design, stays) |
| `ExtractionPipeline` (268 LOC) | Becomes extraction pipeline components |
| `StoragePipeline` (217 LOC) | Becomes store StageCommand |
| `EnrichmentPipeline` (122 LOC) | Becomes enrich StageCommand |
| `IngestionRegistry` (285 LOC) | Becomes component registry |
| `IngestionObserver` (331 LOC) | Emits events at stage transitions → metrics |
| `Ontology` model (225 LOC) | Schema portion of PipelineConfig.ontology |
| `OntologyIR` (278 LOC) | Import/export format for ontology state |
| `OntologyLLMExtractor` (499 LOC) | Becomes llm_extract StageCommand |
| `OntologyLab.jsx` (554 LOC) | Evolves into pipeline ontology viewer |
| `RedisStreamQueue` | Becomes event-bus transport implementation |
| Pattern CRUD endpoints | Kept for direct admin ops, manual corrections |
| PromptProvider + prompts.json | Three-layer prompt resolution (keep, extend) |

---

## 4. Implementation Phases

### Phase 1: Pipeline Foundation — COMPLETE

**Goal:** Replace three orchestrators with one unified pipeline. Establish the core abstractions that all subsequent phases build on.

**Status:** COMPLETE. All deliverables shipped. 109 unit tests pass.

**Deliverables:**

| # | Deliverable | Files | Status |
|---|------------|-------|--------|
| 1.1 | `StageCommand` protocol (execute/undo) | `smartmemory/pipeline/protocol.py` | Done |
| 1.2 | `PipelineState` dataclass (serializable, checkpointable) | `smartmemory/pipeline/state.py` | Done |
| 1.3 | `PipelineConfig` hierarchy (nested dataclasses, per-workspace) | `smartmemory/pipeline/config.py` | Done |
| 1.4 | `PipelineRunner` with `InProcessTransport` | `smartmemory/pipeline/runner.py` | Done |
| 1.5 | `run()`, `run_to()`, `run_from()`, `undo_to()` API | `smartmemory/pipeline/runner.py` | Done |
| 1.6 | Separate ontology FalkorDB graph (`ws_{id}_ontology`) | `smartmemory/graph/ontology_graph.py` | Done |
| 1.7 | Seed 14 entity types (three-tier status: seed/provisional/confirmed) | `smartmemory/graph/ontology_graph.py` | Done |
| 1.8 | Wrap existing stages as StageCommands (classify, coreference, extract, store, link, enrich, ground, evolve) | `smartmemory/pipeline/stages/` | Done |
| 1.9 | `SmartMemory.ingest()` delegates to `Pipeline.run()` | `smartmemory/smart_memory.py` | Done |
| 1.10 | Delete `FastIngestionFlow` (502 LOC, unused) | `smartmemory/memory/ingestion/` | Done |

**Acceptance criteria:** All met.

**Dependencies:** None.

---

### Phase 2: Extraction Stages — COMPLETE

**Goal:** Implement the new extraction stages that bring ontology-grounded extraction into the pipeline.

**Status:** COMPLETE. 4 native stages replace `ExtractStage`. Pipeline goes from 8 to 11 stages. 188 unit tests pass.

**Deliverables:**

| # | Deliverable | Files | Status |
|---|------------|-------|--------|
| 2.1 | `classify` StageCommand (memory type classification) | `smartmemory/pipeline/stages/classify.py` | Done (Phase 1) |
| 2.2 | `coreference` StageCommand (pronoun resolution) | `smartmemory/pipeline/stages/coreference.py` | Done (Phase 1) |
| 2.3 | `simplify` StageCommand | `smartmemory/pipeline/stages/simplify.py` | Done |
| | — clause splitting, relative clause extraction, passive→active, appositive extraction | | |
| | — 4 config flags, `min_token_count`, shared spaCy dep parse | | |
| 2.4 | `entity_ruler` StageCommand (spaCy NER + label mapping) | `smartmemory/pipeline/stages/entity_ruler.py` | Done |
| 2.5 | `llm_extract` StageCommand (wraps `LLMSingleExtractor`) | `smartmemory/pipeline/stages/llm_extract.py` | Done |
| 2.6 | `ontology_constrain` StageCommand | `smartmemory/pipeline/stages/ontology_constrain.py` | Done |
| | — entity merge (ruler + LLM), type validation, provisional creation, relation filtering | | |
| 2.7 | Migrate hardcoded prompts to `prompts.json` | — | Deferred to Phase 3 |

**Deferred items:**
- 2.7 (prompt migration) — deferred to Phase 3. Self-learning promotion flow (Redis Streams) also deferred.
- Studio frontend changes deferred — only backend metadata/models updated.
- `passive_to_active` transform is a no-op placeholder (full rewriting needs a dedicated rewriter).

**Implementation notes:**
- `PipelineState.simplified_text: Optional[str]` renamed to `simplified_sentences: List[str]`
- `SimplifyConfig` rewritten with 4 boolean transform flags (was `enabled` + `model`)
- `EntityRulerConfig` extended with `pattern_sources`, `min_confidence`, `spacy_model`
- `LLMExtractConfig` extended with `max_relations`
- `ConstrainConfig` extended with `domain_range_validation`
- `PromotionConfig` extended with `reasoning_validation`, `min_frequency`, `min_confidence`
- `ExtractStage` and `test_extract.py` deleted
- Studio: `SimplifyRequest`, `EntityRulerRequest`, `OntologyConstrainRequest` added to `models/pipeline.py`
- Studio: `pipeline_info.py` updated with 4 new stage descriptions replacing `extractor_pipeline`

**Dependencies:** Phase 1 (complete).

---

### Phase 3: Storage + Post-Processing Stages — COMPLETE

**Goal:** Complete the pipeline with storage, linking, enrichment, and evolution stages. Add metrics emission.

**Status:** COMPLETE. Pipeline metrics emission added. Normalization deduplicated. Studio extraction preview rewired to v2 PipelineRunner. ExtractorPipeline deprecated (full deletion deferred to Phase 7). 8 new metrics tests pass.

**Deliverables:**

| # | Deliverable | Files | Status |
|---|------------|-------|--------|
| 3.1 | `store` StageCommand | `smartmemory/pipeline/stages/store.py` | Done (Phase 1) |
| 3.2 | `link` StageCommand | `smartmemory/pipeline/stages/link.py` | Done (Phase 1) |
| 3.3 | `enrich` StageCommand | `smartmemory/pipeline/stages/enrich.py` | Done (Phase 1) |
| 3.4 | `evolve` StageCommand | `smartmemory/pipeline/stages/evolve.py` | Done (Phase 1) |
| 3.5 | Pipeline metrics emission via Redis Streams | `smartmemory/pipeline/metrics.py` | Done |
| | — `PipelineMetricsEmitter` with fire-and-forget `EventSpooler` integration | | |
| | — Per-stage latency, entity/relation counts, error tracking | | |
| | — Wired into `PipelineRunner` via optional `metrics_emitter` callback | | |
| 3.6 | ExtractorPipeline deprecation + Studio v2 preview | `smart-memory-studio/`, `smartmemory/memory/pipeline/extractor.py` | Done (scoped) |
| | — Deprecation warning added to `ExtractorPipeline` | | |
| | — `get_v2_runner()` in Studio `pipeline_registry.py` | | |
| | — `run_extraction_v2()` using `PipelineRunner.run_to()` | | |
| | — `transaction.py` extraction preview uses v2 with v1 fallback | | |
| | — Full deletion deferred to Phase 7 (~15 Studio files still import old pipeline) | | |
| 3.7 | Normalization deduplication | `smartmemory/memory/pipeline/enrichment.py`, `storage.py` | Done |
| | — Removed weak `_sanitize_relation_type()`, replaced with canonical `sanitize_relation_type()` | | |

**Acceptance criteria:** All met.

**Dependencies:** Phase 2 (complete).

---

### Phase 4: Self-Learning Loop (**COMPLETE** — 2026-02-06)

**Goal:** Implement the feedback loop where LLM discoveries grow the EntityRuler, closing the quality gap between fast tier and LLM tier over time.

**Status:** COMPLETE. 55 new tests, 252 total pipeline_v2 tests passing. See `docs/plans/reports/phase-4-self-learning-loop-report.md`.

**Deliverables:**

| # | Deliverable | Files |
|---|------------|-------|
| 4.1 | Promotion flow: provisional → confirmed via PromotionConfig gates | `smartmemory/ontology/promotion.py` |
| 4.2 | EntityRuler pattern growth from confirmed types | `smartmemory/pipeline/stages/entity_ruler.py` |
| | — Diff: LLM entities − ruler entities = new patterns | |
| | — Quality gate: confidence, frequency, type consistency | |
| | — Hot-reload ruler on pattern change (Redis pub/sub) | |
| 4.3 | Entity-pair cache (reuse known LLM relations for repeated entity pairs) | `smartmemory/ontology/entity_pair_cache.py` |
| 4.4 | Type-pair validation in `ontology_constrain` | integrated in 2.6 |
| 4.5 | Reasoning validation (Level 2) | `smartmemory/ontology/reasoning_validation.py` |
| | — Statistical pre-filter passes → reasoning LLM validates promotion | |
| | — ReasoningTrace stored as `reasoning` memory type with `reasoning_domain="ontology"` | |
| | — AssertionChallenger wired for entity-type contradictions | |
| 4.6 | Three pattern layers: seed (global) → learned global (promoted) → learned tenant | `smartmemory/ontology/pattern_store.py` |
| | — `is_global=True` for seed and promoted patterns | |
| | — `workspace_id` scoping for tenant patterns via SecureSmartMemory | |

**Acceptance criteria:**
- After 100+ memories, new EntityRuler patterns appear in ontology graph
- Entity-pair cache reuses known relations on repeated entity pairs
- Reasoning traces explain promotion decisions and are queryable as memories
- Promotion respects PromotionConfig parameters per workspace

**Dependencies:** Phase 3. **Parallelizable with:** Phases 5 and 6.

---

### Phase 5: Service + API

**Goal:** Expose pipeline through REST API. Add event-bus transport for production async execution.

**Deliverables:**

| # | Deliverable | Files |
|---|------------|-------|
| 5.1 | PipelineConfig management routes | `memory_service/api/routes/pipeline.py` |
| | `GET /memory/pipeline/configs` — list named configs | |
| | `POST /memory/pipeline/configs` — save named config | |
| | `GET /memory/pipeline/configs/{name}` — load config | |
| | `PUT /memory/pipeline/configs/{name}` — update config | |
| 5.2 | Pattern admin routes | `memory_service/api/routes/ontology.py` |
| | `GET /memory/ontology/patterns` — list with stats | |
| | `POST /memory/ontology/patterns` — manual add/override | |
| | `DELETE /memory/ontology/patterns/{id}` — remove | |
| 5.3 | Ontology status routes | |
| | `GET /memory/ontology/status` — convergence metrics, pattern counts | |
| | `POST /memory/ontology/import` — import OntologyIR (seeding, migration) | |
| | `GET /memory/ontology/export` — export as OntologyIR | |
| 5.4 | `EventBusTransport` (Redis Streams between stages) | `smartmemory/pipeline/transport/event_bus.py` |
| | — Serialize PipelineState → publish to stream | |
| | — Per-stage consumer groups for horizontal scaling | |
| | — Same semantics as InProcessTransport, just RPC over message queue | |
| 5.5 | Retry with undo | `smartmemory/pipeline/runner.py` |
| | — Per-stage retry policy (max_retries, backoff, fallback) | |
| | — Failed stage → undo partial work → retry or fallback | |
| 5.6 | Finish Studio prompt editing UI (API exists, React incomplete) | `smart-memory-studio/` |

**Acceptance criteria:**
- Named pipeline configs saveable/loadable per workspace
- Event-bus transport runs full pipeline asynchronously
- Stage failures retry with undo, then fallback
- Pattern admin allows manual corrections outside pipeline

**Dependencies:** Phase 3. **Parallelizable with:** Phases 4 and 6.

---

### Phase 6: Insights + Observability — COMPLETE

**Status:** COMPLETE (2026-02-06). See `reports/phase-6-insights-observability-report.md`.

**Goal:** Build the metrics aggregation pipeline and Insights dashboard showing pipeline, ontology, and extraction quality metrics. Merged with Decision Memory frontend work.

**Deliverables:**

| # | Deliverable | Files |
|---|------------|-------|
| 6.1 | Metrics aggregation consumer (Redis Streams → pre-aggregated data) | `smartmemory/pipeline/metrics_consumer.py` |
| | — Reads metric events from Redis Streams | |
| | — Aggregates into time-bucketed data (5min buckets) | |
| | — Writes pre-aggregated metrics to Redis for fast dashboard reads | |
| 6.2 | Pipeline metrics dashboard | `smart-memory-insights/` |
| | — Per-stage latency (classify: 2ms, entity_ruler: 4ms, llm_extract: 740ms) | |
| | — Throughput (memories/min, by stage) | |
| | — Error/retry rates per stage | |
| | — Stage bottleneck identification | |
| | — Queue depth per stage (event-bus mode) | |
| 6.3 | Ontology metrics dashboard | `smart-memory-insights/` |
| | — Type registry growth over time (provisional → confirmed) | |
| | — EntityRuler pattern count and coverage rate | |
| | — Convergence curve (ruler quality approaching LLM quality) | |
| | — Promotion/rejection rates through quality gate | |
| | — Type status distribution (seed / provisional / confirmed counts) | |
| 6.4 | Extraction quality dashboard | `smart-memory-insights/` |
| | — Entity count per memory, by type | |
| | — Relation count per memory | |
| | — Confidence distribution across extractions | |
| | — Provisional vs confirmed type ratio | |
| | — LLM vs ruler attribution (which extractor found what) | |
| 6.5 | Replace stub `getExtractionStats()` / `getExtractionOperations()` with real data | `smart-memory-insights/` |

**Acceptance criteria:**
- Dashboard shows real metrics (not stubs) within 5 minutes of pipeline activity
- Convergence curve visible: ruler pattern count grows, LLM-only discovery rate drops
- Stage bottlenecks identifiable from dashboard

**Dependencies:** Phase 3. **Parallelizable with:** Phases 4 and 5.

---

### Phase 7: Studio Pipeline UI

**Status:** COMPLETE. Learning dashboard, PipelineConfig editor, breakpoint debug runner, prompt UI polish. 7.5/7.6 (benchmarking) deferred to Phase 8. See `docs/plans/reports/phase-7-studio-pipeline-ui-report.md`.

**Goal:** Rebuild Studio's pipeline configuration around PipelineConfig. This is not a refactor of the existing 1336-line `MemoryConfigurationPanel.jsx` — it is a new design driven by the PipelineConfig shape.

**Scope:**

| # | Component | Description | Status |
|---|-----------|-------------|--------|
| 7.1 | PipelineConfig editor | One section per stage (~10 stages), each independently editable | Done |
| 7.2 | Breakpoint execution UI | Run-to, inspect intermediate state, modify params, resume | Done |
| 7.3 | Learning page (ontology viewer) | Type registry, convergence dashboard, pattern browser, promotion queue, activity feed | Done |
| 7.4 | Prompt editing UI | Polish: copy-to-clipboard, character count, timestamp | Done |
| 7.5 | Benchmarking workflow | Batch execution + comparison across parameter sets | Deferred to Phase 8 |
| 7.6 | Grid search / parameter tuning | Automated parameter sweep using breakpoint + replay | Deferred to Phase 8 |

**Dependencies:** Phase 5 (needs API routes).

---

### Phase 8: Hardening

**Goal:** Production-readiness. Tests, benchmarks, documentation, and deferred features.

**Deliverables:**

| # | Deliverable |
|---|------------|
| 8.1 | Pipeline integration tests (full run, breakpoints, undo, retry) |
| 8.2 | Stage unit tests (each stage independently) |
| 8.3 | Self-learning tests (promotion flow, EntityRuler growth, convergence) |
| 8.4 | Benchmark suite integration in Studio |
| 8.5 | Documentation (CLAUDE.md, README, CHANGELOG updates) |
| 8.6 | HITL review UI (if `human_review: True` is needed by this point) |
| 8.7 | Synthetic test datasets for 3-5 representative parameter combinations |

**Dependencies:** Phase 4 + Phase 7.

---

## 5. Dependencies & Critical Path

```
Phase 1 (Pipeline Foundation)
     │
     ▼
Phase 2 (Extraction Stages)
     │
     ▼
Phase 3 (Storage + Post-Processing)
     │
     ├──────────────────┬──────────────────┐
     ▼                  ▼                  ▼
Phase 4             Phase 5            Phase 6
(Self-Learning)     (Service + API)    (Insights)
     │                  │
     └────────┬─────────┘
              ▼
Phase 7 (Studio Pipeline UI — MAJOR)
              │
              ▼
Phase 8 (Hardening)
```

| Phase | Can Start After | Parallelizable With |
|-------|----------------|---------------------|
| 1: Pipeline Foundation | Immediately | — |
| 2: Extraction Stages | Phase 1 | — |
| 3: Storage + Post-Processing | Phase 2 | — |
| 4: Self-Learning | Phase 3 | Phase 5, 6 |
| 5: Service + API | Phase 3 | Phase 4, 6 |
| 6: Insights | Phase 3 | Phase 4, 5 |
| 7: Studio UI | Phase 5 (needs API) | — |
| 8: Hardening | Phase 4 + 7 | — |

**Critical path:** 1 → 2 → 3 → 4 → 7 → 8

---

## 6. Cross-Cutting Concerns

### Pattern Storage

All patterns stored in FalkorDB (not JSONL, not MongoDB, not SQLite):
- **Entity patterns** are metadata properties on entity nodes (not separate documents)
- **Type-pair priors** are edges between `:EntityType` nodes in ontology graph
- **Entity-pair cache** needs no new storage — the data graph IS the cache
- **No vector embeddings on patterns** — all lookups are structural (exact match, Cypher index)

**OSS vs Hosted:**
- **OSS (core library):** Ships EntityRuler mechanism empty. Self-learning builds patterns from scratch.
- **Hosted service:** Has seed data as service-level ops concern. Generated from training corpus via Groq extraction + quality gate.

### Prompt Management

Three-layer resolution: PromptProvider resolves MongoDB override > `prompts.json` default. Pipeline only reads materialized PipelineConfig at runtime.

**Migration tasks:** Move `SINGLE_CALL_PROMPT` (llm_single.py) and `EXTRACTION_SYSTEM_PROMPT` (reasoning.py) into `prompts.json`.

### Wikidata Integration

- REST API for 95% of lookups (single entity by name/QID)
- SPARQL for type hierarchy queries only (`P31/P279*` chains)
- Runs in `enrich` stage (async/event-bus) — latency invisible to user
- **Scaling escape hatch:** If we outgrow 200 req/s rate limit, download Wikidata dump

### Relations

- **No RelationRuler.** Relations are semantic, not lexical — LLM territory.
- **Keep:** Type-pair priors (filter impossible relations) in `ontology_constrain`
- **Keep:** Entity-pair cache (reuse known LLM relations) — grows with self-learning
- **Skip:** Dep-parse templates (low ceiling ~70%, fragile, high maintenance)
- Accept: fast tier handles entities (96.9%), enrichment tier handles relations (85-88%)

---

## 7. Decision Log

All architectural decisions from the Q0-Q14 review, for traceability.

| # | Decision | Choice | Rationale |
|---|----------|--------|-----------|
| Q0 | What IS the ontology? | Triples in FalkorDB. Three jobs (classify/predict/constrain). Three layers (Global TBox → Tenant Soft-TBox → Tenant ABox). 14 seed types + 100+ discoverable. Parameterized process. | Field research (NELL, GATE, SPIRES, Wikontic, Text2KGBench). 25 use cases as 10-parameter space. |
| Q1 | Pattern storage backend | FalkorDB for all patterns. Entity patterns as node metadata. No JSONL, no MongoDB. OSS ships empty. Seeds are service-level ops. | Patterns ARE graph entities. No vector embeddings needed. |
| Q2 | Ingestion architecture | One pipeline with breakpoint execution. StageCommand protocol. Three orchestrators → one. ~1,100 LOC deleted. | Debugger breakpoint metaphor. Studio calls core directly. |
| Q3 | Worker deployment | Subsumed by Q2. Enrichment worker = event-bus runner for enrichment StageCommand. | No separate enrichment architecture needed. |
| Q4 | Existing ontology infra | Absorb into pipeline. Separate FalkorDB graphs. All data ontology-governed. Three-tier types. PromotionConfig (Option D, C default). HITL deferred. | Ontology is pipeline input + output + persistent knowledge. Separate graphs = no bleed. |
| Q5 | Insights metrics | Pipeline emits metrics via Redis Streams. Pre-aggregated by consumer. Three metric categories. | Event-driven, decoupled. Same infra as event-bus transport. |
| Q6 | Studio 61KB panel | Rebuild around PipelineConfig. MAJOR discussion deferred to implementation. | Pipeline architecture fundamentally changes the panel. Don't refactor — redesign. |
| Q7 | Quality gate scope | Subsumed by Q4. PromotionConfig in PipelineConfig, per-workspace, tunable via Studio. | Global defaults + per-workspace overrides. No separate quality gate abstraction. |
| Q8 | Wikidata API strategy | REST + SPARQL. FalkorDB canonical, Redis write-through cache. Scaling escape hatch: local dump. | Simple, free. Both updated in same write operation. |
| Q9 | Prompt selection | Reframed as prompt management. Three-layer resolution: prompts.json → MongoDB → PipelineConfig. Migrate hardcoded prompts. Finish Studio UI. | PromptProvider feeds PipelineConfig, not runtime. |
| Q10 | ExtractionConfig | Composite stage config within PipelineConfig. Nested Pydantic dataclasses. `simplify` stage added. Future: config registry/visitor if needed. | Each sub-stage independently breakpointable. |
| Q11 | Ontology foundational? | YES. First principle. Every other decision assumes this. | Two code paths = double testing. Optional features get less adoption. |
| Q12 | Benchmark w/ ontology | No RelationRuler. Type-pair validation + entity-pair cache only. Benchmarking is a Studio workflow. | Relations are semantic (LLM territory). Dep-parse ceiling too low. |
| Q13 | Phase ordering | Complete rewrite (v5). 8 phases. Implementation gated on sign-off. | Old phases obsolete after pipeline architecture decisions. |
| Q14 | Reasoning integration | Level 2 (statistical pre-filter + reasoning validation). Level 1 at launch, Level 2 in Phase 4. AssertionChallenger wired to ontology. First-class models are BOTH memory types AND entity types. | Don't follow NELL verbatim — we have LLMs. Dual-axis bridges extraction and specialized behavior. |

---

## 8. Anti-Patterns

Things that were tested and proven to not work. Do not revisit.

| Approach | Why It Failed | Revisit Only If |
|----------|--------------|-----------------|
| Progressive prompting (4 variants) | LLM anchors to draft, all variants worse than standalone | Fundamentally different model architecture |
| GLiNER2 + GLiREL | GLiNER NER at 34% recall; GLiREL useless without good NER | GLiNER v3+ with better NER |
| REBEL (end-to-end) | 3.9s latency, 62% R-F1 — dominated on all axes | Never |
| NuExtract tiny | 100% precision, 45% recall | NuExtract 2.0 with larger model |
| NuNER Zero | Broken with gliner >= 0.2.x | Package compatibility fixed upstream |
| spaCy trf for fast tier | 33ms vs 4ms, sm+ruler dominates (96.9% > 94.6%) | Never |
| Bigger local models (70B) | Hermes-70B (87.3%) < Gemma-27B (95.3%) | Never for extraction |
| Reasoning/thinking models | DeepSeek-R1, OLMo-think both underperform — overhead, no quality gain | Never for extraction |
| RelationRuler / dep-parse templates | Semantic relations need LLMs. Dep-parse ceiling ~65-70% | Never |
| GPT-5-mini | 93.4% E-F1, 55.2% R-F1, ~40s, $0.002. Over-extracts. | Never — Groq dominates |
| `ontology_enabled` toggle | Contradicts foundational principle. Two code paths = double testing. | Never |

---

## 9. Benchmark Reference

Top results from 53 configurations tested (see [Evidence Base](design/evidence-base.md) for full data):

| Model | Type | Latency | E-F1 | R-F1 | Cost |
|-------|------|---------|------|------|------|
| GPT-4o-mini | LLM API | 4.3s | 100% | 91.3% | ~$0.001 |
| Groq Llama-3.3-70b | LLM API | 740ms | 97.7% | 85-88% | ~$0.0003 |
| spaCy sm + EntityRuler | Local NLP | 4ms | 96.9% | 65.1% | $0 |
| Gemma-3-27b-it | Local LLM | 31s | 95.3% | 86.7% | $0 |

**Architecture implications:**
- Fast tier (entity_ruler): 96.9% E-F1 at 4ms — within 0.8% of Groq on entities
- Enrichment tier (llm_extract): 85-88% R-F1 — only LLMs can do good relation extraction
- Self-learning: EntityRuler grows from LLM feedback, fast tier improves over time
- Zero-cost variant: Gemma-3-27b-it validated for background enrichment ($0, beats Groq on relations)
