# Ontology-Grounded Extraction: Implementation Plan

**Date:** 2026-02-05
**Status:** All decisions finalized. Implementation gated on sign-off.
**Source documents:**
- `2026-02-05-ontology-strategic-implementation-plan.md` — Q&A decisions (Q0–Q14)
- `2026-02-05-ontology-grounded-extraction-design.md` — Architecture vision
- `2026-02-05-extraction-benchmark-findings.md` — 53-config benchmark results
- `2026-02-05-two-tier-extraction-pipeline-design.md` — Early two-tier design (superseded)
- `2026-02-05-use-cases-reference.md` — 25 use cases as parameter space

---

## 1. Principles

Five principles govern every decision in this plan:

1. **Ontology is foundational.** Always on, no toggle. A fresh install is "ontology cold" (minimal patterns), never "ontology off." Every extraction runs with ontology context.

2. **One pipeline.** Unified architecture with breakpoint execution, command pattern with undo, serializable state. Three current orchestrators (`MemoryIngestionFlow`, `FastIngestionFlow`, Studio pipeline) collapse into one.

3. **Parameters govern process.** Use cases are parameter combinations, not categories. `PipelineConfig` is the bridge between Studio and core. 25 use cases mapped to a 10-parameter space — no hardcoded use-case logic.

4. **Reasoning validates.** Self-learning uses reasoning (not just NELL-era statistics) to validate ontology decisions. Statistical pre-filter + LLM reasoning validation.

5. **Discovery over declaration.** The system infers its own configuration from usage patterns. Users don't pick a use case — the ontology discovers domain vocabulary, relation depth, and parameter tuning from data.

---

## 2. Architecture Overview

### 2.1 The Pipeline

```
                    PipelineConfig (per-workspace, tunable via Studio)
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│                Pipeline (linear, StageCommands)                   │
│                                                                  │
│  text ─► classify ─► coreference ─► simplify ─► entity_ruler    │
│              │            │             │             │           │
│              ▼            ▼             ▼             ▼           │
│         PipelineState flows through, accumulating at each stage  │
│              │            │             │             │           │
│              ▼            ▼             ▼             ▼           │
│  ─► llm_extract ─► ontology_constrain ─► store ─► link          │
│                                                      │           │
│                                          ─► enrich ─► evolve    │
│                                                                  │
│  Each stage: StageCommand.execute(state, config) → state        │
│              StageCommand.undo(state) → state                   │
└──────────────────────────────────────────────────────────────────┘
                          │
                          ▼
                ┌───────────────────┐
                │   Pipeline Runner  │
                │                   │
                │  breakpoints:     │  ← where to stop/resume (empty = run all)
                │    list[str]      │
                │                   │
                │  transport:       │  ← how state moves between stages
                │    in-process |   │     in-process = function call
                │    event-bus      │     event-bus = Redis Streams
                └───────────────────┘
```

### 2.2 Ontology Layers

```
┌──────────────────────────────────────────────┐
│        Global TBox (curated, seed)            │
│  14 seed entity types, type-pair priors,      │
│  mutual exclusion constraints.                │
│  Read-only for tenants. Service-level ops.    │
├──────────────────────────────────────────────┤
│     Tenant Soft-TBox (auto-promoted)          │
│  Learned types promoted via quality gate.     │
│  Feeds EntityRuler + LLM prompt schema.       │
│  Three-tier status: seed → provisional →      │
│  confirmed.                                   │
├──────────────────────────────────────────────┤
│       Tenant ABox (the actual graph)          │
│  Entities, relations, memories.               │
│  All data ontology-governed — every entity    │
│  links to a type in the ontology graph.       │
└──────────────────────────────────────────────┘
```

### 2.3 Storage Separation

Ontology and data live in **separate FalkorDB graphs** per workspace:
- `ws_{id}_ontology` — entity types, relation types, type-pair priors, patterns, constraints
- `ws_{id}_data` — actual entities, relations, memories

No label filtering needed, no bleed risk, no existing queries to audit. The pipeline config loader reads from the ontology graph; all other code reads from the data graph.

### 2.4 Dual-Axis Type System

Two orthogonal type systems, cleanly separated:

| Axis | Classifies | Examples | Governs |
|------|-----------|----------|---------|
| **Memory type** | The container | working, semantic, episodic, procedural, zettel, reasoning, opinion, observation, decision | Which pipeline stages and evolvers run |
| **Entity type** | The content | Person, Organization, Technology, Concept, Event, Decision, Claim | Which patterns and constraints apply |

First-class models (Decision, Opinion, ReasoningTrace, Observation) are **both** memory types and entity types. When the LLM extracts a Decision entity, it creates a full `Decision` model with evidence chain AND inserts it as a graph entity. This is the bridge between extraction and specialized behavior.

---

## 3. Data Model

### 3.1 Ontology Graph Schema (`ws_{id}_ontology`)

**Entity type nodes:**
```cypher
(:EntityType {
    name: "Technology",
    status: "seed",              -- seed | provisional | confirmed
    source: "system",            -- system | llm | user
    confidence: 0.95,
    created_at: timestamp,
    confirmed_at: timestamp | null,
    frequency: 0,                -- times seen in data
    examples: ["Python", "Kubernetes", "Docker"]
})
```

**Type-pair prior edges:**
```cypher
(:EntityType {name: "Person"})
    -[:LIKELY_RELATION {label: "founded", weight: 0.8}]->
(:EntityType {name: "Organization"})
```

**Mutual exclusion constraints:**
```cypher
(:EntityType {name: "Technology"})
    -[:MUTUALLY_EXCLUSIVE]->
(:EntityType {name: "Person"})
```

**Entity patterns (metadata on entity nodes):**
Entity patterns are properties on entity nodes — not separate documents:
```cypher
(:EntityPattern {
    name: "FastAPI",
    type: "Technology",
    pattern_source: "seed",      -- seed | llm_learned | user_added
    pattern_confidence: 0.95,
    is_global: true,
    frequency: 12
})
```

### 3.2 Data Graph Schema (`ws_{id}_data`)

Unchanged from current SmartMemory graph model. Entities, relations, memories with the existing node/edge structure. The only addition: every entity node includes an `ontology_type` property that references a type in the ontology graph.

### 3.3 Seed Entity Types (14)

**Generic (8, from existing `OntologyNode` classes):**
Person, Organization, Location, Concept, Event, Tool, Skill, Document

**SmartMemory-native (6, bridging first-class models):**
Decision, Claim, Action, Metric, Process, Project

The 100+ types in `entity_types.py` are the discoverable layer — domain-specific types that emerge from usage. The 14 seed types get EntityRuler patterns with 10-15 examples each. The 100+ list serves as vocabulary hints for the LLM prompt.

### 3.4 Three-Tier Type Status

| Status | Meaning | How created | Transition |
|--------|---------|-------------|------------|
| `seed` | Curated starting types | Shipped with system | — |
| `provisional` | Seen but unvalidated | Auto-created by `ontology_constrain` on first encounter of unknown type | → confirmed (via PromotionConfig) |
| `confirmed` | Validated through gates | Passes configurable quality criteria | — |

When extraction discovers an unknown type, `ontology_constrain` creates it as `provisional` in the ontology graph **before** storing data. Every entity in the data graph links to a type that exists in the ontology. No ungoverned data.

---

## 4. Pipeline Design

### 4.1 PipelineState

Serializable, accumulates as pipeline progresses, checkpointable.

```python
@dataclass
class PipelineState:
    # Input
    text: str
    raw_metadata: dict

    # After classify
    memory_type: str | None = None

    # After coreference
    resolved_text: str | None = None

    # After simplify
    simplified_clauses: list[str] = field(default_factory=list)

    # After entity_ruler
    ruler_entities: list[Entity] = field(default_factory=list)

    # After llm_extract
    llm_entities: list[Entity] = field(default_factory=list)
    llm_relations: list[Relation] = field(default_factory=list)

    # After ontology_constrain
    entities: list[Entity] = field(default_factory=list)         # merged + filtered
    relations: list[Relation] = field(default_factory=list)       # merged + filtered
    rejected: list[Entity] = field(default_factory=list)          # filtered out
    promotion_candidates: list[Entity] = field(default_factory=list)

    # After store
    item_id: str | None = None
    entity_ids: dict[str, str] = field(default_factory=dict)

    # After link
    links: list[Link] = field(default_factory=list)

    # After enrich
    enrichments: list[dict] = field(default_factory=list)

    # After evolve
    evolutions: list[dict] = field(default_factory=list)

    # Meta
    stage_history: list[str] = field(default_factory=list)
    stage_timings: dict[str, float] = field(default_factory=dict)
```

### 4.2 StageCommand Protocol

```python
class StageCommand(Protocol):
    name: str

    def execute(self, state: PipelineState, config: ComponentConfig) -> PipelineState:
        """Run stage, return new state."""

    def undo(self, state: PipelineState) -> PipelineState:
        """Revert to pre-execution state."""
```

Two undo modes:
- **Preview/tuning (Studio, grid search):** Stages compute but don't commit to storage. Undo = discard computed state. Trivial.
- **Production (committed to DB):** Stage wrote to FalkorDB/Redis. Undo = clean up what was written. Real rollback per stage.

### 4.3 PipelineConfig

Per-workspace, saved in FalkorDB, tunable via Studio.

```
PipelineConfig
├── name: str                      ("default", "high-precision", "bulk-import")
├── workspace_id: str
├── mode: str                      ("sync" | "async" | "preview")
├── retry: RetryConfig
│
├── parameters (cross-cutting)
│   ├── domain_vocabulary: str     (general | tech | medical | legal | ...)
│   ├── relation_depth: str        (shallow | medium | deep)
│   ├── temporal_sensitivity: str  (low | medium | high)
│   ├── contradiction_tolerance: str
│   └── confidence_requirement: str
│
├── ontology: OntologyConfig       (loaded from FalkorDB on init)
│   └── entity_types, relation_types, patterns, type-pair priors, constraints
│
├── classify: ClassifyConfig
│   └── model, fallback_type, confidence_threshold
│
├── coreference: CoreferenceConfig
│   └── enabled, model
│
├── simplify: SimplifyConfig
│   ├── clause_splitting: bool
│   ├── relative_clause_extraction: bool
│   ├── passive_to_active: bool
│   └── appositive_extraction: bool
│
├── extraction: ExtractionConfig           ← composite stage config
│   ├── entity_ruler: EntityRulerConfig
│   │   └── enabled, pattern_sources, min_confidence
│   ├── llm_extract: LLMExtractConfig
│   │   └── model, prompt (via PromptProvider), temperature, max_entities
│   ├── ontology_constrain: ConstrainConfig
│   │   ├── domain_range_validation: bool
│   │   └── promotion: PromotionConfig
│   │       ├── reasoning_validation: bool = True
│   │       ├── min_frequency: int = 1
│   │       ├── min_confidence: float = 0.8
│   │       └── human_review: bool = False
│   ├── enrichment_tier: str | None        ("groq" | "gemma" | None)
│   └── self_learning_enabled: bool
│
├── store: StoreConfig
│   └── embed_model, vector_dims, dedup_threshold
│
├── link: LinkConfig
│   └── similarity_threshold, max_links, cross_type_linking
│
├── enrich: EnrichConfig
│   ├── wikidata: WikidataConfig
│   ├── sentiment: SentimentConfig
│   ├── temporal: TemporalConfig
│   └── topic: TopicConfig
│
└── evolve: EvolveConfig
    └── enabled_evolvers, decay_rates, synthesis_threshold
```

**Config hierarchy:** Nested Pydantic/dataclasses. Each node is typed, has defaults, is serializable and validatable.

**Future abstraction note:** If Studio implementation reveals the need for generic config traversal (render any subtree as a form, diff from default, find all configs with a `model` field), add a config registry/visitor pattern. Nested dataclasses don't prevent this upgrade.

### 4.4 Pipeline Runner

One runner, two orthogonal config axes: breakpoints + transport.

```python
class PipelineRunner:
    breakpoints: list[str]
    transport: Transport           # InProcessTransport | EventBusTransport

    def run(self, text, config) -> PipelineState:
    def run_to(self, text, config, stop_after) -> PipelineState:
    def run_from(self, state, config, start_from, stop_after=None) -> PipelineState:
    def undo_to(self, state, target) -> PipelineState:
```

| Use Case | Breakpoints | Transport | Effect |
|----------|-------------|-----------|--------|
| Dev/CI | none | in-process | Full pipeline, function calls |
| Production sync | none | in-process | Full pipeline with retry |
| Production async | none | event-bus | Each stage is a Redis Stream consumer |
| Studio tuning | user-set | in-process | `run_to()`, `run_from()`, undo, iterate |
| Grid search | set + replay | in-process | `run_to()` once, `run_from()` N times |

Stages don't know which transport they're on. The runner handles retry, distribution, checkpointing.

### 4.5 Retry Policy

Per-stage in PipelineConfig:
```python
stage_config:
  llm_extractor:
    max_retries: 3
    retry_delay: exponential
    on_failure: fallback    # skip | retry | abort | fallback
    fallback_to: "entity_ruler"
```

Failed stage → undo partial work → retry (or fallback).

---

## 5. Pipeline Stages

### 5.1 classify

Classifies memory type from text. Uses existing classification model.

**Input:** `PipelineState.text`
**Output:** `PipelineState.memory_type`
**Config:** `ClassifyConfig` — model, fallback_type, confidence_threshold

### 5.2 coreference

Resolves pronouns and references in text. Uses existing coreference resolution.

**Input:** `PipelineState.text`
**Output:** `PipelineState.resolved_text`
**Config:** `CoreferenceConfig` — enabled (bool), model

### 5.3 simplify

**New stage.** Syntactic preprocessing to make sentences flatter and more amenable to machine parsing.

Four operations, each independently configurable, all sharing the same spaCy dep parse:

| Operation | What it does | Example |
|-----------|-------------|---------|
| **Clause splitting** | Split compound sentences at conjunctions | "X founded Y and Z acquired W" → two clauses |
| **Relative clause extraction** | Extract embedded clauses | "X, who founded Y, joined Z" → "X founded Y" + "X joined Z" |
| **Passive→active** | Convert passive voice | "Y was founded by X" → "X founded Y" |
| **Appositive extraction** | Extract appositives as facts | "X, the CEO of Y, ..." → "X is CEO of Y" |

**Input:** `PipelineState.resolved_text`
**Output:** `PipelineState.simplified_clauses`
**Config:** `SimplifyConfig` — four boolean flags

One stage (not four), because all operations share the same spaCy dep tree parse. No reason to parse four times.

### 5.4 entity_ruler

spaCy sm + EntityRuler. Pattern-matching NER at sub-millisecond cost.

**Input:** `PipelineState.resolved_text` (or simplified clauses)
**Output:** `PipelineState.ruler_entities`
**Config:** `EntityRulerConfig` — enabled, pattern_sources, min_confidence

**Benchmark result:** 96.9% entity F1 at 4ms, within 0.8% of Groq.

Key design: sm+ruler beats trf+ruler (96.9% > 94.6%) because the weaker NER model doesn't fight the ruler's precise patterns.

### 5.5 llm_extract

LLM-based extraction. Runs the optimized `SINGLE_CALL_PROMPT` (100% E-F1, 89.3% R-F1).

**Input:** `PipelineState.resolved_text`
**Output:** `PipelineState.llm_entities`, `PipelineState.llm_relations`
**Config:** `LLMExtractConfig` — model, prompt (resolved via PromptProvider), temperature, max_entities

In production sync mode, this stage may be skipped (entity_ruler provides fast results) with async enrichment queued. In preview/tuning mode, it runs synchronously.

### 5.6 ontology_constrain

Merges entity_ruler and llm_extract outputs, validates against ontology, creates provisional types for unknowns.

**Input:** `ruler_entities`, `llm_entities`, `llm_relations`
**Output:** `entities` (merged+filtered), `relations` (merged+filtered), `rejected`, `promotion_candidates`
**Config:** `ConstrainConfig` — promotion (PromotionConfig), domain_range_validation

Key behaviors:
1. **Merge** ruler + LLM entities (LLM takes priority on conflicts)
2. **Validate** entity types against ontology graph — reject types violating mutual exclusion
3. **Type-pair validation** — reject relations between entity type pairs with no prior
4. **Provisional creation** — unknown types create `provisional` entries in ontology graph before data stores
5. **Collect promotion candidates** — entities passing statistical pre-filter go into `PipelineState.promotion_candidates`

### 5.7 store

Stores memory item + entity nodes + relation edges in FalkorDB data graph.

**Input:** `entities`, `relations`, `memory_type`, text
**Output:** `item_id`, `entity_ids`
**Config:** `StoreConfig` — embed_model, vector_dims, dedup_threshold

### 5.8 link

Creates cross-references between the new memory and existing graph.

**Input:** `item_id`, `entity_ids`
**Output:** `links`
**Config:** `LinkConfig` — similarity_threshold, max_links, cross_type_linking

### 5.9 enrich

Enriches entities with external data. Includes Wikidata integration.

**Input:** `entity_ids`
**Output:** `enrichments`
**Config:** `EnrichConfig` — wikidata, sentiment, temporal, topic sub-configs

**Wikidata strategy:**
- REST API for 95% of lookups (single entity by name/QID)
- SPARQL for type hierarchy queries only (`P31/P279*` chains)
- Runs async — 100-200ms latency invisible to user
- **FalkorDB is canonical, Redis is write-through read cache.** Check FalkorDB first → if miss, call Wikidata API → write to both FalkorDB + Redis in same code path. Redis eviction is harmless.
- Scaling escape hatch: download Wikidata dump if we outgrow 200 req/s rate limit

### 5.10 evolve

Runs memory evolution (episodic decay, opinion reinforcement, etc.).

**Input:** `item_id`, current graph state
**Output:** `evolutions`
**Config:** `EvolveConfig` — enabled_evolvers, decay_rates, synthesis_threshold

---

## 6. Self-Learning System

### 6.1 Loop

```
Ingestion:
  text → Pipeline.run(text, config) → stored memory + entities

Background (event bus):
  promotion_candidates (from PipelineState)
    → statistical pre-filter (confidence > 0.8, frequency > 2)
    → reasoning validation (Phase 4+: LLM reasons about promotion)
    → quality gate passes → write pattern to Tenant Soft-TBox
    → EntityRuler reloaded with new pattern
    → next ingestion benefits

Feedback:
  extraction quality metrics (Insights) + user corrections
    → adjust PipelineConfig parameters
    → reasoning traces stored as memories (queryable)
```

### 6.2 PromotionConfig

Parameterized gates — Option D with C as default:

```python
@dataclass
class PromotionConfig:
    reasoning_validation: bool = True   # C behavior (default)
    min_frequency: int = 1              # 1 = confirm on first high-confidence
    min_confidence: float = 0.8
    human_review: bool = False          # HITL gate (deferred — no UI yet)
```

Configurable via Studio per workspace to behave like:
- **Option A** (immediate): `reasoning_validation=False, min_frequency=1, min_confidence=0.5`
- **Option B** (frequency): `reasoning_validation=False, min_frequency=3`
- **Option C** (reasoning, default): `reasoning_validation=True, min_frequency=1`
- **Strict**: `reasoning_validation=True, min_frequency=3, human_review=True`

EntityRuler pattern creation is a separate, higher bar than type confirmation.

### 6.3 Reasoning Integration Levels

| Level | When | What happens |
|-------|------|-------------|
| **Level 1** (Phase 1-3) | At launch | Statistical gates decide. Reasoning traces logged async for auditability. NELL baseline with observability. |
| **Level 2** (Phase 4+) | After real promotion data exists | Statistical pre-filter eliminates noise. Candidates that pass go through LLM reasoning validation. AssertionChallenger wired to ontology contradictions. |
| **Never** | — | Level 3 (reasoning as sole decision maker). Statistics as pre-filter is essential for cost control. |

Ontology reasoning traces stored as `memory_type="reasoning"` with `reasoning_domain="ontology"`. The system can explain its own knowledge structure: "Why did you classify FastAPI as Technology?"

### 6.4 Convergence

EntityRuler grows logarithmically:
- **Cold start** (0 memories): 14 seed types, ~50 base patterns. 96.9% E-F1.
- **Early learning** (100-1K memories): Rapid pattern growth.
- **Steady state** (1K-10K memories): New patterns slow. Most text has known entities.
- **Convergence** (10K+ memories): Ruler covers 99%+ of domain vocabulary.

### 6.5 Relation Strategy

**No RelationRuler.** Relations are semantic, not lexical. EntityRuler works because entity recognition is pattern matching. Relation extraction requires understanding meaning — that's LLM territory.

What we keep:
- **Type-pair priors** (filter impossible relations) — part of `ontology_constrain`, cheap, reduces FPs
- **Entity-pair cache** (reuse known LLM relations) — trivial, grows with self-learning
- **No dep-parse templates** — low ceiling (~65% → ~70%), fragile, not worth maintaining

Accept: entities are fast-tier territory (96.9%), relations are LLM territory (85-88%).

---

## 7. Prompt Management

### 7.1 Three-Layer Design

| Layer | Storage | Role |
|-------|---------|------|
| `prompts.json` | Source code (versioned) | Hardcoded defaults, ships with system |
| MongoDB | Per-tenant (Studio-managed) | Workspace overrides, user edits via Studio UI |
| PipelineConfig | Runtime (materialized) | What actually executes in the pipeline |

Resolution at pipeline start: `PipelineConfig.llm_extract.prompt` gets populated by resolving through PromptProvider (MongoDB override > `prompts.json` default). The pipeline only reads PipelineConfig at runtime.

### 7.2 Existing Infrastructure

Already built:
- `PromptProvider` abstraction with DI (`ConfigPromptProvider`, `MongoPromptProvider`)
- Three-tier override: user > workspace > config default
- Studio API: full CRUD (`/prompts/available`, `/prompts/config/{path}`, render, override, delete, hot-reload)

Not built:
- Studio UI (Redux store exists, React components incomplete)

Broken:
- `SINGLE_CALL_PROMPT` hardcoded in `llm_single.py` — violates prompt architecture
- `EXTRACTION_SYSTEM_PROMPT` hardcoded in `reasoning.py` — same violation

### 7.3 Implementation Tasks

1. Migrate hardcoded prompts into `prompts.json`
2. Wire PipelineConfig population to use PromptProvider resolution chain
3. Finish Studio prompt editing UI

---

## 8. Observability

### 8.1 Metrics Emission

Pipeline emits structured metrics via **Redis Streams** (event-driven, decoupled):
- Runner emits metric events to a Redis Stream metrics channel after each stage
- Lightweight metrics consumer aggregates into time-bucketed data (5min buckets)
- Pre-aggregated metrics written to Redis for fast dashboard reads
- Uses existing Redis Streams infrastructure (same as event-bus transport)

### 8.2 Metric Categories

**Pipeline metrics (from PipelineState):**
- Per-stage latency (classify: 2ms, entity_ruler: 4ms, llm_extract: 740ms, etc.)
- Throughput (memories/min, by stage)
- Error/retry rates per stage
- Stage bottleneck identification
- Queue depth per stage (event-bus mode)

**Ontology metrics (from ontology graph):**
- Type registry growth over time (provisional → confirmed)
- EntityRuler pattern count and coverage rate (ruler hits vs LLM-only discoveries)
- Convergence curve (how fast ruler approaches LLM quality)
- Promotion/rejection rates through quality gate
- Type status distribution (seed / provisional / confirmed counts)

**Extraction quality (from ontology_constrain output):**
- Entity count per memory, by type
- Relation count per memory
- Confidence distribution across extractions
- Provisional vs confirmed type ratio in new extractions
- LLM vs ruler attribution (which extractor found what)

---

## 9. Multi-Tenancy & Privacy

### 9.1 Scoping

All graph access through `SecureSmartMemory` — unchanged from current architecture. Both ontology graph and data graph scoped by `workspace_id`.

### 9.2 Pattern Layers

| Layer | Scope | Written by | Examples |
|-------|-------|-----------|----------|
| **Seed patterns** | Global (`is_global=True`) | System | 14 types, ~50 base patterns |
| **Learned global** | Global (promoted) | System (quality gate) | Patterns passing cross-tenant frequency |
| **Learned tenant** | Per-workspace | Self-learning loop | Domain-specific discoveries |

### 9.3 Privacy Rules

- **Entity vocabulary** (name → type) — shareable globally (vocabulary, no information content)
- **Type-pair priors** — shareable globally (pre-built from Wikidata stats)
- **Entity-pair relations from user text** — ALWAYS tenant-scoped (reveals proprietary processes)
- **Graph structure** (relation topology) — ALWAYS tenant-scoped

### 9.4 OSS vs Hosted

| Capability | OSS | Hosted |
|-----------|-----|--------|
| Fast tier (spaCy + rulers) | Always on, $0 | Always on |
| Seed patterns | Shipped with package | Shipped + pre-warmed |
| Self-learning | User provides LLM (own GPU / API key) | Managed |
| Wikidata | Public API (rate-limited) | Pre-cached, instant |
| Multi-tenant | Single user | Team workspaces |

OSS ships the EntityRuler mechanism (empty). No seed data in the core library. Hosted service has seeds as a service-level ops concern.

---

## 10. Existing Code: What Changes

### 10.1 What Gets Deleted (~1,100 LOC)

| Current | LOC | Why |
|---------|-----|-----|
| `FastIngestionFlow` | 502 | Unused; async mode is a config flag on the unified pipeline |
| `ExtractorPipeline` (Studio) | 491 | Studio calls core directly via `run_to()`/`run_from()` |
| Duplicated normalization/entity-ID/vector code | ~130 | Consolidated into unified pipeline |
| `ontology_enabled` toggle logic | scattered | Ontology is foundational — no toggle |

### 10.2 What Gets Absorbed

| Existing | Becomes |
|----------|---------|
| `MemoryIngestionFlow` (sync) | `Pipeline.run(text, config)` |
| `FastIngestionFlow` (async) | `Pipeline.run(text, config)` with `config.mode = "async"` |
| Studio Pipeline (preview) | `Pipeline.run_to()` / `Pipeline.run_from()` |
| Ontology CRUD routes | PipelineConfig management routes |
| Ontology inference endpoint | `llm_extract` + `ontology_constrain` stages |
| `OntologyManager` (198 LOC) | Splits: config loading → PipelineConfig loader; inference → pipeline stages |
| `OntologyStorage` (FileSystem) | Deprecated — ontology in FalkorDB graph |
| Studio `OntologyConfigSection` toggle | Removed. Params in stage configs. Dedicated ontology viewer for management. |
| 3 Studio integration endpoints | Pipeline breakpoint execution |

### 10.3 What Stays

| Existing | Role |
|----------|------|
| `SmartMemory.add()` | Recursion guard for internal stages (correct design) |
| `Ontology` model (225 LOC) | Schema portion of `PipelineConfig.ontology` |
| `OntologyIR` (278 LOC) | Import/export format (seeding, backup, cross-tenant sharing) |
| `OntologyLLMExtractor` (499 LOC) | Becomes `llm_extract` StageCommand |
| `ExtractionPipeline`, `StoragePipeline`, etc. | Become pipeline stage implementations |
| `IngestionRegistry` | Becomes component registry for unified pipeline |
| `IngestionObserver` | Emits events at stage transitions |
| Pattern CRUD endpoints | Kept for admin ops, manual corrections |

---

## 11. API Routes

### 11.1 New Routes

```
# PipelineConfig management
GET  /memory/pipeline/configs              → List named configs
POST /memory/pipeline/configs              → Save named config
GET  /memory/pipeline/configs/{name}       → Load config
PUT  /memory/pipeline/configs/{name}       → Update config

# Pattern admin
GET  /memory/ontology/patterns             → List patterns with stats
POST /memory/ontology/patterns             → Manual pattern add/override
DELETE /memory/ontology/patterns/{id}      → Remove pattern

# Ontology state
GET  /memory/ontology/status               → Convergence metrics, pattern counts, learning rate
POST /memory/ontology/import               → Import OntologyIR (seeding, migration)
GET  /memory/ontology/export               → Export current ontology as OntologyIR
```

### 11.2 Existing Routes

Unchanged. Ontology routes that existed for CRUD become PipelineConfig management routes. Pattern CRUD kept for admin ops.

---

## 12. UI Integration

### 12.1 Studio

**Ontology params:** Scattered into stage configs where they're used — EntityRuler confidence in entity_ruler config, promotion thresholds in ontology_constrain config, etc.

**Dedicated ontology viewer:** For management concerns:
- Type registry browser (seed/provisional/confirmed counts)
- Convergence dashboard (ruler pattern growth over time)
- Pattern browser (search, edit, delete patterns)
- Import/export
- Pruning tools
- Graph analytics (type distribution, relation statistics)

**Pipeline config panel:** MAJOR redesign deferred to Phase 7. The existing 1336-line `MemoryConfigurationPanel.jsx` is a reference for what configs exist, but the structure is completely new (one section per ~10 stages, breakpoint controls, etc.). Requires its own design session.

### 12.2 Insights

Consumes pre-aggregated metrics from Redis (pipeline metrics, ontology metrics, extraction quality — see Section 8.2). Building the dashboard is implementation timing, not architecture. The data will be there.

### 12.3 Web

Entity types shown in extraction results and graph visualization. Users don't see ontology config.

---

## 13. Implementation Phases

```
Phase 1 (Pipeline Foundation)
     │
     ▼
Phase 2 (Extraction Stages)
     │
     ▼
Phase 3 (Storage + Post-Processing Stages)
     │
     ├──────────────────┬──────────────────┐
     ▼                  ▼                  ▼
Phase 4             Phase 5            Phase 6
(Self-Learning)     (Service + API)    (Insights)
     │                  │
     ▼                  ▼
Phase 7 (Studio Pipeline UI — MAJOR, own design session)
     │
     ▼
Phase 8 (Hardening)
```

### Phase 1: Pipeline Foundation

**Goal:** Replace three orchestrators with one unified pipeline.

**Deliverables:**
- `StageCommand` protocol (`execute`/`undo`)
- `PipelineState` dataclass (serializable, all fields from Section 4.1)
- `PipelineConfig` hierarchy (nested Pydantic, all configs from Section 4.3)
- `PipelineRunner` with breakpoints + transport config axes
- `InProcessTransport` (function calls)
- Separate ontology FalkorDB graph (`ws_{id}_ontology`)
- Seed ontology loader (14 types, three-tier status)
- Migrate `MemoryIngestionFlow` to new pipeline (same behavior, new architecture)
- All existing tests pass against new pipeline

**Deletes:** Nothing yet — old orchestrators coexist until Phase 2 validates stages.

### Phase 2: Extraction Stages

**Goal:** Implement all extraction stages as StageCommands.

**Deliverables:**
- `ClassifyStageCommand`
- `CoreferenceStageCommand`
- `SimplifyStageCommand` (clause splitting, relative clause extraction, passive→active, appositive extraction — rules on spaCy dep tree)
- `EntityRulerStageCommand` (spaCy sm + EntityRuler, patterns from ontology graph)
- `LLMExtractStageCommand` (wraps `OntologyLLMExtractor`, uses PromptProvider)
- `OntologyConstrainStageCommand` (type validation, provisional creation, PromotionConfig)
- Migrate hardcoded prompts (`SINGLE_CALL_PROMPT`, `EXTRACTION_SYSTEM_PROMPT`) to `prompts.json`
- Wire PipelineConfig prompt resolution through PromptProvider

**Deletes:** `ExtractorPipeline` in Studio (491 LOC). Studio now calls core `run_to()`.

### Phase 3: Storage + Post-Processing Stages

**Goal:** Complete all remaining pipeline stages. Pipeline is fully functional.

**Deliverables:**
- `StoreStageCommand`
- `LinkStageCommand`
- `EnrichStageCommand` (Wikidata: REST + SPARQL, FalkorDB canonical + Redis write-through)
- `EvolveStageCommand`
- Pipeline metrics emission (Redis Streams — latency, counts, errors per stage)
- Delete old orchestrators (`MemoryIngestionFlow`, `FastIngestionFlow`)

**Deletes:** `FastIngestionFlow` (502 LOC), old `MemoryIngestionFlow` (473 LOC), ~130 LOC duplicated code.

### Phase 4: Self-Learning Loop (parallel with 5, 6)

**Goal:** Ontology grows automatically from LLM discoveries.

**Deliverables:**
- Promotion flow (provisional → confirmed via PromotionConfig gates)
- EntityRuler pattern growth from confirmed types
- Entity-pair cache (reuse known LLM relations for repeated entity pairs)
- Type-pair validation in `ontology_constrain`
- Level 2 reasoning validation (statistical pre-filter + LLM reasoning)
- AssertionChallenger wired to ontology type contradictions
- Ontology reasoning traces stored as `reasoning` memory type

### Phase 5: Service + API (parallel with 4, 6)

**Goal:** Expose pipeline and ontology management via REST API.

**Deliverables:**
- PipelineConfig management routes (save/load/update named configs)
- Pattern admin routes (list/add/delete)
- Ontology status/import/export routes
- `EventBusTransport` (Redis Streams between stages, per-stage consumer groups)
- Retry with undo (failed stage → undo → retry/fallback)
- Worker entrypoint (`python -m smartmemory.pipeline.worker --stage <name>`)

### Phase 6: Insights + Observability (parallel with 4, 5)

**Goal:** Real metrics replace stubs in Insights dashboard.

**Deliverables:**
- Pipeline metrics aggregation consumer (Redis Streams → pre-aggregated 5min buckets)
- Insights dashboard UI (pipeline metrics, ontology metrics, extraction quality)
- Convergence monitoring (ruler pattern growth curve)

### Phase 7: Studio Pipeline UI (MAJOR — requires own design session)

**Goal:** Full Studio UI for pipeline configuration and tuning.

**Deliverables:**
- PipelineConfig editor (per-stage hierarchical configs)
- Breakpoint execution UI (run-to, inspect state, modify params, resume)
- Ontology viewer (type registry, convergence, pattern browser, pruning, graph analytics)
- Prompt editing UI (complete existing API integration)
- Benchmarking workflow (batch execution + comparison)
- Grid search / parameter tuning

**Note:** This is a full UI redesign. The existing `MemoryConfigurationPanel.jsx` (1336 lines) is reference only. Design the new panel from PipelineConfig shape. Requires its own design session before implementation.

### Phase 8: Hardening

**Goal:** Production-ready quality.

**Deliverables:**
- Tests: pipeline, stages, self-learning, breakpoints, undo, serialization
- Benchmark suite integration in Studio
- Documentation (CLAUDE.md, README, CHANGELOG, architecture docs)
- HITL review UI (if `human_review: True` is needed)
- Synthetic test datasets for 3-5 representative parameter combinations

### Phase Dependencies

| Phase | Can Start After | Parallelizable With |
|-------|----------------|---------------------|
| 1: Pipeline Foundation | Immediately | — |
| 2: Extraction Stages | Phase 1 | — |
| 3: Storage + Post | Phase 2 | — |
| 4: Self-Learning | Phase 3 | 5, 6 |
| 5: Service + API | Phase 3 | 4, 6 |
| 6: Insights | Phase 3 | 4, 5 |
| 7: Studio UI | Phase 5 (needs API) | — |
| 8: Hardening | Phase 4 + 7 | — |

**Critical path:** 1 → 2 → 3 → 5 → 7 → 8

---

## 14. Benchmark Evidence

All architecture decisions backed by 53-config benchmark. Key results:

### Entity Extraction (NER)

| Configuration | Latency | E-F1 | Cost |
|--------------|---------|------|------|
| GPT-4o-mini | 4,300ms | 100% | ~$0.001 |
| Groq Llama-3.3-70b | 740ms | 97.7% | ~$0.0003 |
| **spaCy sm + EntityRuler** | **4ms** | **96.9%** | **$0** |
| spaCy trf + EntityRuler | 33ms | 94.6% | $0 |
| Gemma-3-27b-it (local) | 31s | 95.3% | $0 |
| spaCy sm (baseline) | 5ms | 86.0% | $0 |

### Relation Extraction (RE)

| Configuration | R-F1 | Notes |
|--------------|------|-------|
| GPT-4o-mini | 91.3% | Best overall, expensive |
| Groq Llama-3.3-70b | 85-88% | Best cost/quality |
| Gemma-3-27b-it | 86.7% | Beats Groq on relations, $0 |
| spaCy sm + EntityRuler | 65.1% | Dep-parse ceiling |

### Key Findings

- EntityRuler adds +10.9 E-F1 points at zero latency cost
- sm+ruler beats trf+ruler (96.9% > 94.6%) — weaker model doesn't fight ruler
- Progressive prompting (4 variants) all worse than LLM standalone — anchoring bias
- Bigger ≠ better for local: Gemma-27B > QWQ-56B > Hermes-70B
- Reasoning models (DeepSeek-R1, OLMo-think) don't help extraction

---

## 15. Dead Ends (Don't Revisit)

| Approach | Why it failed | Revisit only if |
|----------|--------------|-----------------|
| Progressive prompting | Anchoring bias, 4 variants all worse | Fundamentally different architecture |
| GLiNER + GLiREL | GLiNER NER at 34% recall | GLiNER v3+ |
| REBEL (end-to-end) | 3.9s, 62% R-F1 — dominated on all axes | Never |
| NuExtract tiny | 100% precision, 45% recall | NuExtract 2.0 |
| NuNER Zero | Broken with gliner >= 0.2.x | Upstream fix |
| spaCy trf for fast tier | 33ms vs 4ms, ruler makes sm better | Never |
| RelationRuler (dep-parse) | 65% → ~70% ceiling, fragile | Never |
| GPT-5-mini | 93.4% E-F1, 55.2% R-F1, over-extracts | Never |
| Bigger local models | Hermes-70B < Gemma-27B | Never |
| Reasoning models for extraction | Overhead, no quality gain | Never |
| Forking spaCy for RelationRuler | Maintenance burden | Contribute upstream |

---

## 16. Decision Log

| # | Decision | Rationale |
|---|----------|-----------|
| D1 | Ontology is foundational, always on | Self-learning is the differentiator. Optional = never enabled. |
| D2 | FalkorDB for all patterns (no JSONL, no MongoDB) | Patterns are graph entities; OSS ships empty; hosted seeds globally |
| D3 | One pipeline with breakpoint execution | Debugger metaphor: run to breakpoint, inspect, modify, resume |
| D4 | Separate FalkorDB graphs (ontology vs data) | Zero bleed, no queries to audit |
| D5 | All data ontology-governed | Unknown types create provisional entries before data stores |
| D6 | PromotionConfig (Option D, C default) | Parameterized gates encompass all strategies |
| D7 | No RelationRuler | Relations are semantic (LLM territory). Type-pair validation + entity-pair cache only. |
| D8 | Simplify stage (one stage, 4 flags) | Shared dep parse, each operation configurable |
| D9 | Redis Streams for metrics | Event-driven, decoupled, same infra as event-bus transport |
| D10 | FalkorDB canonical + Redis write-through for Wikidata | No sync concern — both updated in same write |
| D11 | Three-layer prompt management | prompts.json → MongoDB → PipelineConfig. PromptProvider feeds config. |
| D12 | ExtractionConfig is composite stage config | Groups sub-stages, each independently breakpointable |
| D13 | Level 2 reasoning (Phase 4+) | Statistical pre-filter + LLM reasoning. Level 1 (audit trail) at launch. |
| D14 | Studio panel redesign deferred | Pipeline architecture fundamentally changes the panel. Own design session. |
| D15 | First-class models are both memory types AND entity types | Bridge between extraction and specialized behavior |
| D16 | HITL deferred | `human_review: True` holds at provisional until review UI exists |
| D17 | Benchmarking is a Studio workflow | Same mechanism as tuning: breakpoint + grid search |
| D18 | Entity-pair relations NEVER shared globally | Process shapes are proprietary even when entities are public |
| D19 | Nested dataclasses now, config registry later | Don't abstract until Studio reveals need for generic traversal |
| D20 | Phase plan gated on sign-off | Review before implementation begins |
