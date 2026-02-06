# Ontology-Grounded Extraction: Strategic Implementation Plan v4

**Date:** 2026-02-05 (revised v4)
**Predecessor:** `2026-02-05-ontology-grounded-extraction-design.md`
**Status:** All questions resolved. Architecture defined. Ready for implementation planning.

---

## Overview

This plan implements the ontology-grounded extraction architecture across four repositories:
- **smart-memory** (core library)
- **smart-memory-service** (REST API)
- **smart-memory-insights** (observability dashboard)
- **smart-memory-studio** (pipeline lab UI)

**Key principles (v4):**
1. **Ontology is foundational** — always on, no toggle. "Ontology cold" (minimal patterns), not "ontology off."
2. **One pipeline** — unified architecture with breakpoint execution, command pattern with undo, serializable state.
3. **Parameters govern process** — use cases are parameter combinations. `PipelineConfig` is the bridge between Studio and core.
4. **Reasoning validates** — self-learning uses reasoning (not just statistics) to validate ontology decisions.
5. **Discovery over declaration** — system infers its own configuration from usage patterns.

---

## High-Level Architecture

### The Pipeline

```
                        PipelineConfig (per-workspace, tunable via Studio)
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                     Pipeline (linear, StageCommands)                      │
│                                                                          │
│  text ──► classify ──► coreference ──► simplify ──► entity_ruler ──► llm_extract │
│               │             │               │               │            │
│               ▼             ▼               ▼               ▼            │
│          PipelineState flows through, accumulating at each stage          │
│               │             │               │               │            │
│               ▼             ▼               ▼               ▼            │
│         ──► ontology_constrain ──► store ──► link ──► enrich ──► evolve │
│                                                                          │
│  Each stage: StageCommand.execute(state, config) -> state                │
│              StageCommand.undo(state) -> state                           │
└──────────────────────────────────────────────────────────────────────────┘
                              │
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
                    │    event-bus      │     event-bus = Redis Streams (serialized state, RPC)
                    └───────────────────┘
                              │
               ┌──────────────┼──────────────┐
               ▼              ▼              ▼
          no breakpoints  no breakpoints  breakpoints set
          in-process      event-bus       (either transport)
          (dev/CI/sync)   (prod async)    (Studio/tuning/grid search)
```

### Ontology Woven Into Pipeline

```
┌─────────────────────────────────────────────────────┐
│              Global TBox (FalkorDB)                  │
│  Seed entity types (14): Person, Org, Technology,    │
│  Concept, Event, Location, Document, Tool, Skill,    │
│  Decision, Claim, Action, Metric, Process, Project   │
│  + type-pair priors, mutual exclusion constraints     │
│  + 100+ discoverable types in vocabulary             │
│  Read-only for tenants. Service-level ops.           │
├─────────────────────────────────────────────────────┤
│           Tenant Soft-TBox (FalkorDB)                │
│  Learned patterns promoted via quality gate:          │
│    Phase 1-5: statistical pre-filter + audit trail    │
│    Phase 6+: reasoning validation (Level 2)           │
│  Feeds EntityRuler + LLM prompt schema               │
│  AssertionChallenger wired for type contradictions    │
├─────────────────────────────────────────────────────┤
│             Tenant ABox (FalkorDB)                   │
│  The actual graph. Entities, relations, memories.    │
│  Soft-TBox is a materialized view derived from this. │
└─────────────────────────────────────────────────────┘
         │                              ▲
         │ loads into PipelineConfig     │ self-learning writes back
         ▼                              │
    ┌──────────┐                   ┌──────────┐
    │entity_ruler│                 │quality_gate│
    │llm_extract │◄── ontology ──►│reasoning   │
    │constrain   │    context      │promotion   │
    └──────────┘                   └──────────┘
```

### PipelineConfig (the parameter set)

```python
@dataclass
class PipelineConfig:
    """Saved per workspace. Tuned via Studio. Loaded by Pipeline at runtime."""

    # Identity
    name: str                          # Named config (e.g., "default", "high-precision", "bulk-import")
    workspace_id: str                  # Tenant scoping

    # Ontology context (loaded from FalkorDB on init)
    ontology: OntologyConfig           # Entity types, relation types, patterns, type-pair priors, constraints

    # Per-stage configs
    classify: ClassifyConfig           # Classification model, thresholds
    coreference: CoreferenceConfig     # Enabled, model
    entity_ruler: EntityRulerConfig    # Confidence threshold, pattern sources
    llm_extract: LLMExtractConfig      # Model, prompt, temperature, max_tokens
    ontology_constrain: ConstrainConfig # Strictness, which types to enforce
    store: StoreConfig                 # Entity dedup strategy
    link: LinkConfig                   # Similarity threshold, strategies
    enrich: EnrichConfig               # Which enrichers, aggressiveness
    evolve: EvolveConfig               # Which evolvers, thresholds

    # Pipeline-level
    mode: str = "sync"                 # sync | async | preview
    retry: RetryConfig                 # Per-stage retry policy (max_retries, backoff, fallback)

    # Parameter space (discovered from usage, governs process)
    domain_vocabulary: str = "general"
    relation_depth: str = "shallow"
    temporal_sensitivity: str = "medium"
    contradiction_tolerance: str = "medium"
    confidence_requirement: str = "medium"
    scope: str = "personal"
```

### PipelineState (flows through stages)

```python
@dataclass
class PipelineState:
    """Serializable. Accumulates as pipeline progresses. Checkpointable."""

    # Input
    text: str
    raw_metadata: dict

    # After classify
    memory_type: str | None = None

    # After coreference
    resolved_text: str | None = None

    # After entity_ruler
    ruler_entities: list[Entity] = field(default_factory=list)

    # After llm_extract
    llm_entities: list[Entity] = field(default_factory=list)
    llm_relations: list[Relation] = field(default_factory=list)

    # After ontology_constrain
    entities: list[Entity] = field(default_factory=list)       # merged + filtered
    relations: list[Relation] = field(default_factory=list)     # merged + filtered
    rejected: list[Entity] = field(default_factory=list)        # filtered out (for inspection)
    promotion_candidates: list[Entity] = field(default_factory=list)  # for self-learning

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
    stage_history: list[str] = field(default_factory=list)     # stages completed
    stage_timings: dict[str, float] = field(default_factory=dict)  # latency per stage
```

### StageCommand Protocol

```python
class StageCommand(Protocol):
    name: str

    def execute(self, state: PipelineState, config: ComponentConfig) -> PipelineState:
        """Run stage, return new state. Side effects only in production mode."""

    def undo(self, state: PipelineState) -> PipelineState:
        """Revert to pre-execution state. Clean up any side effects."""
```

### Runner (one, configurable)

```python
class PipelineRunner:
    """One runner. Two orthogonal config axes: breakpoints + transport."""

    breakpoints: list[str]         # where to stop (empty = run all)
    transport: Transport           # in-process | event-bus (Redis Streams)

    def run(self, text, config) -> PipelineState:          # full run (no breakpoints)
    def run_to(self, text, config, stop_after) -> PipelineState:  # run to breakpoint
    def run_from(self, state, config, start_from, stop_after=None) -> PipelineState:  # resume
    def undo_to(self, state, target) -> PipelineState:     # rollback

class InProcessTransport(Transport):
    """Function calls. state = stage.execute(state, config). Direct return."""

class EventBusTransport(Transport):
    """Redis Streams. Serialize state → publish → consumer picks up → deserialize → execute.
    Same semantics as in-process, just RPC over message queue.
    Per-stage consumer groups for horizontal scaling."""
```

### Self-Learning Loop

```
Ingestion:
  text → Pipeline.run(text, config) → stored memory + entities

Background (event bus):
  promotion_candidates (from PipelineState)
    → statistical pre-filter (confidence > 0.8, frequency > 2)
    → reasoning validation (Phase 6+, LLM reasons about promotion)
    → quality gate passes → write pattern to Tenant Soft-TBox (FalkorDB)
    → EntityRuler reloaded with new pattern
    → next ingestion benefits from learned pattern

Feedback:
  extraction quality metrics (Insights) + user corrections
    → adjust PipelineConfig parameters
    → reasoning traces stored as memories (queryable: "why did you classify X?")
```

### Dual-Axis Entity Types

```
Memory Type axis (classifies the container):
  working | semantic | episodic | procedural | zettel | reasoning | opinion | observation | decision

Entity Type axis (classifies content, extractable + linkable):
  Generic: Person, Organization, Location, Concept, Event, Tool, Skill, Document
  SmartMemory-native: Decision, Claim, Action, Metric, Process, Project
  Discoverable: 100+ types in entity_types.py (emerge from usage)

Bridge: When LLM extracts a Decision entity → creates Decision model with evidence chain
        → available through decision API AND as graph entity referencing other memories
```

---

## Resolved Questions: Detailed Analysis

All 15 questions resolved. Each includes options analysis, the decision, and reasoning.

---

### Q0. What IS the Ontology? — RESOLVED

**Status:** Research complete. SmartMemory-specific questions resolved through parameterized use case analysis (25 use cases, 10-parameter space). See `2026-02-05-use-cases-reference.md`.

**Why this blocks everything:** Every other question references "ontology" — pattern storage (Q1), extraction paths (Q2), enrichment (Q3), existing infrastructure reuse (Q4), Studio UI (Q6), config (Q10). Without a concrete model, each question defines ontology implicitly and potentially inconsistently.

---

#### Research Finding: The Data Model Is Triples (Settled)

Every major ontology system (OWL/RDF, Wikidata, Schema.org, NELL) converges on the same answer: `(subject, predicate, object)` with metadata. FalkorDB already stores triples as nodes and edges. **No custom data model needed. The graph IS the ontology.**

The OWL TBox/ABox distinction maps directly:
- **TBox** (schema) = entity types, relation types, constraints → global ontology
- **ABox** (instances) = actual entities and relations from user's memories → tenant graph

---

#### Research Finding: Three Jobs, One Pipeline (Settled)

The ontology's job is not one thing — it's three phases of the extraction pipeline:

| Phase | When | What It Does | Evidence |
|-------|------|-------------|----------|
| **CLASSIFY** | During extraction | Type entities via EntityRuler patterns | GATE gazetteer pattern; our benchmark: +10.9 E-F1 |
| **PREDICT** | Before LLM extraction | Generate valid relation set for type pairs, structure LLM prompt | SPIRES/OntoGPT: grounding accuracy 1-3% → 97-100% with schema guidance |
| **CONSTRAIN** | After extraction | Validate triples against domain/range rules, reject invalid | Wikontic: +7.1 F1 from post-validation; 96.5% schema-consistent output |

**Different tiers use different phases:**

**Fast tier (4ms, spaCy + EntityRuler):**
```
Text → CLASSIFY (EntityRuler types entities)
     → CONSTRAIN (filter SVO triples against valid type-pair relations)
     → Output
```
Prediction skipped — dep-parse can't use it (finds what syntax reveals, nothing more).

**LLM enrichment tier (async):**
```
Text → CLASSIFY (reuse fast tier's entity types)
     → PREDICT (ontology generates valid relation set for type pairs)
     → EXTRACT (LLM uses predictions as output schema + "also find anything else")
     → CONSTRAIN (post-validate, flag novel relations for ontology review)
     → Output
```

**Critical design rule:** Predictions must be presented as a **vocabulary** (valid relation types), NOT a draft to confirm. The benchmark proved progressive prompting (draft → refine) fails due to anchoring bias. Schema-guided prompting (here are valid types, select what applies) succeeds.

---

#### Research Finding: Three-Layer Architecture for Multi-Tenant (Settled)

```
┌─────────────────────────────────────────────────┐
│           Global TBox (curated, seed-generated)  │
│  "Kubernetes is Technology" (universal)           │
│  Read-only for tenants. Quality-gated.           │
│  Generated from Wikidata/training corpus by ops. │
├─────────────────────────────────────────────────┤
│        Tenant Soft-TBox (auto-promoted)          │
│  Emergent from tenant's graph data               │
│  "FastAPI → Technology" (per-workspace)           │
│  Auto-promoted via confidence × frequency         │
│  Feeds EntityRuler + LLM prompt schema           │
├─────────────────────────────────────────────────┤
│            Tenant ABox (the actual graph)         │
│  Entities, relations, memories                    │
│  "(Sarah, works_at, Acme)" (instance facts)      │
│  No separate store — graph IS the data           │
└─────────────────────────────────────────────────┘
```

- **Global TBox**: Position A (curated). Shared. Updated by ops. Backed by Wikidata.
- **Tenant Soft-TBox**: Position B (automatic promotion). Per-workspace. Confidence × frequency × consistency gate. This IS the EntityRuler self-learning loop, generalized beyond entity patterns to type assertions and relation schemas.
- **Tenant ABox**: Position C (graph is truth). No separate store. Soft-TBox is a materialized view derived from it.

---

#### Research Finding: Minimum Viable Ontology (Settled)

NELL bootstrapped from **10-15 seed examples per category** + **mutual exclusion constraints**. You do NOT need relations, axioms, or full hierarchy on day one.

Minimum seed:
- A set of entity types (bounded, ~10-20 for SmartMemory's domain)
- A few examples per type (10-15)
- Mutual exclusion rules (a Technology is never a Person)

---

#### Research Finding: Self-Learning Works But Has Known Failure Modes (Settled)

From NELL (ran for years, 80M+ beliefs):
- **75% of categories** reached 90-99% precision autonomously
- **25% degraded** to 25-60% precision — researchers couldn't predict which ones
- **Error propagation** is #1 risk (one bad classification cascades)
- **Mutual exclusion constraints** are the strongest defense
- **Multi-source agreement** (2+ independent extractors) is the best quality gate
- **5-minute periodic human review** per category dramatically improved quality

Quality gate stack (ordered by effectiveness):
1. Multi-source agreement (2+ methods agree)
2. Confidence threshold (>0.75 minimum, >0.90 for promotion)
3. Mutual exclusion (prevents cascading errors)
4. Frequency minimum (seen at least N times)
5. Rate limiting (max K promotions per cycle)
6. Human review queue (for items scoring 0.75-0.90)

---

#### Research Finding: Key Evidence for Design Decisions

| System | Finding | Implication for SmartMemory |
|--------|---------|---------------------------|
| **GATE gazetteer** | Ontology vocabulary → lookup list → deterministic matching | Our EntityRuler IS this pattern. Already validated. |
| **SPIRES/OntoGPT** | Schema in LLM prompt → 97-100% grounding accuracy | LLM enrichment tier should use ontology as prompt schema |
| **Wikontic** | Post-extraction type validation → +7.1 F1, 96.5% consistent | Fast tier needs domain/range validation filter |
| **Text2KGBench** | Schema-constrained: 0.77 F1 vs unconstrained: 0.07 F1 | Pre-guidance has MORE impact than post-validation |
| **NELL** | 10-15 seeds + mutual exclusion → viable bootstrap | Minimal seed is small; mutual exclusion is essential |
| **AutoSchemaKG** | LLM-based type abstraction → 95% alignment with human schemas | LLM can validate promotion candidates |
| **Text2Onto** | Probabilistic Ontology Model — confidence on everything | Never commit a learned fact as absolute truth |

---

#### SmartMemory-Specific Questions — RESOLVED (via parameterized use case analysis)

Resolved by analyzing 25 use cases as parameter combinations (see `2026-02-05-use-cases-reference.md`).

**1. What entity types do we seed with?**

The codebase already has a rich type system — far more than just 7:

| Layer | Count | What's There |
|-------|-------|-------------|
| `EntityNodeType` enum | 7 | person, organization, location, concept, event, tool, skill |
| `OntologyNode` typed classes | 8 | Above + Document (each with typed fields, `to_memory_item()`) |
| `entity_types.py` domain list | 100+ | Financial, legal, healthcare, education, government, logistics, commerce, arts, security, etc. |
| `RelationType` enum | 30+ | Professional, spatial, temporal, causal, hierarchical, semantic, logical, knowledge, document, project |
| First-class models | 4 | Decision, ReasoningTrace, OpinionMetadata, ObservationMetadata |

**Seed entity types (14 minimum):**

*Generic (standard KG — already in EntityNodeType):*
Person, Organization, Location, Concept, Event, Tool, Skill, Document (8, from OntologyNode classes)

*SmartMemory-native (reflecting first-class models and capabilities):*
- **Decision** — has DecisionModel, manager, 11 API endpoints
- **Claim/Assertion** — things tracked by AssertionChallenger
- **Action/Task** — trackable items (in entity_types.py as `task`, `milestone`)
- **Metric/Measurement** — quantitative observations (in entity_types.py as `measurement`)
- **Process/Procedure** — has Procedure model, `procedural` memory type
- **Project** — common anchor across most use cases (in entity_types.py)

The 100+ types in `entity_types.py` are the DISCOVERABLE layer — domain-specific types that emerge from usage. The 14 seed types are what get EntityRuler patterns with 10-15 examples each. The 100+ list serves as a vocabulary hint for the LLM enrichment tier prompt.

**Open question:** Should the 4 first-class models (Decision, ReasoningTrace, Opinion, Observation) be entity types in the ontology, or are they a separate axis (memory types)? Currently they're memory types. Making them ALSO entity types means a Decision extracted from text gets both a memory type AND an entity type. See Q14 (reasoning integration).

**2. What are the actual promotion thresholds?**

Thresholds are a function of the `confidence_requirement` parameter. Default: confidence >0.8, frequency >2, multi-source agreement. At launch: single global default. Per-parameter tuning deferred until empirical data exists. Design the `QualityGate` class with `get_thresholds(namespace)` so overrides are easy to add.

**3. How does entity type ontology interact with memory type system?**

Orthogonal axes. Memory type (episodic, semantic, procedural...) classifies the MEMORY — governs which pipeline stages and evolvers run. Entity type (person, technology, concept...) classifies the CONTENT — governs which patterns and constraints apply. Both are parameters in the parameter space. No special interaction needed — they're independent concerns.

**4. Should users see the ontology?**

Governed by the user's context:
- **Maya users:** See entity types in graph visualization. Don't see ontology config.
- **Studio users:** Full ontology visibility (pattern browser, convergence stats, type hierarchy).
- **API consumers:** Ontology metadata in extraction results, optional field.
- **At launch:** Entity types shown in extraction results. Full management UI is Phase 4.

**5. What's the demotion strategy?**

At launch: **none**. Patterns persist. Confidence naturally decays on non-use (existing decay evolvers already handle this for memories; extend to patterns). Active demotion (patterns with 0 hits in N days → inactive) added later when we have data on domain shift rates across use cases.

**6. Where does the type-pair relation index live?**

Resolved in Q1: FalkorDB edges between `:EntityType` nodes (global). Tenant-specific additions come from graph statistics (query `MATCH (a)-[r]->(b) RETURN labels(a), type(r), labels(b), count(*) ORDER BY count DESC`). In-memory cache refreshed on pattern change.

---

#### Design Principle: Parameterized Ontology (from use case analysis)

**Use cases are parameter combinations, not categories.** 25 use cases were compiled (see reference doc) and analyzed. They're points in a parameter space — not discrete things to hard-code.

Key parameters that govern process:

| Parameter | Range | Governs |
|-----------|-------|---------|
| Domain vocabulary | tech/medical/legal/financial/academic/personal | Entity types, patterns |
| Entity focus | people/concepts/orgs/projects/events | EntityRuler priority |
| Relation depth | shallow (1-hop) → deep (multi-hop) | Graph traversal, extraction granularity |
| Temporal sensitivity | high → low | Temporal extractor activation |
| Contradiction tolerance | low (legal) → high (brainstorming) | Assertion challenger, confidence thresholds |
| Confidence requirement | high (medical) → low (casual) | Quality gates, promotion thresholds |
| Scope | personal → team → org | Cross-user promotion, TBox sharing |

**Discovery over declaration:** Users don't pick a use case. The system infers parameters from usage patterns:
- Emerging entity types (Case, Statute → legal domain)
- Query patterns (multi-hop → deep relation depth)
- Contradiction frequency (high → low tolerance needed)

**Implications for ontology:**
- Seed entity types are GENERAL (person, org, technology, concept, event, location, document) — not per-use-case
- Parameters configure the extraction pipeline at runtime
- The ontology discovers its own configuration as data flows through
- Quality gates, enrichment aggressiveness, and self-learning speed are all functions of parameters
- Parameters can be per-workspace (team setting) or per-user

#### Parameterization Decisions (resolved)

1. **At launch:** All parameters exist with sensible defaults (domain=general, entity_focus=balanced, relation_depth=shallow, temporal_sensitivity=medium, contradiction_tolerance=medium, confidence_requirement=medium, scope=personal, ingestion_pattern=stream). No parameter requires explicit user setting. Discovery kicks in after ~50-100 memories.

2. **Per-workspace** (matches existing tenant model via `SecureSmartMemory` scoping). Per-user overrides are Phase 7+.

3. **Both automatic AND visible in Studio.** Auto-inferred, displayed with inference confidence, manually overridable. Most users never touch it. This is the "advanced tuning" section of Studio.

4. **Feedback loop:** Extraction quality metrics (from Insights dashboard) + user entity/relation corrections → adjust relevant parameters. At launch: basic tracking (entity/relation count trends over time). Full adaptive feedback loop is Phase 6+.

5. **Testing dataset:** Not at launch. Phase 7 creates synthetic datasets for 3-5 representative parameter combinations (e.g., high-confidence medical, shallow-depth CRM, deep-investigation journalism) for regression testing.

---

#### What exists in code today (multiple partial models)

| Model | Location | What It Represents | Reusable? |
|-------|----------|--------------------|-----------|
| `OntologyIR` (278 lines) | `smartmemory/ontology/models.py` | Full IR: Concepts, Relations, Attributes, Taxonomy, Constraints | Partially — Concept model maps to entity types, Relation model maps to relation types. Taxonomy and Constraints are TBox. |
| `Ontology` (225 lines) | `smartmemory/ontology/ontology.py` | Simpler: entity types, relationship types, rules, validation | Yes — this IS the TBox, just needs connection to extraction |
| `ExtractionConfig.ontology_*` | `memory/pipeline/config.py` | Config flags (unused) | Replace — remove `ontology_enabled` toggle per Q11 |
| `OntologyExtractor` (499 lines) | `extraction/extractor.py` | LLM-based extractor using NodeType/RelationType registries | Yes — becomes the LLM enrichment tier |
| `OntologyManager` (198 lines) | `smartmemory/ontology/manager.py` | CRUD + one-way inference (extraction → ontology) | Extend — add ontology → extraction direction |
| EntityRuler patterns (benchmark) | `tests/benchmark_model_quality.py` | Flat `(pattern, label)` tuples | Move to production — these become the classification layer |

---

### Q1. Pattern Storage Backend — DECIDED

**Constraint from user:** Not Redis (volatile, not appropriate for persistent domain knowledge).

**Decision: FalkorDB for all patterns (entity + relation). No JSONL in repo. No MongoDB.**

#### Key Clarifications (from discussion)

**OSS vs Hosted distinction:**
- **OSS (core library):** Ships the EntityRuler *mechanism* (empty). No seed data. Self-learning builds patterns from scratch as memories are ingested.
- **Hosted service:** Has seed data as a service-level ops concern. Seeds are global, loaded once into FalkorDB, shared across all workspaces. Generated from a training corpus via Groq extraction + quality gate.

**Three pattern layers:**
1. **Seed patterns (global):** `is_global=True` in FalkorDB. Generated by ops team from training corpus. Loaded once. Shared by all workspaces. Written only by system.
2. **Learned global (promoted):** Started as tenant patterns, promoted via quality/frequency gate. Also `is_global=True`. System-written only.
3. **Learned tenant (per-workspace):** Scoped by `workspace_id` via `SecureSmartMemory`. Grown by self-learning loop per tenant.

**Pattern storage model — entity patterns:**
- Entity patterns are **metadata properties on entity nodes** that already exist in FalkorDB.
- A pattern is NOT a separate document. It's a `pattern_type`, `pattern_source`, `pattern_confidence` property set on an entity node.
- Example: entity node `(:Entity {name: "FastAPI", type: "technology", pattern_source: "seed", pattern_confidence: 0.95, is_global: true})`
- This avoids dual storage — the entity IS the pattern.

**Pattern storage model — relation patterns (four types):**

| Signal | Storage | Scope | Vector Embedded? | Count |
|--------|---------|-------|-------------------|-------|
| Type-pair priors | FalkorDB edges between `:EntityType` nodes | Global | No | ~50-200 |
| Dep-parse templates | FalkorDB `:RelationTemplate` nodes | Global (promotable) | No | ~100-500 |
| Entity-pair relation cache | Existing graph edges (already there) | Tenant | No (entities are) | N/A |
| Graph pattern inference | Not stored — Cypher queries at runtime | Tenant | N/A | N/A |

**Type-pair priors** are edges between EntityType nodes:
```
(:EntityType {name: "person"})-[:LIKELY_RELATION {label: "founded", weight: 0.8}]->(:EntityType {name: "organization"})
```

**Dep-parse templates** are learned pattern nodes:
```
(:RelationTemplate {pattern: "nsubj_dobj", verb_lemma: "found", source_type: "person",
                    target_type: "organization", relation: "founded", confidence: 0.9, is_global: true})
```

**Entity-pair relation cache** needs no new storage — the graph IS the cache. When the LLM extracts `Google -developed-> Kubernetes`, that edge is already stored. The RelationRuler queries existing edges to bias future extractions.

**No vector embeddings on any patterns or relation edges.** All pattern lookups are structural:
- Entity patterns: exact match on entity name → type lookup
- Type-pair priors: exact match on `(source_type, target_type)` → Cypher index
- Dep-parse templates: exact match on `(dep_pattern, verb_lemma)` → Cypher index
- Entity-pair cache: exact match on `(entity_a, entity_b)` → graph edge traversal

Embeddings serve the *search entry point* (finding relevant memories and entities via vector similarity). Relations are found by *graph traversal* from those entry points, not by vector search.

**Fuzzy verb matching** (e.g., "founded" vs "established") uses a verb synonym table (~200-500 lemmas), not embeddings. The long tail gets caught by the LLM enrichment tier.

#### Why not the other options?

- **JSONL in repo:** Seed data is service-level ops, not core library. OSS ships empty. No JSONL to commit.
- **MongoDB:** Only used by the service layer for auth metadata. Core library doesn't depend on MongoDB. Patterns belong in the graph with the entities they describe.
- **SQLite:** New dependency. Patterns are graph data — a relational DB adds conceptual mismatch.
- **JSONL on disk:** Single-node only. Doesn't scale to multi-worker deployment. No queryability.

#### Sub-answers
- **Q1a. Tenant scoping?** Same as all graph data — `SecureSmartMemory` scopes by `workspace_id`.
- **Q1b. Global patterns?** `is_global=True` property. Read by all tenants, written only by system (seeding or promotion gate).
- **Q1c. Pattern loading?** Eager on first request per tenant, cached in-process (spaCy EntityRuler). Hot-reload via Redis pub/sub on pattern change.
- **Q1d. Phase 0A impact?** Seeding is a service-level ops task, not a library task. Script runs Groq on training corpus → quality gate → load into FalkorDB as global entity nodes with pattern metadata.

---

### Q2. Ingestion Architecture & Extraction Unification — DECIDED

**Status:** Resolved. One pipeline with breakpoint-style execution. Studio calls core directly.

---

#### What exists (code exploration)

**Current orchestrators (~5,600 LOC total):**

| Component | Lines | What It Does |
|-----------|-------|-------------|
| `MemoryIngestionFlow` | 473 | Sync full pipeline (classify → extract → store → link → enrich → ground) |
| `FastIngestionFlow` | 502 | Async variant (fast path + Redis enqueue for background stages) |
| Studio Pipeline | ~250 | Separate per-stage preview/rollback in `smart-memory-studio/`, NOT in core |
| `SmartMemory.ingest()` | 222 | Router/dispatcher |
| `SmartMemory.add()` | 74 | Simple storage (recursion guard for internal stages — correct, stays) |

**Delegated modules (clean, no duplication within):**
- `ExtractionPipeline` (268 LOC) — extractor selection, fallback chain, entity conversion
- `StoragePipeline` (217 LOC) — vector store, graph storage, entity nodes
- `EnrichmentPipeline` (122 LOC) — enricher execution, derivative items
- `IngestionRegistry` (285 LOC) — lazy loading, fallback order
- `IngestionObserver` (331 LOC) — event emission, performance metrics

**Duplication between orchestrators:** ~130 LOC (normalization, entity ID mapping, vector store saving)

**Four extraction components (from earlier analysis):**
- `ExtractionPipeline` (ingestion) — 268 LOC
- `ExtractorPipeline` (Studio) — 491 LOC
- `FastIngestionFlow` — 502 LOC (unused)
- `OntologyExtractor` — 499 LOC (called by both)

~500 lines of identical code between the first two.

---

#### Decision: One Pipeline, Breakpoint Execution

**The debugger breakpoint metaphor IS the spec:**

- **Run to breakpoint** — pipeline executes up to a stage, state captured
- **Inspect state** — see what the pipeline produced so far
- **Modify parameters** — change config for the next stage(s)
- **Resume** — replay from breakpoint to the next breakpoint, or to the end
- **Repeat** — until tuned

Think of it like setting breakpoints in code and dynamically modifying variables, then resuming execution.

**Pipeline is linear** (for now — DAG is a future possibility, but the tuning mechanics would be the same).

**State flows through stages:**

```
text
  → classify         → state: {text, memory_type}
  → coreference      → state: {resolved_text, memory_type}
  → entity_ruler     → state: {+ ruler_entities}
  → llm_extractor    → state: {+ llm_entities}
  → ontology_constrain → state: {+ filtered_entities, rejected}
  → store            → state: {+ item_id, entity_ids}
  → link             → state: {+ links}
  → enrich           → state: {+ enrichments}
  → evolve           → state: {+ evolutions}
```

**Tuning is flexible — manual AND automated:**
- Tune a single stage (stop before it, modify params, run just that stage)
- Tune multiple stages (stop, modify, run the next N stages)
- Tune the rest of the pipeline (stop, modify, run to end)
- All are the same operation: resume from checkpoint with config

**Automated tuning (grid search, hyperparameter optimization):**
The same API supports both manual Studio interaction and automated parameter sweeps. `PipelineState` is serializable, `run_from()` is stateless (pure function: state + config → new state). Compute a checkpoint once, replay N times with different parameter combinations:

```python
# Compute checkpoint once
state = pipeline.run_to(text, config, stop_after="entity_ruler")

# Grid search over LLM extractor params
for params in param_grid:
    result = pipeline.run_from(state, params, start_from="llm_extractor")
    score = evaluate(result, gold_standard)

best_config = max(results, key=lambda x: x[1])
```

This is how benchmark_model_quality.py evolves: instead of running full pipelines per extractor, it runs to a checkpoint and sweeps parameters from there. Same evaluation framework, dramatically faster iteration.

**Implementation: Command pattern with undo (Option D).**

Rollback is required to iterate the same stage: run it, inspect results, tweak params, undo, run again. Like breakpoints with the ability to replay from the same point repeatedly.

```python
class StageCommand(Protocol):
    def execute(self, state: PipelineState, config: ComponentConfig) -> PipelineState:
        """Run the stage, return new state."""
    def undo(self, state: PipelineState) -> PipelineState:
        """Revert to pre-execution state."""
```

**Two undo modes:**
- **Preview/tuning (Studio, grid search):** Stages compute but don't commit to storage. Undo = discard computed state, checkpoint unchanged. Trivial. This is the common case.
- **Production (committed to DB):** Stage wrote to FalkorDB/Redis. Undo = clean up what was written. Real rollback logic per stage. Retry after undo for transient failures.

This generalizes Studio's existing ChangeSet model (preview → commit/rollback) to all pipeline stages. Studio's ChangeSet becomes one implementation of the StageCommand undo pattern.

**Retry policy (per-stage in PipelineConfig):**
```python
stage_config:
  llm_extractor:
    max_retries: 3
    retry_delay: exponential
    on_failure: fallback  # skip | retry | abort | fallback
    fallback_to: "entity_ruler"
```
Failed stage → undo partial work → retry (or fallback). LLM stages especially need this (network failures, rate limits, timeouts).

**Execution model is orthogonal to stage implementation.**

One runner, two configuration axes:

| "Mode" | Breakpoints | Transport | Effect |
|--------|-------------|-----------|--------|
| Dev/CI | none | in-process | Full pipeline, function calls |
| Production sync | none | in-process | Full pipeline with retry |
| Production async | none | event-bus | Each stage is a Redis Stream consumer, scales independently |
| Studio/tuning | user-set | in-process | `run_to()`, `run_from()`, undo, iterate |
| Grid search | set + replay | in-process | `run_to()` once, `run_from()` N times |

Stages don't know which transport they're on. They take state + config, return new state (or undo). The runner handles retry, distribution, checkpointing. Event bus is just a transport detail — serialized state over Redis Streams is a form of RPC. The stage semantics are identical.

**Horizontal scaling (event-bus transport):**
`PipelineState` is serializable. With event-bus transport, each stage is a consumer group on a Redis Stream:
```
classify workers → [stream] → extract workers → [stream] → store workers → ...
```
Stage N completes → serializes state → publishes. Stage N+1 worker picks up → runs → publishes. Bottleneck stages (LLM extraction) scale independently. This is what `FastIngestionFlow`'s async pattern was — generalized to all stages via transport config, not a separate orchestrator.

**Core provides:**
- `PipelineState` — serializable state object that accumulates as pipeline progresses
- `StageCommand` — protocol with `execute(state, config) -> state` and `undo(state) -> state`
- `PipelineRunner` — protocol for execution strategy (sequential, event-bus, breakpoint)
- `Pipeline.run(text, config)` — full run (production `ingest()`)
- `Pipeline.run_to(text, config, stop_after="entity_ruler")` — run to breakpoint, return state
- `Pipeline.run_from(state, config, start_from="llm_extractor")` — resume from checkpoint
- `Pipeline.run_from(state, config, start_from="llm_extractor", stop_after="ontology_constrain")` — run a range
- `Pipeline.undo_to(state, target="entity_ruler")` — rollback to a previous stage

**Studio calls core directly.** No separate Studio pipeline. No duplicated extraction code. Studio's UI maps to:
- "Preview extraction" → `Pipeline.run_to(text, config, stop_after="ontology_constrain")`
- "Tune EntityRuler" → `Pipeline.run_to(text, config, stop_after="entity_ruler")`, tweak params, `Pipeline.run_from(state, new_config, start_from="entity_ruler", stop_after="entity_ruler")`
- "See full pipeline with new params" → `Pipeline.run_from(state, new_config, start_from="entity_ruler")`

**Three current orchestrators collapse to one:**

| Current | Becomes |
|---------|---------|
| `MemoryIngestionFlow` (sync) | `Pipeline.run(text, config)` |
| `FastIngestionFlow` (async) | `Pipeline.run(text, config)` with `config.mode = "async"` (enqueue after store) |
| Studio Pipeline (preview) | `Pipeline.run_to()` / `Pipeline.run_from()` calls from Studio API |

**What gets deleted:**
- `FastIngestionFlow` (502 LOC) — unused, async mode is a config flag
- `ExtractorPipeline` in Studio (491 LOC) — Studio calls core directly
- ~130 LOC of duplicated normalization/entity-ID/vector code
- Studio's separate pipeline routes replaced by thin wrappers around core

**What stays:**
- `SmartMemory.add()` — recursion guard for internal stages (correct design)
- Delegated modules (ExtractionPipeline, StoragePipeline, etc.) — become pipeline components
- `IngestionRegistry` — becomes component registry for the unified pipeline
- `IngestionObserver` — emits events at each stage transition

---

#### Connection to Parameters

`PipelineConfig` IS the parameter set from the use cases reference doc. The 10-dimensional parameter space (domain vocabulary, relation depth, temporal sensitivity, contradiction tolerance, etc.) maps to config fields that govern how each pipeline component behaves.

Studio tunes `PipelineConfig`. Core runs with `PipelineConfig`. Named configs are saved per workspace. This is the bridge between Studio and core.

---

#### Connection to Ontology

`OntologyExtractionService` from the earlier Q2 analysis is NOT a separate service — it's just the extraction pipeline components (entity_ruler, llm_extractor, ontology_constrain) running with ontology-aware config. Ontology is foundational (Q11), so ALL extraction runs through these components. No separate ontology extraction path.

---

### Q3. Enrichment Worker Deployment Model — SUBSUMED BY Q2

**Status:** No longer a separate question. The enrichment worker is just the event-bus runner executing the enrichment `StageCommand`.

**Previous recommendation (Redis consumer group, flexible deployment)** is still correct — it's now the production async runner for ANY stage, not just enrichment.

**What exists and remains useful:**
- `RedisStreamQueue` with consumer groups, DLQ, namespacing — becomes the event-bus runner implementation
- `.for_enrich()` and `.for_ground()` class helpers — generalize to `.for_stage(stage_name)`
- docker-compose separate service containers — becomes the deployment model for stage workers

**Key details (from original analysis, still valid):**
- Workers need FalkorDB access (read text, update entities/relations, add patterns)
- Same Docker image as API, different entrypoint (`python -m smartmemory.pipeline.worker --stage enrichment`)
- Dev: in-process sequential runner. Production: event-bus runner with per-stage consumer groups
- Auto-scaling: not initially. Monitor queue depth per stage. Add workers where bottlenecked.
- DLQ handles stage failures that exceed retry policy

---

### Q4. Existing Ontology Infrastructure — DECIDED: Absorb Into Pipeline

**Status:** Resolved. Ontology infrastructure is absorbed into the unified pipeline architecture, not extended as a separate system.

**Context shift from Q2:** With the unified pipeline (StageCommands, PipelineConfig, breakpoint execution), ontology is no longer a separate system to extend. It's woven into the pipeline as:
1. **Input** — entity types, patterns, type-pair priors loaded into `PipelineConfig.ontology`
2. **Output** — extraction produces promotion candidates captured in `PipelineState`
3. **Persistent knowledge** — lives in FalkorDB (per Q1)

#### Key decisions (from Q4 walkthrough)

**Separate FalkorDB graphs** — ontology lives in a separate named graph (`ws_{id}_ontology`) from data (`ws_{id}_data`). No label filtering needed, no bleed risk, no existing queries to audit. The pipeline config loader reads from the ontology graph; all other code reads from the data graph.

**All data is ontology-governed** — when extraction discovers an unknown type, `ontology_constrain` creates it as `provisional` in the ontology graph BEFORE the data stores. Every entity in the data graph links to a type that exists in the ontology. No ungoverned data.

**Three-tier type status:**
| Status | Meaning | How it got there |
|---|---|---|
| `seed` | Curated starting types (14) | Shipped with system |
| `provisional` | Seen but not yet validated | Auto-created on first encounter by `ontology_constrain` |
| `confirmed` | Validated through promotion gates | Passes configurable quality criteria |

**Promotion config (Option D — parameterized gates, C as default):**
```python
@dataclass
class PromotionConfig:
    reasoning_validation: bool = True     # C behavior (default)
    min_frequency: int = 1                # 1 = confirm on first high-confidence extraction
    min_confidence: float = 0.8
    human_review: bool = False            # HITL gate (deferred — no review UI yet)
```
- Default (Option C): reasoning validates on first occurrence → confirmed
- Configurable via Studio per workspace to behave like A (immediate), B (frequency), or strict (B+C+HITL)
- EntityRuler pattern creation is a separate, higher bar than type confirmation

**HITL (human-in-the-loop)** — new paradigm, not yet built. `human_review: True` holds types at provisional until review UI exists. Deferred to later phase.

#### What gets absorbed

| Existing (776 LOC routes, ~1,200 LOC core) | Becomes |
|---------------------------------------------|---------|
| Ontology CRUD routes (create/load/update/delete) | `PipelineConfig` management routes (save/load named configs) |
| Ontology inference endpoint | Subsumed by `llm_extract` + `ontology_constrain` stages. "Dry run" = pipeline with breakpoint. |
| `OntologyManager` (198 LOC) | Splits: config loading → `PipelineConfig` loader (reads ontology graph); inference → pipeline stages |
| `OntologyStorage` (FileSystem impl) | Deprecated — ontology in separate FalkorDB graph (`ws_{id}_ontology`) per Q1 |
| Studio `OntologyConfigSection.jsx` toggle | Removed. Ontology params scattered into stage configs (entity_ruler, ontology_constrain, llm_extract). Separate dedicated ontology view for inspection/management (type registry, pattern browser, convergence, import/export, pruning, graph analytics). |
| 3 Studio integration endpoints | Replaced by pipeline breakpoint execution (Studio calls `Pipeline.run_to()`) |

#### What stays useful

| Existing | Role in New Architecture |
|----------|--------------------------|
| `Ontology` model (225 LOC) | Schema portion of `PipelineConfig.ontology` — entity types, relation types, rules, validation |
| `OntologyIR` (278 LOC) | Import/export format for ontology state (seeding, backup, cross-tenant sharing) |
| `OntologyLLMExtractor` (499 LOC) | Becomes the LLM extractor `StageCommand` |
| Pattern CRUD endpoints | Kept for direct admin ops, manual corrections (outside pipeline context) |
| `OntologyLab.jsx` (554 LOC) | Evolves into pipeline ontology viewer — reads from FalkorDB, shows patterns + convergence |

#### Remaining service routes (after absorption)

```
# PipelineConfig management (new, replaces ontology CRUD)
GET  /memory/pipeline/configs              → List named configs
POST /memory/pipeline/configs              → Save named config
GET  /memory/pipeline/configs/{name}       → Load config
PUT  /memory/pipeline/configs/{name}       → Update config

# Pattern admin (kept, subset of old ontology routes)
GET  /memory/ontology/patterns             → List patterns with stats
POST /memory/ontology/patterns             → Manual pattern add/override
DELETE /memory/ontology/patterns/{id}      → Remove pattern

# Ontology state (kept, restructured)
GET  /memory/ontology/status               → Convergence metrics, pattern counts, learning rate
POST /memory/ontology/import               → Import OntologyIR (seeding, migration)
GET  /memory/ontology/export               → Export current ontology as OntologyIR
```

---

### Q5. Insights Extraction Stats — Currently Stubs

**Context:** `getExtractionStats()` and `getExtractionOperations()` in `smartMemoryClient.js` return placeholder data. Redis event stream captures extraction events but no aggregation exists.

**Key question:** Implement real metrics first (before ontology UI), or build ontology metrics on top of stubs?

#### Option A: Implement real metrics first

Build aggregation pipeline from Redis event streams before adding ontology-specific metrics.

**Pros:** Foundation is solid before adding ontology layer. Avoids building on stubs. Events already captured — just needs aggregation.
**Cons:** Delays ontology work. Insights is the lowest-priority component (observation, not action).

#### Option B: Build ontology metrics on top of stubs

Add ontology-specific metrics alongside existing stubs. Replace all stubs together when ready.

**Pros:** Faster to start ontology work. Stubs can show mock data for demo/testing. Metrics can be implemented incrementally.
**Cons:** Dashboard shows fake data for some metrics and real data for others. Confusing UX.

#### Option C: Implement real metrics as part of ontology rollout (Phase 5)

Make real extraction metrics a parallel workstream during Phase 4-5. Not a prerequisite, not deferred — concurrent.

**Pros:** Metrics available when ontology features land. No blocking dependency. Natural parallel work (different codebase: React vs Python).
**Cons:** Two teams/streams needed for parallelism.

#### Decision: Pipeline emits metrics; Insights UI is implementation timing

The pipeline architecture naturally produces the data Insights needs via `PipelineState` (stage timings, entity counts, promotion events, confidence distributions). The architecture decision is:

1. **Pipeline emits structured metrics** — each stage run produces metric events (latency, counts, errors, ontology events)
2. **Plumbing: Option 1 — Redis Streams (event-driven)**
   - Runner emits metric events to a Redis Stream metrics channel after each stage
   - Lightweight metrics consumer aggregates into time-bucketed data (5min buckets, existing pattern)
   - Pre-aggregated metrics written to Redis for fast dashboard reads
   - Decoupled: pipeline doesn't wait for aggregation. Uses existing Redis Streams infrastructure (same as event-bus transport from Q2).
   - Not Option 2 (direct write — couples pipeline to metrics store, adds latency) or Option 3 (lazy aggregation — expensive on every dashboard load, doesn't scale)
3. **Insights UI consumes when built** — pre-aggregated data will be there. Building the dashboard is a prioritization/implementation concern, not an architecture blocker.

#### Insights UI metric categories (design reference)

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

### Q6. Studio MemoryConfigurationPanel.jsx — 61KB / 1336 Lines

**Context:** This component is the hub that merges ontology state into extraction config. Adding more ontology UI here makes it worse.

**Agent finding:** MemoryConfigurationPanel manages:
- Ontology state (lines 187-194): profile, enabled, constraints, LLM params
- Extraction state: pipeline configs with extraction sub-config
- The critical side-effect (lines 196-215): automatically merges ontology settings into `pipelineConfigs.extraction`
- 4 tab sections: extraction, enrichment, grounding, evolution

**If ontology is foundational:** The side-effect that conditionally merges ontology into extraction becomes the default path. No conditional — ontology is always part of extraction. This actually simplifies the code (remove the toggle logic).

#### Option A: Refactor before adding ontology features

Break MemoryConfigurationPanel into smaller components, then add ontology-integrated extraction UI.

**Pros:** Clean architecture from the start. Easier to maintain. Each component under 200 lines.
**Cons:** Delays ontology work by 2-3 days for a refactor that's orthogonal to the goal.

#### Option B: Add ontology features, refactor later

Add the minimal ontology integration, mark refactoring as tech debt.

**Pros:** Faster to ship. Ontology features not blocked by UI cleanup.
**Cons:** 61KB becomes 70KB+. Technical debt compounds. Harder to refactor later with more functionality embedded.

#### Option C: Refactor AS PART OF ontology integration

The refactoring IS the ontology integration. Move OntologyConfigSection into the extraction tab. Remove the toggle. Simplify the merge side-effect. Extract each tab section into its own file. This is the same work as "integrate ontology" — just done properly.

**Pros:** Refactoring serves the goal directly. Two birds, one stone. Result is smaller, cleaner components that naturally express "ontology is foundational."
**Cons:** Slightly more upfront thought about component boundaries.

#### Decision: **Rebuild around PipelineConfig — MAJOR DISCUSSION DEFERRED TO IMPLEMENTATION**

The pipeline architecture (Q2) fundamentally changes this panel. It's no longer "refactor 4 tabs + remove ontology toggle." The panel becomes a **PipelineConfig editor**:
- One section per stage (~10 stages, not 4 tabs)
- Each stage config editable independently
- Breakpoint controls (set where to stop, preview intermediate state)
- Promotion config lives under `ontology_constrain` stage
- Pipeline-level params (domain_vocabulary, relation_depth, etc.) at top

The existing 1336-line panel is a **reference for what configs exist**, but the structure is new. This is a full Studio UI redesign discussion that requires its own design session at implementation time.

**Do not attempt to refactor the existing panel. Design the new one from PipelineConfig shape.**

---

### Q7. Quality Gate Thresholds — Global or Per-Tenant?

**Context:** Self-learning quality gate filters before patterns are added to EntityRuler: confidence > 0.8, length > 3 chars, frequency > 1.

#### Option A: Global defaults only

Same thresholds for all tenants. Hardcoded or in config file.

**Pros:** Simple. Consistent behavior. One set of thresholds to tune and validate.
**Cons:** Some domains need tighter gates (medical: higher confidence), some need looser (casual notes: accept more entities). One size doesn't fit all at scale.

#### Option B: Per-tenant overrides from day one

Tenants can customize thresholds via API/UI.

**Pros:** Flexible. Domain-specific tuning. Power users happy.
**Cons:** More complexity. Thresholds UI needed. Most users won't touch defaults. Premature optimization.

#### Option C: Global defaults, per-tenant overrides added later

Start with global. Add per-tenant override capability in a later phase when we have real user feedback.

**Pros:** Ship fast with sane defaults. Add customization when users actually ask for it. YAGNI principle.
**Cons:** Must design the system to support overrides later (not hard — config with fallback to defaults).

#### Decision: **Subsumed by Q4 — PromotionConfig in PipelineConfig**

Quality gate thresholds are the `PromotionConfig` inside `PipelineConfig`, which is per-workspace and tunable via Studio:
```python
@dataclass
class PromotionConfig:
    reasoning_validation: bool = True
    min_frequency: int = 1
    min_confidence: float = 0.8
    human_review: bool = False
```
Global defaults baked in. Per-workspace overrides via PipelineConfig (saved in FalkorDB). Studio exposes them under `ontology_constrain` stage config. No separate quality gate abstraction needed.

---

### Q8. Wikidata API Strategy

**Context:** Need to link extracted entities to world knowledge. Wikidata is the chosen source (structured triples vs Wikipedia's prose).

#### Option A: REST API with Redis caching

Use `wikidata.org/w/rest.php/wikibase/v1/` for entity lookups. Cache results in Redis with 7-day TTL.

| Aspect | Assessment |
|--------|-----------|
| **Rate limits** | 200 req/s for bots. Generous for our scale. |
| **Latency** | ~100-200ms per entity lookup. Hidden behind async. |
| **Coverage** | Full Wikidata — 100M+ entities. |
| **Caching** | Redis with 7-day TTL. Entities rarely change. |
| **Cost** | Free. |

**Pros:** Simple, full coverage, free, generous rate limits.
**Cons:** Rate limits could bite during bulk import. 100ms per entity adds up.

#### Option B: SPARQL for complex queries

Use `query.wikidata.org/sparql` for type hierarchy and multi-hop queries.

| Aspect | Assessment |
|--------|-----------|
| **Rate limits** | 60s timeout per query. Stricter than REST. |
| **Power** | Can query type hierarchies, property statistics, related entities in one call. |
| **Complexity** | SPARQL is hard to write and debug. |
| **Use case** | Type hierarchy (P31 instance_of + P279 subclass_of chains). Batch entity lookups. |

**Pros:** Powerful for type hierarchies and batch queries.
**Cons:** Complex, slower for simple lookups, 60s timeout.

#### Option C: Local Wikidata dump

Download and index a subset of Wikidata locally.

| Aspect | Assessment |
|--------|-----------|
| **Size** | Full dump: ~100GB compressed, ~1TB decompressed. |
| **Subset** | Entity types + basic properties: ~5GB. |
| **Latency** | <1ms (local). |
| **Freshness** | Monthly dumps available. |
| **Complexity** | ETL pipeline, storage, indexing, update mechanism. |

**Pros:** No rate limits, instant lookups.
**Cons:** Massive setup effort. Storage overhead. Not justified at our scale.

#### Decision: **Option A (REST) + Option B (SPARQL for type hierarchies only)**

1. REST API for 95% of lookups: single entity by name/QID. Simple, fast enough behind async enrichment stage.
2. SPARQL for type hierarchy queries only: `P31/P279*` chains to get "Kubernetes → software → technology → abstract entity".
3. Runs in `enrich` stage (async/event-bus) — 100-200ms latency invisible to user.
4. **Scaling escape hatch:** If we outgrow rate limits (200 req/s), download Wikidata dump and process locally. Not justified at current scale.

**Storage: FalkorDB is canonical, Redis is write-through read cache.**
- `enrich` stage checks FalkorDB first: entity already enriched?
- If no → call Wikidata API → write to FalkorDB (canonical) + Redis (cache) in same code path
- If yes → done (Redis serves hot reads, FalkorDB is fallback)
- No separate sync concern — both updated in same write operation
- Redis eviction/miss is harmless — FalkorDB has the data

---

### Q9. Which Prompt for Seeding?

**Context:** The seeding script generates 5K-10K base EntityRuler patterns by running LLM extraction on a diverse corpus.

**Agent finding:** The optimized `SINGLE_CALL_PROMPT` is already in `llm_single.py`. It achieved 100% E-F1 on both Groq and Gemma. The optimization is model-agnostic.

#### Option A: Standard Groq prompt (unoptimized)

Use the original Groq extraction prompt without the entity splitting and implicit relation improvements.

**Pros:** None. The optimized prompt is strictly better.
**Cons:** Misses compound entities. Lower quality seed data.

#### Option B: Optimized SINGLE_CALL_PROMPT (current default)

Use the prompt that achieved 100% E-F1 on both Groq and Gemma.

**Pros:** Best quality extraction for seeding. Model-agnostic. Already the default in production code.
**Cons:** None.

#### Decision: **Reframed — prompt management, not prompt selection**

The optimized `SINGLE_CALL_PROMPT` is the right default. But the real question is prompt management in the pipeline architecture.

**What exists today:**
- `prompts.json` — single source of truth for all prompts (dot-notation paths)
- `PromptProvider` abstraction with dependency injection (`ConfigPromptProvider` file-based, `MongoPromptProvider` Studio/MongoDB)
- Three-tier override resolution: user override > workspace override > config default
- Studio API: full CRUD endpoints (`/prompts/available`, `/prompts/config/{path}`, render, override, delete, hot-reload)
- Studio UI: **incomplete** — Redux store exists, React components not built
- Architecture doc: `docs/PROMPT_ARCHITECTURE.md`

**What's broken:**
- `SINGLE_CALL_PROMPT` hardcoded in `llm_single.py` — violates `PROMPT_ARCHITECTURE.md`
- `EXTRACTION_SYSTEM_PROMPT` hardcoded in `reasoning.py` — same violation
- Studio prompt editing UI not finished

**Prompt management design (confirmed):**

Three layers, each with a clear role:
| Layer | Storage | Role |
|---|---|---|
| `prompts.json` | Source code (versioned) | Hardcoded defaults, ships with system |
| MongoDB | Per-tenant (Studio-managed) | Workspace overrides, user edits via Studio UI |
| PipelineConfig | Runtime (materialized) | What actually executes in the pipeline |

Resolution: PipelineConfig gets populated at pipeline start by resolving through PromptProvider (MongoDB override > `prompts.json` default). The pipeline only reads PipelineConfig at runtime — never calls PromptProvider directly. PromptProvider is the resolution mechanism that feeds PipelineConfig, not a separate runtime system.

**Implementation tasks:**
- Migrate hardcoded prompts (`SINGLE_CALL_PROMPT` in llm_single.py, `EXTRACTION_SYSTEM_PROMPT` in reasoning.py) into `prompts.json`
- Finish Studio prompt editing UI (API endpoints exist, React components incomplete)
- Wire PipelineConfig population to use PromptProvider resolution chain

---

### Q10. ExtractionConfig — Extend or New Config?

**Context:** `ExtractionConfig` currently has:
- `extractor_name`, `ontology_extraction` (bool), `enable_fallback_chain`, `legacy_extractor_support`, `max_extraction_attempts`
- `ontology_profile`, `ontology_enabled`, `ontology_constraints` — already exist but unused

**Missing for our architecture:**
- `enrichment_tier: Optional[str]` — "groq" | "gemma" | "custom" | None
- `self_learning_enabled: bool`
- `pattern_namespace: Optional[str]` — for tenant scoping
- `quality_gate_config: Optional[QualityGateConfig]`

#### Option A: Extend existing ExtractionConfig

Add new fields directly to ExtractionConfig.

**Pros:** Single config object. Simple. All extraction config in one place.
**Cons:** Config grows larger. Users who don't use ontology still see ontology fields (but if ontology is foundational, everyone uses it).

#### Option B: New OntologyExtractionConfig wrapping ExtractionConfig

Create a wrapper config that inherits from or contains ExtractionConfig.

**Pros:** Separation of concerns. Base config stays lean.
**Cons:** Wrapper adds indirection. If ontology is foundational, the separation is artificial.

#### Option C: Remove the split — ExtractionConfig IS ontology-aware

If ontology is foundational, there's no "extraction without ontology." Remove `ontology_enabled` flag. Remove `ontology_extraction` boolean. Extraction always includes ontology context. Add the missing fields directly.

**Pros:** Reflects the architectural truth — ontology is always on. Simplifies config (fewer booleans). Clean.
**Cons:** Breaking change for any code that checks `ontology_enabled` (but that code is unused/stubbed anyway).

#### Decision: **ExtractionConfig is a composite stage config within PipelineConfig**

ExtractionConfig groups the extraction sub-stages. Each sub-stage is its own StageCommand with execute/undo, independently breakpointable.

```
PipelineConfig
├── name, workspace_id, mode, retry
├── parameters (cross-cutting)
│   └── domain_vocabulary, relation_depth, temporal_sensitivity,
│       contradiction_tolerance, confidence_requirement
├── classify: ClassifyConfig
│   └── model, fallback_type, confidence_threshold
├── coreference: CoreferenceConfig
│   └── enabled, model
├── simplify: SimplifyConfig
│   └── clause_splitting, relative_clause_extraction, passive_to_active, appositive_extraction (all bool)
├── extraction: ExtractionConfig                    ← composite stage config
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
│   ├── wikidata: WikidataConfig, sentiment: SentimentConfig
│   └── temporal: TemporalConfig, topic: TopicConfig
└── evolve: EvolveConfig
    └── enabled_evolvers, decay_rates, synthesis_threshold
```

Old `ontology_enabled`, `ontology_extraction` booleans removed — ontology is foundational.

**Config hierarchy:** Nested Pydantic/dataclasses for now. Each node is a typed config with defaults, serializable, validatable.

**Future abstraction note:** If Studio implementation reveals the need for generic config traversal (render any subtree as a form, diff any config from default, find all configs with a `model` field), add a config registry/visitor pattern then. Nested dataclasses don't prevent this upgrade. Trigger: when hand-coding Studio UI per config type becomes repetitive or when we need config inheritance/composition across workspaces.

---

### Q11. Should Ontology Be Foundational or Optional? [NEW]

**Context from user:** "If ontology is a given it would be integrated from the start yes or no? Not just extractors working in isolation but always with respect to ontology."

**Context from design doc (Section 1.1):** "The ontology is core architecture, like the graph or the embedding store. It exists in every deployment."

This question was implicit in the design doc but not explicit in the implementation plan. The v2 plan treated ontology as add-on (Studio UI in Phase 4, extraction without ontology as default path). This contradicts the design doc's core principle.

#### Option A: Foundational (always on)

Every extraction happens with ontology context. No toggle. Fresh install is "ontology cold" (minimal patterns), not "ontology off."

**Implications:**
- Remove `ontology_enabled` toggle from config and Studio
- Extraction results always include ontology type information
- Studio extraction tab always shows entity types from ontology hierarchy
- ExtractionConfig loses ontology booleans (Q10 → Option C)
- Phase ordering changes: ontology context is Phase 1, not Phase 4 (Q13)

**Pros:** Consistent with design doc. Self-learning starts from day one. Every memory contributes to improving extraction. Users see unified behavior.
**Cons:** Can't disable ontology for debugging. Slight cold-start overhead (loading empty patterns is nearly free, but still a code path).

#### Option B: Optional (current plan)

Ontology is an enhancement that can be toggled on/off. Default: on.

**Implications:**
- Keep `ontology_enabled` toggle
- Two code paths: with and without ontology
- Studio has toggle in OntologyConfigSection
- Testing doubles: with and without ontology

**Pros:** Flexibility. Backward compatible. Can ship extraction improvements without ontology dependency.
**Cons:** Two code paths means double testing. "Optional" features get less adoption. If most users leave it on default (on), the toggle adds complexity for no benefit. Contradicts design doc principle.

#### Decision: **Option A (foundational) — already decided, first principle of v4**

This is the foundational premise of the entire plan. Every decision in Q2-Q10 assumes ontology is always on. No toggle. "Ontology cold" not "ontology off."

---

### Q12. Benchmark Non-LLM Extractors WITH Ontology Constraints [NEW]

**Context from user:** "That also begs benchmarking non-LLM extractors with ontologies."

**Current state:** The 53-config benchmark tested extractors in isolation. EntityRuler adds domain vocabulary but not ontology constraints. We never tested:
- EntityRuler + ontology type constraints (reject entities whose type doesn't match ontology hierarchy)
- EntityRuler + RelationRuler type-pair priors (does constraining to valid type pairs improve R-F1?)
- spaCy dep-parse + ontology-informed relation filtering

**Why this matters:** The benchmarked 65.1% R-F1 for spaCy+ruler is without any ontology constraints. The RelationRuler design (Section 6 of design doc) proposes type-pair priors + dep-parse templates + entity-pair cache. These are ontology-grounded signals. If they push R-F1 from 65% toward 80%, the gap between fast tier and LLM enrichment tier narrows significantly, potentially changing the architecture.

#### Option A: Benchmark before implementing

Create a benchmark that adds ontology constraints (type-pair priors, type validation) to the EntityRuler extractor and measures the quality improvement.

**What to test:**
1. EntityRuler + type-pair priors (static lookup table): Does constraining to valid relation types for entity type pairs reduce false positives?
2. EntityRuler + dep-parse template matching: Does the RelationRuler multi-signal scorer improve R-F1?
3. EntityRuler + entity-pair cache: Does memorizing known entity pairs help on repeated mentions?
4. Full stack: EntityRuler + all RelationRuler signals vs LLM

**Effort:** 2-3 days (extend existing benchmark framework).

**Pros:** Data-driven architecture decision. If R-F1 jumps from 65% to 80%+, the enrichment tier becomes less critical. If it stays at 65%, confirms LLM enrichment is essential.
**Cons:** Delays implementation by 2-3 days. The benchmark gold set only has 16 test cases — may not be statistically significant for relation quality.

#### Option B: Skip benchmark, implement as designed

Trust the architecture design. EntityRuler + RelationRuler + enrichment tier. Measure quality after implementation with real user data.

**Pros:** Faster to ship. Real-world data is better than synthetic benchmarks anyway.
**Cons:** Building RelationRuler without knowing if it actually improves quality. Could waste effort on a component that doesn't help.

#### Option C: Benchmark in parallel with Phase 1

Start implementing EntityRuler (Phase 1A) while simultaneously benchmarking ontology-constrained extraction. Use benchmark results to decide whether to invest in RelationRuler (Phase 4 of design doc) or skip it.

**Pros:** No delay. Benchmark informs later phases. Phase 1 (EntityRuler core) is validated already — benchmark only affects RelationRuler decisions.
**Cons:** Benchmark might invalidate some Phase 1 design decisions (unlikely — EntityRuler is already validated).

#### Decision: **No RelationRuler. Type-pair validation + entity-pair cache only. Benchmarking is a Studio concern.**

**RelationRuler analysis:** Relations are semantic, not lexical. EntityRuler works because entity recognition is pattern matching ("Python" → Technology). Relation extraction requires understanding meaning ("joined" → WORKS_AT). That's why spaCy R-F1 is 65% while E-F1 is 97%.

| Signal | Verdict | Reasoning |
|---|---|---|
| Type-pair priors (filter impossible relations) | **Keep** — part of `ontology_constrain` | Cheap, reduces false positives, already in the stage |
| Entity-pair cache (reuse known LLM relations) | **Keep** — trivial to build | Caches LLM results for repeated entity pairs, grows with self-learning |
| Dep-parse templates | **Skip** | Low ceiling, fragile, high maintenance. The 65% → ~70% gain isn't worth a dedicated component |

Accept that relations are LLM territory. Fast tier handles entities (96.9%). Enrichment tier handles relations (85-88%). Entity-pair cache helps the fast tier reuse known relations over time.

**Benchmarking:**
- Orthogonal to implementation, can run in parallel
- Swappability built into PipelineConfig (swap extractors/configs without code changes)
- **Benchmarking IS a Studio workflow** — same mechanism as tuning: breakpoint execution + grid search, automated over multiple inputs with comparison UI
- Initial benchmarks use `benchmark_model_quality.py`. Future benchmarking lives in Studio.

---

### Q13. Phase Ordering If Ontology Is Foundational [NEW]

**Context from user:** "If ontology is foundational, does the phase ordering change? Currently ontology UI is Phase 4 (after core + queue). Should it be Phase 1 alongside core?"

**Current phase ordering:**
```
Phase 0A (seeding) → Phase 1 (EntityRuler) → Phase 2 (async queue) → Phase 3 (Wikidata)
                                                                    → Phase 4 (Studio/Service UI)
                                                                    → Phase 5 (Insights)
                     Phase 6 (self-learning) → Phase 7 (tests/docs)
```

**Problem:** If ontology is foundational, Studio should show ontology-grounded extraction from Phase 1. Users tuning extraction in Studio should always see results relative to ontology. Waiting until Phase 4 for Studio integration means Phases 1-3 are built and tested without Studio feedback.

#### Option A: Studio in Phase 1 (ontology from the start)

Phase 1 includes both EntityRuler core AND Studio integration. Studio's extraction tab shows ontology-grounded results from day one.

**Pros:** Foundational principle is real from day one. Studio users immediately see ontology context. Tests and development happen with the full loop: core ↔ Studio ↔ API.
**Cons:** Phase 1 becomes much larger (3-4 days → 6-8 days). Core + UI + API all in one phase. Risk of scope creep.

#### Option B: Keep current ordering, but make Studio ontology-aware in Phase 1

Add a minimal Studio change in Phase 1: remove the ontology toggle, show entity types from EntityRuler in extraction results. Full Studio UI features (pattern management, convergence dashboard) stay in Phase 4.

**Pros:** Phase 1 stays focused on core. Minimal Studio change (remove toggle, show types) takes hours, not days. Foundational principle is visible from day one without scope creep.
**Cons:** Full Studio ontology features are still deferred. But this is fine — the principle is established, features are incremental.

#### Option C: Keep current ordering exactly

Phase 4 is when Studio gets ontology features. Phase 1-3 are backend-only.

**Pros:** Clear separation. Backend first, UI second. Common engineering pattern.
**Cons:** Violates the foundational principle. Studio users tune extraction without ontology context for 3 phases. Discovered integration issues late.

#### Decision: **Phase plan rewrite required — v5 below. GATE: review and sign off before implementation begins.**

---

## Revised Phase Plan (v5)

Complete rewrite based on unified pipeline architecture (Q2), absorption (Q4), and all decisions Q1-Q14.

**IMPLEMENTATION GATE: This phase plan must be reviewed and signed off before implementation begins.**

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
Phase 8 (Hardening: tests, benchmarks, docs, HITL)
```

### Phase 1: Pipeline Foundation
- StageCommand protocol (execute/undo) + PipelineState + PipelineConfig (nested dataclasses)
- One runner (breakpoints + transport config axes)
- Separate ontology FalkorDB graph (`ws_{id}_ontology`)
- Seed ontology (14 types, three-tier status: seed/provisional/confirmed)
- Graph label constants (`labels.py`)
- Migrate existing orchestrators to new pipeline
- Existing tests pass against new pipeline

### Phase 2: Extraction Stages
- `classify` StageCommand
- `coreference` StageCommand
- `simplify` StageCommand (clause splitting, relative clause extraction, passive→active, appositive extraction — rules on spaCy dep tree)
- `entity_ruler` StageCommand (spaCy sm + EntityRuler)
- `llm_extract` StageCommand
- `ontology_constrain` StageCommand (type validation, provisional type creation, PromotionConfig)
- Migrate hardcoded prompts to `prompts.json`

### Phase 3: Storage + Post-Processing Stages
- `store` StageCommand
- `link` StageCommand
- `enrich` StageCommand (Wikidata: FalkorDB canonical + Redis write-through cache)
- `evolve` StageCommand
- Pipeline metrics emission (Redis Streams)

### Phase 4: Self-Learning Loop
- Promotion flow (provisional → confirmed via configurable PromotionConfig gates)
- EntityRuler pattern growth from confirmed types
- Entity-pair cache (reuse known LLM relations)
- Type-pair validation in `ontology_constrain`
- Reasoning validation integration (Level 2 — statistical pre-filter + reasoning)

### Phase 5: Service + API (can parallel with Phase 4)
- PipelineConfig management routes (save/load named configs)
- Pattern admin routes
- Ontology status/import/export routes
- Event-bus transport (Redis Streams between stages)
- Retry with undo

### Phase 6: Insights + Observability (can parallel with Phase 4)
- Pipeline metrics aggregation consumer (Redis Streams → pre-aggregated data)
- Dashboard UI (pipeline metrics, ontology metrics, extraction quality metrics)
- Convergence monitoring

### Phase 7: Studio Pipeline UI (MAJOR — requires own design session)
- PipelineConfig editor (per-stage hierarchical configs)
- Breakpoint execution UI (run-to, inspect state, modify params, resume)
- Ontology viewer (type registry, convergence dashboard, pattern browser, pruning, graph analytics)
- Prompt editing UI (complete existing API integration)
- Benchmarking workflow (batch execution + comparison)
- Grid search / parameter tuning

### Phase 8: Hardening
- Tests (pipeline, stages, self-learning, breakpoints, undo)
- Benchmark suite integration in Studio
- Documentation
- HITL review UI (if `human_review: True` is needed)

### Dependencies

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

### Q14. Reasoning Integration in Self-Learning — HOW DEEP? [NEW]

**Context:** The self-learning quality gates (Q7) are currently NELL-era statistical heuristics: confidence thresholds, frequency counts, multi-source agreement. SmartMemory already has `ReasoningTrace`, `AssertionChallenger`, and reasoning as a first-class memory type. The question is whether reasoning should be integrated into ontology self-learning decisions, and if so, how deeply.

**Why this matters:** NELL's 25% category degradation was a black box — categories degraded and researchers couldn't predict which ones or diagnose why. They had no reasoning, just statistical accumulation. One bad classification cascaded because there was no mechanism to ask "does this make sense?" We have LLMs and a reasoning infrastructure. We should be able to do better.

---

#### The Integration Spectrum

**Level 0: No reasoning (NELL-style, current plan)**
Statistical quality gates only. Confidence > 0.8, frequency > 2, multi-source agreement. Promotion is a threshold check.

- **Advantage:** Simple, fast, no LLM cost for promotions
- **Risk:** Same failure modes as NELL. Degradation is a black box. No audit trail.

**Level 1: Reasoning as audit trail**
Statistical gates remain the decision maker. But when a pattern is promoted, a reasoning step runs asynchronously to explain WHY: "FastAPI promoted to Technology because: seen 5 times, always in tech context, co-occurs with Flask/Django (both Technology), Wikidata confirms." ReasoningTrace attached to the promotion event.

- **Advantage:** Promotions are auditable. When degradation happens, you can read the traces and find the root cause. No change to promotion speed or cost (reasoning is async, after the fact).
- **Risk:** Reasoning is post-hoc — it explains decisions already made. Doesn't prevent bad promotions.

**Level 2: Reasoning as quality gate (pre-filter passes, reasoning validates)**
Statistical pre-filter eliminates obvious noise (frequency < 2, confidence < 0.5). Candidates that pass go through a reasoning validation step (async LLM): "Should this pattern be promoted? Consider evidence, contradictions, semantic consistency." ReasoningTrace is the decision record.

- **Advantage:** Reasoning PREVENTS bad promotions, not just explains them. Catches semantic drift, context mismatches, and cascading errors that statistics miss. Existing `AssertionChallenger` can validate promotions against contradicting evidence.
- **Risk:** LLM cost per promotion candidate. Latency (async, so not blocking, but slower feedback loop). LLM itself can be wrong.

**Level 3: Reasoning as the decision maker (statistics are input, not gates)**
No statistical gates at all. Every promotion candidate goes through reasoning with all available evidence: graph context, co-occurrence patterns, Wikidata alignment, contradiction history. The reasoning trace IS the quality gate. Statistics are input features, not thresholds.

- **Advantage:** Maximum intelligence. Catches everything statistics miss. Full provenance for every ontology change. The system can explain every decision it made about its own knowledge.
- **Risk:** Expensive (LLM call per candidate). Overkill for obvious promotions ("Python" → Technology). LLM hallucination becomes a risk vector for ontology quality.

---

#### Sub-questions — RESOLVED

**SQ1. Does reasoning apply to demotion?**
**Yes, symmetrically.** If promotion has a reasoning trace ("FastAPI promoted to Technology because..."), demotion should too ("Pattern X hasn't been seen in 30 days, workspace shifted from tech to legal, recommend demotion"). The trace makes demotion diagnosable and reversible. Same Level 1→2 phasing as promotion.

**SQ2. Does AssertionChallenger connect to ontology?**
**Yes.** Currently AssertionChallenger detects contradictions between memories ("claim A contradicts claim B"). Extend to entity-type contradictions: "Entity X classified as Technology in 8 memories but Organization in 2 — challenge." Same contradiction detection mechanism, applied to ontology classifications. Wired in Phase 6 alongside reasoning validation.

**SQ3. Should ontology reasoning traces be stored as `reasoning` memory type?**
**Yes.** The system's reasoning about its own ontology becomes searchable memory. "Why did you classify FastAPI as Technology?" → search reasoning memories tagged with ontology context → return the trace. The system can explain its own knowledge structure. Implementation: ontology reasoning traces get `memory_type="reasoning"` + metadata tag `reasoning_domain="ontology"`.

**SQ4. First-class models as entity types?**
**Yes — Decision, Opinion, ReasoningTrace, Observation should be BOTH memory types AND entity types.** This is the bridge between extraction and specialized models.

Currently: "We decided to use PostgreSQL" → memory classified as `decision`, Decision model created, entities extracted: PostgreSQL (Technology), MySQL (Technology). The decision itself is NOT an entity — it's invisible to the entity graph.

With dual-axis: same text extracts THREE things: PostgreSQL (Technology), MySQL (Technology), AND "Use PostgreSQL over MySQL" (Decision). The Decision entity links to the Decision model (with reinforce/contradict/supersede behavior). Future memories can reference it: "Remember the database decision?" → extracts a reference to that Decision entity.

| First-Class Model | As Entity Type | What It Enables |
|---|---|---|
| **Decision** | "We decided X" → Decision entity | Other memories reference past decisions. Decision graph navigable. Type-pair priors: Person --MADE--> Decision, Decision --AFFECTS--> Project. |
| **Opinion** | "Sarah thinks X" → Opinion entity | Opinions extractable, attributable to persons, reinforceable across memories. |
| **ReasoningTrace** | "The analysis shows X because Y" → Reasoning entity | Chains of reasoning are extractable, referenceable, and critiqueable. |
| **Observation** | "I noticed pattern X" → Observation entity | Patterns become entities tracked, confirmed, or refuted over time. |

**Architectural implication:** Extraction and specialized models converge. When the LLM tier extracts a Decision entity, it creates a full `Decision` model with evidence chain, makes it available through the decision API, AND inserts it as a graph entity. The SmartMemory-native seed types aren't just NER labels — they're bridges to first-class behavior.

---

#### Recommendation: **Level 2 (reasoning validates, statistics pre-filter)** — but not at launch

**Phase 1-5:** Level 1 (reasoning as audit trail). Statistical gates make promotion decisions. Reasoning traces logged asynchronously for auditability. This is the NELL baseline with observability.

**Phase 6 (self-learning):** Upgrade to Level 2. Reasoning validates promotion candidates that pass statistical pre-filter. AssertionChallenger wired into ontology contradictions. ReasoningTraces stored as `reasoning` memory type — making the system's ontology decisions queryable.

**Never:** Level 3. Statistics as pre-filter is essential for cost control. The LLM shouldn't evaluate obvious promotions. But the reasoning layer should catch the non-obvious ones that NELL would get wrong.

**Reasoning:** The phased approach means we ship with NELL-equivalent quality (proven to work 75% of the time) plus audit trails (Level 1), then upgrade to reasoning-validated quality (Level 2) once we have real promotion data to reason about. We don't need the reasoning infrastructure to be perfect from day one — we need the hooks in place so it can be wired in.

---

## Decision Summary

| Question | Recommendation | Key Reasoning |
|----------|---------------|---------------|
| Q0. What IS the ontology? | **RESOLVED** — Triples in FalkorDB. Three jobs (classify/predict/constrain). Three layers (global TBox → tenant soft-TBox → tenant ABox). 14 seed types (8 generic + 6 SmartMemory-native), 100+ discoverable, parameterized process. | Field research validated. 25 use cases analyzed as 10-parameter space. |
| Q1. Pattern storage | **DECIDED** — FalkorDB for all (entity patterns as node metadata, relation patterns as edges/nodes). No JSONL in repo. Seeds are service-level ops (global, loaded once). No vector embeddings on patterns. | Patterns ARE graph entities; OSS ships empty; hosted seeds globally |
| Q2. Ingestion path | **DECIDED** — One pipeline with breakpoint execution. `PipelineState` flows through linear stages. `run_to()`/`run_from()` for Studio tuning. Three orchestrators collapse to one. ~1,100 LOC deleted. | Debugger breakpoint metaphor: run to breakpoint, inspect state, modify params, resume. Studio calls core directly. |
| Q3. Worker model | **Subsumed by Q2** — enrichment worker is just the event-bus runner for enrichment StageCommand. Redis consumer groups, DLQ, same image different entrypoint. | Execution model is orthogonal to stage implementation; no separate enrichment architecture needed. |
| Q4. Existing ontology | **DECIDED — Absorb into pipeline.** Separate FalkorDB graphs (data vs ontology). All data ontology-governed (provisional types on first encounter). Three-tier status (seed/provisional/confirmed). PromotionConfig (Option D, C as default). HITL deferred. Studio: params scattered in stage configs, dedicated ontology viewer for management. | Ontology isn't separate — it's pipeline config input, state output, persistent knowledge. Separate graphs = no bleed, no queries to audit. |
| Q5. Insights stubs | **DECIDED — Pipeline emits metrics via Redis Streams.** Pre-aggregated by consumer. Three metric categories (pipeline, ontology, extraction quality). UI is implementation timing. | Event-driven, decoupled. Same Redis Streams infrastructure as event-bus transport. |
| Q6. Studio 61KB | **DECIDED — Rebuild around PipelineConfig.** MAJOR discussion deferred to implementation. Old panel is reference, not starting point. | Pipeline architecture fundamentally changes the panel. Don't refactor — redesign. |
| Q7. Quality gate | **Subsumed by Q4** — PromotionConfig in PipelineConfig. Per-workspace, tunable via Studio. | Global defaults + per-workspace overrides. No separate quality gate abstraction. |
| Q8. Wikidata API | **DECIDED — REST + SPARQL.** FalkorDB canonical, Redis write-through cache. Scaling escape hatch: local dump. | Simple, free. Cache avoids repeated API calls. FalkorDB is source of truth. |
| Q9. Which prompt | **DECIDED — Prompt management system.** prompts.json (defaults) → MongoDB (per-tenant overrides) → PipelineConfig (runtime). Migrate hardcoded prompts. Finish Studio UI. | Three-layer resolution. PromptProvider feeds PipelineConfig, not runtime. |
| Q10. Config | **DECIDED — ExtractionConfig is composite stage config** within PipelineConfig. Nested dataclasses. `simplify` stage added (clause splitting, relative clause extraction, passive→active, appositive extraction). Future: config registry/visitor if needed. | Hierarchy is natural nesting. Each sub-stage independently breakpointable. |
| Q11. Foundational? | **DECIDED — YES, first principle of v4.** | Every other decision assumes this. |
| Q12. Benchmark w/ ontology | **DECIDED — No RelationRuler.** Type-pair validation + entity-pair cache only. Skip dep-parse templates. Benchmarking is a Studio workflow (breakpoint + grid search). | Relations are semantic (LLM territory). Dep-parse ceiling too low for the maintenance cost. |
| Q13. Phase ordering | **DECIDED — Complete rewrite (v5).** 8 phases based on pipeline architecture. GATE: review and sign off before implementation begins. | Old phases obsolete. Pipeline foundation first, then stages, then self-learning. |
| Q14. Reasoning integration | **DECIDED** — Level 2 recommended (statistical pre-filter + reasoning validation). Level 1 (audit trail) at launch, Level 2 (reasoning validates) in Phase 6. AssertionChallenger wired to ontology. First-class models (Decision, Opinion, etc.) are BOTH memory types AND entity types. | Don't follow NELL verbatim — we have LLMs. Dual-axis model bridges extraction and specialized behavior. |
