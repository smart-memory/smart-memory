# Extraction Stages: Technical Specification

**Date:** 2026-02-06
**Status:** Design document
**Part of:** Ontology-Grounded Extraction Implementation

---

## Overview

This document specifies the 10 pipeline stages in SmartMemory's ontology-grounded extraction system. Each stage implements the `StageCommand` protocol with `execute(state, config) -> state` and `undo(state) -> state` methods.

**Pipeline flow:**

```
text → classify → coreference → simplify → entity_ruler → llm_extract
     → ontology_constrain → store → link → enrich → evolve
```

All stages are linear and checkpoint-able. Studio can run to any stage, inspect intermediate state, modify config, and resume.

---

## PipelineState

The `PipelineState` dataclass accumulates data as it flows through the pipeline. Each stage reads from and writes to specific fields.

```python
@dataclass
class PipelineState:
    # Input
    raw_text: str
    memory_type: str | None
    metadata: dict[str, Any]

    # classify outputs
    classified_memory_type: str
    classification_confidence: float

    # coreference outputs
    resolved_text: str | None

    # simplify outputs
    simplified_sentences: list[str]

    # entity_ruler outputs
    ruler_entities: list[Entity]

    # llm_extract outputs
    llm_entities: list[Entity]
    llm_relations: list[Relation]

    # ontology_constrain outputs
    entities: list[Entity]  # merged + validated
    relations: list[Relation]  # merged + validated
    rejected: list[Entity | Relation]
    promotion_candidates: list[PromotionCandidate]

    # store outputs
    stored_memory_id: str
    stored_entity_ids: dict[str, str]  # entity name -> graph ID

    # link outputs
    linked_entities: list[str]  # IDs of linked/existing entities
    cross_references: list[tuple[str, str, float]]  # (entity_id, ref_id, similarity)

    # enrich outputs
    enriched_entities: dict[str, dict[str, Any]]  # entity_id -> enrichment data

    # evolve outputs
    evolved_memory_type: str | None
    evolution_actions: list[str]

    # Metadata
    pipeline_id: str
    stage_timings: dict[str, float]
    stage_errors: dict[str, str | None]
    created_at: datetime
    updated_at: datetime
```

---

## Stage 1: classify

**Purpose:** Determines memory type from content (working, semantic, episodic, procedural, zettel, reasoning, opinion, observation, decision).

### Inputs (from PipelineState)

- `raw_text: str` - The input content
- `metadata: dict[str, Any]` - Optional user-provided metadata with `memory_type` hint

### Outputs (added to PipelineState)

- `classified_memory_type: str` - Determined memory type
- `classification_confidence: float` - Confidence score (0.0-1.0)

### Config (from PipelineConfig)

```python
@dataclass
class ClassifyConfig:
    model: str = "heuristic"  # "heuristic" | "llm"
    fallback_type: str = "semantic"
    confidence_threshold: float = 0.7
    content_indicators: dict[str, list[str]] = field(default_factory=dict)
```

### Implementation Notes

- **Heuristic mode** (default): Keyword matching based on `content_indicators`
- **LLM mode**: GPT-4o-mini classification for ambiguous cases
- If `metadata['memory_type']` is provided, use it with `confidence=0.9`
- If confidence < `confidence_threshold`, use `fallback_type`
- Extended types (reasoning, opinion, observation, decision) detected via specialized keywords

### Undo Behavior

Clears `classified_memory_type` and `classification_confidence` from state.

---

## Stage 2: coreference

**Purpose:** Resolves pronouns and vague references to explicit entity names ("he said" → "John said").

### Inputs (from PipelineState)

- `raw_text: str` - Original text (or `simplified_sentences` joined, if stage reordered)

### Outputs (added to PipelineState)

- `resolved_text: str` - Text with pronouns replaced by explicit entity references

### Config (from PipelineConfig)

```python
@dataclass
class CoreferenceConfig:
    enabled: bool = True
    model: str = "fastcoref"  # "fastcoref" | "spacy"
    device: str = "auto"  # "auto" | "mps" | "cuda" | "cpu"
    min_text_length: int = 50
```

### Implementation Notes

- **fastcoref** (default): Uses `fastcoref` library with F-coref model
- **spacy**: Uses spaCy's `neuralcoref` or experimental coreference component
- If `enabled=False`, `resolved_text = raw_text` (pass-through)
- Skip if `len(raw_text) < min_text_length` (no pronouns in very short text)
- Device selection: `auto` tries MPS → CUDA → CPU in order

### Undo Behavior

Clears `resolved_text` from state.

---

## Stage 3: simplify

**Purpose:** Makes sentences flatter for machine parsing (dep-parse, not LLM). Shared spaCy dep parse for 4 transformations.

### Inputs (from PipelineState)

- `resolved_text: str` (if coreference ran) or `raw_text: str`

### Outputs (added to PipelineState)

- `simplified_sentences: list[str]` - Flattened sentences, one simple assertion per line

### Config (from PipelineConfig)

```python
@dataclass
class SimplifyConfig:
    clause_splitting: bool = True
    relative_clause_extraction: bool = True
    passive_to_active: bool = True
    appositive_extraction: bool = True
```

### Implementation Notes

This is ONE stage with 4 config flags that share a single spaCy dep-parse pass:

1. **clause_splitting** - Compound sentences → simple sentences
   - "X and Y" → two sentences
   - "X but Y" → two sentences
   - Split on coordinating conjunctions at sentence root

2. **relative_clause_extraction** - Relative clauses → standalone assertions
   - "Python, which was invented by Guido, is popular" → "Python is popular. Python was invented by Guido."
   - Extract `relcl` dependencies

3. **passive_to_active** - Passive voice → active voice
   - "was founded by Steve Jobs" → "Steve Jobs founded"
   - Detect `nsubjpass` + `auxpass` patterns

4. **appositive_extraction** - Appositive phrases → separate assertions
   - "Tim Cook, CEO of Apple, said X" → "Tim Cook is CEO of Apple. Tim Cook said X."
   - Extract `appos` dependencies

**Key constraint:** All 4 transforms share the SAME spaCy dep parse (run once). Each transform reads the dependency tree and applies its rule set independently.

**Output format:** List of simple sentences, deduplicated. Each sentence is a single assertion with minimal nesting.

### Undo Behavior

Clears `simplified_sentences` from state.

---

## Stage 4: entity_ruler

**Purpose:** spaCy sm + EntityRuler pattern matching. Fast tier extraction (4ms, 96.9% E-F1).

### Inputs (from PipelineState)

- `simplified_sentences: list[str]` (if simplify ran) or `resolved_text` or `raw_text`

### Outputs (added to PipelineState)

- `ruler_entities: list[Entity]` - Entities extracted via pattern matching

### Config (from PipelineConfig)

```python
@dataclass
class EntityRulerConfig:
    enabled: bool = True
    pattern_sources: list[str] = field(default_factory=lambda: ["seed", "learned"])
    min_confidence: float = 0.8
```

### Implementation Notes

- Uses spaCy `sm` model + EntityRuler component
- Patterns loaded from ontology graph (`ws_{id}_ontology`)
- Three pattern layers: `seed` (global, shipped) → `learned_global` (promoted) → `learned_tenant` (workspace-specific)
- Pattern format: spaCy's `patterns` JSON (exact match, lemma, POS, dependency patterns)
- Each match produces an `Entity` with `source="entity_ruler"` and confidence from pattern metadata
- Filter entities with `confidence < min_confidence`

**Benchmark:** 96.9% E-F1 at 4ms (within 0.8% of Groq LLM on entities).

### Undo Behavior

Clears `ruler_entities` from state.

---

## Stage 5: llm_extract

**Purpose:** LLM-based extraction (Groq default, Gemma for self-hosted). Enrichment tier (740ms, 85-88% R-F1).

### Inputs (from PipelineState)

- `simplified_sentences: list[str]` or `resolved_text` or `raw_text`
- Ontology schema (from ontology graph) - valid entity types and relation types for prompt

### Outputs (added to PipelineState)

- `llm_entities: list[Entity]` - Entities extracted by LLM
- `llm_relations: list[Relation]` - Relations extracted by LLM

### Config (from PipelineConfig)

```python
@dataclass
class LLMExtractConfig:
    model: str = "groq/llama-3.3-70b-versatile"  # or "ollama/gemma-3-27b-it"
    prompt: str | None = None  # Resolved via PromptProvider if None
    temperature: float = 0.1
    max_entities: int = 20
    max_relations: int = 30
```

### Implementation Notes

- **Prompt resolution:** Three-layer lookup via `PromptProvider`
  1. `config.prompt` (runtime override)
  2. MongoDB per-workspace prompt (Studio custom prompts)
  3. `prompts.json` default (`extraction.llm_extract`)

- **Schema-guided extraction:** Valid entity types and relation types from ontology graph injected into prompt as vocabulary
  - Entity types: `["Person", "Organization", "Technology", "Concept", ...]`
  - Relation types: `["works_for", "founded_by", "uses", "related_to", ...]`

- **Output format:** Structured JSON from LLM with entities and relations
  - Entity: `{ "name": "...", "type": "...", "confidence": 0.9 }`
  - Relation: `{ "source": "...", "target": "...", "type": "...", "confidence": 0.85 }`

- **NEVER progressive/refinement** - Clean text extraction only (progressive prompting proven worse, anchoring bias)

- **Recommended models:**
  - Groq Llama-3.3-70b: 97.7% E-F1, 85-88% R-F1, 740ms, ~$0.0003
  - Gemma-3-27b-it: 95.3% E-F1, 86.7% R-F1, 31s, $0 (validated for background enrichment)

### Undo Behavior

Clears `llm_entities` and `llm_relations` from state.

---

## Stage 6: ontology_constrain

**Purpose:** Merges ruler + LLM entities, validates against ontology, creates provisional types for unknowns, filters impossible relations.

### Inputs (from PipelineState)

- `ruler_entities: list[Entity]` - From entity_ruler stage
- `llm_entities: list[Entity]` - From llm_extract stage
- `llm_relations: list[Relation]` - From llm_extract stage
- Ontology graph (`ws_{id}_ontology`) - Type registry, type-pair priors

### Outputs (added to PipelineState)

- `entities: list[Entity]` - Merged, deduplicated, validated entities
- `relations: list[Relation]` - Validated relations (type-pair filtering applied)
- `rejected: list[Entity | Relation]` - Items that failed validation
- `promotion_candidates: list[PromotionCandidate]` - New types for self-learning flow

### Config (from PipelineConfig)

```python
@dataclass
class ConstrainConfig:
    promotion: PromotionConfig
    domain_range_validation: bool = True

@dataclass
class PromotionConfig:
    reasoning_validation: bool = True
    min_frequency: int = 1
    min_confidence: float = 0.8
    human_review: bool = False  # Deferred - no HITL UI yet
```

### Implementation Notes

**Entity merging:**
1. Deduplicate `ruler_entities + llm_entities` by name (fuzzy match, 0.9 threshold)
2. Ruler entities take precedence (confirmed patterns > LLM speculation)
3. For LLM-only entities, check if high confidence (`>= min_confidence`)

**Type validation:**
1. For each entity type, check existence in ontology graph
2. If type exists with `status="confirmed"` or `status="seed"`, accept entity
3. If type exists with `status="provisional"`, accept entity
4. If type does NOT exist:
   - Create provisional type in ontology graph BEFORE data stores
   - Add to `promotion_candidates` for self-learning
   - Accept entity (ontology is always-on, every entity has a type)

**Type-pair validation (relations):**
1. For each relation `(source_entity, relation_type, target_entity)`:
   - Get entity types: `source_type`, `target_type`
   - Query ontology graph for type-pair prior: `(source_type, relation_type, target_type)`
   - If prior exists with `weight > 0.0`, accept relation
   - If prior exists with `weight = 0.0` (mutual exclusion), reject relation → add to `rejected`
   - If no prior exists, accept relation (assume possible) and track for learning

**Promotion candidate creation:**
- `PromotionCandidate` has: `entity_type`, `entity_name`, `confidence`, `source`, `context`, `timestamp`
- Sent to background self-learning flow via Redis Streams (asynchronous)

### Undo Behavior

Clears `entities`, `relations`, `rejected`, `promotion_candidates` from state.

---

## Stage 7: store

**Purpose:** Entity deduplication, graph storage, vector embedding.

### Inputs (from PipelineState)

- `entities: list[Entity]` - Validated entities from ontology_constrain
- `relations: list[Relation]` - Validated relations
- `classified_memory_type: str` - Memory type
- `raw_text: str` - Original content
- `metadata: dict[str, Any]` - User metadata

### Outputs (added to PipelineState)

- `stored_memory_id: str` - FalkorDB node ID for the memory
- `stored_entity_ids: dict[str, str]` - Entity name → FalkorDB node ID mapping

### Config (from PipelineConfig)

```python
@dataclass
class StoreConfig:
    embed_model: str = "text-embedding-3-small"
    vector_dims: int = 1536
    dedup_threshold: float = 0.95
```

### Implementation Notes

**Storage steps:**
1. **Entity deduplication** - Check existing entities in graph via embedding similarity
   - Query vector index with `dedup_threshold`
   - Reuse existing entity node if match found
   - Create new entity node if no match

2. **Memory node creation**
   - Create `:Memory` node with `memory_type`, `content`, `metadata`, `embedding`
   - Embedding generated via `embed_model` (OpenAI or local)

3. **Entity node creation/linking**
   - For each entity in `entities`:
     - Create or merge `:Entity` node with `name`, `type`, `confidence`
     - Create `(Memory)-[:MENTIONS]->(Entity)` edge

4. **Relation edge creation**
   - For each relation in `relations`:
     - Create edge `(EntityA)-[:relation_type]->(EntityB)` with `confidence`, `source="memory:{stored_memory_id}"`

5. **Scope enforcement**
   - All nodes tagged with `workspace_id`, `user_id` from `ScopeProvider`
   - SecureSmartMemory wrapper ensures tenant isolation

### Undo Behavior

Deletes stored memory node and created entity/relation nodes from graph (rollback).

---

## Stage 8: link

**Purpose:** Cross-reference linking to existing memories and entities.

### Inputs (from PipelineState)

- `stored_memory_id: str` - The memory just stored
- `stored_entity_ids: dict[str, str]` - Entity IDs from store stage
- `entities: list[Entity]` - Entity details for semantic matching

### Outputs (added to PipelineState)

- `linked_entities: list[str]` - IDs of entities linked to existing entities
- `cross_references: list[tuple[str, str, float]]` - (entity_id, existing_entity_id, similarity)

### Config (from PipelineConfig)

```python
@dataclass
class LinkConfig:
    similarity_threshold: float = 0.8
    max_links: int = 10
    cross_type_linking: bool = True
```

### Implementation Notes

**Linking steps:**
1. **Semantic search** - For each entity, search for similar entities in graph
   - Vector similarity search via entity embeddings
   - Filter by `similarity >= similarity_threshold`
   - Limit to `max_links` per entity

2. **Cross-reference creation**
   - Create `(Entity)-[:SIMILAR_TO]->(ExistingEntity)` edges with `similarity` property
   - Bidirectional if `similarity > 0.9` (strong match)

3. **Cross-type linking** (if `cross_type_linking=True`)
   - Link entities to related memories of different types
   - Example: Link a Person entity to all episodic memories mentioning them

4. **Working memory activation**
   - If any linked entity appears in working memory, update access timestamp
   - Working memories with recent access decay slower (managed by evolvers)

### Undo Behavior

Deletes created cross-reference edges from graph.

---

## Stage 9: enrich

**Purpose:** External knowledge enrichment (Wikidata, sentiment, temporal, topic analysis).

### Inputs (from PipelineState)

- `stored_entity_ids: dict[str, str]` - Entity IDs to enrich
- `entities: list[Entity]` - Entity details (names, types)
- `stored_memory_id: str` - Memory ID for enrichment attribution

### Outputs (added to PipelineState)

- `enriched_entities: dict[str, dict[str, Any]]` - Entity ID → enrichment data

### Config (from PipelineConfig)

```python
@dataclass
class EnrichConfig:
    wikidata: WikidataConfig
    sentiment: SentimentConfig | None = None
    temporal: TemporalConfig | None = None
    topic: TopicConfig | None = None

@dataclass
class WikidataConfig:
    enabled: bool = True
    api_mode: str = "rest"  # "rest" | "sparql" | "hybrid"
    cache_backend: str = "redis"  # "redis" | "falkordb" | "both"
    max_requests_per_entity: int = 3
    timeout: float = 5.0
```

### Implementation Notes

**Wikidata enrichment:**
- **REST API** (95% of lookups): Single entity lookup by name/QID
  - `https://www.wikidata.org/w/api.php?action=wbsearchentities&search={entity_name}`
- **SPARQL** (5% of lookups): Type hierarchy queries only
  - `P31` (instance of) + `P279*` (subclass of) chains
- **Caching:** Write-through to Redis (fast reads) + FalkorDB (canonical)
  - Both updated in same write operation (no sync concern)
  - TTL: 7 days for REST API results, indefinite for SPARQL type hierarchies
- **Scaling escape hatch:** If we exceed 200 req/s rate limit, switch to local Wikidata dump

**Sentiment enrichment:**
- Analyze sentiment of memory content (positive/negative/neutral + score)
- Store as property on Memory node

**Temporal enrichment:**
- Extract dates, times, durations from content
- Create `:TemporalEntity` nodes and `[:OCCURRED_AT]` edges

**Topic enrichment:**
- Extract topics via LDA or zero-shot classification
- Create `:Topic` nodes and `[:ABOUT]` edges

**Enrichment runs asynchronously (event-bus mode) in production** - latency invisible to user.

### Undo Behavior

Deletes enrichment data from entity nodes (removes added properties and edges).

---

## Stage 10: evolve

**Purpose:** Memory evolution (working→episodic, episodic→semantic, decay, synthesis).

### Inputs (from PipelineState)

- `stored_memory_id: str` - The memory to potentially evolve
- `classified_memory_type: str` - Current memory type
- Existing graph state (related memories, usage patterns, timestamps)

### Outputs (added to PipelineState)

- `evolved_memory_type: str | None` - New memory type if evolved
- `evolution_actions: list[str]` - Actions taken (e.g., "promoted to episodic", "decayed")

### Config (from PipelineConfig)

```python
@dataclass
class EvolveConfig:
    enabled_evolvers: list[str] = field(default_factory=lambda: [
        "working_to_episodic",
        "episodic_to_semantic",
        "episodic_decay",
        "opinion_synthesis"
    ])
    decay_rates: dict[str, float] = field(default_factory=lambda: {
        "working": 0.1,   # 10% confidence loss per day
        "episodic": 0.05  # 5% confidence loss per day
    })
    synthesis_threshold: float = 0.8
```

### Implementation Notes

**Evolution rules (implemented as plugins):**
1. **working_to_episodic** - Working memory → episodic after time threshold (e.g., 24 hours)
2. **episodic_to_semantic** - Repeated episodic patterns → semantic knowledge
3. **working_to_procedural** - Step-by-step working memories → procedural memory
4. **episodic_to_zettel** - High-quality episodic insights → permanent zettel notes
5. **episodic_decay** - Old episodic memories lose confidence over time
6. **opinion_synthesis** - Contradicting opinions → synthesized opinion with nuance
7. **opinion_reinforcement** - Repeated opinions → increased confidence
8. **observation_synthesis** - Related observations → generalized observation
9. **decision_confidence** - Decision outcomes update confidence

**Execution:**
- Each evolver is a plugin with `evolve(memory, log=None)` method
- Evolvers run in order specified in `enabled_evolvers`
- Evolvers can:
  - Change memory type (promote/demote)
  - Update confidence scores
  - Create new derived memories (synthesis)
  - Delete memories (aggressive decay)

**Logging:**
- All evolution actions logged to `ReasoningTrace` (audit trail)
- Domain: `reasoning_domain="evolution"`

### Undo Behavior

Reverts memory type changes and deletes derived memories created by evolvers.

---

## Stage Execution Order

```
┌────────────┐
│  classify  │  Memory type classification
└─────┬──────┘
      ▼
┌────────────┐
│ coreference│  Pronoun resolution
└─────┬──────┘
      ▼
┌────────────┐
│  simplify  │  Sentence flattening (4 transforms, 1 dep parse)
└─────┬──────┘
      ▼
┌────────────┐
│entity_ruler│  Fast pattern matching (4ms, 96.9% E-F1)
└─────┬──────┘
      ▼
┌────────────┐
│llm_extract │  LLM enrichment tier (740ms, 85-88% R-F1)
└─────┬──────┘
      ▼
┌────────────┐
│  ontology  │  Merge, validate, constrain, promote
│ _constrain │
└─────┬──────┘
      ▼
┌────────────┐
│   store    │  Graph storage + vector embedding
└─────┬──────┘
      ▼
┌────────────┐
│    link    │  Cross-reference linking
└─────┬──────┘
      ▼
┌────────────┐
│   enrich   │  Wikidata, sentiment, temporal, topic
└─────┬──────┘
      ▼
┌────────────┐
│   evolve   │  Memory type evolution, decay, synthesis
└────────────┘
```

**Key properties:**
- Linear flow (no branching)
- Each stage independently testable
- Breakpoint execution supported between any two stages
- Undo works in reverse order
- Async execution (event-bus) preserves order via stage sequencing

---

## Config Resolution

**PipelineConfig hierarchy:**

```python
@dataclass
class PipelineConfig:
    name: str
    workspace_id: str
    mode: str  # "sync" | "async" | "preview"
    retry: RetryConfig

    # Stage configs
    classify: ClassifyConfig
    coreference: CoreferenceConfig
    simplify: SimplifyConfig
    extraction: ExtractionConfig  # Composite containing:
        # - entity_ruler: EntityRulerConfig
        # - llm_extract: LLMExtractConfig
        # - ontology_constrain: ConstrainConfig
    store: StoreConfig
    link: LinkConfig
    enrich: EnrichConfig
    evolve: EvolveConfig
```

**Storage:** Saved per workspace in FalkorDB. Named configs ("default", "high-precision", "bulk-import").

**Prompt management:** Three layers:
1. `prompts.json` - Default prompts shipped with core library
2. MongoDB - Per-workspace prompt overrides (editable via Studio)
3. PipelineConfig - Runtime prompt value (resolved at pipeline start)

---

## Metrics Emission

Each stage emits metrics via Redis Streams:

```python
{
    "pipeline_id": "...",
    "stage": "entity_ruler",
    "latency_ms": 4.2,
    "success": true,
    "entities_extracted": 12,
    "confidence_avg": 0.91,
    "timestamp": "2026-02-06T10:30:45Z"
}
```

Metrics consumed by aggregation worker and displayed in Insights dashboard.

---

## Benchmark Reference

| Stage | Latency | Quality (E-F1 / R-F1) | Cost |
|-------|---------|----------------------|------|
| entity_ruler | 4ms | 96.9% / 65.1% | $0 |
| llm_extract (Groq) | 740ms | 97.7% / 85-88% | ~$0.0003 |
| llm_extract (Gemma) | 31s | 95.3% / 86.7% | $0 |

**Fast tier (entity_ruler):** Handles entities with 96.9% quality at 200x speed vs LLM.
**Enrichment tier (llm_extract):** Handles relations with 85-88% quality (only LLMs can reason semantically).
**Self-learning:** EntityRuler grows from LLM feedback, converging to 99%+ domain coverage after ~10k memories.

---

## Anti-Patterns

These approaches were tested and proven not to work. Do not revisit.

| Approach | Why It Failed |
|----------|--------------|
| Progressive prompting | LLM anchors to draft, all 4 variants worse than standalone extraction |
| spaCy trf for fast tier | 33ms vs 4ms, sm+ruler dominates (96.9% > 94.6%) |
| RelationRuler / dep-parse | Semantic relations need LLMs. Dep-parse ceiling ~65-70% R-F1 |
| Bigger local models (70B) | Hermes-70B (87.3%) < Gemma-27B (95.3%) for extraction |
| GLiNER2 + GLiREL | GLiNER NER at 34% recall; GLiREL useless without good NER |
| REBEL end-to-end | 3.9s latency, 62% R-F1 - dominated on all axes |

---

## Implementation Notes

1. **Stages are synchronous** - Only service layer (PipelineRunner) handles async execution
2. **StageCommand is a protocol** - No base class, just type hint for duck typing
3. **PipelineState is serializable** - JSON-encodable dataclass for checkpointing
4. **Undo is optional** - Not all stages need to implement undo (e.g., classify)
5. **Errors propagate** - Stage failures bubble up to PipelineRunner for retry logic
6. **Ontology is foundational** - Every entity has a type, even if provisional
7. **Self-learning is asynchronous** - Promotion happens in background, doesn't block ingestion

---

## File Locations

| Component | Path |
|-----------|------|
| Pipeline stages | `smartmemory/pipeline/stages/` |
| Stage protocol | `smartmemory/pipeline/protocol.py` |
| PipelineState | `smartmemory/pipeline/state.py` |
| PipelineConfig | `smartmemory/pipeline/config.py` |
| PipelineRunner | `smartmemory/pipeline/runner.py` |
| Ontology graph | `smartmemory/graph/ontology_graph.py` |
| Self-learning | `smartmemory/ontology/promotion.py` |
| Prompts | `smartmemory/prompts/prompts.json` |
| Service routes | `memory_service/api/routes/pipeline.py` |

---

## See Also

- [Pipeline Architecture](pipeline-architecture.md) - StageCommand, PipelineRunner, breakpoint execution
- [Ontology Model](ontology-model.md) - Type system, promotion, TBox/ABox
- [Self-Learning](self-learning.md) - EntityRuler growth, reasoning validation
- [Evidence Base](evidence-base.md) - Benchmark results, research findings
- [Implementation Plan](../2026-02-05-implementation-plan.md) - Phase-by-phase build plan
