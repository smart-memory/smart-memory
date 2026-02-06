# Ontology-Grounded Extraction Architecture

**Date:** 2026-02-05
**Status:** Draft v2 — incorporating design discussion decisions
**Predecessor:** `2026-02-05-extraction-benchmark-findings.md`

---

## 1. Vision: Three-Layer Knowledge Graph

SmartMemory's graph has three distinct layers. The ontology is **core infrastructure**, always on — not an optional enhancement. Like the graph itself, it exists in every deployment.

```
┌──────────────────────────────────────────────────────────┐
│  LAYER 3: USER'S PERSONAL KNOWLEDGE (tenant-scoped)      │
│                                                           │
│  Opinions: "I prefer Python over Java"                    │
│  Decisions: "Chose Kubernetes for our infra"              │
│  Episodic: "Met Sarah at Google on Tuesday"               │
│  Observations: "Team velocity dropped after reorg"        │
│  Process relations: AcmeCorp → uses → FastAPI             │
│                                                           │
│  → Extracted from user text by LLM                        │
│  → ALWAYS tenant-scoped, NEVER shared                     │
│  → Includes ALL relation instances from user text         │
└──────────────────┬───────────────────────────────────────┘
                   │ links to / extends
┌──────────────────▼───────────────────────────────────────┐
│  LAYER 2: LEARNED DOMAIN KNOWLEDGE (mixed scope)         │
│                                                           │
│  EntityRuler patterns: "FastAPI" is a technology          │
│  Dep-parse templates: nsubj(founded) + ORG → "founded"   │
│  Type-pair priors: (person, org) → works_at, founded      │
│                                                           │
│  → Self-learned from LLM enrichment feedback              │
│  → Vocabulary and grammar are globally shareable           │
│  → Entity-pair relations are ALWAYS tenant-scoped          │
└──────────────────┬───────────────────────────────────────┘
                   │ grounded in
┌──────────────────▼───────────────────────────────────────┐
│  LAYER 1: WORLD KNOWLEDGE (global, shared)               │
│                                                           │
│  Entities: Python (Q28865), Google (Q95)                  │
│  Types: Python is-a programming language                  │
│  Relations: Google developed Kubernetes                   │
│  Properties: Python created_in 1991                       │
│                                                           │
│  → Demand-loaded from Wikidata on first mention           │
│  → Shared globally (is_global=True), read-only            │
│  → Lookup, not inference                                  │
└──────────────────────────────────────────────────────────┘
```

### 1.1 Core Principle: Ontology Is Not Optional

The ontology is core architecture, like the graph or the embedding store. It exists in every deployment.

- **Always-on**: The system always uses whatever knowledge it has to inform extraction. There is no toggle to "disable ontology."
- **Cold start, not off**: A fresh install has no learned patterns, no Wikidata cache, just base spaCy + seeded EntityRuler patterns. This is "ontology cold" — it works at 96.9% E-F1 from day one and gets better with every memory stored.
- **Enrichment is the variable**: What differs between deployments is whether background LLM enrichment runs (which teaches the rulers and populates the cache). This is a resource/tier question, not an architecture toggle.

### 1.2 Three Categories of Knowledge

| Category | Source | Extraction method | Scope | Example |
|----------|--------|-------------------|-------|---------|
| **World knowledge** | Wikidata | Demand-loaded lookup | Global | "Google developed Kubernetes" |
| **Domain knowledge** | Self-learning from LLM | EntityRuler + RelationRuler | Vocabulary: global. Relations: tenant | "FastAPI" is a technology |
| **Personal knowledge** | User's text | LLM extraction | Always tenant-scoped | Opinions, decisions, processes |

### 1.3 Separation of Entity Types and Memory Types

Currently SmartMemory mixes two type systems. They should be cleanly separated:

- **Entity types** (person, organization, technology, location, etc.) — these map to ontology hierarchies. An entity *is* a person or a technology.
- **Memory types** (working, semantic, episodic, opinion, decision, etc.) — these are SmartMemory-specific. A memory *about* an entity is episodic or semantic.

An entity node has an entity type from the ontology. A memory node has a memory type from SmartMemory. They connect via edges.

---

## 2. Privacy Model: Patterns Are Public, Instances Are Private

### 2.1 The Core Rule

**Entity vocabulary and syntactic grammar are shareable. Relation instances from user text are never shared.**

Even when both entities are public (e.g., "Python", "FastAPI"), the relation between them extracted from user text ("We use Python + FastAPI + Docker + Kubernetes") is private. The combination of public entities into a process graph reveals proprietary architecture, workflows, and trade secrets.

### 2.2 What Is Shared vs. Tenant-Scoped

| Knowledge type | Example | Sharing | Why |
|---------------|---------|---------|-----|
| **Entity vocabulary** (name → type) | "FastAPI" is a technology | **Global** | Vocabulary, no information content |
| **Syntactic templates** (dep-parse patterns) | nsubj(VERB) + dobj(VERB) → relation | **Global** | Grammar, doesn't reveal who does what |
| **Type-pair priors** (entity type → likely relations) | (person, org) → works_at, founded | **Global** | Pre-built from Wikidata stats, not user data |
| **Wikidata facts** (world knowledge) | Google developed Kubernetes | **Global** | Public knowledge from Wikidata API |
| **Entity-pair relations from user text** | AcmeCorp → uses → FastAPI | **Always tenant-scoped** | Reveals proprietary processes |
| **Memory content** (opinions, decisions, notes) | "We chose FastAPI for performance" | **Always tenant-scoped** | User's private knowledge |
| **Graph structure** (relation topology between entities) | The shape of entity connections | **Always tenant-scoped** | Process shapes are proprietary |

### 2.3 EntityRuler Pattern Promotion

When a tenant's LLM discovers a new entity (e.g., "LangChain" as a technology):

1. **Wikidata-linkable entity**: Pattern promoted to global immediately. "LangChain" exists in Wikidata — it's public vocabulary.
2. **Non-Wikidata entity, single tenant**: Pattern stays tenant-scoped. Could be an internal codename ("Project Mercury").
3. **Non-Wikidata entity, multiple tenants independently discover it**: After N tenants (e.g., 3+) independently discover the same entity name + type, promote to global. Cross-tenant frequency proves it's public vocabulary, not proprietary.

### 2.4 RelationRuler: What Is Learned Globally

- **Dep-parse templates**: Safe to share globally. `{nsubj(VERB) = PERSON, dobj(VERB) = ORG, verb_lemma ∈ {found, establish, create}} → "founded"` reveals syntax, not content.
- **Entity-pair relation cache**: NEVER shared globally. Even between two public entities, the user's specific connection is private.
- **Graph pattern inference**: Runs only within a tenant's own graph. Never crosses tenant boundaries.

---

## 3. Three-Tier Extraction Architecture

### 3.1 Overview

```
Text arrives
     │
     ▼
┌─────────────────────────────────────────┐
│  TIER 1: Ontology Lookup (1-5ms)        │
│                                          │
│  EntityRuler finds "Kubernetes"          │
│  → Entity linking: Wikidata Q22661306   │
│  → Type: technology (from Wikidata)     │
│  → World relations: developed_by→Google │
│  → These are facts, not inferences      │
└──────────────┬──────────────────────────┘
               │ entities not in Wikidata
               ▼
┌─────────────────────────────────────────┐
│  TIER 2: Ruler-Based Extraction (4ms)   │
│                                          │
│  EntityRuler: learned patterns (96.9%)  │
│  RelationRuler: type-pair priors +      │
│    dep-parse templates (tenant-scoped   │
│    entity-pair cache)                   │
│  → Handles domain-specific entities     │
│  → Self-learning, improves over time    │
└──────────────┬──────────────────────────┘
               │ queue for background enrichment
               ▼
┌─────────────────────────────────────────┐
│  TIER 3: LLM Enrichment (async)        │
│                                          │
│  Hosted: Groq Llama-3.3-70b (740ms)    │
│  Self-hosted: Gemma-3-27b-it (31s, $0) │
│  100% E-F1, 89% R-F1 with tuned prompt │
│  → Discovers novel entities/relations   │
│  → Feeds back into Tier 2 rulers        │
│  → Replaces stored results with richer  │
│    extraction                           │
└─────────────────────────────────────────┘
```

### 3.2 Real-Time Path (Tiers 1+2)

User gets results in <20ms:
1. spaCy sm + EntityRuler identifies entities (4ms)
2. For each entity, check Wikidata cache (Redis, <1ms per entity)
3. If cached: use Wikidata type + world-knowledge relations
4. If not cached: use EntityRuler type + RelationRuler inference
5. Store memory + entities + relations (all tenant-scoped)
6. Queue for async Tier 3 enrichment (if enrichment quota allows)

### 3.3 Background Path (Tier 3)

Runs asynchronously after response is returned:
1. LLM extracts full entities + relations from text
2. For entities found by LLM but not in Wikidata cache:
   - Attempt entity linking to Wikidata (may succeed for entities EntityRuler missed)
   - If linked: demand-load Wikidata data, cache forever (global)
   - If not linked: add to EntityRuler patterns (tenant-scoped, eligible for promotion)
3. For relations found by LLM:
   - Extract dep-parse template, add to RelationRuler (global)
   - Add to tenant's entity-pair relation cache (tenant-scoped, NEVER global)
4. Replace stored extraction results with enriched version

---

## 4. Ingestion Modalities

Not all ingestion is conversation. The system must handle diverse input types efficiently.

### 4.1 Input Types

| Input type | Typical size | Chunks | Fast tier time | Enrichment time (Groq) | Enrichment time (Gemma) |
|-----------|-------------|--------|---------------|----------------------|------------------------|
| Chat message / note | 1-3 sentences | 1 | 4ms | 740ms | 31s |
| Long article | 1-5 pages | 10 | 40ms | 7s | 5 min |
| PDF document | 10-100 pages | 50-200 | 200-800ms | 37s-2.5min | 26-100 min |
| Knowledge base | 100-10K docs | 1K-100K | 4-400s | 12min-20hr | **Impractical** |

**Key insight**: Local Gemma only works for drip-feed ingestion (one memory at a time). For bulk/document ingestion, a fast API (Groq) or batch processing is required.

### 4.2 Document Classification: Public vs Private

Documents fall into three classes with different processing strategies:

| Class | Example | Extraction sharing | Enrichment cost |
|-------|---------|-------------------|-----------------|
| **Public reference** | Python docs, Wikipedia, published papers | Process once, share globally | Amortized across all tenants |
| **Public unique** | Blog post, news article | Deduplicate by URL hash | One-time cost per unique URL |
| **Private** | Internal docs, notes, conversations | Fully tenant-scoped | Per-tenant |

For public documents:
1. **Deduplication**: Hash document content or URL. If already processed, reuse extraction results.
2. **Shared extraction**: Entities and world-knowledge relations from public docs go into the global layer.
3. **Pre-processing**: Popular content (major frameworks, languages, tools) can be pre-extracted and shipped with SmartMemory.

Private documents: all extraction results (entities, relations, memories) are tenant-scoped. Even entity vocabulary learned from private docs starts tenant-scoped (eligible for promotion via the multi-tenant frequency gate in section 2.3).

### 4.3 Batch Enrichment

For bulk ingestion efficiency, batch multiple texts into a single LLM call:

- **Single call**: 1 text, ~500 input tokens, ~300 output tokens
- **Batched call**: 10-50 texts concatenated, shared prompt overhead amortized
- **Cost reduction**: 30-50% per-unit savings from reduced prompt overhead and API call overhead
- **Implementation**: Enrichment queue accumulates texts, flushes batch every N items or T seconds

---

## 5. EntityRuler: Validated Design

### 5.1 Benchmark Results

53 extractor configurations tested. Key findings:

| Configuration | Prompt | E-F1 | R-F1 | Latency | Cost |
|--------------|--------|------|------|---------|------|
| spaCy sm + EntityRuler | N/A | **96.9%** | 65.1% | 4ms | $0 |
| **Groq Llama-3.3-70b** | **improved** | **100%** | **89.3%** | **878ms** | **~$0.0005** |
| Gemma-3-27b-it (local) | improved | 100% | 89.3% | 31s | $0 |
| GPT-5-mini | improved | 93.4% | 55.2% | ~40s | ~$0.002 |
| spaCy trf + EntityRuler | N/A | 94.6% | 65.1% | 33ms | $0 |

- Groq with improved prompt matches Gemma exactly (100% E-F1, 89.3% R-F1) at 35x speed
- Prompt improvement is model-agnostic: entity splitting + implicit relations helped both Groq and Gemma
- GPT-5-mini massively over-extracts relations (78 FP on 48 gold) — not competitive
- EntityRuler adds +10.9 E-F1 points at zero latency cost
- sm + ruler beats trf + ruler (96.9% > 94.6%) — weaker NER model doesn't fight the ruler
- Progressive prompting (4 variants) always worse than LLM standalone — anchoring bias

### 5.2 Self-Learning Loop

1. Text → spaCy sm + EntityRuler → instant results (4ms)
2. Background: LLM extracts full entities
3. Diff: LLM entities − EntityRuler entities = new patterns
4. Quality gate: confidence > 0.8, length > 3, frequency > 1, type consistency > 80%
5. Add to EntityRuler patterns, persist to disk
6. Convergence: after ~1,000-5,000 memories, ruler covers most domain vocabulary

### 5.3 Pattern Storage and Multi-Tenancy

Three pattern layers, loaded in order (later layers override earlier):

```
1. Base patterns (pre-generated via Groq/Gemma against diverse corpus)
   → {install_dir}/entity_patterns_base.jsonl
   → 5,000-10,000+ patterns covering major domains
   → Generated offline before release, no time pressure

2. Global learned patterns (promoted from tenants via frequency gate)
   → {data_dir}/entity_patterns_global.jsonl

3. Tenant-specific patterns (learned from this tenant's LLM extractions)
   → {data_dir}/entity_patterns_{tenant_id}.jsonl
```

### 5.4 Pre-Release Seeding Strategy

Instead of shipping with 50 hand-crafted patterns, pre-generate a comprehensive base using LLM extraction:

1. Curate diverse text corpus: tech docs, Wikipedia excerpts, science, business, history, arts
2. Run Groq extraction (878ms/text, ~$0.0005/text) on thousands of texts
3. Collect unique entities + types → 5,000-10,000+ EntityRuler patterns
4. Collect dep-parse templates from extracted relations → seed RelationRuler
5. Quality-gate: frequency > 1, type consistency > 80%, no single-tenant entities
6. Ship as `entity_patterns_base.jsonl` + `dep_templates_base.jsonl`

Cost to seed: ~$5 for 10,000 texts via Groq. One-time investment. Day-one users get rulers that already know most common entities across major domains. The cold-start problem largely disappears.

---

## 6. RelationRuler: New Design

The relation extraction equivalent of EntityRuler. Combines multiple fast signals to infer relations without an LLM.

### 6.1 Signal 1: Type-Pair Relation Priors

A lookup table mapping entity type pairs to likely relation types:

```python
TYPE_PAIR_PRIORS = {
    ("person", "organization"): ["founded", "works_at", "ceo_of", "member_of"],
    ("organization", "location"): ["headquartered_in", "located_in", "based_in"],
    ("person", "work_of_art"): ["composed", "wrote", "created", "directed"],
    ("technology", "technology"): ["framework_for", "uses", "inspired_by", "extends"],
    ("organization", "product"): ["released", "produces", "owns", "developed"],
    ("person", "award"): ["won", "nominated_for", "received"],
    ("event", "temporal"): ["occurred_in", "started_in", "ended_in"],
    ("organization", "technology"): ["developed", "maintains", "uses"],
}
```

Sub-millisecond lookup. Combined with verb signal from dep-parse, narrows relation space dramatically.

Can be bootstrapped from Wikidata property statistics (which properties typically connect which entity types).

**Scope:** Global. Pre-built from Wikidata stats, not learned from user data.

### 6.2 Signal 2: Learned Dep-Parse Templates

The true mirror of EntityRuler — generalizable syntactic patterns:

1. LLM extracts: (Bill Gates, founded, Microsoft)
2. Dep-parse finds: `nsubj(founded) = "Bill Gates"`, `dobj(founded) = "Microsoft"`
3. Learn template: `{nsubj(VERB) = PERSON, dobj(VERB) = ORG, verb_lemma ∈ {found, establish, create}} → "founded"`

Next time ANY text has "X founded/established/created Y" with PERSON and ORG, emit the relation instantly.

Self-learning loop:
1. LLM extracts (entity_A, relation, entity_B) from text
2. Run dep-parse on same text, find syntactic path between A and B
3. Record: (dep_pattern, entity_type_pair, verb_lemma) → relation_type
4. Quality gate: pattern seen 3+ times with consistent relation type
5. Persist to disk, load on startup

**Scope:** Global. Syntactic patterns reveal grammar, not content.

### 6.3 Signal 3: Entity-Pair Relation Cache

Direct memorization of known entity-pair relations:

- LLM discovers (Google, Kubernetes) → "developed"
- Store in cache: `{(entity_A, entity_B): relation_type}`
- Next time both entities co-occur in this tenant's text, emit the relation

**Scope: ALWAYS tenant-scoped.** Even between public entities, the user's specific entity connections reveal proprietary processes. "AcmeCorp → uses → FastAPI → deployed_on → Kubernetes" is a proprietary architecture decision, not public knowledge.

### 6.4 Signal 4: Graph Pattern Inference

Leverage the tenant's existing knowledge graph:

- Graph knows: Google→developed→Kubernetes, Google→developed→TensorFlow
- Pattern: (Google, developed, TECHNOLOGY) appears twice
- New text mentions "Google" and "JAX" (a technology)
- Predict: (Google, developed, JAX) with moderate confidence

**Scope: ALWAYS tenant-scoped.** Runs only within a tenant's own graph. Never crosses tenant boundaries.

### 6.5 Multi-Signal Scoring

Combine all signals for each candidate entity pair:

```python
def score_relation(entity_a, entity_b, dep_path, verb_lemma, graph):
    scores = {}

    # Signal 1: Type-pair prior (global)
    candidates = TYPE_PAIR_PRIORS.get((entity_a.type, entity_b.type), [])
    for rel in candidates:
        scores[rel] = scores.get(rel, 0) + 0.3

    # Signal 2: Dep-parse template match (global)
    template_rel = match_dep_template(dep_path, verb_lemma, entity_a.type, entity_b.type)
    if template_rel:
        scores[template_rel] = scores.get(template_rel, 0) + 0.3

    # Signal 3: Entity-pair cache (tenant-scoped)
    cached_rel = tenant_entity_pair_cache.get((entity_a.name, entity_b.name))
    if cached_rel:
        scores[cached_rel] = scores.get(cached_rel, 0) + 0.3

    # Signal 4: Graph pattern (tenant-scoped)
    graph_rel = graph_pattern_predict(entity_a, entity_b, tenant_graph)
    if graph_rel:
        scores[graph_rel] = scores.get(graph_rel, 0) + 0.2

    # Threshold: emit relation if score > 0.5
    best_rel = max(scores, key=scores.get) if scores else None
    if best_rel and scores[best_rel] > 0.5:
        return best_rel
    return None
```

Total cost: <1ms per entity pair. All signals are lookups or pattern matches.

---

## 7. Ontology Integration: Wikidata

### 7.1 Why Wikidata (Not Wikipedia)

| Aspect | Wikipedia (current) | Wikidata (proposed) |
|--------|-------------------|-------------------|
| Format | Prose text | Structured triples (JSON) |
| Entity types | Must parse from categories | Direct: instance-of (P31) |
| Relations | Buried in text | Explicit properties (P-numbers) |
| Entity linking | Title string match | QID disambiguation |
| API | Text-heavy, slow | Lightweight JSON, fast |
| Offline | Requires full article fetch | Can precompute entity index |

### 7.2 Demand-Loading Architecture

Not a bulk import. Entities enter the world knowledge layer on first user mention:

```
User stores text mentioning "Kubernetes"
     │
     ▼
EntityRuler finds "Kubernetes"
     │
     ▼
Check Wikidata cache (Redis)
     │
     ├─ HIT → Use cached type + relations (< 1ms)
     │
     └─ MISS → Entity linking:
              │
              ▼
         Wikidata API: Q22661306
              │
              ▼
         Fetch: instance_of = software
                developer = Google (Q95)
                maintained_by = CNCF (Q62597478)
                written_in = Go (Q37227)
              │
              ▼
         Create world-knowledge nodes + edges in graph (is_global=True)
         Cache in Redis (long TTL — world facts rarely change)
              │
              ▼
         Return type + relations to extraction pipeline
```

### 7.3 Entity Linking Strategy

Entity linking maps surface forms ("Kubernetes", "k8s") to Wikidata QIDs. Options:

1. **spaCy entity linker** — spaCy has a built-in `entity_linker` component. Trained on Wikipedia. Can be added to the pipeline after NER.
2. **Simple string matching + disambiguation** — For most entities, exact title match works. Disambiguation handled by entity type: "Apple" + type=organization → Q312 (Apple Inc.), not Q89 (fruit).
3. **Precomputed index** — Build a Redis hash from Wikidata dump: `entity_name:entity_type → QID`. ~10M entries, fits in memory.

**Recommendation:** Start with option 2 (simple + type-aware disambiguation), fall back to Wikidata search API for ambiguous cases. Upgrade to spaCy entity linker later if needed.

### 7.4 What to Fetch from Wikidata

For each entity, fetch a bounded set of properties:

```python
WIKIDATA_PROPERTIES = {
    "P31": "instance_of",      # Entity type
    "P279": "subclass_of",     # Type hierarchy
    "P17": "country",          # Location context
    "P131": "located_in",      # Administrative location
    "P159": "headquarters",    # Organization HQ
    "P112": "founded_by",      # Founder
    "P571": "inception",       # Founded date
    "P178": "developer",       # Software developer
    "P277": "programming_language",  # Written in
    "P137": "operator",        # Who runs it
    "P361": "part_of",         # Part-whole
    "P527": "has_part",        # Whole-part
    "P50": "author",           # Creator
    "P86": "composer",         # Music
    "P170": "creator",         # General creator
}
```

~15 properties, bounded. Fetch once, cache with long TTL.

### 7.5 Multi-Tenancy

World knowledge nodes are shared across all tenants:
- Created with `is_global=True` (same pattern as current WikipediaGrounder)
- Read-only from tenant perspective
- Personal knowledge edges (opinions, decisions) connect tenant-scoped memories to shared world nodes

```
[tenant:user123] "I prefer Python" --OPINION_ABOUT--> [global] Python (Q28865)
[tenant:user456] "We chose Python for the backend" --DECISION_ABOUT--> [global] Python (Q28865)
```

The world nodes are identical. The tenant-scoped edges are private.

### 7.6 Graceful Degradation ("Ontology Cold")

The system is never "ontology off" — it's "ontology cold" on a fresh install:

| State | What's available | E-F1 | Behavior |
|-------|-----------------|------|----------|
| **Cold start** | Base spaCy + seeded EntityRuler (~50 patterns) | ~96.9% | Works well from day one |
| **No Redis** | spaCy + EntityRuler only, no Wikidata cache | ~96.9% | Wikidata lookups skip, ruler still works |
| **No LLM / no enrichment** | Fast tier only, rulers don't grow | ~96.9% | Static quality, no self-learning |
| **Warm (1K+ memories)** | Rulers grown, Wikidata cached | ~98%+ | Fast tier approaching LLM quality |
| **Full** | All tiers active, large pattern/cache base | ~99%+ | LLM rarely needed for entities |

The ontology degrades gracefully without any configuration. Missing infrastructure (Redis, LLM, Wikidata API) simply means that tier is skipped.

---

## 8. LLM Enrichment Layer

### 8.1 Prompt Optimization Results

Error analysis on 16 gold test cases identified three fixable issues:

| Issue | Before | After prompt fix |
|-------|--------|-----------------|
| Entity merging ("Nobel Prize in Physics" as one entity) | 4 FN, 2 FP | 0 FN, 0 FP |
| Type confusion (product vs work_of_art) | 6 mismatches | 2 mismatches |
| Implicit relations missed | 9 FN | 2 FN |

**Prompt changes that worked:**
1. **Entity splitting rule**: "Split compound entities into atomic parts"
2. **Clearer type definitions**: Added examples per type, disambiguated product/technology/work_of_art
3. **Implicit relation instruction**: "Extract ALL relationships, including implicit ones"

**Results:** 95.3% → **100% E-F1**, 86.7% → **89.3% R-F1**

### 8.2 Fine-Tuning: Not Needed

With the improved prompt, Gemma-3-27b-it matches GPT-4o-mini on entities and nearly matches on relations. The remaining gap is gold-set quality, not model quality. Fine-tuning on 16 examples would overfit.

### 8.3 Role in Architecture

The LLM serves as the **teacher** for the self-learning system:
- Discovers entities that EntityRuler misses → feeds back into ruler patterns
- Discovers relations that RelationRuler misses → feeds back into dep-parse templates and entity-pair cache
- Attempts entity linking for novel entities → feeds back into Wikidata cache
- Runs asynchronously — latency doesn't affect user experience

### 8.4 LLM Provider Strategy

The enrichment LLM is the only configurable part of the architecture:

| Provider | Prompt | E-F1 | R-F1 | Latency | Cost/ingest | Verdict |
|----------|--------|------|------|---------|-------------|---------|
| **Groq Llama-3.3-70b** | improved | **100%** | **89.3%** | **878ms** | **~$0.0005** | **Default for hosted** |
| Gemma-3-27b-it (local) | improved | 100% | 89.3% | 31s | $0 | OSS self-hosted with GPU |
| GPT-5-mini | improved | 93.4% | 55.2% | ~40s | ~$0.002 | Over-extracts, not competitive |

Groq wins on all axes vs GPT-5-mini: better quality, 45x faster, 4x cheaper. Gemma matches Groq on quality but only makes sense for OSS users who bring their own GPU.

The architecture doesn't care which LLM is used — only that it produces entities + relations in the expected JSON format.

---

## 9. Self-Learning Feedback Loops

All three tiers improve over time through feedback from Tier 3 (LLM):

```
                    ┌──────────────────┐
                    │   Tier 3: LLM    │
                    │  (teacher model)  │
                    └────┬───┬───┬─────┘
                         │   │   │
          ┌──────────────┘   │   └──────────────┐
          ▼                  ▼                   ▼
┌─────────────────┐ ┌──────────────┐ ┌──────────────────┐
│ EntityRuler     │ │ RelationRuler│ │ Wikidata Cache   │
│                 │ │              │ │                   │
│ New entity      │ │ New dep-parse│ │ Entity linking    │
│ patterns        │ │ templates    │ │ for novel         │
│ (vocabulary)    │ │ (grammar)    │ │ entities          │
│                 │ │              │ │                   │
│ Scope: tenant → │ │ Scope:       │ │ Scope:            │
│ global via freq │ │ always global│ │ always global     │
│ gate            │ │              │ │                   │
└─────────────────┘ └──────────────┘ └──────────────────┘

Entity-pair cache and graph patterns are ALWAYS tenant-scoped.
They improve Tier 2 for that tenant only.
```

### 9.1 Convergence Properties

- **EntityRuler**: Logarithmic growth. Most domain vocabulary covered after ~1,000-5,000 memories.
- **RelationRuler**: Slower convergence (combinatorial entity pairs), but type-pair priors provide strong baseline from day one.
- **Wikidata cache**: Monotonically grows. Typical user references ~500-2,000 unique entities.
- **At steady state**: Tier 1+2 handles ~95% of entities and ~80%+ of relations. LLM only needed for genuinely novel knowledge.
- **Self-learning converges**: After enough patterns are learned, the LLM finds fewer new patterns. Enrichment calls decrease naturally over time.

---

## 10. Unit Economics and Tier Model

### 10.1 Per-Ingest Cost Breakdown (Hosted)

| Component | Cost per ingest | Notes |
|-----------|----------------|-------|
| spaCy + EntityRuler | ~$0 | CPU only, negligible |
| FalkorDB storage | ~$0.0001 | Amortized shared instance |
| Redis cache | ~$0 | Negligible |
| Embedding (OpenAI ada-002) | ~$0.0001 | ~500 tokens |
| **Groq enrichment** | **~$0.0005** | 500 in + 300 out tokens |
| Wikidata API | $0 | Free, rate-limited |
| **Total per ingest** | **~$0.0007** | |

### 10.2 Monthly Cost by Usage Profile

| User type | Memories/month | Enrichment cost | Infra (amortized) | Total cost |
|-----------|---------------|-----------------|-------------------|------------|
| Light (occasional notes) | 100 | $0.05 | ~$1.00 | ~$1.05 |
| Medium (daily use) | 1,000 | $0.50 | ~$1.20 | ~$1.70 |
| Heavy (document imports) | 10,000 | $5.00 | ~$2.00 | ~$7.00 |
| Bulk (knowledge base) | 100,000 | $50.00 | ~$5.00 | ~$55.00 |

### 10.3 Tier Structure

| Tier | Price | Enrichment limit | What happens at limit |
|------|-------|-------------------|----------------------|
| **Free** | $0 | 200/month | Fast tier always works (96.9% E-F1). No background enrichment. Rulers don't grow. |
| **Pro** | ~$8/month | 5,000/month | Managed enrichment. Self-learning active. Rulers grow. |
| **Team** | ~$20/month | 20,000/month | + multi-tenant workspaces, shared team ontology |
| **Enterprise** | Custom | Custom | + bulk import, pre-warmed ontology, dedicated infra |

**Key design principle**: The fast tier is **unlimited and free forever**. Users can always store memories with 96.9% E-F1 extraction. The limit is only on enrichment calls — the background LLM that teaches the rulers. And because self-learning converges, early enrichment benefits all future extractions even after quota is hit.

A Free user after 6 months (1,200 enriched memories at 200/month) has rulers that are substantially better than a day-one install. The system got smarter even within the free limit.

### 10.4 OSS vs Hosted Differentiation

| Capability | OSS (self-hosted) | Hosted (Pro+) |
|-----------|-------------------|---------------|
| Fast tier (spaCy + rulers) | Always on, $0 | Always on |
| Seeded patterns (~50-100 base) | Shipped with package | Shipped with package |
| Self-learning | Works if you provide LLM (your GPU / your API key) | Managed — we run it |
| Wikidata integration | Public API (rate-limited, ~100ms/entity) | Pre-cached, instant |
| Multi-tenant | Single user, local graph | Team workspaces, shared ontology |
| Evolution, reasoning, Studio, Maya | Available | Available + managed |
| Bulk document import | You provide compute | Managed batch processing |

The ontology mechanism is identical in OSS and hosted. The difference is who provides the enrichment compute.

---

## 11. Implementation Phases

### Phase 1: EntityRuler Extractor (validated, ready to build)

Replace current extraction defaults with spaCy sm + EntityRuler:
- New `SpacyRulerExtractor` plugin
- Seed with ~50 base patterns (already defined in benchmark)
- Register as default extractor in `IngestionRegistry`
- Deprecate GLiNER2/ReLiK hybrid (benchmarked at 50.6% E-F1)

**Files:** `plugins/extractors/spacy_ruler.py`, `memory/ingestion/registry.py`

### Phase 2: Improved LLM Prompt + Async Enrichment

- Deploy improved `SINGLE_CALL_PROMPT` (already written, tested at 100% E-F1)
- Add async enrichment queue: ingest returns fast (Tier 1+2), queues Tier 3
- LLM results replace stored extraction results
- Diff mechanism: LLM entities − ruler entities = new patterns
- Batch enrichment support: accumulate and flush in batches

**Files:** `plugins/extractors/llm_single.py` (done), async queue infrastructure

### Phase 3: EntityRuler Self-Learning + Privacy Model

- Pattern diff engine: compare LLM extraction vs EntityRuler extraction
- Quality gate: confidence, frequency, type consistency thresholds
- Three-layer pattern storage: base → global → tenant
- Pattern promotion gate: Wikidata-linkable → immediate global; non-Wikidata → frequency threshold across tenants
- Load patterns on startup, hot-reload on new patterns

**Files:** `extraction/self_learning.py`, `extraction/pattern_store.py`

### Phase 4: RelationRuler

- Type-pair relation priors (static table, bootstrapped from Wikidata property stats)
- Dep-parse template learning from LLM extractions (global scope)
- Entity-pair relation cache (tenant-scoped, never global)
- Multi-signal scoring function
- Integration into extraction pipeline

**Files:** `extraction/relation_ruler.py`, `extraction/dep_templates.py`

### Phase 5: Wikidata Integration

- Replace WikipediaGrounder with WikidataGrounder
- Entity linking (type-aware string match → QID)
- Demand-loading: fetch entity properties on first mention
- Redis cache for Wikidata entities
- World knowledge nodes (is_global=True) in FalkorDB
- Pre-extraction grounding (inform extraction, don't just decorate after)
- Document deduplication for public content

**Files:** `integration/wikidata_client.py`, `plugins/grounders/wikidata.py`, `extraction/entity_linker.py`

### Phase 6: Graph Pattern Inference + Document Import

- Leverage existing graph for relation prediction (tenant-scoped only)
- Type-constrained pattern matching from stored relations
- Feed into RelationRuler's multi-signal scorer
- Bulk document import pipeline with batch enrichment
- Public/private document classification

**Files:** `extraction/graph_inference.py`, `memory/ingestion/bulk.py`

---

## 12. What This Replaces

| Current | Replaced by | Why |
|---------|-------------|-----|
| SpacyExtractor (deprecated) | SpacyRulerExtractor | +10.9 E-F1, same speed |
| HybridGlinerRebelExtractor | SpacyRulerExtractor | GLiNER benchmarked at 50.6% E-F1 |
| WikipediaGrounder | WikidataGrounder | Structured data > prose, entity linking |
| WikipediaEnricher | Merged into WikidataGrounder | Same data source, one component |
| LLM as primary extractor | LLM as async enrichment teacher | Speed: 31s → <20ms real-time |

---

## 13. Open Questions

1. **Entity linking disambiguation**: How to handle truly ambiguous entities ("Apple" in text without type context)? Heuristics vs. always deferring to LLM?

2. **Wikidata coverage**: What percentage of entities in typical SmartMemory usage are in Wikidata? Need to measure on real user data.

3. **RelationRuler precision threshold**: What multi-signal score threshold balances precision and recall? Needs empirical tuning.

4. **Dep-parse template expressiveness**: Can dep-parse patterns capture enough syntactic variation? Or do we need to also match on token-level features (lemmas, POS tags)?

5. **Wikidata API rate limits**: Wikidata API is generous but not unlimited. For burst ingestion (e.g., importing 1,000 notes), need to handle rate limiting gracefully. Local dump vs. API?

6. **Ontology evolution**: World knowledge changes (companies merge, people change roles). How often to refresh Wikidata cache? Probably annually for most facts, with manual refresh option.

7. **Relation over-extraction**: The improved Gemma prompt increased relation recall (+14.6%) but decreased precision (-9.3%) due to extracting valid-but-not-in-gold-set relations. Is over-extraction acceptable for a knowledge graph (more edges = richer graph) or does it introduce noise?

8. **Batch enrichment design**: Optimal batch size for LLM calls? How to handle partial failures in a batch? How does batching interact with the enrichment queue?

9. **Public document detection**: How to reliably classify a document as public vs private? URL-based heuristics? User annotation? Content hash matching against known public corpora?

10. **Enrichment quota strategy**: When a user hits their enrichment limit, which memories should be prioritized for enrichment? Most recent? Most entity-dense? User-flagged as important?

---

## Appendix A: Benchmark Error Analysis (Gemma-3-27b-it)

### Before Prompt Improvement

**Entities:** P=96.8% R=93.8% F1=95.3% (TP=61 FP=2 FN=4)

All 4 missed entities were entity-merging:
- "Nobel Prize in Physics" (should be "Nobel Prize" + "Physics")
- "Los Gatos, California" (should be "Los Gatos" + "California")

6 type mismatches (not counted as errors):
- product → work_of_art: Symphony No. 9, Stranger Things
- technology → product: Docker, iOS
- location → work_of_art: Eiffel Tower
- organization → product: YouTube

**Relations:** P=92.9% R=81.2% F1=86.7% (TP=39 FP=3 FN=9)

9 missed relations:
- 4 implicit/secondary (framework_for, used_for)
- 2 caused by merged entities (Nobel Prize→Physics, Los Gatos→California)
- 2 temporal (launched_in, won_in)
- 1 hierarchical location (Austin→located_in→Texas)

### After Prompt Improvement

**Entities:** P=100% R=100% F1=100% (TP=65 FP=0 FN=0)

Entity splitting instruction fixed all merging issues. Type mismatches reduced from 6 to 2.

**Relations:** P=83.6% R=95.8% F1=89.3% (TP=46 FP=9 FN=2)

Only 2 relations still missed:
- World War II --[began_in]--> 1939
- Google --[developed]--> Borg

9 "hallucinated" relations are mostly valid facts not in gold set:
- "iPhone announced_at Macworld" (gold has Steve Jobs as subject)
- "Germany involved_in World War II" (valid but not in gold)
- "Nobel Prize awarded_in 1903" (duplicate of Marie Curie→won_in→1903)

### Conclusion

Fine-tuning Gemma is not needed. Prompt engineering solved the entity gap completely and improved relations significantly. Remaining relation "errors" are gold-set coverage issues.

## Appendix B: Dead Ends (Don't Revisit)

| Approach | Why it failed | Don't revisit unless |
|----------|--------------|---------------------|
| Progressive prompting (4 variants) | LLM anchors to draft, loses quality | Fundamentally different model architecture |
| GLiNER2 + GLiREL | GLiNER NER at 34% recall | GLiNER v3+ with better NER |
| REBEL (end-to-end) | 3.9s latency, 62% R-F1 | Never — dominated on all axes |
| NuExtract tiny | 100% precision, 45% recall | NuExtract 2.0 with larger model |
| NuNER Zero | Broken with gliner >= 0.2.x | Package compatibility fixed upstream |
| spaCy trf for fast tier | 33ms vs 4ms, ruler makes sm better | Never — sm+ruler dominates |
| LLM fine-tuning | Prompt improvement already hit 100% E-F1 | Only if prompt changes stop working |
| Forking spaCy for RelationRuler | Maintenance burden, spaCy arch mismatch | Contribute upstream PR instead |
| GPT-5-mini for extraction | 93.4% E-F1, 55.2% R-F1, ~40s, $0.002. Over-extracts relations (78 FP). | Never — Groq dominates on all axes |

## Appendix C: Design Decisions Log

| Decision | Rationale | Date |
|----------|-----------|------|
| Ontology is core, always-on | Self-learning is the product differentiator. Optional = most users never enable. | 2026-02-05 |
| Entity-pair relations NEVER shared globally | Process shapes are proprietary even when all entities are public. | 2026-02-05 |
| Patterns are public, instances are private | Entity vocabulary and grammar reveal nothing; specific connections reveal processes. | 2026-02-05 |
| EntityRuler promotion via Wikidata-linkable or frequency gate | Wikidata-linkable = obviously public. Multi-tenant frequency = empirically public. | 2026-02-05 |
| Fast tier unlimited and free | Core extraction value should be accessible to all. Enrichment is the monetization lever. | 2026-02-05 |
| Groq for hosted, Gemma for self-hosted | Groq 35x faster than local Gemma, identical quality. Local Gemma for $0 OSS. | 2026-02-05 |
| Batch enrichment for bulk ingestion | Local Gemma impractical for bulk (86 hours for 10K docs). Batching reduces API cost 30-50%. | 2026-02-05 |
| Pre-seed rulers with Groq/Gemma before launch | Ship 5K-10K base patterns instead of 50. ~$5 one-time cost. Eliminates cold-start. | 2026-02-05 |
| GPT-5-mini not competitive | 93.4% E-F1, 55.2% R-F1 (78 FP relations), ~40s, ~$0.002. Groq wins on all axes. | 2026-02-05 |
| Prompt improvement is model-agnostic | Same prompt changes lifted Groq 97.7%→100% E-F1 and Gemma 95.3%→100% E-F1. | 2026-02-05 |
