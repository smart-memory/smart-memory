# Two-Tier Extraction Pipeline: Fast Local + Async LLM Refinement

**Date:** 2026-02-05
**Status:** Design
**Author:** SmartMemory Team

## Problem

The current extraction pipeline runs LLM extraction synchronously on every `ingest()` call. This creates three scaling bottlenecks:

1. **Throughput**: LLM APIs are rate-limited (Groq: ~30 RPM free tier). At 1,000 concurrent users (~17 ingestions/sec), LLM extraction cannot keep up.
2. **Latency**: Even fast LLMs (Groq llama-3.3-70b at 561ms) add half a second to every ingest. At scale, this compounds.
3. **Cost**: At 1M ingestions/month, LLM extraction costs $300-$1,000/month. This eats into product margin.

Meanwhile, local extractors (spaCy trf: 35ms, 89% E-F1, 64% R-F1) are fast and free but produce lower-quality relations.

## Solution

**Always run fast local extraction first. Optionally refine with LLM asynchronously.**

The LLM refinement prompt includes the local extraction results as a draft, changing the LLM's job from generation to verification/correction. This is faster, cheaper, and more accurate than extraction from scratch.

```
SYNC PATH (every ingest, <50ms extraction):
  text → local extractor → full pipeline (store, embed, link, enrich, evolve) → return

ASYNC PATH (queued, rate-limited, paid accounts):
  text + local results → LLM refinement → merge into existing graph → enrich NEW entities only
```

## Architecture

### Current Pipeline (8 stages)

```
1. Input Adaptation    → MemoryItem
1.5 Coreference        → resolved text
2. Extraction          → entities[], relations[]
3. Storage             → graph nodes
4. Relationships       → relation edges
5. Linking             → cross-references
6. Vector/Graph        → embeddings
7. Enrichment          → sentiment, topics, temporal, etc.
7.5 Grounding          → Wikipedia provenance
8. Grounding Edges     → GROUNDED_IN edges
```

### Modified Pipeline

```
1. Input Adaptation    → MemoryItem
1.5 Coreference        → resolved text
2a. LOCAL Extraction   → entities[], relations[]  (spaCy/GLiNER2/best-local)
3-8. Full pipeline     → graph built, enriched, embedded
    ↓
    Queue LLM refinement job (if enabled for account)
    ↓
2b. LLM Refinement    → corrections[], additions[]  (async, from queue)
9. Merge               → update graph with LLM improvements
10. Incremental Enrich → enrich only NEW entities from LLM
```

Stages 1-8 are unchanged. Stages 2b, 9, 10 are new.

### Key Principle: Refinement, Not Replacement

The LLM does NOT re-extract from scratch. It receives the local extraction as a draft and outputs corrections and additions. This means:

- **Fewer output tokens** → faster, cheaper LLM calls
- **Better accuracy** → verification is easier than generation
- **Trivial merge** → LLM uses the same entity names as the local extractor
- **Works with smaller models** → "review and correct" is simpler than "extract everything"

## Detailed Design

### 2a. Local Extraction (Sync Path)

The local extractor is selected by configuration. "spaCy" is a placeholder for whatever benchmarks best for fast local extraction. Candidates:

| Extractor | E-F1 | R-F1 | Latency | Status |
|-----------|------|------|---------|--------|
| spaCy trf | 88.9% | 64.2% | 35ms | Benchmarked |
| spaCy sm | 86.0% | 48.7% | 5ms | Benchmarked |
| GLiNER2 (fastino/gliner2-base-v1) | ? | ? | ? | **Untested** |
| GLiNER2 + ReLiK hybrid | ? | ? | ? | **Untested** |
| NuMind NuNER | ? | ? | ? | **Untested** |
| NuExtract | ? | ? | ? | **Untested** |

The local extractor MUST:
- Run in <100ms for typical text (<500 words)
- Require no external API calls
- Produce entities with names and types
- Produce relations as (subject, predicate, object) triples

**Configuration:**

```python
# In ExtractionConfig
@dataclass
class ExtractionConfig:
    # Existing fields...
    local_extractor: str = "spacy_trf"        # Fast local extractor for sync path
    llm_refinement_enabled: bool = False       # Enable async LLM refinement
    llm_refinement_model: str = "groq"         # LLM model for refinement
    llm_refinement_priority: str = "normal"    # normal | high | low
```

### 2b. LLM Refinement Prompt

The refinement prompt includes the local extraction as context:

```
You are reviewing an automated knowledge graph extraction. The initial extraction
was performed by a fast local model and may have errors or omissions.

TEXT:
{text}

INITIAL EXTRACTION (automated, may contain errors):
Entities:
{entities_json}

Relations:
{relations_json}

TASK: Review and improve this extraction.
1. Verify each entity - remove false positives, fix types
2. Add missing entities the local model missed
3. Verify each relation - remove incorrect ones
4. Add missing relations between entities
5. Use EXACT entity names from the initial extraction when referring to existing entities
6. For NEW entities, use the most specific canonical name

Return JSON with the COMPLETE corrected extraction:
{
  "entities": [
    {"name": "exact name", "entity_type": "type", "confidence": 0.95,
     "status": "verified|corrected|new|removed"}
  ],
  "relations": [
    {"subject": "entity name", "predicate": "relationship", "object": "entity name",
     "status": "verified|corrected|new|removed"}
  ]
}
```

The `status` field on each entity/relation enables precise merge:
- `verified`: local extraction was correct, no changes needed
- `corrected`: local extraction had an error (wrong type, wrong name), corrected
- `new`: LLM found something the local extractor missed
- `removed`: local extraction was a false positive, should be deleted

### 9. Merge Strategy

When LLM refinement results arrive, merge into the existing graph:

```python
def merge_llm_refinement(memory_id: str, llm_result: dict, graph: SmartGraph):
    """Merge LLM refinement results into existing graph for a memory."""

    existing_entities = graph.get_entities_for_memory(memory_id)
    existing_relations = graph.get_relations_for_memory(memory_id)

    for entity in llm_result["entities"]:
        match entity["status"]:
            case "verified":
                # Update confidence score, keep everything else
                update_entity_confidence(entity, confidence=entity["confidence"])

            case "corrected":
                # Update entity type or name in-place
                matched = match_entity(entity, existing_entities)
                if matched:
                    update_entity(matched, new_type=entity["entity_type"],
                                  new_name=entity["name"])

            case "new":
                # Create new entity node, run enrichers on it
                new_id = create_entity(entity)
                new_entity_ids.append(new_id)

            case "removed":
                # Mark as low-confidence or delete
                matched = match_entity(entity, existing_entities)
                if matched:
                    mark_low_confidence(matched)

    # Same pattern for relations: verified/corrected/new/removed
    for relation in llm_result["relations"]:
        # ... similar merge logic

    return new_entity_ids  # For incremental enrichment
```

Entity matching uses `get_canonical_key()` from `smartmemory/utils/deduplication.py` for fuzzy matching between local and LLM entity names.

### 10. Incremental Enrichment

After merge, only NEW entities (status="new") need enrichment:

```python
new_entity_ids = merge_llm_refinement(memory_id, llm_result, graph)

if new_entity_ids:
    # Run enrichers only on new entities
    for enricher in enrichment_pipeline.get_enrichers():
        enricher.enrich_entities(new_entity_ids)

    # Run linkers to connect new entities to existing graph
    linking.link_entities(new_entity_ids)

    # Run grounders on new entities
    grounder.ground_entities(new_entity_ids)
```

Verified and corrected entities keep their existing enrichment. No wasted work.

### Queue Integration

Uses existing `RedisStreamQueue` infrastructure:

```python
# In ingest() sync path, after stage 8:
if extraction_config.llm_refinement_enabled:
    q = RedisStreamQueue.for_extraction_refinement()
    q.enqueue({
        "job_type": "llm_refine",
        "memory_id": item_id,
        "text": resolved_text,
        "local_extraction": {
            "entities": local_entities_json,
            "relations": local_relations_json,
        },
        "model": extraction_config.llm_refinement_model,
        "priority": extraction_config.llm_refinement_priority,
    })
```

New queue stream: `smartmemory:jobs:extraction_refinement:{namespace}`

### Background Worker

```python
class ExtractionRefinementWorker:
    """Consumes LLM refinement jobs from queue."""

    def __init__(self, queue: RedisStreamQueue, rate_limiter: RateLimiter):
        self.queue = queue
        self.rate_limiter = rate_limiter

    def process_job(self, job: dict):
        # 1. Build refinement prompt with local extraction
        prompt = build_refinement_prompt(job["text"], job["local_extraction"])

        # 2. Call LLM (rate-limited)
        self.rate_limiter.wait()
        llm_result = call_llm(prompt, model=job["model"])

        # 3. Merge into graph
        new_ids = merge_llm_refinement(job["memory_id"], llm_result, graph)

        # 4. Incremental enrichment on new entities
        if new_ids:
            enrich_entities(new_ids)

        # 5. Update memory status
        memory.update_status("llm_refined", notes=f"Added {len(new_ids)} entities")
```

### Status Model

Extended status tracking on MemoryItem:

| Status | Meaning |
|--------|---------|
| `created` | Raw text stored |
| `extracted` | Local extraction complete, full pipeline done |
| `enriched` | Enrichment complete |
| `llm_queued` | LLM refinement job queued |
| `llm_refined` | LLM refinement merged, incremental enrichment done |

The `metadata` dict tracks extraction provenance:

```python
metadata = {
    "extraction_source": "spacy_trf",          # Which local extractor
    "extraction_status": "extracted",           # Current extraction state
    "llm_refinement_status": "queued",          # pending | queued | complete | skipped
    "llm_refinement_model": "groq/llama-3.3-70b-versatile",
    "local_entity_count": 3,
    "llm_entity_additions": 2,                  # After refinement
    "llm_entity_corrections": 1,
    "llm_relation_additions": 3,
}
```

## Product Integration

### Account Tiers

| Tier | Local Extraction | LLM Refinement | Quality |
|------|-----------------|----------------|---------|
| Free | Best local (always) | No | ~64-89% R-F1 |
| Pro | Best local (always) | Yes (queued, shared rate limit) | ~87% R-F1 |
| Enterprise | Best local (always) | Yes (priority, dedicated rate limit) | ~91% R-F1 |

### Configuration

Per-workspace setting in service layer:

```python
# In smart-memory-service, workspace config
workspace_config = {
    "extraction": {
        "local_extractor": "spacy_trf",
        "llm_refinement_enabled": True,         # Paid feature
        "llm_refinement_model": "groq",
        "llm_refinement_rate_limit": 30,         # RPM per workspace
    }
}
```

### API Response

```json
// POST /memory/ingest
{
    "item_id": "mem_abc123",
    "extraction_status": "extracted",
    "llm_refinement": "queued",
    "entities_found": 3,
    "relations_found": 1
}

// Later, via webhook or polling:
{
    "item_id": "mem_abc123",
    "extraction_status": "llm_refined",
    "entities_found": 5,
    "relations_found": 4,
    "llm_additions": {"entities": 2, "relations": 3, "corrections": 1}
}
```

## Benchmark-Driven Decisions

### What We Know (benchmarked, 16 tests each)

Best cloud LLM extractors (for refinement step):
- GPT-4o-mini: 100% E-F1, 91.3% R-F1, 4.3s, ~$0.001/call
- Groq llama-3.3-70b: 97.7% E-F1, 87.1% R-F1, 561ms, ~$0.0003/call
- Groq kimi-k2-instruct-0905: 95.4% E-F1, 88.4% R-F1, 3.7s

Best local extractors (for sync path):
- spaCy trf: 88.9% E-F1, 64.2% R-F1, 35ms
- spaCy sm: 86.0% E-F1, 48.7% R-F1, 5ms

### What We Don't Know Yet (needs benchmarking)

- GLiNER2 (`fastino/gliner2-base-v1`) - in codebase, never benchmarked
- GLiNER2 + ReLiK hybrid - in codebase, never benchmarked
- NuMind NuNER / NuExtract - not in codebase
- **Refinement prompt quality** - does LLM-refining-spaCy beat LLM-from-scratch?

### Required Benchmark Before Implementation

Add to `tests/benchmark_model_quality.py`:

1. **GLiNER2 standalone** - entity extraction quality and speed
2. **GLiNER2 + ReLiK hybrid** - full extraction quality
3. **Refinement prompt benchmark** - spaCy draft + LLM refinement vs LLM from scratch:
   - Speed comparison (output tokens, latency)
   - Quality comparison (E-F1, R-F1)
   - Cost comparison (token count)

The refinement benchmark is critical. If refinement is not faster or better than from-scratch, the design simplifies to just async LLM extraction without the local draft.

## Implementation Plan

### Phase 1: Benchmark Missing Extractors (1-2 days)
- Add GLiNER2, GLiNER2+ReLiK, NuMind tools to benchmark
- Add refinement prompt benchmark (spaCy + LLM vs LLM alone)
- Determine best local extractor

### Phase 2: Refinement Prompt & Merge Logic (2-3 days)
- New `LLMRefinementExtractor` plugin
- Refinement prompt with draft extraction input
- Merge logic with entity matching via canonical keys
- Incremental enrichment for new entities only

### Phase 3: Queue Integration (1-2 days)
- New Redis stream for refinement jobs
- `ExtractionRefinementWorker` background consumer
- Rate limiter per workspace
- Status tracking on MemoryItem metadata

### Phase 4: Pipeline Integration (2-3 days)
- Modify `MemoryIngestionFlow` to queue refinement after stage 8
- Add `llm_refinement_enabled` to ExtractionConfig
- Ensure local extractor is always the sync path default
- Update extraction fallback chain: local first, LLM refinement async

### Phase 5: Service Layer (1-2 days)
- Workspace-level LLM refinement configuration
- API response includes refinement status
- Webhook/polling for refinement completion
- Account tier enforcement

### Phase 6: Testing & Docs (1-2 days)
- Integration tests for full two-tier flow
- Load tests for queue throughput
- Update docs, CHANGELOG, README

**Total estimate: 8-14 days**

## Files Modified

| File | Change |
|------|--------|
| `smartmemory/memory/pipeline/config.py` | Add `local_extractor`, `llm_refinement_*` to ExtractionConfig |
| `smartmemory/memory/ingestion/flow.py` | Queue refinement job after stage 8 |
| `smartmemory/plugins/extractors/llm_refinement.py` | **NEW** - LLM refinement extractor with draft prompt |
| `smartmemory/memory/pipeline/merge.py` | **NEW** - Merge logic for refinement results |
| `smartmemory/memory/pipeline/refinement_worker.py` | **NEW** - Background queue consumer |
| `smartmemory/observability/events.py` | Add `RedisStreamQueue.for_extraction_refinement()` |
| `smartmemory/models/memory_item.py` | Document refinement metadata fields |
| `memory_service/api/routes/crud.py` | Return refinement status in API response |
| `tests/benchmark_model_quality.py` | Add refinement prompt benchmark |

## Open Questions

1. **Should removed entities be deleted or just low-confidence?** Keeping them with low confidence is safer (LLM might be wrong about removal). Deletion is cleaner.

2. **What if LLM refinement contradicts enrichment?** E.g., local extractor finds "Einstein" → enricher adds Wikipedia link → LLM says "Einstein" was a false positive. Do we undo the Wikipedia link? Proposal: mark low-confidence, don't delete enrichment.

3. **Retry strategy for failed refinement jobs?** Exponential backoff, max 3 retries, then DLQ. Use existing RedisStreamQueue DLQ support.

4. **Should refinement be retriggerable?** If user re-edits the memory text, should we re-run refinement? Yes, but cancel any pending refinement job first.

5. **Ontology-constrained refinement?** Future enhancement: include SmartMemory's edge schema in the refinement prompt to constrain relation types. Deferred to separate design.
