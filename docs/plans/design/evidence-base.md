# Evidence Base

**Parent:** [Implementation Plan](../2026-02-05-implementation-plan.md)
**Status:** Reference document (benchmark data + research findings)

---

## 1. Benchmark Summary

53 extraction configurations benchmarked on a curated evaluation corpus. Full results in `tests/benchmark_model_quality.py` and `docs/plans/2026-02-05-extraction-benchmark-findings.md`.

### 1.1 Key Results

| Configuration | E-F1 | R-F1 | Latency | Cost | Role |
|---------------|------|------|---------|------|------|
| **spaCy sm + EntityRuler** | 96.9% | 65.1% | 4ms | $0 | Fast tier (production) |
| **spaCy trf + EntityRuler** | 94.6% | 63.2% | 33ms | $0 | Rejected (sm+ruler better) |
| **Groq Llama-3.3-70b** | 97.7% | 85-88% | 740ms | ~$0.0003 | Enrichment tier (production) |
| **Gemma-3-27b-it** | 95.3% | 86.7% | 31s | $0 | Zero-cost enrichment (validated) |
| **GPT-4o-mini** | 100% | 89.3% | 1.2s | ~$0.001 | Benchmark reference only |

### 1.2 EntityRuler Impact

The EntityRuler component is the single most important architectural decision:

| Configuration | E-F1 | R-F1 | Delta from base |
|---------------|------|------|-----------------|
| spaCy sm (no ruler) | 86.0% | 48.7% | — |
| spaCy sm + EntityRuler | 96.9% | 65.1% | +10.9 E-F1, +16.4 R-F1 |
| spaCy trf (no ruler) | 88.2% | 51.3% | — |
| spaCy trf + EntityRuler | 94.6% | 63.2% | +6.4 E-F1, +11.9 R-F1 |

Key findings:
- EntityRuler adds +10.9 E-F1 and +16.4 R-F1 at zero latency cost
- sm + ruler (96.9%) beats trf + ruler (94.6%) — ruler patterns are more precise than statistical overrides
- sm + ruler is within 0.8% of Groq LLM on entities at 200x speed

### 1.3 Model Quality Ranking

**Cloud models (extraction quality):**
1. GPT-4o-mini: 100% E-F1, 89.3% R-F1 ($0.001/call)
2. Groq Llama-3.3-70b: 97.7% E-F1, 85-88% R-F1 ($0.0003/call)
3. Groq Llama-3.1-8b: 92.1% E-F1, 74.2% R-F1 ($0.0001/call)

**Local models (extraction quality):**
1. Gemma-3-27b-it: 95.3% E-F1, 86.7% R-F1 (31s, $0)
2. QWQ-32b: 93.8% E-F1, 82.1% R-F1 (45s, $0)
3. Hermes-3-70b: 87.3% E-F1, 78.4% R-F1 (60s, $0)

**Bigger is NOT better for local extraction**: Gemma-27B > QWQ-56B > Hermes-70B.

---

## 2. Anti-Patterns (Proven Failures)

These approaches were tested and proven not to work. Do not revisit.

### 2.1 Progressive Prompting

Four variants tested: LLM refining spaCy draft, LLM adding to spaCy results, LLM validating spaCy results, LLM filling gaps.

| Variant | E-F1 | R-F1 | vs. LLM Standalone |
|---------|------|------|--------------------|
| Refine | 96.2% | 78.5% | -3.8 E, -10.8 R |
| Add | 95.1% | 81.2% | -4.9 E, -8.1 R |
| Validate | 97.7% | 85.0% | -2.3 E, -4.3 R |
| Fill gaps | 98.5% | 86.8% | -1.5 E, -2.5 R |
| **LLM standalone** | **100%** | **89.3%** | — |

**Root cause:** Anchoring bias. The LLM anchors to the draft and fails to correct errors or discover entities the draft missed. All 4 variants worse than standalone.

**Decision:** LLM always extracts from scratch. Never feed spaCy output as a "draft."

### 2.2 spaCy trf for Fast Tier

| Model | E-F1 | Latency | Verdict |
|-------|------|---------|---------|
| sm + ruler | 96.9% | 4ms | Winner |
| trf + ruler | 94.6% | 33ms | 8x slower AND worse |

trf model's statistical predictions conflict with ruler patterns. sm model defers to ruler more cleanly.

### 2.3 Local Specialized Models

| Model | Task | Quality | Latency | Issue |
|-------|------|---------|---------|-------|
| REBEL | End-to-end RE | 62% R-F1 | 3.9s | Slow and bad |
| GLiNER2 | NER | 34% recall | 1.2s | Misses most entities |
| GLiREL | RE | N/A | N/A | Useless without good NER |
| NuExtract | NER | Low recall | 2.1s | Extraction-focused but misses |
| NuNER | NER | N/A | N/A | Incompatible with pipeline |

**Decision:** No specialized local models. spaCy + EntityRuler for fast tier, LLMs for enrichment.

### 2.4 RelationRuler (Dep-Parse Templates)

Dep-parse relation extraction has a fundamental ceiling:

| Approach | R-F1 | Issue |
|----------|------|-------|
| Dep-parse SVO | ~50% | Only captures subject-verb-object |
| Dep-parse + rules | ~65% | Still misses semantic relations |
| LLM standalone | 85-88% | Semantic reasoning required |

Relations are semantic, not syntactic. "X developed Y" and "X is the creator of Y" and "Y was built by X" all express the same relation but have completely different syntax.

**Decision:** No dep-parse relation templates. Type-pair priors (cheap lookup) for filtering; LLM for actual relation extraction.

---

## 3. Research Foundations

### 3.1 NELL (Never-Ending Language Learner)

CMU project, 12+ years continuous operation. Key lessons:

**Success metrics:**
- 75% of categories reached 90-99% precision autonomously
- 25% degraded to 25-60% precision over time
- System could not predict which categories would degrade

**Error propagation:**
- #1 failure mode: single bad promotion cascades
- "New York" classified as Person → creates Person-Location relations → poisons pattern extraction
- Requires mutual exclusion constraints as primary defense

**Best quality gate:**
- Multi-source agreement (2+ independent extractors must agree)
- Single-source discoveries are high-noise
- Dramatically reduces false promotions

**Human-in-the-loop:**
- 5 minutes/day of human review caught 95%+ of error-prone promotions
- Disproportionate ROI for minimal effort
- SmartMemory defers this to Phase 4+ (Studio review UI)

### 3.2 Applied NELL Lessons

| NELL Finding | SmartMemory Mitigation |
|-------------|------------------------|
| Error propagation | Mutual exclusion constraints in ontology |
| Category degradation | Type consistency gate (>80% same classification) |
| Multi-source agreement | EntityRuler + LLM + Wikidata must agree |
| HITL ROI | Deferred review UI in Studio (Phase 4+) |

### 3.3 EntityRuler Research

spaCy's EntityRuler is a pattern-matching component added to the NER pipeline. Key properties:

- **Exact match patterns:** Zero false positives for known entities
- **Rule-based precedence:** Ruler matches override statistical model predictions
- **Zero latency cost:** Pattern matching adds <1ms to pipeline
- **Hot-reloadable:** Patterns can be updated without model retraining
- **Composable:** Works with any spaCy model (sm, md, lg, trf)

SmartMemory's innovation: self-learning feedback loop where LLM-discovered entities feed back into EntityRuler patterns, making the fast tier progressively better.

### 3.4 Ontology Design Principles

**Separate TBox and ABox:**
- TBox (terminological box): Entity types, relation types, constraints
- ABox (assertional box): Actual entity instances, relation instances
- SmartMemory maps these to separate FalkorDB graphs per workspace

**Open vs. Closed World:**
- Open world: unknown entities are possible (default for ABox)
- Closed world: only declared types exist (default for TBox seed types)
- SmartMemory: hybrid — TBox is closed for seed types, open for provisional types

**Three-tier type status (seed/provisional/confirmed):**
- Borrowed from ontology learning literature
- Seed types provide stable foundation
- Provisional types enable discovery without commitment
- Confirmation gates prevent quality degradation

---

## 4. Self-Learning Convergence Model

### 4.1 Expected Learning Curve

Based on domain vocabulary analysis and benchmark extrapolation:

```
Ruler hit rate (%)
100 |                              ___________________
    |                           __/
    |                        __/
    |                     __/
 97 |____________________/
    |__/
    |
    +--+--------+--------+--------+--------+--------+--
     0    1K      5K      10K     20K     50K
              Memories processed
```

| Milestone | Memories | Ruler Hit Rate | Enrichment Rate |
|-----------|----------|----------------|-----------------|
| Cold start | 0 | ~70% (seed only) | 100% |
| Early learning | 1K | ~90% | ~30% |
| Rapid growth | 5K | ~95% | ~15% |
| Near convergence | 10K | ~97% | ~8% |
| Steady state | 20K+ | ~99% | ~3% |

### 4.2 Cost Projection

As ruler hit rate increases, enrichment tier calls decrease:

| Memories | Enrichment Rate | LLM Calls/1K | Cost/1K (Groq) |
|----------|----------------|---------------|-----------------|
| 0-1K | 100% | 1,000 | $0.30 |
| 1K-5K | 30% | 300 | $0.09 |
| 5K-10K | 15% | 150 | $0.045 |
| 10K-20K | 8% | 80 | $0.024 |
| 20K+ | 3% | 30 | $0.009 |

At steady state: ~$0.009 per 1,000 memories (effectively free).

### 4.3 Gemma Zero-Cost Path

For cost-sensitive deployments, Gemma-3-27b-it replaces Groq:

| Metric | Groq | Gemma-3-27b-it |
|--------|------|----------------|
| E-F1 | 97.7% | 95.3% |
| R-F1 | 85-88% | 86.7% |
| Latency | 740ms | 31s |
| Cost | ~$0.0003/call | $0 |

Gemma slightly better on relations, slightly worse on entities, 40x slower but free. Suitable for background enrichment where latency doesn't matter.

---

## 5. Existing Code Inventory

Code that will be absorbed or replaced by the new pipeline:

### 5.1 Replaced (Delete After Migration)

| File | Lines | Replacement |
|------|-------|-------------|
| `memory/pipeline/ingestion_flow.py` | 473 | `Pipeline.run(text, config)` |
| `memory/pipeline/fast_ingestion_flow.py` | 502 | Deleted (async is config flag) |
| Studio `ExtractorPipeline` | 491 | Deleted (Studio calls core) |
| Duplicated normalization | ~130 | Consolidated in stages |
| **Total** | **~1,600** | **Single pipeline** |

### 5.2 Kept (Wrapped as StageCommands)

| Component | Role | Notes |
|-----------|------|-------|
| `HybridExtractor` | Entity extraction | Wrapped by entity_ruler + llm_extract stages |
| `ReasoningExtractor` | Reasoning traces | Wrapped by evolve stage |
| `DecisionExtractor` | Decision parsing | Wrapped by classify stage |
| `BasicEnricher` | Enrichment | Wrapped by enrich stage |
| `WikipediaGrounder` | Wikidata lookup | Part of enrich stage |
| All evolvers (9) | Memory evolution | Wrapped by evolve stage |

### 5.3 Reused Infrastructure

| Component | Location | Reuse |
|-----------|----------|-------|
| `RedisStreamQueue` | `memory/pipeline/redis_stream_queue.py` | Generalized for all stages |
| `PromptProvider` | `memory/pipeline/prompt_provider.py` | Used by llm_extract stage |
| `SmartGraph` | `graph/smartgraph.py` | Used by store, link stages |
| `ScopeProvider` | `scope_provider.py` | Injected at pipeline start |
| `SchemaValidator` | `graph/models/schema_validator.py` | Used by ontology_constrain |

---

## 6. Use Cases as Parameters

25 use cases analyzed and mapped to a 10-parameter space. Users don't pick a use case; PipelineConfig parameters are set (or auto-tuned) to match their needs.

### 6.1 Parameter Space

| Parameter | Type | Range | Default |
|-----------|------|-------|---------|
| `domain_vocabulary` | str | None, "medical", "legal", ... | None |
| `relation_depth` | int | 1-5 | 3 |
| `temporal_sensitivity` | float | 0.0-1.0 | 0.5 |
| `contradiction_tolerance` | float | 0.0-1.0 | 0.3 |
| `confidence_requirement` | float | 0.5-0.99 | 0.7 |
| `extraction.enrichment_tier` | str | "groq", "gemma", None | "groq" |
| `extraction.self_learning_enabled` | bool | true/false | true |
| `enrich.wikidata.enabled` | bool | true/false | true |
| `evolve.enabled_evolvers` | list | subset of 9 evolvers | 4 default |
| `store.dedup_threshold` | float | 0.8-0.99 | 0.95 |

### 6.2 Example Use Case Mappings

| Use Case | Key Parameters |
|----------|---------------|
| Personal knowledge management | Default (balanced extraction) |
| Medical notes | `confidence_requirement=0.9`, `domain_vocabulary="medical"` |
| Legal research | `contradiction_tolerance=0.1`, `confidence_requirement=0.95` |
| Meeting notes | `temporal_sensitivity=0.8`, episodic evolvers enabled |
| Bulk import (Wikipedia) | `enrichment_tier=None`, `self_learning_enabled=False` |
| High-precision research | `confidence_requirement=0.95`, `enrichment_tier="groq"` |
| Cost-optimized | `enrichment_tier="gemma"` or `enrichment_tier=None` |

Full mapping: `docs/plans/2026-02-05-use-cases-reference.md` (25 use cases).

---

## 7. Decision Log

All architectural decisions from the Q0-Q14 design review:

| ID | Decision | Rationale |
|----|----------|-----------|
| Q0 | Ontology is foundational, not optional | Prevents semantic drift, enables self-learning |
| Q1 | `simplify` = ONE stage with 4 flags sharing one dep-parse | Performance: single parse, 4 transforms |
| Q2 | Separate FalkorDB graphs (ontology + data) | Zero tenant bleed, clean backup/restore |
| Q3 | Three-layer TBox: Global, Tenant Soft-TBox, Tenant ABox | Global for shared vocabulary, tenant for learned |
| Q4 | Statistical pre-filter for promotion (Option C) | NELL research: statistics handle 80-90% of candidates |
| Q5 | Dual-axis type system (memory types + entity types) | Orthogonal classification, bridge types connect them |
| Q6 | 14 seed entity types (8 generic + 6 SmartMemory-native) | Minimum viable, discoverable types emerge from usage |
| Q7 | Entity-pair relations NEVER shared globally | Privacy: relation instances reveal proprietary architecture |
| Q8 | Wikidata: REST (95%) + SPARQL (5%), write-through cache | REST for entity lookup, SPARQL only for type hierarchies |
| Q9 | Entity-pair cache = graph edges, not separate store | Avoid data duplication, graph IS the cache |
| Q10 | No RelationRuler dep-parse (use type-pair priors + LLM) | Dep-parse ceiling ~65% R-F1, semantic gap too large |
| Q11 | PipelineConfig as nested Pydantic with named configs | Type-safe, serializable, per-workspace |
| Q12 | Three-layer prompt management (prompts.json → MongoDB → config) | Layered overrides, Studio-editable |
| Q13 | Event-bus via Redis Streams with consumer groups | Horizontal scaling, DLQ, exactly-once via acknowledgment |
| Q14 | Metrics via Redis Streams to Insights dashboard | Fire-and-forget, bounded streams, time-bucketed aggregation |

---

## See Also

- [Pipeline Architecture](pipeline-architecture.md) — Technical spec for pipeline components
- [Ontology Model](ontology-model.md) — Type system, promotion, privacy
- [Extraction Stages](extraction-stages.md) — Per-stage specifications
- [Self-Learning](self-learning.md) — Promotion flow, convergence
- [Service & API](service-api.md) — Routes, transport, prompts
- [Metrics & Observability](metrics-observability.md) — Dashboard, alerting
- [Benchmark Data](../2026-02-05-extraction-benchmark-findings.md) — Raw benchmark results
- [Use Cases Reference](../2026-02-05-use-cases-reference.md) — 25 use cases analysis
- [Implementation Plan](../2026-02-05-implementation-plan.md) — Phase-by-phase build plan
