# Self-Learning Ontology System

**Date:** 2026-02-06
**Status:** Design specification
**Audience:** Implementation engineers

---

## 1. Overview

SmartMemory's self-learning loop enables the fast extraction tier to improve continuously by learning from LLM discoveries. The system operates on a simple feedback mechanism: LLM extracts entities and relations from user text, differences are computed against what the fast tier (EntityRuler + RelationRuler) found, high-quality patterns are promoted to the rulers, and the fast tier improves for all future ingestions.

### 1.1 Core Mechanism

```
Text arrives
    │
    ▼
Fast tier: spaCy sm + EntityRuler (4ms, 96.9% E-F1)
    │
    ▼
Store results, return to user (<20ms)
    │
    ▼
Background: LLM enrichment (31s local / 740ms Groq, 100% E-F1)
    │
    ▼
Diff: LLM entities - Ruler entities = new patterns
    │
    ▼
Quality gate: confidence > 0.8, frequency > 1, type consistency > 80%
    │
    ▼
Promote patterns to EntityRuler
    │
    ▼
Ruler grows → fast tier improves → LLM needed less frequently
```

### 1.2 Convergence Timeline

The system follows a predictable improvement curve as patterns accumulate:

| Memories processed | Ruler coverage | Fast tier quality | LLM enrichment rate |
|-------------------|----------------|-------------------|---------------------|
| Day 1 (0) | Seeded patterns (~5K base) | 96.9% E-F1 | 100% (all memories) |
| ~1K | Most common domain entities | ~98% E-F1 | ~60% |
| ~5K | 95% of domain vocabulary | ~99% E-F1 | ~20% |
| ~10K+ | 99%+ coverage | Near LLM parity | <10% (novel only) |

Convergence is logarithmic. Most learning happens in the first 1,000 memories. By 10,000 memories, the system has learned the user's domain vocabulary and the LLM is only needed for genuinely novel entities.

---

## 2. Promotion Flow

### 2.1 Three-Tier Type Status

Entity types progress through three lifecycle states:

```
┌──────────────────────────────────────────────────────────────┐
│  SEED: Pre-installed patterns shipped with SmartMemory       │
│                                                               │
│  - Base ~5K patterns from diverse corpus extraction          │
│  - Global scope, loaded from {install_dir}/patterns_base.jsonl│
│  - Covers common entities across major domains               │
└─────────────────────────┬────────────────────────────────────┘
                          │ first occurrence in user text
                          ▼
┌──────────────────────────────────────────────────────────────┐
│  PROVISIONAL: Discovered by LLM, pending validation          │
│                                                               │
│  - Entity exists in user text but not in seed patterns       │
│  - Stored in ontology graph with status=provisional          │
│  - May be proprietary term, typo, or emerging vocabulary     │
│  - Quality gate decides whether to promote                   │
└─────────────────────────┬────────────────────────────────────┘
                          │ passes quality gate
                          ▼
┌──────────────────────────────────────────────────────────────┐
│  CONFIRMED: Validated pattern, added to EntityRuler          │
│                                                               │
│  - Pattern written to disk: global or tenant-scoped          │
│  - Loaded on ruler hot-reload, available for all subsequent │
│  - Promotion tracked in ontology graph (promoted_at, stats) │
└──────────────────────────────────────────────────────────────┘
```

### 2.2 Promotion Configuration

Promotion behavior is controlled by `PromotionConfig`:

```python
@dataclass
class PromotionConfig:
    """Configuration for pattern promotion to EntityRuler."""

    # Quality gates
    min_confidence: float = 0.8
    min_frequency: int = 1
    min_type_consistency: float = 0.8
    min_name_length: int = 3

    # Validation
    reasoning_validation: bool = True

    # Human review (deferred for Phase 1)
    human_review: bool = False

    # Global promotion
    wikidata_linkable_auto_promote: bool = True
    cross_tenant_frequency_threshold: int = 3
```

### 2.3 Default Behavior (Option C)

The system implements a **balanced approach** between statistical gates and reasoning validation:

1. **Statistical pre-filter**: Confidence > 0.8, frequency > 1, type consistency > 80%, name length > 3
2. **Reasoning validation**: ReasoningTrace validates promotion on first high-confidence occurrence
3. **Immediate promotion**: Once validated, entity type moves from provisional to confirmed
4. **Async reasoning trace**: Reasoning runs in background, doesn't block promotion

This strikes the balance between quality control and responsiveness. Low-quality patterns are filtered statistically. High-quality patterns are validated with reasoning, logged for auditability, and promoted immediately.

### 2.4 Promotion Candidates from Pipeline

The extraction pipeline emits promotion candidates via `PipelineState`:

```python
@dataclass
class PipelineState:
    """State object passed through pipeline stages."""

    text: str
    memory_item: MemoryItem
    entities: list[Entity]
    relations: list[Relation]

    # Extraction metadata
    extraction_source: Literal["ruler", "llm", "hybrid"]
    llm_confidence: float | None = None

    # Promotion candidates
    promotion_candidates: list[PromotionCandidate] = field(default_factory=list)
```

```python
@dataclass
class PromotionCandidate:
    """Entity type discovered by LLM but not in EntityRuler."""

    entity_name: str
    entity_type: str
    confidence: float
    source_memory_id: str
    discovered_at: datetime

    # Context for reasoning validation
    context_text: str
    co_occurring_entities: list[str]
```

When LLM extraction finds an entity not in the ruler, a `PromotionCandidate` is added to `PipelineState.promotion_candidates`. The background enrichment worker processes these candidates through the quality gate.

---

## 3. EntityRuler Pattern Growth

### 3.1 Pattern Extraction

After LLM enrichment completes, the system diffs LLM results against ruler results to identify new patterns:

```python
def extract_new_patterns(llm_entities: list[Entity],
                        ruler_entities: list[Entity]) -> list[EntityPattern]:
    """Diff LLM extraction vs ruler extraction to find new patterns."""

    # Normalize entity names for comparison
    ruler_names = {normalize_entity_name(e.name) for e in ruler_entities}

    new_patterns = []
    for entity in llm_entities:
        normalized_name = normalize_entity_name(entity.name)

        # Entity found by LLM but not by ruler
        if normalized_name not in ruler_names:
            pattern = EntityPattern(
                text=entity.name,
                label=entity.type,
                pattern=[{"LOWER": token.lower()} for token in entity.name.split()],
                confidence=entity.confidence,
                source="llm_discovery",
                discovered_at=datetime.utcnow(),
            )
            new_patterns.append(pattern)

    return new_patterns
```

### 3.2 Quality Gate Criteria

Not all LLM discoveries become patterns. The quality gate filters for high-confidence, consistent entities:

```python
def passes_quality_gate(pattern: EntityPattern,
                       config: PromotionConfig,
                       historical_stats: EntityStats | None) -> bool:
    """Determine if pattern meets promotion criteria."""

    # Criterion 1: Confidence threshold
    if pattern.confidence < config.min_confidence:
        return False

    # Criterion 2: Name length (filter single chars, abbreviations without context)
    if len(pattern.text) < config.min_name_length:
        return False

    # Criterion 3: Not in common word blocklist
    if pattern.text.lower() in COMMON_WORD_BLOCKLIST:
        return False

    # Criterion 4: Frequency and type consistency (requires historical stats)
    if historical_stats:
        if historical_stats.occurrence_count < config.min_frequency:
            return False

        # Type consistency: entity classified as same type >80% of occurrences
        type_consistency = historical_stats.type_counts.get(pattern.label, 0) / historical_stats.occurrence_count
        if type_consistency < config.min_type_consistency:
            return False

    # Criterion 5: Reasoning validation (if enabled)
    if config.reasoning_validation:
        reasoning_result = validate_pattern_with_reasoning(pattern, historical_stats)
        if not reasoning_result.is_valid:
            return False

    return True
```

```python
COMMON_WORD_BLOCKLIST = {
    # Pronouns
    "i", "you", "he", "she", "it", "we", "they",
    # Articles
    "a", "an", "the",
    # Prepositions
    "in", "on", "at", "to", "for", "with", "from",
    # Conjunctions
    "and", "or", "but", "if", "when", "while",
    # Common verbs
    "is", "are", "was", "were", "be", "been", "have", "has", "had",
    # ... ~200 total entries
}
```

### 3.3 Pattern Storage and Multi-Tenancy

Patterns are stored in three layers, loaded in order (later layers override earlier):

```
1. Seed patterns (pre-generated, shipped with SmartMemory)
   {install_dir}/entity_patterns_base.jsonl
   - 5,000-10,000+ patterns
   - Global scope
   - Generated offline before release via Groq/Gemma extraction on diverse corpus

2. Learned global patterns (promoted from tenant patterns via frequency gate)
   {data_dir}/entity_patterns_global.jsonl
   - Patterns validated across multiple tenants
   - Wikidata-linkable entities promoted immediately
   - Non-Wikidata entities promoted after N tenants independently discover them

3. Tenant-specific patterns (learned from this tenant's LLM extractions)
   {data_dir}/entity_patterns_{tenant_id}.jsonl
   - Tenant-scoped patterns not yet promoted to global
   - May include proprietary terminology, internal code names, domain jargon
```

Pattern promotion from tenant to global:

```python
def should_promote_to_global(pattern: EntityPattern,
                             cross_tenant_stats: CrossTenantStats,
                             config: PromotionConfig) -> bool:
    """Determine if tenant pattern should be promoted to global scope."""

    # Immediate promotion: entity exists in Wikidata
    if config.wikidata_linkable_auto_promote:
        if cross_tenant_stats.wikidata_qid is not None:
            return True

    # Frequency-based promotion: N+ tenants independently discovered this entity
    if cross_tenant_stats.tenant_count >= config.cross_tenant_frequency_threshold:
        # Verify type consistency across tenants (prevent cross-tenant pollution)
        if cross_tenant_stats.dominant_type_ratio >= 0.8:
            return True

    return False
```

### 3.4 Hot Reload via Redis Pub/Sub

When new patterns are added, all active service instances reload their EntityRuler without restart:

```python
class EntityRulerManager:
    """Manages EntityRuler lifecycle with hot-reload support."""

    def __init__(self, redis_client: Redis):
        self.redis = redis_client
        self.ruler = self._build_ruler()

        # Subscribe to pattern update events
        self.pubsub = self.redis.pubsub()
        self.pubsub.subscribe("smartmemory:patterns:updated")

        # Start background listener
        threading.Thread(target=self._listen_for_updates, daemon=True).start()

    def _listen_for_updates(self):
        """Background thread listening for pattern updates."""
        for message in self.pubsub.listen():
            if message["type"] == "message":
                # Pattern file changed, rebuild ruler
                self.ruler = self._build_ruler()
                logger.info("EntityRuler reloaded with updated patterns")

    def _build_ruler(self) -> spacy.pipeline.EntityRuler:
        """Load all pattern layers and build EntityRuler."""
        patterns = []

        # Layer 1: Seed patterns
        patterns.extend(load_patterns(PATTERNS_BASE_PATH))

        # Layer 2: Global learned patterns
        patterns.extend(load_patterns(PATTERNS_GLOBAL_PATH))

        # Layer 3: Tenant patterns (if in tenant context)
        if self.tenant_id:
            patterns.extend(load_patterns(f"entity_patterns_{self.tenant_id}.jsonl"))

        ruler = nlp.add_pipe("entity_ruler")
        ruler.add_patterns(patterns)
        return ruler
```

Pattern persistence triggers pub/sub notification:

```python
def persist_patterns(patterns: list[EntityPattern], scope: Literal["global", "tenant"]):
    """Write patterns to disk and notify all service instances."""

    if scope == "global":
        path = PATTERNS_GLOBAL_PATH
    else:
        path = f"entity_patterns_{tenant_id}.jsonl"

    # Atomic write
    with open(path, "a") as f:
        for pattern in patterns:
            f.write(json.dumps(pattern.to_dict()) + "\n")

    # Notify all instances
    redis_client.publish("smartmemory:patterns:updated", scope)
```

---

## 4. Entity-Pair Cache

### 4.1 Mechanism

When the LLM extracts a relation `(entity_A, relation_type, entity_B)`, the relation is stored in the data graph as an edge. On subsequent text mentioning both `entity_A` and `entity_B` in the same tenant's context, the fast tier can reuse the known relation without invoking the LLM.

This requires no new storage infrastructure. The graph IS the cache. The RelationRuler simply traverses the graph to check for existing edges between co-occurring entities.

```python
def check_entity_pair_cache(entity_a: Entity,
                           entity_b: Entity,
                           graph: SmartGraph) -> Relation | None:
    """Check if a relation between entity_a and entity_b already exists in the graph."""

    # Query: find edges between these two entities in this tenant's scope
    query = f"""
        MATCH (a:Entity {{name: $entity_a, tenant_id: $tenant_id}})
             -[r]-
             (b:Entity {{name: $entity_b, tenant_id: $tenant_id}})
        RETURN type(r) as relation_type, r.confidence as confidence
    """

    result = graph.query(query, {
        "entity_a": entity_a.name,
        "entity_b": entity_b.name,
        "tenant_id": graph.scope_provider.tenant_id,
    })

    if result:
        # Relation exists, reuse it
        return Relation(
            source=entity_a,
            target=entity_b,
            relation_type=result[0]["relation_type"],
            confidence=result[0]["confidence"],
            source="entity_pair_cache",
        )

    return None
```

### 4.2 Privacy Model

Entity-pair relations are **always tenant-scoped**. Even when both entities are public (e.g., "Python", "FastAPI"), the relation extracted from user text ("We use Python + FastAPI + Docker + Kubernetes") reveals proprietary architecture decisions.

The combination of public entities into a process graph is proprietary. Relation instances never cross tenant boundaries.

```python
# Valid: Tenant A's graph
(Python) --[USES]--> (FastAPI)
(FastAPI) --[DEPLOYED_ON]--> (Kubernetes)

# Valid: Tenant B's graph
(Python) --[USES]--> (Django)
(Django) --[DEPLOYED_ON]--> (AWS Lambda)

# Invalid: Cross-tenant relation reuse
# Tenant B cannot inherit Tenant A's entity-pair relations
# Even though all entities are public, the connections are proprietary
```

### 4.3 Cache Scope

The entity-pair cache operates at the workspace level by default, respecting the same isolation boundaries as all other SmartMemory data:

- **WORKSPACE**: Team members share entity-pair relations (default)
- **USER**: Personal entity-pair relations, not shared with team
- **TENANT**: Organization-wide (rare, only for enterprise deployments)

---

## 5. What Was Skipped (and Why)

### 5.1 No RelationRuler Dep-Parse Templates

The original design included learning syntactic templates for relation extraction (e.g., `nsubj(founded) + PERSON → ORG = "founded"`). This was removed from Phase 1 for three reasons:

1. **Relations are semantic, not syntactic**. Dep-parse patterns capture syntax ("X founded Y") but miss semantic variations ("X is the founder of Y", "X established Y", "Y was created by X"). The ceiling for dep-parse relation extraction is ~65-70% R-F1.

2. **High maintenance burden**. Syntactic templates are fragile. They break with passive voice, complex sentences, and domain-specific phrasings. Every language requires separate templates.

3. **LLM dominates**. Groq Llama-3.3-70b achieves 89.3% R-F1 at 740ms. The fast tier (dep-parse + type-pair priors) achieves 65% R-F1. The 24-point gap is too large to ignore. For relations, the LLM is the source of truth.

**Decision**: Phase 1 uses entity-pair cache only. Dep-parse templates deferred indefinitely. If relation extraction performance becomes a bottleneck, revisit with transformer-based relation extraction models, not dep-parse.

### 5.2 No Progressive Prompting

Four variants of progressive prompting were benchmarked: LLM refines spaCy draft, LLM adds to spaCy results, LLM validates spaCy results, LLM fills gaps in spaCy results. All four performed worse than standalone LLM extraction.

**Root cause**: Anchoring bias. The LLM anchors to the draft and fails to correct errors or discover entities the draft missed.

**Benchmark results**:

| Configuration | E-F1 | R-F1 |
|--------------|------|------|
| LLM standalone | 100% | 89.3% |
| Progressive (refine) | 96.2% | 78.5% |
| Progressive (add) | 95.1% | 81.2% |
| Progressive (validate) | 97.7% | 85.0% |
| Progressive (fill gaps) | 98.5% | 86.8% |

Best progressive prompt still 3 points below standalone on relations.

**Decision**: Progressive prompting abandoned. LLM always extracts from scratch in enrichment tier.

### 5.3 Type-Pair Priors Kept

Unlike dep-parse templates, **type-pair priors remain** in the architecture. These are simple lookup tables mapping entity type pairs to likely relation types:

```python
TYPE_PAIR_PRIORS = {
    ("person", "organization"): ["founded", "works_at", "ceo_of", "member_of"],
    ("organization", "technology"): ["developed", "maintains", "uses"],
    ("person", "work_of_art"): ["composed", "wrote", "created", "directed"],
}
```

These priors are:
- **Cheap**: Sub-millisecond hash lookup
- **Pre-built**: Bootstrapped from Wikidata property statistics
- **Useful**: Reduce false positives by filtering impossible type pairs

Type-pair priors provide a weak signal (30% confidence boost) in RelationRuler's multi-signal scoring, but they don't make extraction decisions alone.

---

## 6. Reasoning Integration

### 6.1 Three Levels of Reasoning

SmartMemory's reasoning system operates at three levels of decision-making authority:

**Level 1: Reasoning as Audit Trail (launch, Phases 1-3)**

Statistical gates make promotion decisions. Reasoning provides explanation after the fact.

```python
def promote_pattern_with_audit(candidate: PromotionCandidate,
                               config: PromotionConfig) -> PromotionResult:
    """Promote pattern with reasoning trace logged asynchronously."""

    # Statistical gate decides
    if not passes_quality_gate(candidate, config):
        return PromotionResult(promoted=False, reason="quality_gate_failed")

    # Promotion happens immediately
    persist_pattern(candidate.to_pattern())

    # Reasoning logged async (audit trail)
    background_tasks.add_task(log_promotion_reasoning, candidate)

    return PromotionResult(promoted=True, reasoning_pending=True)
```

**Level 2: Reasoning Validates (Phase 4+)**

Statistical pre-filter narrows candidates. Reasoning LLM validates promotion.

```python
def promote_pattern_with_validation(candidate: PromotionCandidate,
                                   config: PromotionConfig) -> PromotionResult:
    """Reasoning validates promotion before committing."""

    # Statistical pre-filter
    if not statistical_prefilter(candidate, config):
        return PromotionResult(promoted=False, reason="prefilter_failed")

    # Reasoning validation
    reasoning = validate_pattern_with_reasoning(candidate)
    if not reasoning.is_valid:
        return PromotionResult(
            promoted=False,
            reason="reasoning_rejected",
            reasoning_trace=reasoning,
        )

    # Promotion after validation
    persist_pattern(candidate.to_pattern())
    persist_reasoning_trace(reasoning)

    return PromotionResult(promoted=True, reasoning_trace=reasoning)
```

```python
def validate_pattern_with_reasoning(candidate: PromotionCandidate) -> ReasoningTrace:
    """Use reasoning LLM to validate pattern promotion."""

    prompt = f"""
    Evaluate whether this entity should be added to the ontology:

    Entity: {candidate.entity_name}
    Type: {candidate.entity_type}
    Confidence: {candidate.confidence}
    Context: {candidate.context_text}

    Consider:
    1. Is this entity name unambiguous and specific?
    2. Is the type classification correct?
    3. Would this pattern generalize well to future text?
    4. Are there any quality concerns (typos, overfitting, noise)?

    Provide a structured reasoning trace and final verdict (accept/reject).
    """

    response = reasoning_llm.generate(prompt)
    return parse_reasoning_trace(response, domain="ontology")
```

**Level 3: Reasoning as Decision Maker (never)**

Full reasoning for every promotion candidate. Overkill for obvious patterns. Risk of LLM hallucination blocking valid promotions.

This level is explicitly rejected. Reasoning augments decision-making but never fully replaces statistical gates.

### 6.2 ReasoningTrace Storage

All promotion reasoning traces are stored as `reasoning` memory type with metadata tag `reasoning_domain="ontology"`:

```python
@dataclass
class OntologyReasoningTrace(ReasoningTrace):
    """Reasoning trace for ontology promotion decision."""

    # Base ReasoningTrace fields
    reasoning_type: str = "ontology_promotion"
    steps: list[ReasoningStep] = field(default_factory=list)
    conclusion: str = ""
    confidence: float = 0.0

    # Ontology-specific context
    entity_name: str
    entity_type: str
    promotion_decision: Literal["accept", "reject"]
    statistical_gate_passed: bool

    # Metadata
    metadata: dict = field(default_factory=lambda: {
        "reasoning_domain": "ontology",
        "memory_type": "reasoning",
    })
```

This enables the system to explain its own knowledge structure:

```python
# Query: "Why is 'FastAPI' classified as a technology?"
results = memory.search(
    "FastAPI technology classification",
    filters={"reasoning_domain": "ontology"},
)

# Returns reasoning trace showing:
# - Statistical confidence score
# - Type consistency across occurrences
# - Wikidata validation
# - Reasoning LLM verdict with step-by-step explanation
```

### 6.3 AssertionChallenger Extension

SmartMemory's existing `AssertionChallenger` detects contradictions in opinions and observations. This is extended to handle entity-type contradictions:

```python
def detect_entity_type_contradiction(entity_name: str,
                                    graph: SmartGraph) -> Contradiction | None:
    """Detect if entity is classified inconsistently across memories."""

    query = f"""
        MATCH (m:Memory)-[:MENTIONS]->(e:Entity {{name: $entity_name}})
        RETURN e.type as entity_type, count(*) as count
        ORDER BY count DESC
    """

    results = graph.query(query, {"entity_name": entity_name})

    if len(results) > 1:
        dominant_type = results[0]["entity_type"]
        dominant_count = results[0]["count"]

        # Calculate type consistency
        total_count = sum(r["count"] for r in results)
        consistency = dominant_count / total_count

        # Flag if consistency < 80%
        if consistency < 0.8:
            return Contradiction(
                type="entity_type_inconsistency",
                entity=entity_name,
                dominant_type=dominant_type,
                dominant_ratio=consistency,
                conflicting_types=[(r["entity_type"], r["count"]) for r in results[1:]],
            )

    return None
```

When a contradiction is detected, the system:
1. Flags the entity for review
2. Emits a reasoning task to resolve the contradiction
3. Demotes the provisional pattern until resolved
4. Logs the contradiction in a `reasoning` memory with domain `ontology`

---

## 7. Convergence Monitoring

### 7.1 Key Metrics

The following metrics are tracked to measure self-learning progress:

```python
@dataclass
class OntologyMetrics:
    """Metrics for ontology self-learning convergence."""

    # Pattern counts
    total_patterns: int
    seed_patterns: int
    learned_global_patterns: int
    learned_tenant_patterns: int

    # Coverage rates
    ruler_hit_rate: float  # % entities found by ruler vs LLM
    entity_pair_cache_hit_rate: float  # % relations reused from cache

    # Convergence indicators
    new_patterns_per_1k_memories: float
    patterns_last_updated_at: datetime

    # Quality metrics
    avg_pattern_confidence: float
    avg_type_consistency: float

    # Enrichment efficiency
    enrichment_rate: float  # % memories requiring LLM enrichment
    avg_enrichment_latency: float
```

### 7.2 Convergence Curve

The system emits convergence metrics to Redis Streams for visualization in the Insights dashboard:

```python
def emit_convergence_metrics(memory_count: int, metrics: OntologyMetrics):
    """Publish convergence metrics for monitoring."""

    redis_client.xadd(
        "smartmemory:metrics:ontology",
        {
            "timestamp": datetime.utcnow().isoformat(),
            "memory_count": memory_count,
            "ruler_hit_rate": metrics.ruler_hit_rate,
            "cache_hit_rate": metrics.entity_pair_cache_hit_rate,
            "enrichment_rate": metrics.enrichment_rate,
            "new_patterns": metrics.new_patterns_per_1k_memories,
        },
    )
```

Expected convergence curve:

```
Ruler hit rate (%)
100 |                    ________________
    |                 __/
    |              __/
    |           __/
 97 |________/
    |
    +----------------------------------------
     0     1K    5K    10K   20K   50K
              Memories processed
```

### 7.3 Steady State Definition

The system has reached steady state when:

- **Ruler hit rate > 95%**: Fast tier handles 95%+ of entities without LLM
- **Cache hit rate > 80%**: Entity-pair cache resolves 80%+ of relations
- **New patterns < 5 per 1K memories**: Pattern discovery rate has plateaued
- **Enrichment rate < 10%**: LLM only needed for genuinely novel text

At steady state, the fast tier handles almost all extraction. The LLM enrichment tier still runs for quality assurance but rarely discovers new patterns.

---

## 8. Demotion Strategy

### 8.1 Phase 1: No Demotion

At launch, patterns persist indefinitely. Confidence naturally decays on non-use through existing confidence scoring mechanisms, but patterns remain in the EntityRuler.

**Rationale**:
- Unknown domain shift rates in production
- Pattern storage cost is negligible (<1MB per 10K patterns)
- False negatives (removing good patterns) are worse than false positives (keeping stale patterns)
- Need production data before implementing demotion logic

### 8.2 Future: Inactivity-Based Demotion

After sufficient production telemetry, implement demotion for patterns with zero hits in N days:

```python
@dataclass
class PatternStats:
    """Usage statistics for a pattern."""

    pattern_id: str
    last_matched_at: datetime | None
    total_matches: int
    matches_last_30d: int
    matches_last_90d: int

    def is_inactive(self, inactive_days: int = 180) -> bool:
        """Check if pattern has had no matches for N days."""
        if self.last_matched_at is None:
            return True

        days_since_match = (datetime.utcnow() - self.last_matched_at).days
        return days_since_match > inactive_days
```

Inactive patterns are not deleted, but moved to a separate file:

```
{data_dir}/entity_patterns_global.jsonl       # Active patterns
{data_dir}/entity_patterns_global_inactive.jsonl  # Dormant patterns
```

Inactive patterns can be reactivated if matched again in future text.

### 8.3 Reasoning Trace Requirement

Any demotion action (active → inactive, confirmed → provisional) requires a reasoning trace, symmetric with promotion:

```python
def demote_pattern_with_reasoning(pattern: EntityPattern,
                                  reason: str) -> DemotionResult:
    """Demote pattern with reasoning trace for auditability."""

    # Generate reasoning trace
    reasoning = ReasoningTrace(
        reasoning_type="ontology_demotion",
        conclusion=f"Pattern demoted: {reason}",
        steps=[
            ReasoningStep(
                type="evidence",
                content=f"Pattern '{pattern.text}' last matched {pattern.stats.last_matched_at}",
            ),
            ReasoningStep(
                type="decision",
                content=f"Inactive for {pattern.stats.days_since_match} days → demote to inactive",
            ),
        ],
        confidence=1.0,
        metadata={"reasoning_domain": "ontology"},
    )

    # Move pattern to inactive
    move_pattern_to_inactive(pattern)
    persist_reasoning_trace(reasoning)

    return DemotionResult(demoted=True, reasoning_trace=reasoning)
```

This ensures all ontology changes (promotion and demotion) are explained and auditable.

---

## 9. NELL Lessons Applied

The Never-Ending Language Learner (NELL) project at CMU ran for 12+ years, learning world knowledge from web text. Key lessons from NELL inform SmartMemory's design:

### 9.1 Success Rate

75% of NELL's categories reached 90-99% precision autonomously. 25% degraded to 25-60% precision over time. The system could not predict which categories would degrade.

**SmartMemory mitigation**:
- Multi-source agreement: EntityRuler + LLM + Wikidata (2+ extractors must agree)
- Type consistency gate: entity must be classified identically >80% of occurrences
- Reasoning validation: high-stakes promotions validated by reasoning LLM
- Contradiction detection: AssertionChallenger flags inconsistent classifications

### 9.2 Error Propagation

NELL's #1 failure mode: a single bad promotion (e.g., "New York" classified as a person) cascades. The bad entity then participates in relation extraction, creating bad patterns, which extract more bad entities.

**SmartMemory mitigation**:
- Mutual exclusion constraints in ontology: an entity can only have one type
- Type-pair priors filter impossible relations: (location, person) cannot have "founded" relation
- Wikidata grounding catches obvious errors: "New York" has Wikidata QID, type is location (unambiguous)
- Statistical gates prevent low-confidence promotions from entering the system

### 9.3 Multi-Source Agreement Quality Gate

NELL's best quality gate: require 2+ independent extractors to agree before promotion. Single-source discoveries are high-noise.

**SmartMemory implementation**:
- Phase 1: LLM is single source (high precision model, 100% E-F1 in benchmark)
- Phase 2+: Require EntityRuler + Wikidata linkable OR EntityRuler + cross-tenant frequency
- Phase 3+: Full multi-source: EntityRuler + LLM + Wikidata + historical consistency

### 9.4 Human-in-the-Loop (Deferred)

NELL achieved dramatic quality improvements with 5 minutes of human review per day. A human labeled 10-20 candidate promotions as correct/incorrect, and the system learned from the feedback.

**SmartMemory design** (for Phase 4+):
- `PromotionConfig.human_review = True` gates high-impact promotions
- High-impact defined as: cross-tenant promotion, >100 expected future matches, or low reasoning confidence
- Promotion candidate queued for human review, surfaces in admin UI
- Human labels as accept/reject, system learns reviewer preferences

5-minute daily HITL can catch 95%+ of error-prone promotions before they enter production.

---

## 10. Implementation Phases

This design will be implemented across multiple phases:

### Phase 1: EntityRuler Self-Learning (Phases 1-3 in implementation plan)
- Pattern extraction from LLM diff
- Statistical quality gates
- Three-layer pattern storage (seed, global, tenant)
- Hot-reload via Redis pub/sub
- Async reasoning audit trail

### Phase 2: Entity-Pair Cache (Phase 4)
- Graph traversal for entity-pair relation lookup
- Tenant-scoped cache (never global)
- Integration into RelationRuler multi-signal scoring

### Phase 3: Reasoning Validation (Phase 4)
- Reasoning LLM validates promotion candidates
- ReasoningTrace storage with `reasoning_domain="ontology"`
- AssertionChallenger extension for entity-type contradictions

### Phase 4: Convergence Monitoring (Phase 5)
- Metrics emission to Redis Streams
- Insights dashboard visualizations
- Convergence curve tracking

### Phase 5: Human-in-the-Loop (Phase 6)
- Admin UI for promotion candidate review
- Human feedback loop
- Reviewer preference learning

---

## Appendix A: File Locations

| Component | File Path |
|-----------|-----------|
| Pattern storage | `smartmemory/extraction/pattern_store.py` |
| Pattern diff | `smartmemory/extraction/self_learning.py` |
| Quality gates | `smartmemory/extraction/quality_gates.py` |
| Promotion config | `smartmemory/extraction/promotion_config.py` |
| EntityRuler manager | `smartmemory/extraction/ruler_manager.py` |
| Entity-pair cache | `smartmemory/extraction/entity_pair_cache.py` |
| Reasoning validation | `smartmemory/reasoning/ontology_validation.py` |
| AssertionChallenger | `smartmemory/reasoning/assertion_challenger.py` (existing) |
| Convergence metrics | `smartmemory/monitoring/ontology_metrics.py` |
| HITL review | `memory_service/api/routes/admin/pattern_review.py` (future) |

---

## Appendix B: Configuration Example

```python
# Default configuration (launch)
default_config = PromotionConfig(
    min_confidence=0.8,
    min_frequency=1,
    min_type_consistency=0.8,
    min_name_length=3,
    reasoning_validation=True,
    human_review=False,
    wikidata_linkable_auto_promote=True,
    cross_tenant_frequency_threshold=3,
)

# Conservative configuration (high-stakes domains)
conservative_config = PromotionConfig(
    min_confidence=0.9,
    min_frequency=3,
    min_type_consistency=0.9,
    min_name_length=4,
    reasoning_validation=True,
    human_review=True,  # HITL for all promotions
    wikidata_linkable_auto_promote=True,
    cross_tenant_frequency_threshold=5,
)

# Aggressive configuration (fast vocabulary growth)
aggressive_config = PromotionConfig(
    min_confidence=0.7,
    min_frequency=1,
    min_type_consistency=0.7,
    min_name_length=2,
    reasoning_validation=False,  # Audit only
    human_review=False,
    wikidata_linkable_auto_promote=True,
    cross_tenant_frequency_threshold=2,
)
```

---

## Appendix C: Metrics Dashboard Mockup

The Insights dashboard displays convergence metrics:

```
┌─────────────────────────────────────────────────────────────┐
│  Ontology Self-Learning                                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Total Patterns: 8,432                                      │
│    - Seed: 5,000                                            │
│    - Global learned: 2,104                                  │
│    - Tenant: 1,328                                          │
│                                                             │
│  Coverage                                                   │
│    Ruler hit rate: 97.2% (target: 95%)                     │
│    Entity-pair cache: 84.3% (target: 80%)                  │
│                                                             │
│  Convergence                                                │
│    New patterns (last 1K): 12 (trending down)              │
│    Enrichment rate: 8.4% (LLM rarely needed)               │
│                                                             │
│  [Graph: Ruler hit rate over time]                         │
│  100% |                    _____________                    │
│       |                 __/                                 │
│   97% |________/                                            │
│       +--------------------------------------------          │
│        0     1K    5K    10K   20K                          │
│                                                             │
│  Status: Converged (steady state reached at 8.2K memories) │
└─────────────────────────────────────────────────────────────┘
```

---

**End of Document**
