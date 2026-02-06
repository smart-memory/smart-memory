# Ontology Model Technical Specification

**Date:** 2026-02-06
**Status:** Design Document
**Source:** Extracted from `2026-02-05-ontology-strategic-implementation-plan.md` and `2026-02-05-ontology-grounded-extraction-design.md`

---

## 1. Core Principle

The ontology is **foundational infrastructure**, not an optional feature. It exists in every SmartMemory deployment, identical in architecture to the graph database and embedding store.

### 1.1 Always On, Not Toggle

- **No "ontology off" mode**: The system always uses whatever knowledge it has to inform extraction.
- **"Ontology cold" not "ontology off"**: A fresh installation has minimal learned patterns but works at 96.9% entity F1 score immediately with seeded EntityRuler patterns.
- **No configuration toggle**: There is no `ontology_enabled` flag. The ontology is core architecture.

### 1.2 The Data Model Is Triples

Every major ontology system (OWL/RDF, Wikidata, Schema.org, NELL) uses `(subject, predicate, object)` triples with metadata. FalkorDB already stores triples as nodes and edges.

**Key design decision: The graph IS the ontology. No custom data model needed.**

The OWL TBox/ABox distinction maps directly:
- **TBox** (schema): Entity types, relation types, constraints
- **ABox** (instances): Actual entities and relations from user memories

---

## 2. Separate FalkorDB Graphs

SmartMemory uses **two isolated FalkorDB graphs per workspace** for complete separation of concerns.

### 2.1 Graph Architecture

```
Workspace: ws_a1b2c3

┌─────────────────────────────────────┐
│  ws_a1b2c3_ontology                 │
│                                     │
│  - Entity type definitions          │
│  - Relation type schemas            │
│  - EntityRuler patterns             │
│  - Type-pair priors                 │
│  - Mutual exclusion constraints     │
│  - Promotion metadata               │
│                                     │
│  Read by: Pipeline config loader    │
│  Written by: Self-learning system   │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│  ws_a1b2c3_data                     │
│                                     │
│  - Entity instances                 │
│  - Relations between entities       │
│  - Memory nodes                     │
│  - Embeddings                       │
│  - Temporal metadata                │
│                                     │
│  Read/write by: All user operations │
└─────────────────────────────────────┘
```

### 2.2 Zero Bleed By Construction

- **No label filtering needed**: Ontology types and data entities cannot collide because they exist in different graphs.
- **Clean namespace separation**: `:EntityType` nodes in ontology graph, `:Entity` nodes in data graph.
- **Pipeline isolation**: Config loader reads from `_ontology` graph; all other operations read from `_data` graph.
- **Simpler queries**: No complex filtering logic to separate schema from data.

### 2.3 Connection Points

The two graphs connect only through the pipeline:

1. **Pipeline startup**: Loads `PipelineConfig` from ontology graph
2. **Extraction**: Uses loaded config (entity types, patterns, constraints) to process text
3. **Storage**: Writes entities and relations to data graph
4. **Self-learning**: Analyzes data graph, writes promoted patterns back to ontology graph

---

## 3. Three-Layer Architecture

The ontology has three conceptual layers, all stored in the `_ontology` graph.

### 3.1 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│  Global TBox (curated, seed-generated, read-only for tenants)│
│                                                              │
│  Seed entity types (14 minimum):                            │
│    - Generic: Person, Organization, Location, Concept,      │
│      Event, Tool, Skill, Document                           │
│    - SmartMemory-native: Decision, Claim, Action,           │
│      Metric, Process, Project                               │
│                                                              │
│  Type-pair relation priors (bootstrapped from Wikidata)     │
│  Mutual exclusion constraints (Technology ≠ Person)         │
│                                                              │
│  Updated by: Service operators, never tenants               │
│  Shared across: All workspaces (read-only)                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ extends
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Tenant Soft-TBox (auto-promoted from usage)                │
│                                                              │
│  Learned entity patterns: "FastAPI" → Technology            │
│  Learned relation templates: nsubj(founded)+PERSON+ORG      │
│  Type status: seed | provisional | confirmed                │
│  Promotion metadata: confidence, frequency, source          │
│                                                              │
│  Feeds: EntityRuler patterns + LLM prompt schema            │
│  Scope: Per-workspace (tenant-isolated)                     │
│  Auto-promoted via: PromotionConfig quality gates           │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ derived from
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Tenant ABox (the actual graph data)                        │
│                                                              │
│  Entity instances: Sarah (Person), Acme Corp (Org)          │
│  Relation instances: Sarah --[works_at]--> Acme Corp        │
│  Memory nodes: Opinions, Decisions, Observations            │
│                                                              │
│  Storage: ws_{id}_data graph (NOT in _ontology)             │
│  Soft-TBox is materialized view derived from this data      │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Layer Responsibilities

**Global TBox:**
- **Content**: Entity types, relation types, mutual exclusion constraints, seed patterns
- **Source**: Curated by service operators, generated from Wikidata and training corpus
- **Scope**: Global, shared read-only across all tenants
- **Size**: 14 seed entity types, 100+ discoverable types in vocabulary, 30+ relation types
- **Updates**: Service-level operations only, versioned deployments

**Tenant Soft-TBox:**
- **Content**: Learned patterns promoted from this workspace's extraction history
- **Source**: Auto-promoted from LLM enrichment feedback via quality gates
- **Scope**: Per-workspace, tenant-isolated
- **Size**: Grows with usage, typical workspace has 500-5000 learned patterns
- **Updates**: Continuous background self-learning, user corrections

**Tenant ABox:**
- **Content**: The actual knowledge graph (entities, relations, memories)
- **Source**: User ingestion, extraction pipeline output
- **Scope**: Per-workspace, tenant-isolated
- **Size**: Unbounded, grows with user data
- **Updates**: Every ingest operation

---

## 4. Seed Entity Types

### 4.1 Minimum Seed Set (14 Types)

**8 Generic Types** (standard knowledge graph):
1. **Person** - Human individuals
2. **Organization** - Companies, institutions, groups
3. **Location** - Geographic places, addresses
4. **Concept** - Abstract ideas, theories, principles
5. **Event** - Occurrences, meetings, milestones
6. **Tool** - Software, physical tools, instruments
7. **Skill** - Competencies, abilities, expertise areas
8. **Document** - Files, papers, specifications, references

**6 SmartMemory-Native Types** (domain-specific):
9. **Decision** - Choices made, with evidence chains
10. **Claim** - Assertions tracked by AssertionChallenger
11. **Action** - Tasks, todos, trackable work items
12. **Metric** - Quantitative measurements, KPIs
13. **Process** - Procedures, workflows, sequences
14. **Project** - Initiatives, efforts, bounded work

### 4.2 Discoverable Types (100+)

The file `smartmemory/ontology/entity_types.py` contains 100+ domain-specific types:
- **Financial**: invoice, transaction, account, budget, portfolio
- **Legal**: case, statute, contract, regulation, precedent
- **Healthcare**: diagnosis, treatment, symptom, medication, procedure
- **Education**: course, assignment, curriculum, degree, certification
- **Logistics**: shipment, route, warehouse, inventory, delivery

**Role in system**: Vocabulary hints for LLM enrichment tier prompt. Not included in seed patterns. Emerge from usage and get promoted to Tenant Soft-TBox when quality gates pass.

### 4.3 Pattern Examples per Type

Each seed type includes 10-15 examples for EntityRuler pattern generation:

```python
# Person examples
["Sarah Chen", "Dr. Martinez", "Alice Johnson", "Bob Smith", ...]

# Technology examples
["Kubernetes", "React", "PostgreSQL", "Docker", "FastAPI", ...]

# Decision examples
["decided to migrate", "chose Python", "selected AWS", ...]
```

**Sourced from**: Pre-release seeding via LLM extraction on diverse text corpus (~10,000 texts, $5 via Groq).

---

## 5. Three-Tier Type Status

Every entity type has a lifecycle status governing its usage in extraction and promotion.

### 5.1 Status Definitions

```python
class TypeStatus(Enum):
    SEED = "seed"              # Shipped with release, global TBox
    PROVISIONAL = "provisional" # Auto-created on first encounter
    CONFIRMED = "confirmed"     # Passed quality gate, in Soft-TBox
```

### 5.2 Status Transitions

```
                    First encounter
                    (LLM extracts new type)
                           │
                           ▼
┌──────────────────────────────────────────┐
│         PROVISIONAL                      │
│  - Usable immediately (not blocked)      │
│  - Tracked in promotion candidates       │
│  - Subject to quality gates              │
└──────────────────────────────────────────┘
                           │
                           │ Meets quality gate
                           │ (frequency, confidence, reasoning)
                           ▼
┌──────────────────────────────────────────┐
│         CONFIRMED                        │
│  - Promoted to Tenant Soft-TBox          │
│  - Feeds EntityRuler patterns            │
│  - Used in LLM prompt schema             │
└──────────────────────────────────────────┘
                           │
                           │ (Optional) Multi-tenant gate
                           │ (Wikidata-linkable OR frequency across tenants)
                           ▼
┌──────────────────────────────────────────┐
│         SEED (promoted to Global)        │
│  - Shared across all workspaces          │
│  - Read-only for tenants                 │
│  - Updated via service deployment        │
└──────────────────────────────────────────┘
```

### 5.3 Key Design Rule

**All data is ontology-governed.** Unknown entity types create provisional ontology entries BEFORE data stores. No entity can exist without a type.

This prevents the "open world" problem where unconstrained extraction creates semantic drift.

---

## 6. PromotionConfig

Promotion thresholds are parameterized per workspace, tunable via Pipeline Studio.

### 6.1 Configuration Model

```python
@dataclass
class PromotionConfig:
    """Quality gates for promoting provisional types to confirmed."""

    # Validation strategy (default: statistical pre-filter)
    reasoning_validation: bool = False  # False: stats only, True: LLM validates

    # Statistical thresholds
    min_frequency: int = 3              # Seen at least N times
    min_confidence: float = 0.8         # Average confidence > threshold
    type_consistency: float = 0.85      # Same type assigned >= 85% of time

    # Multi-source agreement (NELL's best gate)
    require_agreement: bool = True      # 2+ extractors must agree

    # Human-in-the-loop (deferred until review UI exists)
    human_review: bool = False          # Hold at provisional for manual review

    # Rate limiting (prevent cascading errors)
    max_promotions_per_day: int = 100   # Cap on auto-promotions
```

### 6.2 Default Profiles

Four pre-configured profiles corresponding to `confidence_requirement` parameter:

| Profile | min_frequency | min_confidence | reasoning_validation | Use Case |
|---------|--------------|----------------|---------------------|----------|
| **Low** | 2 | 0.7 | False | Casual notes, brainstorming |
| **Medium** (default) | 3 | 0.8 | False | General knowledge work |
| **High** | 5 | 0.9 | True | Professional, medical |
| **Critical** | 10 | 0.95 | True + human_review | Legal, compliance |

### 6.3 Option C as Default (Statistical Pre-Filter)

From research (Q4 in strategic plan), Option C selected:
- **Statistical pre-filter** (frequency, confidence, consistency) eliminates 80-90% of candidates
- **Audit trail** for all promotions (stored as reasoning traces)
- **Manual review** capability exposed in Studio (future)
- **Reasoning validation** opt-in for high-confidence domains (Option D features available but not default)

Reasoning validation (Level 2 from research) added in Phase 6+ when reasoning infrastructure is integrated.

### 6.4 Storage

PromotionConfig stored in workspace's `_ontology` graph as properties on `:Config` node. Loaded by pipeline at startup. Hot-reloadable when changed in Studio.

---

## 7. Dual-Axis Type System

SmartMemory has two independent type systems that classify different aspects of knowledge.

### 7.1 Memory Types (Container Classification)

```python
class MemoryType(Enum):
    # Core types (MemoryNodeType enum)
    WORKING = "working"
    SEMANTIC = "semantic"
    EPISODIC = "episodic"
    PROCEDURAL = "procedural"
    ZETTEL = "zettel"

    # Extended types (string values)
    REASONING = "reasoning"
    OPINION = "opinion"
    OBSERVATION = "observation"
    DECISION = "decision"
```

**Purpose**: Classifies the memory container. Governs which pipeline stages run, which evolvers apply, retention policies, search strategies.

**Example**: "We decided to use PostgreSQL for better data integrity" → memory_type = `decision`

### 7.2 Entity Types (Content Classification)

```python
class EntityType:
    # Defined in ontology graph as :EntityType nodes
    # Examples: Person, Organization, Technology, Concept, Decision, Metric
```

**Purpose**: Classifies extracted entities. Governs which EntityRuler patterns apply, which relation types are valid, which constraints check.

**Example**: "PostgreSQL" → entity_type = `Technology`

### 7.3 The Bridge

Some types exist on BOTH axes:

| Type | As Memory Type | As Entity Type |
|------|----------------|----------------|
| **Decision** | DecisionModel with causal tracking | Extractable from text, linkable entity |
| **Opinion** | OpinionMetadata with reinforcement | Extractable belief, confidence-weighted |
| **Observation** | ObservationMetadata with synthesis | Extractable fact, temporal-tracked |
| **Reasoning** | ReasoningTrace with step-by-step | Extractable rationale, chain-linked |

### 7.4 Bridge Behavior

When LLM extracts a Decision entity from text:
1. Creates Decision memory (memory_type = "decision")
2. Creates Decision entity in graph (entity_type = "Decision")
3. Links entity to memory with evidence chain
4. Decision is queryable via decision API AND as graph entity
5. Available for graph traversal: `MATCH (d:Entity {type: "Decision"})-[:AFFECTS]->(e:Entity)`

**Key insight**: The dual classification enables both structured tracking (via models) and graph connectivity (via entities).

### 7.5 Independence

The two axes are **orthogonal**:
- An episodic memory can contain Person, Technology, and Decision entities
- A semantic memory can contain Concept and Metric entities
- Memory type governs process, entity type governs structure
- No special interaction logic needed

---

## 8. Pattern Storage

Patterns are stored as **metadata on graph nodes** in the `_ontology` graph. No separate storage system.

### 8.1 Entity Patterns

Stored as properties on entity nodes in ontology graph:

```cypher
CREATE (:Entity {
  name: "FastAPI",
  type: "Technology",
  pattern_type: "exact_match",
  pattern_source: "llm_extraction",
  pattern_confidence: 0.92,
  pattern_frequency: 15,
  first_seen: "2026-01-15T10:30:00Z",
  last_seen: "2026-02-01T14:22:00Z",
  promotion_status: "confirmed"
})
```

**Pattern types:**
- `exact_match`: "FastAPI" → Technology
- `prefix_match`: "Dr." → Person
- `suffix_match`: " Inc." → Organization
- `regex`: `/[A-Z]{2,5}\d+/` → (product codes)
- `compound`: "Nobel Prize in Physics" → Award + Field

### 8.2 Type-Pair Priors

Stored as edges between `:EntityType` nodes in ontology graph:

```cypher
CREATE (person:EntityType {name: "Person"})
CREATE (org:EntityType {name: "Organization"})
CREATE (person)-[:VALID_RELATION {
  relation_type: "founded",
  confidence: 0.95,
  frequency: 1247,
  source: "global_seed"
}]->(org)
```

**Loaded at startup**, cached in-memory as lookup table:

```python
TYPE_PAIR_PRIORS = {
    ("Person", "Organization"): ["founded", "works_at", "ceo_of"],
    ("Organization", "Location"): ["headquartered_in", "located_in"],
    ("Technology", "Technology"): ["uses", "extends", "inspired_by"],
    # ... 30+ pairs
}
```

### 8.3 Entity-Pair Relation Cache

**NOT stored in ontology graph.** Lives in data graph as edges between entity instances.

```cypher
# In ws_{id}_data graph
CREATE (google:Entity {name: "Google", type: "Organization"})
CREATE (k8s:Entity {name: "Kubernetes", type: "Technology"})
CREATE (google)-[:developed {
  confidence: 0.98,
  source: "user_text",
  first_mentioned: "2026-01-20T09:15:00Z"
}]->(k8s)
```

**Key privacy rule: Entity-pair relations from user text are NEVER shared globally**, even if both entities are public. The connection reveals proprietary architecture.

The "cache" is just the data graph itself. No separate storage needed.

### 8.4 Pattern Loading

Three layers loaded in sequence (later layers override earlier):

```python
def load_entity_patterns(workspace_id: str) -> EntityRuler:
    ruler = EntityRuler()

    # Layer 1: Global seed patterns (shipped with release)
    ruler.load_patterns("global_seed")

    # Layer 2: Global learned patterns (promoted from tenants)
    ruler.load_patterns("global_learned")

    # Layer 3: Tenant-specific patterns
    ruler.load_patterns(f"tenant_{workspace_id}")

    return ruler
```

All patterns stored in `_ontology` graph with `scope` property:
- `scope: "global_seed"` - Read-only, shipped
- `scope: "global_learned"` - Promoted via multi-tenant gate
- `scope: "tenant_{id}"` - Workspace-specific

### 8.5 No Vector Embeddings on Patterns

All pattern lookups are **structural queries**, not semantic search:
- Entity patterns: string matching (exact, prefix, suffix, regex)
- Type-pair priors: graph edge lookup by entity types
- Entity-pair cache: graph traversal by entity IDs

Vector embeddings only used for memory content search, never for ontology lookups.

---

## 9. Privacy Model

The ontology's privacy model governs what knowledge is shareable across workspaces.

### 9.1 Core Privacy Rule

**Entity vocabulary and syntactic grammar are shareable. Entity-pair relations from user text are NEVER shared.**

Even when both entities are public (e.g., "Python", "FastAPI"), the relation between them extracted from user text is private. The combination of public entities into a process graph reveals proprietary architecture.

### 9.2 Sharing Matrix

| Knowledge Type | Example | Scope | Rationale |
|---------------|---------|-------|-----------|
| **Entity vocabulary** | "FastAPI" → Technology | Global (shareable) | Vocabulary has no information content |
| **Syntactic templates** | nsubj(VERB)+ORG → relation | Global (shareable) | Grammar doesn't reveal who does what |
| **Type-pair priors** | (Person, Org) → works_at | Global (shareable) | Pre-built from Wikidata statistics |
| **Wikidata facts** | Google developed Kubernetes | Global (shareable) | Public world knowledge |
| **Entity-pair relations** | AcmeCorp → uses → FastAPI | Always tenant-scoped | Reveals proprietary processes |
| **Memory content** | "We chose FastAPI for..." | Always tenant-scoped | User's private knowledge |
| **Graph topology** | Shape of entity connections | Always tenant-scoped | Process architecture is proprietary |

### 9.3 Pattern Promotion Privacy Gates

When LLM discovers a new entity (e.g., "LangChain" as Technology):

**Gate 1: Wikidata Linkable**
- Check if entity exists in Wikidata
- If yes: Promote to global immediately (public knowledge)
- If no: Continue to Gate 2

**Gate 2: Multi-Tenant Frequency**
- Check if N+ independent tenants discovered same entity+type
- Default threshold: N=3 tenants
- If yes: Promote to global (empirically proven public)
- If no: Remains tenant-scoped

**Examples:**
- "Kubernetes" → Wikidata Q22661306 → immediate global promotion
- "Project Mercury" → no Wikidata, single tenant → stays tenant-scoped
- "LangChain" → no Wikidata initially, 5 tenants discover independently → promote to global after threshold

### 9.4 Relation Template Privacy

**Dep-parse templates are globally shareable:**

```python
{
  "pattern": "nsubj(VERB)=PERSON, dobj(VERB)=ORG, verb_lemma in {found, establish}",
  "relation": "founded",
  "source": "learned_global"
}
```

This reveals syntax (grammar), not content (who founded what).

**Entity-pair relation cache is NEVER global:**

Even though template is shared, the specific instances are not:
- Template: `(Person, Organization, founded)` → shared
- Instance: `(Bill Gates, Microsoft, founded)` → tenant-scoped
- Instance: `(Sarah, AcmeCorp, founded)` → tenant-scoped, NEVER crosses to other tenants

### 9.5 Graph Pattern Inference Isolation

Graph pattern inference (Signal 4 in RelationRuler) runs ONLY within tenant's own data graph:
- Query: `MATCH (a:Entity)-[r]->(b:Entity) WHERE ... RETURN pattern`
- Scope: `ws_{id}_data` graph only
- Result: Never crosses tenant boundaries

Example:
- Tenant A's graph shows: Google → developed → Kubernetes, TensorFlow, JAX
- Tenant B's graph cannot see Tenant A's patterns
- Both can learn from global type-pair priors (Person, Org) → founded

---

## 10. Mutual Exclusion Constraints

From NELL research: Mutual exclusion is the strongest defense against error propagation in self-learning systems.

### 10.1 Constraint Definition

```python
@dataclass
class MutualExclusionConstraint:
    """Entity types that cannot overlap."""

    type_a: str          # e.g., "Technology"
    type_b: str          # e.g., "Person"
    reason: str          # Human-readable explanation
    confidence: float    # How certain (1.0 for seed constraints)

    def validate(self, entity: Entity) -> bool:
        """Check if entity violates this constraint."""
        return entity.type not in {self.type_a, self.type_b}
```

### 10.2 Seed Constraints

Shipped with Global TBox, curated from domain knowledge:

| Type A | Type B | Reason |
|--------|--------|--------|
| Technology | Person | Software cannot be a human |
| Organization | Location | A company is not a place |
| Person | Event | A person is not an occurrence |
| Concept | Tool | Abstract ideas are not physical instruments |
| Document | Person | Files are not humans |
| Metric | Location | Measurements are not places |

### 10.3 Constraint Application

Applied in `ontology_constrain` pipeline stage (after extraction):

```python
def constrain_entities(
    entities: list[Entity],
    constraints: list[MutualExclusionConstraint]
) -> tuple[list[Entity], list[Entity]]:
    """
    Returns:
        (accepted_entities, rejected_entities)
    """
    accepted = []
    rejected = []

    for entity in entities:
        violations = [c for c in constraints if not c.validate(entity)]

        if violations:
            entity.rejection_reason = f"Violates: {violations[0].reason}"
            rejected.append(entity)
        else:
            accepted.append(entity)

    return accepted, rejected
```

Rejected entities stored in `PipelineState.rejected` for inspection and audit.

### 10.4 Constraint Learning

Self-learned constraints (Phase 6+):
1. Observe co-occurrence patterns in data graph
2. If entity X never appears with type A and B simultaneously across large sample
3. Generate provisional constraint: (A, B) mutually exclusive
4. Quality gate: confidence >0.95, sample size >1000
5. Promote to Soft-TBox (tenant-scoped)

Global promotion: Only via service operator review, never automatic.

### 10.5 Constraint Violation Handling

When LLM extracts entity with conflicting types:

**Option 1: Reject (default)**
- Discard entity, log violation, increment error counter
- If error rate high: flag for manual review

**Option 2: Prefer higher confidence**
- If same entity extracted twice with different types, keep higher confidence
- Mark as ambiguous, candidate for reasoning validation

**Option 3: Human-in-the-loop**
- When `PromotionConfig.human_review = True`
- Queue violation for manual type resolution in Studio

---

## Summary

The ontology model provides:

1. **Foundational architecture** - Always on, no toggle, works from day one
2. **Clean separation** - Two FalkorDB graphs per workspace (ontology + data)
3. **Three-layer design** - Global TBox, Tenant Soft-TBox, Tenant ABox
4. **Minimal seed** - 14 entity types, 10-15 examples each, mutual exclusion constraints
5. **Progressive status** - Seed → Provisional → Confirmed type lifecycle
6. **Parameterized quality gates** - PromotionConfig tunable per workspace
7. **Dual-axis types** - Memory types (container) + Entity types (content)
8. **Graph-native storage** - Patterns as node properties, no separate store
9. **Privacy by design** - Vocabulary shared, instances isolated
10. **Error prevention** - Mutual exclusion constraints block cascading failures

This design delivers ontology-grounded extraction with zero architectural debt, clean multi-tenancy, and self-learning that converges to high quality through proven quality gates.
