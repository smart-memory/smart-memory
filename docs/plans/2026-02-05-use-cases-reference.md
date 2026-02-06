# SmartMemory Use Cases Reference

**Date:** 2026-02-05
**Purpose:** Comprehensive use case inventory for ontology and pipeline design decisions.
**Framing:** SmartMemory is a persistent memory layer for ALL AI use cases — not just chatbots, not just notes.

---

## Design Principle: Use Cases Are Parameter Combinations, Not Categories

Use cases are not discrete things to enumerate. They're points in a parameter space. The parameters govern process — they configure extraction, quality gates, self-learning, and enrichment behavior.

### The Parameter Space

| Parameter | Range | What It Governs |
|-----------|-------|-----------------|
| **Domain vocabulary** | tech, medical, legal, financial, academic, personal | Entity types to seed, patterns to learn |
| **Entity focus** | people, concepts, organizations, projects, events | EntityRuler priority, type-pair priors |
| **Relation depth** | shallow (1-hop CRM) → deep (multi-hop investigation) | Graph traversal strategy, extraction granularity |
| **Temporal sensitivity** | high (health, investment) → low (reference library) | Temporal extractor activation, date entity priority |
| **Contradiction tolerance** | low (legal, medical) → high (brainstorming) | Assertion challenger activation, confidence thresholds |
| **Ingestion pattern** | steady stream → bulk import → event-driven | Fast tier vs async-only, queue depth, chunking strategy |
| **Confidence requirement** | high (medical, legal) → low (casual notes) | Quality gate strictness, promotion thresholds, human review |
| **Privacy sensitivity** | high (therapy, health) → low (public research) | Scope isolation level, global promotion eligibility |
| **Scope** | personal → team → organization | Cross-user pattern promotion, shared TBox participation |
| **Memory type mix** | episodic-heavy → semantic-heavy → procedural-heavy | Evolver activation, linking priorities, decay rates |

### Parameters Govern Process

These aren't just descriptive — they configure the pipeline:

```
contradiction_tolerance: low  → assertion challenger runs on every ingest
relation_depth: deep          → graph pattern inference enabled, multi-hop enrichment
temporal_sensitivity: high    → temporal extractor runs, date entities prioritized
confidence_requirement: high  → quality gate thresholds raised, promotion needs approval
scope: team                   → cross-user pattern promotion enabled
ingestion_pattern: bulk       → async-only enrichment, fast tier for typing only
```

### Discovery Over Declaration

Users don't declare their use case. The system discovers parameter configuration from usage patterns:
- High contradiction sensitivity → user clips contradicting sources
- Deep relation depth → user queries multi-hop paths
- Domain vocabulary → emerging entity types (Case, Statute, Ruling → legal)

**The ontology isn't seeded per use case — it discovers its own configuration.** General seed types (person, organization, technology, concept, event, location, document) cover the base. Parameters are inferred, then tuned empirically.

### Resolved Decisions (2026-02-05)

1. **At launch:** All parameters exist with sensible defaults (general/balanced/shallow/medium across the board). No explicit setting needed. Discovery after ~50-100 memories.
2. **Per-workspace** (matches SecureSmartMemory tenant model). Per-user overrides Phase 7+.
3. **Both automatic AND visible in Studio.** Auto-inferred with confidence scores, manually overridable. "Advanced tuning" section — most users ignore it.
4. **Feedback loop:** Extraction quality metrics (Insights) + user corrections. Basic at launch (count trends). Full adaptive loop Phase 6+.
5. **Testing dataset:** Phase 7. Synthetic datasets for 3-5 representative parameter combos (high-confidence medical, shallow CRM, deep investigation).

---

## Use Case Summary Matrix

| # | Use Case | Primary Memory Types | Key Graph Feature | Primary User |
|---|----------|---------------------|-------------------|--------------|
| 1 | Literature Review & Research Synthesis | semantic, zettel | citation network, contradiction | Researcher |
| 2 | Health & Wellness Tracking | episodic, observation | temporal patterns, causality | Individual |
| 3 | Architecture Decision Records | decision, procedural | dependency graph, versioning | Engineer |
| 4 | Investigation Board (Journalism/Analysis) | semantic, episodic | multi-hop traversal, provenance | Journalist/Analyst |
| 5 | Personal CRM / Relationship Management | episodic, opinion | person-network graph | Networker/Founder |
| 6 | Consultant / Freelancer Project Continuity | procedural, episodic | project timeline, per-client | Freelancer |
| 7 | Reflective Journaling & Self-Knowledge | episodic, opinion | emotion patterns, evolution | Individual |
| 8 | Decision Audit Trail & Retrospective | decision, reasoning | prediction vs outcome, bias | Executive/Investor |
| 9 | Web Research Curation & Synthesis | zettel, semantic | source provenance, synthesis | Analyst/Student |
| 10 | Team Onboarding & Institutional Knowledge | procedural, semantic | expertise mapping | Team Lead/HR |
| 11 | Legal Case Research & Precedent Tracking | semantic, decision | precedent chains, contradiction | Lawyer |
| 12 | Product Feedback Synthesis | episodic, observation | aggregation, sentiment temporal | Product Manager |
| 13 | Learning & Study Companion | zettel, semantic | prerequisite chains, gaps | Student |
| 14 | Sales Intelligence & Deal Memory | episodic, opinion, working | objection patterns, sentiment | Sales Team |
| 15 | Content Creator Idea & Reference Library | zettel, semantic | cross-topic discovery | Writer/Creator |
| 16 | Belief & Worldview Evolution Tracker | opinion, reasoning | contradiction timeline | Thinker/Writer |
| 17 | Meeting Note Intelligence & Action Tracking | episodic, working | action extraction, decisions | Manager |
| 18 | Recruiting Pipeline Intelligence | decision, episodic | skill graph, silver medalist | Recruiter |
| 19 | Investment Thesis Tracking | decision, opinion | thesis evolution, contradiction | Investor |
| 20 | Technical Troubleshooting & Runbook Memory | procedural, episodic | causal chains, recurrence | SRE/DevOps |
| 21 | Competitive Intelligence Dashboard | semantic, observation | temporal competitor tracking | Strategy Analyst |
| 22 | Personal Project & Goal Tracking | working, episodic | blocker chains, patterns | Individual |
| 23 | Therapy / Coaching Session Memory | episodic, opinion | theme recurrence, evolution | Client/Practitioner |
| 24 | Conference & Event Knowledge Capture | episodic, observation | contact graph, trend comparison | Attendee |
| 25 | Personalized AI Assistant (Maya) | all types | full graph traversal | Any User |

---

## AI System Consumer Use Cases (Layer 1: How AI consumes memory)

These are not end-user workflows — they're the ways AI systems use persistent memory:

| Consumer | What It Needs From Memory | Memory Types | Graph Operations |
|----------|--------------------------|-------------|-----------------|
| **Chatbot/Assistant** (Maya) | Personal context for responses | all | search, traverse, recall |
| **RAG Pipeline** | Filtered, typed chunks for retrieval | semantic, zettel | vector search + type filtering |
| **Autonomous Agent** | Episodic recall + procedural execution | episodic, procedural | temporal query, procedure lookup |
| **Multi-Agent System** | Shared knowledge between agents | semantic, working | concurrent read, write coordination |
| **Code Assistant** | Project context persistence | procedural, decision | dependency graph, decision recall |
| **Workflow Automation** | Learned procedures, trigger conditions | procedural, observation | pattern match, conditional execution |
| **Analytics/BI** | Aggregated entity/relation statistics | all | aggregation, temporal comparison |
| **Recommendation Engine** | User preferences, historical patterns | opinion, episodic | preference graph, similarity |

---

## Memory Population Patterns (Layer 2: How memory gets filled)

| Pattern | Volume | Latency Requirement | Extraction Complexity |
|---------|--------|--------------------|-----------------------|
| **Direct user input** (notes, chat) | Low (5-50/day) | Real-time (<500ms) | High (conversational, implicit) |
| **Web clipping** (Capture extension) | Medium (10-100/day) | Near real-time | Medium (structured HTML) |
| **API ingestion** (programmatic) | High (bulk possible) | Async OK | Variable |
| **Agent self-logging** | High (continuous) | Async | Low (structured by agent) |
| **Document processing** (bulk import) | Very high (thousands) | Async | High (long-form, complex) |
| **Meeting transcripts** | Medium | Near real-time | High (conversational) |
| **Sensor/metric data** | Very high | Async | Low (structured) |

---

## Detailed Use Cases

### 1. Literature Review & Research Synthesis
- **User**: Researcher, PhD student, analyst
- **Trigger**: Starting a research project or systematic review
- **Workflow**: Ingest papers, extract claims/methods/findings, build citation graph, identify contradictions across studies, synthesize across 50-200 sources
- **Ontology needs**: Entities: Paper, Author, Method, Finding, Claim, Dataset. Relations: CITES, CONTRADICTS, EXTENDS, USES_METHOD, EVALUATES_ON
- **Success**: Complete literature map in hours instead of weeks. No contradicting finding missed.

### 2. Health & Wellness Tracking
- **User**: Individual tracking health, patient managing chronic condition
- **Trigger**: New symptom, medication change, doctor visit
- **Workflow**: Log symptoms/meds/activities, detect temporal patterns (X happens 2 days after Y), prepare for doctor visits with full history
- **Ontology needs**: Entities: Symptom, Medication, Activity, Metric, Condition, Provider. Relations: TRIGGERS, ALLEVIATES, CORRELATES, PRESCRIBED_FOR
- **Success**: Temporal causality surfaces. Doctor gets complete picture.

### 3. Architecture Decision Records
- **User**: Software engineer, tech lead, architect
- **Trigger**: Making a technical choice, reviewing past decisions, onboarding new team member
- **Workflow**: Record decisions with context/alternatives/reasoning, link to affected systems, track outcomes
- **Ontology needs**: Entities: Decision, System, Technology, Tradeoff, Constraint, Person. Relations: CHOSE_OVER, AFFECTS, DEPENDS_ON, SUPERSEDES, DECIDED_BY
- **Success**: No decision is unexplained. New engineers understand why things are the way they are.

### 4. Investigation Board
- **User**: Journalist, analyst, fraud investigator
- **Trigger**: Complex investigation with many entities and connections
- **Workflow**: Ingest documents/sources, extract entities, build connection graph, find hidden links, trace money/influence flows
- **Ontology needs**: Entities: Person, Organization, Document, Event, Location, Amount. Relations: OWNS, FUNDED_BY, CONNECTED_TO, MENTIONED_IN, PRESENT_AT
- **Success**: Multi-hop connections surface. Provenance chain from source to claim is complete.

### 5. Personal CRM / Relationship Management
- **User**: Networker, founder, sales professional
- **Trigger**: Meeting someone new, preparing for a meeting, following up
- **Workflow**: Record interactions, track preferences/interests, get context before meetings
- **Ontology needs**: Entities: Person, Organization, Interest, Event, Location. Relations: WORKS_AT, KNOWS, INTRODUCED_BY, INTERESTED_IN, MET_AT
- **Success**: Every interaction builds on context. No relationship detail lost.

### 6-25: [See summary matrix above — detailed workflows follow same pattern]

---

## Sources
- Mem0 architecture paper (arXiv 2504.19413)
- Google Research: Personal Knowledge Graphs Research Agenda
- NELL: Never-Ending Language Learner (AAAI 2015)
- ICIJ + Neo4j (Panama Papers investigation)
- PKM workflows (Zettelkasten, Second Brain, Obsidian/Roam patterns)
- AI agent memory frameworks survey (Graphlit)
- Patient-centric knowledge graphs (Frontiers in AI)
