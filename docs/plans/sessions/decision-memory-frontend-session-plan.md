# Decision Memory: Frontend + Client Integration — Reference Plan

**Date:** 2026-02-06
**Status:** MERGED INTO PHASE 6 — This plan's work items are now Steps 6-7 in the Phase 6 session plan.
**Prerequisite:** Decision Memory backend COMPLETE (model, manager, queries, extractor, evolver, inference, validation, edge schemas, API routes — all done)
**Estimated scope:** ~800 LOC React, ~150 LOC Python (client SDK)

---

## Current State

The Decision Memory backend is **~95% complete**:

| Component | Status | Location |
|-----------|--------|----------|
| Decision model | DONE | `smartmemory/models/decision.py` |
| DecisionManager | DONE | `smartmemory/decisions/manager.py` |
| DecisionQueries | DONE | `smartmemory/decisions/queries.py` |
| DecisionExtractor plugin | DONE | `smartmemory/plugins/extractors/decision.py` |
| DecisionConfidenceEvolver | DONE | `smartmemory/plugins/evolvers/decision_confidence.py` |
| MemoryValidator | DONE | `smartmemory/validation/memory_validator.py` |
| EdgeValidator | DONE | `smartmemory/validation/edge_validator.py` |
| InferenceEngine | DONE | `smartmemory/inference/engine.py` |
| InferenceRules | DONE | `smartmemory/inference/rules.py` |
| Graph edge schemas | DONE | `smartmemory/graph/models/schema_validator.py` (PRODUCED, SUPERSEDES, CONTRADICTS, INFLUENCES) |
| Service API routes | DONE | `smart-memory-service/api/routes/decisions.py` |
| Unit tests (1400+ lines) | DONE | `tests/unit/decisions/`, `tests/unit/models/test_decision.py`, `tests/unit/test_inference.py` |
| Design spec | DONE | `docs/plans/2026-02-04-decision-memory-causal-tracking-design.md` |

**What remains:**
1. Web UI components (smart-memory-web) — 0%
2. Client SDK methods (smart-memory-client) — 0%
3. Insights dashboard integration — 0% (covered by Phase 6 plan)

---

## Goal

Add Decision memory type support to the web UI and Python client SDK: filtering, display, provenance visualization, and creation.

---

## Step 1: Client SDK — Decision Methods

**File:** `smart-memory-client/smartmemory_client/client.py` — EDIT

Add Decision-specific convenience methods to `SmartMemoryClient`:

```python
# Decision lifecycle
def create_decision(
    self,
    content: str,
    decision_type: str = "inference",
    confidence: float = 0.8,
    source_trace_id: str | None = None,
    evidence_ids: list[str] | None = None,
    domain: str | None = None,
    tags: list[str] | None = None,
) -> dict:
    """Create a new decision."""
    return self._post("/memory/decisions", json={...})

def get_decision(self, decision_id: str) -> dict:
    """Get decision by ID with full metadata."""
    return self._get(f"/memory/decisions/{decision_id}")

def list_decisions(
    self,
    domain: str | None = None,
    decision_type: str | None = None,
    status: str = "active",
    limit: int = 50,
) -> list[dict]:
    """List decisions with optional filtering."""
    return self._get("/memory/decisions", params={...})

def supersede_decision(self, decision_id: str, new_content: str, reason: str, **kwargs) -> dict:
    """Replace a decision with a new one."""
    return self._put(f"/memory/decisions/{decision_id}/supersede", json={...})

def retract_decision(self, decision_id: str, reason: str) -> None:
    """Mark a decision as retracted."""
    return self._delete(f"/memory/decisions/{decision_id}", json={"reason": reason})

def reinforce_decision(self, decision_id: str, evidence_id: str) -> dict:
    """Reinforce a decision with supporting evidence."""
    return self._post(f"/memory/decisions/{decision_id}/reinforce", json={"evidence_id": evidence_id})

def get_provenance_chain(self, decision_id: str) -> list[dict]:
    """Get the full provenance chain for a decision."""
    return self._get(f"/memory/decisions/{decision_id}/provenance")

def get_causal_chain(self, decision_id: str, depth: int = 3) -> list[dict]:
    """Get causal chain traversal."""
    return self._get(f"/memory/decisions/{decision_id}/causal-chain", params={"depth": depth})
```

### Tests

**Create:** `smart-memory-client/tests/test_decision_methods.py` (~10 tests)
- Each method call with mock HTTP responses
- Error handling (404, 400, 500)

---

## Step 2: Web UI — Decision Type in Memory Selectors

**File:** `smart-memory-web/src/components/` — EDIT multiple files

Decision needs to appear in all memory type selectors, filters, and graph visualizations.

### 2a. Memory Type Color & Icon

**File to find:** The component that maps memory types to colors/icons (likely in a constants or theme file).

Add decision to the type map:
```javascript
const MEMORY_TYPE_CONFIG = {
    // ... existing types ...
    decision: {
        color: '#f59e0b',    // amber
        icon: 'Scale',       // or Gavel icon
        label: 'Decision',
    },
};
```

### 2b. Memory Type Filter

Find and update the memory type filter/selector components to include "decision" as a filterable type.

### 2c. Graph Visualization Colors

Update the graph color mapping so Decision nodes render with the amber color in the knowledge graph view.

---

## Step 3: Web UI — Decision Detail Card

**Create:** `smart-memory-web/src/components/decisions/DecisionCard.jsx`

Displays a single decision with its metadata:

```
┌─────────────────────────────────────────┐
│  🟡 Decision: inference    [Active]     │
│                                         │
│  "User prefers dark mode for all apps"  │
│                                         │
│  Confidence: ████████░░ 0.85            │
│  Reinforced: 3x  Contradicted: 1x      │
│  Stability: 0.75                        │
│                                         │
│  Domain: preferences  Tags: #ui, #ux   │
│  Source: reasoning trace #tr_abc123     │
│                                         │
│  [View Provenance] [Supersede] [Retract]│
└─────────────────────────────────────────┘
```

### Props
```typescript
interface DecisionCardProps {
    decision: {
        decision_id: string;
        content: string;
        decision_type: string;
        confidence: number;
        status: string;
        reinforcement_count: number;
        contradiction_count: number;
        domain?: string;
        tags?: string[];
        source_trace_id?: string;
        created_at: string;
    };
    onSupersede?: (id: string) => void;
    onRetract?: (id: string) => void;
    onViewProvenance?: (id: string) => void;
}
```

---

## Step 4: Web UI — Decision List Page/Panel

**Create:** `smart-memory-web/src/components/decisions/DecisionList.jsx`

Filterable list of decisions for the workspace.

### Features
- Filter by: domain, decision_type, status (active/superseded/retracted), confidence range
- Sort by: created_at, confidence, reinforcement_count
- Search by content text
- Pagination (offset/limit)

### Layout
```
┌──────────────────────────────────────────────────┐
│  Decisions                                        │
│  [Domain ▼] [Type ▼] [Status ▼] [Search...    ] │
├──────────────────────────────────────────────────┤
│  DecisionCard                                     │
│  DecisionCard                                     │
│  DecisionCard                                     │
│  ...                                              │
│  [Load More]                                      │
└──────────────────────────────────────────────────┘
```

---

## Step 5: Web UI — Provenance Visualization

**Create:** `smart-memory-web/src/components/decisions/ProvenanceChain.jsx`

Visual display of a decision's provenance: what evidence it was derived from, what reasoning trace produced it, and what it superseded.

### Layout

```
Evidence A ──DERIVED_FROM──┐
Evidence B ──DERIVED_FROM──┤
                           ▼
Reasoning Trace ──PRODUCED── Decision (v1)
                                  │
                             SUPERSEDED
                                  ▼
                             Decision (v2) ← current
```

Use a simple vertical timeline/tree layout (not a full graph). Each node is a clickable card linking to the source memory item.

### Data Flow

1. Call `GET /memory/decisions/{id}/provenance`
2. Response: `{chain: [{item_id, content_summary, memory_type, edge_type, confidence}]}`
3. Render as vertical chain with edge labels

---

## Step 6: Integration into Main App

### Route Registration

**File:** `smart-memory-web/src/` — find router config

Add a `/decisions` route pointing to the decisions list view.

### Navigation

Add "Decisions" to the sidebar/nav with the Scale icon and amber accent.

### Memory Detail View

When viewing a memory item of type "decision", show the DecisionCard + ProvenanceChain instead of the generic memory detail.

---

## Step 7: Tests + Report

### Client Tests
```bash
cd /Users/ruze/reg/my/SmartMemory/smart-memory-client
pytest tests/test_decision_methods.py -v
```

### Frontend Tests
```bash
cd /Users/ruze/reg/my/SmartMemory/smart-memory-web
npm run build
npm run lint
```

### Documentation
- Update `smart-memory-client/README.md` with decision methods
- Update CHANGELOG.md in affected repos

---

## Execution Order

```
Step 1 (Client SDK) → Step 2 (Type selectors) → Step 3 (DecisionCard)
    → Step 4 (DecisionList) → Step 5 (Provenance) → Step 6 (Integration) → Step 7 (Tests)
```

Steps 1 and 2 are independent and can run in parallel.

---

## Key Files Reference

### Existing files to read first
- `smart-memory-service/memory_service/api/routes/decisions.py` — API routes (to match SDK methods)
- `smart-memory-client/smartmemory_client/client.py` — existing SDK client
- `smart-memory-web/src/` — explore component structure
- Look for memory type config/constants file for color/icon mapping
- Look for existing memory list/detail components for pattern reference
- `smart-memory/smartmemory/models/decision.py` — Decision model (to understand fields)
- `smart-memory/smartmemory/decisions/queries.py` — Query patterns
