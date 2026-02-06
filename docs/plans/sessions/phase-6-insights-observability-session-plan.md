# Phase 6: Insights + Observability — Session-Ready Implementation Plan

**Date:** 2026-02-06
**Prerequisite:** Phases 1-4 COMPLETE
**Estimated scope:** ~600 LOC Python (backend), ~1500 LOC React (frontend — includes Decision Memory UI)

**Decisions (from user input):**
- Ontology data: Query **FalkorDB directly** from insights backend (not through main API)
- Decision Memory frontend: **Merged into Phase 6** (avoids two sessions editing same repos)

---

## Goal

Build metrics aggregation and three new dashboard panels in smart-memory-insights (pipeline metrics, ontology metrics, extraction quality), replace stub API endpoints with real data, and add Decision Memory UI components to smart-memory-web (type selectors, DecisionCard, provenance visualization).

---

## Codebase Context

**Frontend stack:** React 18, Vite 6, shadcn/ui (Radix), Recharts 2.15, Framer Motion, Tailwind CSS dark theme.

**Existing pages:** Dashboard, Operations, Graph, Analytics, Extraction, Vector.

**Backend:** FastAPI at `smart-memory-insights/server/observability/api.py`. Uses `DatabaseManager` for Redis + FalkorDB queries.

**API client:** `web/src/api/smartMemoryClient.js` — REST + WebSocket hybrid with subscribe/unsubscribe channels.

**Chart pattern:** Recharts components (`<AreaChart>`, `<LineChart>`, `<Pie>`, `<BarChart>`) with brand color tokens (cyan, mint, purple).

---

## Step 1: Metrics Aggregation Consumer (6.1)

**Create:** `smart-memory/smartmemory/pipeline/metrics_consumer.py` (~200 LOC)

Reads metric events from Redis Streams, aggregates into time-bucketed data, writes pre-aggregated metrics to Redis for fast dashboard reads.

### Architecture

```
Redis Stream (smartmemory:metrics:*)
    ↓
MetricsConsumer (background thread or standalone worker)
    ↓ reads events, aggregates into 5-min buckets
    ↓
Redis Hash/Sorted Set (smartmemory:metrics:agg:{metric_type}:{bucket})
    ↓
Insights API reads pre-aggregated data
```

### Key Class

```python
class MetricsConsumer:
    """Consume pipeline metric events and aggregate into time buckets."""

    BUCKET_SIZE_SECONDS = 300  # 5 minutes
    RETENTION_HOURS = 24       # Keep 24h of buckets

    def __init__(self, redis_client=None, stream_name="smartmemory:metrics:pipeline"):
        ...

    def run(self, max_iterations: int | None = None) -> int:
        """Process metric events. Returns count processed."""
        # Read from stream
        # For each event: extract stage_name, duration_ms, status, entity_count, etc.
        # Aggregate into bucket: HINCRBY for counts, running avg for latencies

    def get_aggregated(self, metric_type: str, hours: int = 1) -> list[dict]:
        """Read pre-aggregated buckets for dashboard consumption."""
        # Return [{timestamp, stage, avg_latency_ms, count, error_count, ...}]
```

### Metric Events Shape

Pipeline stages already emit via `emit_ctx()`:
```python
{
    "event_type": "stage_complete",
    "component": "entity_ruler",
    "data": {
        "stage": "entity_ruler",
        "duration_ms": 4.2,
        "entity_count": 5,
        "relation_count": 3,
        "status": "success"
    }
}
```

### Aggregated Bucket Shape

```python
{
    "timestamp": "2026-02-06T12:00:00Z",  # Bucket start
    "stage": "entity_ruler",
    "count": 42,
    "avg_latency_ms": 3.8,
    "p95_latency_ms": 8.1,
    "error_count": 1,
    "total_entities": 210,
    "total_relations": 126
}
```

---

## Step 2: Backend API Endpoints (6.2-6.5)

**File:** `smart-memory-insights/server/observability/api.py` — EDIT

Replace stubs and add new endpoints.

### New/Updated Endpoints

| Endpoint | Method | Purpose | Status |
|----------|--------|---------|--------|
| `/api/pipeline/metrics` | GET | Per-stage latency, throughput, error rates | NEW |
| `/api/pipeline/bottlenecks` | GET | Identify slowest stages | NEW |
| `/api/ontology/status` | GET | Type counts, pattern counts, convergence | NEW |
| `/api/ontology/growth` | GET | Time series: pattern count over time | NEW |
| `/api/ontology/promotion-rates` | GET | Promotion/rejection counts | NEW |
| `/api/extraction/stats` | GET | Real extraction stats (replace stub) | REPLACE |
| `/api/extraction/operations` | GET | Real extraction events (replace stub) | REPLACE |
| `/api/extraction/quality` | GET | Confidence distribution, type ratios | NEW |
| `/api/extraction/attribution` | GET | LLM vs ruler attribution breakdown | NEW |

### Implementation

Pipeline metrics query the MetricsConsumer's aggregated Redis data.
Ontology metrics query FalkorDB ontology graph directly using Cypher:

```python
# Type status distribution
"MATCH (t:EntityType) RETURN t.status, count(t)"

# Pattern counts by source
"MATCH (p:EntityPattern) RETURN p.source, count(p)"

# Growth over time (from pattern discovered_at timestamps)
"MATCH (p:EntityPattern) RETURN date(p.discovered_at) AS day, count(p) ORDER BY day"
```

Extraction quality queries event stream for recent extractions.

### Response Models

**File:** `smart-memory-insights/server/observability/models.py` — EDIT

```python
class PipelineMetrics(BaseModel):
    stages: list[StageMetric]  # [{stage, avg_latency_ms, count, error_rate, throughput}]
    total_memories_processed: int
    time_range_hours: int

class OntologyStatus(BaseModel):
    type_counts: dict[str, int]     # {seed: N, provisional: N, confirmed: N}
    pattern_counts: dict[str, int]  # {seed: N, promoted: N, llm_discovery: N}
    total_entity_types: int
    total_patterns: int
    convergence_estimate: float     # 0.0-1.0 based on discovery rate decline

class ExtractionQuality(BaseModel):
    avg_confidence: float
    confidence_distribution: list[dict]  # [{range: "0.8-0.9", count: N}]
    provisional_ratio: float             # provisional / total types
    attribution: dict[str, int]          # {entity_ruler: N, llm_extract: N, entity_ruler_learned: N}
```

---

## Step 3: Pipeline Metrics Dashboard Page (6.2)

**Create:** `smart-memory-insights/web/src/pages/Pipeline.jsx`

### Layout

```
┌──────────────────────────────────────────────────────┐
│  Pipeline Performance                    [1H] [6H] [24H] │
├──────────────────────────────────────────────────────┤
│                                                      │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   │
│  │ Total   │ │ Avg     │ │ Error   │ │ Through-│   │
│  │ Runs    │ │ Latency │ │ Rate    │ │ put/min │   │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘   │
│                                                      │
│  ┌────────────────────────────────────────────────┐  │
│  │  Per-Stage Latency (stacked bar chart)         │  │
│  │  classify | coref | simplify | ruler | llm | …│  │
│  └────────────────────────────────────────────────┘  │
│                                                      │
│  ┌────────────────────────────────────────────────┐  │
│  │  Throughput Over Time (area chart)             │  │
│  │  memories/min by stage                         │  │
│  └────────────────────────────────────────────────┘  │
│                                                      │
│  ┌────────────────────────────────────────────────┐  │
│  │  Stage Error Rates (bar chart)                 │  │
│  └────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────┘
```

### Components to Create

| File | Purpose |
|------|---------|
| `web/src/pages/Pipeline.jsx` | Page layout + data fetching |
| `web/src/components/pipeline/StageLatencyChart.jsx` | Horizontal stacked bar |
| `web/src/components/pipeline/ThroughputChart.jsx` | Area chart over time |
| `web/src/components/pipeline/StageErrorChart.jsx` | Bar chart per stage |
| `web/src/components/pipeline/PipelineSummaryCards.jsx` | 4 stat cards |

### Data Fetching

Add to `smartMemoryClient.js`:
```javascript
async getPipelineMetrics(hours = 1) {
    return this._fetch(`/api/pipeline/metrics?hours=${hours}`);
}
```

---

## Step 4: Ontology Metrics Dashboard Page (6.3)

**Create:** `smart-memory-insights/web/src/pages/Ontology.jsx`

### Layout

```
┌──────────────────────────────────────────────────────┐
│  Ontology Health                                     │
├──────────────────────────────────────────────────────┤
│                                                      │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   │
│  │ Total   │ │ Total   │ │ Confirmed│ │ Conver- │   │
│  │ Types   │ │ Patterns│ │ Rate %  │ │ gence   │   │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘   │
│                                                      │
│  ┌──────────────────────┐ ┌──────────────────────┐   │
│  │ Type Status (donut)  │ │ Pattern Layers (donut)│  │
│  │ seed/prov/confirmed  │ │ seed/promoted/tenant  │  │
│  └──────────────────────┘ └──────────────────────┘   │
│                                                      │
│  ┌────────────────────────────────────────────────┐  │
│  │  Convergence Curve (line chart)                │  │
│  │  X: memories processed, Y: new discoveries/100 │  │
│  │  Shows ruler quality approaching LLM quality    │  │
│  └────────────────────────────────────────────────┘  │
│                                                      │
│  ┌────────────────────────────────────────────────┐  │
│  │  Pattern Growth (area chart)                   │  │
│  │  Cumulative pattern count over time            │  │
│  └────────────────────────────────────────────────┘  │
│                                                      │
│  ┌────────────────────────────────────────────────┐  │
│  │  Promotion/Rejection Rates (bar chart)         │  │
│  └────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────┘
```

### Components to Create

| File | Purpose |
|------|---------|
| `web/src/pages/Ontology.jsx` | Page layout + data fetching |
| `web/src/components/ontology/TypeStatusChart.jsx` | Donut: seed/provisional/confirmed |
| `web/src/components/ontology/PatternLayersChart.jsx` | Donut: seed/promoted/tenant |
| `web/src/components/ontology/ConvergenceCurve.jsx` | Line chart: discovery rate decline |
| `web/src/components/ontology/PatternGrowthChart.jsx` | Area chart: cumulative patterns |
| `web/src/components/ontology/PromotionRatesChart.jsx` | Bar chart: accept/reject |
| `web/src/components/ontology/OntologySummaryCards.jsx` | 4 stat cards |

### Data Fetching

Add to `smartMemoryClient.js`:
```javascript
async getOntologyStatus() {
    return this._fetch('/api/ontology/status');
}
async getOntologyGrowth(days = 30) {
    return this._fetch(`/api/ontology/growth?days=${days}`);
}
async getPromotionRates(days = 7) {
    return this._fetch(`/api/ontology/promotion-rates?days=${days}`);
}
```

---

## Step 5: Extraction Quality Dashboard Update (6.4-6.5)

**File:** `smart-memory-insights/web/src/pages/Extraction.jsx` — EDIT

Replace existing stub-driven extraction page with real metrics.

### New Components

| File | Purpose |
|------|---------|
| `web/src/components/extraction/ConfidenceDistribution.jsx` | Histogram of confidence scores |
| `web/src/components/extraction/AttributionChart.jsx` | Pie: ruler vs LLM vs learned |
| `web/src/components/extraction/TypeRatioChart.jsx` | Bar: provisional vs confirmed |

### Updated Data Fetching

Replace stub methods in `smartMemoryClient.js`:
```javascript
async getExtractionStats() {
    return this._fetch('/api/extraction/stats');  // Now returns real data
}
async getExtractionQuality() {
    return this._fetch('/api/extraction/quality');
}
async getExtractionAttribution() {
    return this._fetch('/api/extraction/attribution');
}
```

---

## Step 6: Decision Memory — Client SDK

**File:** `smart-memory-client/smartmemory_client/client.py` — EDIT

Add Decision-specific convenience methods:

```python
def create_decision(self, content, decision_type="inference", confidence=0.8, **kwargs) -> dict
def get_decision(self, decision_id: str) -> dict
def list_decisions(self, domain=None, decision_type=None, status="active", limit=50) -> list[dict]
def supersede_decision(self, decision_id, new_content, reason, **kwargs) -> dict
def retract_decision(self, decision_id, reason) -> None
def reinforce_decision(self, decision_id, evidence_id) -> dict
def get_provenance_chain(self, decision_id) -> list[dict]
def get_causal_chain(self, decision_id, depth=3) -> list[dict]
```

**Tests:** `smart-memory-client/tests/test_decision_methods.py` (~10 tests)

---

## Step 7: Decision Memory — Web UI Components

**Repo:** `smart-memory-web/`

### 7a. Memory Type Config

Add "decision" to memory type color/icon mapping (amber `#f59e0b`, Scale icon).
Update memory type filter/selector components to include "decision".
Update graph visualization colors for Decision nodes.

### 7b. DecisionCard Component

**Create:** `smart-memory-web/src/components/decisions/DecisionCard.jsx`

Displays decision with confidence bar, reinforcement/contradiction counts, stability score, domain/tags, source trace link, and action buttons (View Provenance, Supersede, Retract).

### 7c. DecisionList Component

**Create:** `smart-memory-web/src/components/decisions/DecisionList.jsx`

Filterable list: domain, decision_type, status, confidence range. Sort by created_at/confidence/reinforcements. Search by content. Pagination.

### 7d. ProvenanceChain Component

**Create:** `smart-memory-web/src/components/decisions/ProvenanceChain.jsx`

Vertical timeline showing evidence → reasoning trace → decision chain. Each node clickable to navigate to source memory. Uses `GET /memory/decisions/{id}/provenance` API.

### 7e. Route & Navigation

Add `/decisions` route to smart-memory-web. Add "Decisions" to sidebar with Scale icon and amber accent. When viewing a decision memory item, show DecisionCard + ProvenanceChain.

---

## Step 8: Navigation & Routing (Insights)

**File:** `smart-memory-insights/web/src/pages/index.jsx` — EDIT

Add Pipeline and Ontology to the page router:

```javascript
{ path: '/Pipeline', element: <Pipeline /> },
{ path: '/Ontology', element: <Ontology /> },
```

**File:** `smart-memory-insights/web/src/App.jsx` — EDIT

Add navigation links for Pipeline and Ontology pages.

---

## Step 9: Tests + Report

### Backend Tests

**Create:** `smart-memory/tests/unit/pipeline_v2/test_metrics_consumer.py` (~8 tests)
- Event aggregation into buckets
- Bucket retention/cleanup
- Edge cases (empty streams, malformed events)

**Create:** `smart-memory-insights/tests/test_new_endpoints.py` (~10 tests)
- Pipeline metrics endpoint
- Ontology status/growth/promotion endpoints
- Extraction quality/attribution endpoints

### Client Tests

**Create:** `smart-memory-client/tests/test_decision_methods.py` (~10 tests)
- Each decision method with mock HTTP responses
- Error handling (404, 400, 500)

### Frontend Build Verification

```bash
cd /Users/ruze/reg/my/SmartMemory/smart-memory-web
npm run build && npm run lint

cd /Users/ruze/reg/my/SmartMemory/smart-memory-insights/web
npm run build && npm run lint
```

### Documentation

- Update CHANGELOG.md in all affected repos
- Mark Phase 6 COMPLETE in implementation plan
- Write `docs/plans/reports/phase-6-insights-observability-report.md`
- Update `smart-memory-client/README.md` with decision methods

---

## Execution Order

```
Step 1 (MetricsConsumer) → Step 2 (Backend endpoints) → Steps 3-5 (Insights Frontend, parallelizable)
    ↓
Step 6 (Decision Client SDK) → Steps 7a-7e (Decision Web UI)
    ↓
Step 8 (Insights Navigation) → Step 9 (Tests + Report)
```

Steps 3-5 and Steps 6-7 are independent tracks that can parallelize.

---

## Verification

```bash
# Backend
cd /Users/ruze/reg/my/SmartMemory/smart-memory
PYTHONPATH=. pytest tests/unit/pipeline_v2/test_metrics_consumer.py -v
ruff check --fix smartmemory/ && ruff format smartmemory/

# Insights server
cd /Users/ruze/reg/my/SmartMemory/smart-memory-insights
pytest tests/ -v

# Client SDK
cd /Users/ruze/reg/my/SmartMemory/smart-memory-client
pytest tests/test_decision_methods.py -v

# Insights frontend
cd /Users/ruze/reg/my/SmartMemory/smart-memory-insights/web
npm run build && npm run lint

# Web frontend
cd /Users/ruze/reg/my/SmartMemory/smart-memory-web
npm run build && npm run lint
```

---

## Key Files Reference

### Existing files to read first
- `smart-memory-insights/server/observability/api.py` — existing API endpoints (admin-only)
- `smart-memory-insights/server/observability/models.py` — Pydantic response models
- `smart-memory-insights/server/observability/database.py` — DatabaseManager (Redis + FalkorDB)
- `smart-memory-insights/web/src/api/smartMemoryClient.js` — API client (300+ lines)
- `smart-memory-insights/web/src/pages/Extraction.jsx` — existing extraction page
- `smart-memory-insights/web/src/pages/Analytics.jsx` — reference for chart patterns
- `smart-memory-insights/web/src/components/dashboard/MemoryTypeStats.jsx` — donut chart reference
- `smart-memory-insights/web/src/components/dashboard/OperationsChart.jsx` — area chart reference
- `smart-memory-insights/web/src/components/analytics/PerformanceChartCard.jsx` — dual-axis line chart
- `smart-memory/smartmemory/observability/events.py` — EventStream, RedisStreamQueue
- `smart-memory/smartmemory/observability/instrumentation.py` — emit_ctx() pattern

### Decision Memory files to read
- `smart-memory-service/memory_service/api/routes/decisions.py` — API routes (match SDK methods to these)
- `smart-memory-client/smartmemory_client/client.py` — existing SDK client
- `smart-memory-web/src/` — explore component structure, find memory type config
- `smart-memory/smartmemory/models/decision.py` — Decision model (understand fields)
- `smart-memory/smartmemory/decisions/queries.py` — Query patterns

### Color tokens for charts
```
cyan:   #00d4ff (primary)
mint:   #64ffda (secondary)
purple: #8b5cf6 (tertiary)
amber:  #f59e0b (decisions)
red:    #ef4444
blue:   #3b82f6
```
