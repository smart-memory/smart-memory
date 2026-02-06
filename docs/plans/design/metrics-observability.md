# Metrics and Observability Design

**Date:** 2026-02-06
**Version:** 1.0
**Status:** Design specification
**Parent:** [Implementation Plan](../2026-02-05-implementation-plan.md)

---

## 1. Architecture

### 1.1 Overview

The metrics system provides real-time observability into SmartMemory's ontology-grounded extraction pipeline. It is designed to be:

- **Decoupled**: Pipeline execution never waits for metric collection or aggregation
- **Lightweight**: Minimal overhead on ingestion path (<1ms per stage)
- **Fast**: Dashboard queries return in milliseconds via pre-aggregated data
- **Scalable**: Time-bucketed aggregation prevents unbounded growth
- **Event-driven**: Uses existing Redis Streams infrastructure

### 1.2 Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                 Pipeline Runner                             │
│  (PipelineState flows through StageCommands)                │
│                                                              │
│  classify → coreference → simplify → entity_ruler →         │
│  llm_extract → ontology_constrain → store → link →          │
│  enrich → evolve                                             │
│                                                              │
│  After each stage: emit metric event to Redis Stream        │
└──────────────────────┬──────────────────────────────────────┘
                       │ (async, fire-and-forget)
                       ▼
┌─────────────────────────────────────────────────────────────┐
│           Redis Stream: smartmemory:metrics:pipeline        │
│  Ordered log of metric events from all pipeline executions  │
└──────────────────────┬──────────────────────────────────────┘
                       │ (consumer group: metrics_aggregator)
                       ▼
┌─────────────────────────────────────────────────────────────┐
│          Metrics Aggregation Consumer                       │
│  Reads from stream, aggregates into 5-minute buckets        │
│  Computes: percentiles, counts, distributions                │
│  Writes pre-computed metrics to Redis                       │
└──────────────────────┬──────────────────────────────────────┘
                       │ (writes to Redis)
                       ▼
┌─────────────────────────────────────────────────────────────┐
│          Pre-Aggregated Metrics (Redis)                     │
│  - Hash: metrics:pipeline:{workspace_id}:{timestamp}        │
│  - Hash: metrics:ontology:{workspace_id}                    │
│  - Hash: metrics:extraction:{workspace_id}:{timestamp}      │
└──────────────────────┬──────────────────────────────────────┘
                       │ (HTTP GET)
                       ▼
┌─────────────────────────────────────────────────────────────┐
│          Insights Dashboard API                             │
│  GET /api/metrics/pipeline                                  │
│  GET /api/metrics/ontology                                  │
│  GET /api/metrics/extraction                                │
└─────────────────────────────────────────────────────────────┘
```

### 1.3 Key Design Decisions

**Decision: Redis Streams for event transport**
- Reuses existing infrastructure (same as event-bus transport for async pipeline)
- Ordered, persistent log with consumer groups for horizontal scaling
- Built-in backpressure and acknowledgment semantics
- No additional dependencies

**Decision: 5-minute time buckets**
- Balances granularity with storage efficiency
- Provides near-real-time updates (dashboard updates within 5 minutes)
- Simplifies retention policy (expire old bucket keys automatically)
- Aligns with typical observability time windows

**Decision: Pre-aggregation in consumer, not query-time**
- Dashboard reads are fast (simple Redis hash lookup, no computation)
- Aggregation cost paid once per bucket, not per dashboard load
- Multiple users viewing dashboard don't multiply computation cost
- Trade-off: 5-minute latency on metric updates (acceptable for this use case)

**Decision: Redis for aggregated storage (not FalkorDB)**
- Metrics are time-series data, not graph data
- Redis hashes + sorted sets are ideal for time-bucketed metrics
- TTL support for automatic cleanup
- Sub-millisecond read performance

---

## 2. Metric Categories

### 2.1 Pipeline Metrics

**Source:** PipelineState after each StageCommand.execute()

**Metrics collected:**

- **Per-stage latency**: Duration in milliseconds for each stage
  - Example stages: classify (2ms), entity_ruler (4ms), llm_extract (740ms), store (120ms)
  - Aggregated as: p50, p95, p99 percentiles per time bucket
- **Throughput**: Memories processed per minute
  - Total pipeline throughput
  - Per-stage throughput (for event-bus transport mode)
- **Error/retry rates**: Count of errors and retries per stage
  - Error types: APITimeout, ValidationError, ExtractionError, etc.
  - Retry count and success rate
- **Queue depth** (event-bus mode only): Number of pending events in Redis Stream per stage
- **Bottleneck detection**: Stage with highest latency contribution to total pipeline time

**Aggregation:**
- Time bucket: 5 minutes
- Scope: per workspace
- Storage: Redis hash `metrics:pipeline:{workspace_id}:{timestamp_bucket}`
- TTL: 7 days

**Example aggregated data:**

```json
{
  "timestamp_bucket": "2026-02-06T14:30:00Z",
  "workspace_id": "ws_abc123",
  "total_memories_processed": 142,
  "throughput_per_min": 28.4,
  "stages": {
    "classify": {
      "count": 142,
      "duration_p50_ms": 2.1,
      "duration_p95_ms": 3.8,
      "duration_p99_ms": 5.2,
      "errors": 0,
      "retries": 0
    },
    "entity_ruler": {
      "count": 142,
      "duration_p50_ms": 4.3,
      "duration_p95_ms": 6.1,
      "duration_p99_ms": 8.7,
      "errors": 0,
      "retries": 0
    },
    "llm_extract": {
      "count": 142,
      "duration_p50_ms": 687,
      "duration_p95_ms": 892,
      "duration_p99_ms": 1124,
      "errors": 3,
      "retries": 5,
      "api_timeout_errors": 2,
      "validation_errors": 1
    },
    "ontology_constrain": {
      "count": 142,
      "duration_p50_ms": 12.4,
      "duration_p95_ms": 18.9,
      "duration_p99_ms": 24.1,
      "errors": 0,
      "retries": 0
    },
    "store": {
      "count": 142,
      "duration_p50_ms": 118.2,
      "duration_p95_ms": 156.7,
      "duration_p99_ms": 189.3,
      "errors": 0,
      "retries": 0
    }
  },
  "bottleneck_stage": "llm_extract",
  "total_errors": 3,
  "total_retries": 5
}
```

### 2.2 Ontology Metrics

**Source:** Ontology graph (`ws_{id}_ontology`) and promotion events from `ontology_constrain` stage

**Metrics collected:**

- **Type registry growth**: Count of entity types over time
  - Total types
  - Breakdown by status: seed (14) / provisional / confirmed
- **EntityRuler pattern count**: Number of patterns loaded in ruler
  - Total patterns
  - New patterns added in time bucket
  - Pattern source: seed / learned-global / learned-tenant
- **Coverage rate**: Percentage of entity extractions using ruler vs LLM-only
  - Formula: `ruler_hits / (ruler_hits + llm_only_discoveries)`
  - Indicates how much the ruler has learned
- **Convergence curve**: Ruler quality approaching LLM quality over time
  - Measured via extraction attribution metrics (see 2.3)
  - Convergence target: ruler_coverage > 95%
- **Promotion/rejection rates**: Type status transitions through quality gate
  - `provisional_to_confirmed_count` (successful promotions)
  - `rejected_count` (failed quality gate)
  - Promotion reasons: frequency threshold / confidence threshold / reasoning validation
- **Type status distribution**: Current snapshot of type counts by status

**Aggregation:**
- Type counts: queried from ontology graph on-demand, cached in Redis for 60 seconds
- Pattern counts: queried from ontology graph on-demand, cached
- Promotion events: aggregated per 5-minute bucket
- Convergence history: stored as sorted set `metrics:ontology:convergence:{workspace_id}` (score = timestamp)

**Storage:**
- Current state: Redis hash `metrics:ontology:{workspace_id}` (no TTL, updated in place)
- Promotion history: Redis sorted set `metrics:ontology:promotions:{workspace_id}` (TTL: 30 days)

**Example ontology metrics:**

```json
{
  "timestamp": "2026-02-06T14:35:22Z",
  "workspace_id": "ws_abc123",
  "type_counts": {
    "seed": 14,
    "provisional": 127,
    "confirmed": 43,
    "total": 184
  },
  "entity_ruler": {
    "pattern_count": 43,
    "new_patterns_last_5min": 2,
    "coverage_rate": 0.689,
    "pattern_sources": {
      "seed": 14,
      "learned_global": 12,
      "learned_tenant": 17
    }
  },
  "promotion_events_last_5min": {
    "provisional_to_confirmed": 2,
    "rejected": 0,
    "reasons": {
      "frequency_threshold": 1,
      "confidence_threshold": 1,
      "reasoning_validation": 0
    }
  },
  "convergence": {
    "ruler_coverage_7d_ago": 0.42,
    "ruler_coverage_now": 0.689,
    "trend": "improving",
    "memories_processed": 8200
  }
}
```

### 2.3 Extraction Quality Metrics

**Source:** `ontology_constrain` stage output and PipelineState

**Metrics collected:**

- **Entity count per memory**: Distribution of how many entities are extracted per memory
  - Mean, median, p95
  - Breakdown by entity type (Person, Organization, Technology, Concept, etc.)
- **Relation count per memory**: Distribution of how many relations are extracted
  - Mean, median, p95
- **Confidence distribution**: Histogram of confidence scores across extractions
  - Bins: [0.0-0.5), [0.5-0.7), [0.7-0.8), [0.8-0.9), [0.9-1.0]
  - Shows extraction quality and certainty
- **Provisional vs confirmed type ratio**: New extractions using provisional (unvalidated) types
  - Formula: `provisional_type_usage / total_entity_count`
  - High ratio indicates the domain is still being learned
- **LLM vs ruler attribution**: Which extractor found each entity
  - `ruler_only_count`: Entities found only by EntityRuler (fast tier)
  - `llm_only_count`: Entities found only by LLM (enrichment tier)
  - `both_count`: Entities found by both (ruler + LLM confirmation)
  - Ruler coverage: `(ruler_only + both) / total_entities`

**Aggregation:**
- Time bucket: 5 minutes
- Scope: per workspace
- Storage: Redis hash `metrics:extraction:{workspace_id}:{timestamp_bucket}`
- TTL: 7 days

**Example extraction quality metrics:**

```json
{
  "timestamp_bucket": "2026-02-06T14:30:00Z",
  "workspace_id": "ws_abc123",
  "memories_in_bucket": 142,
  "entity_stats": {
    "count_mean": 7.3,
    "count_median": 6,
    "count_p95": 14,
    "by_type": {
      "Person": 312,
      "Organization": 189,
      "Technology": 156,
      "Concept": 478,
      "Event": 87,
      "Location": 134,
      "other": 231
    }
  },
  "relation_stats": {
    "count_mean": 4.1,
    "count_median": 3,
    "count_p95": 9,
    "total_relations": 582
  },
  "confidence_distribution": {
    "[0.0-0.5)": 12,
    "[0.5-0.7)": 34,
    "[0.7-0.8)": 67,
    "[0.8-0.9)": 189,
    "[0.9-1.0]": 424
  },
  "type_usage": {
    "provisional_ratio": 0.31,
    "confirmed_ratio": 0.69
  },
  "extractor_attribution": {
    "ruler_only": 412,
    "llm_only": 87,
    "both": 227,
    "total_entities": 726,
    "ruler_coverage": 0.881
  }
}
```

---

## 3. Event Schema

### 3.1 Base Event Structure

All metric events emitted to Redis Stream follow this schema:

```python
{
    "event_id": str,           # UUID for traceability
    "timestamp": float,        # Unix timestamp (seconds since epoch)
    "workspace_id": str,       # Tenant scope
    "event_type": str,         # Event category (stage_complete, stage_error, etc.)
    "stage": str,              # Pipeline stage name
    "pipeline_id": str,        # Unique pipeline execution ID
    "data": dict               # Stage-specific metrics
}
```

### 3.2 Event Types and Payloads

**Stage completion event:**

```python
{
    "event_id": "evt_abc123",
    "timestamp": 1738852800.123,
    "workspace_id": "ws_456",
    "event_type": "stage_complete",
    "stage": "entity_ruler",
    "pipeline_id": "pip_xyz789",
    "data": {
        "duration_ms": 4.3,
        "entities_found": 7,
        "patterns_matched": 5,
        "status": "success"
    }
}
```

**Stage error event:**

```python
{
    "event_id": "evt_def456",
    "timestamp": 1738852801.456,
    "workspace_id": "ws_456",
    "event_type": "stage_error",
    "stage": "llm_extract",
    "pipeline_id": "pip_xyz789",
    "data": {
        "duration_ms": 1523,
        "error_type": "APITimeout",
        "error_message": "OpenAI API timeout after 30s",
        "retry_count": 2,
        "max_retries": 3,
        "fallback_used": false
    }
}
```

**Ontology promotion event:**

```python
{
    "event_id": "evt_ghi789",
    "timestamp": 1738852802.789,
    "workspace_id": "ws_456",
    "event_type": "type_promoted",
    "stage": "ontology_constrain",
    "pipeline_id": "pip_xyz789",
    "data": {
        "entity_type": "ProductName",
        "previous_status": "provisional",
        "new_status": "confirmed",
        "promotion_reason": "confidence_threshold",
        "frequency": 8,
        "confidence": 0.92
    }
}
```

**Extraction attribution event:**

```python
{
    "event_id": "evt_jkl012",
    "timestamp": 1738852803.012,
    "workspace_id": "ws_456",
    "event_type": "extraction_attribution",
    "stage": "ontology_constrain",
    "pipeline_id": "pip_xyz789",
    "data": {
        "entity_name": "Claude",
        "entity_type": "Person",
        "extractors": ["entity_ruler", "llm_extract"],
        "confidence": 0.95,
        "type_status": "confirmed"
    }
}
```

**Pipeline completion event:**

```python
{
    "event_id": "evt_mno345",
    "timestamp": 1738852804.345,
    "workspace_id": "ws_456",
    "event_type": "pipeline_complete",
    "stage": null,
    "pipeline_id": "pip_xyz789",
    "data": {
        "total_duration_ms": 892,
        "stages_completed": 10,
        "stages_skipped": 0,
        "stages_failed": 0,
        "memory_type": "episodic",
        "entity_count": 15,
        "relation_count": 6
    }
}
```

### 3.3 Emission Points in Pipeline

Metric events are emitted at the following points:

1. **After each StageCommand.execute() completes** (success or error)
   - Includes stage-specific metrics (entities extracted, duration, etc.)
2. **After type promotion** in `ontology_constrain` stage
   - Captures provisional → confirmed transitions
3. **After extraction merging** in `ontology_constrain` stage
   - Captures ruler vs LLM attribution for each entity
4. **After full pipeline completion**
   - Summary event with total duration and counts

**Emission pattern (fire-and-forget):**

```python
def emit_metric(self, event: dict) -> None:
    """
    Fire-and-forget metric emission.
    Pipeline never blocks on metrics.
    """
    try:
        self.redis.xadd(
            "smartmemory:metrics:pipeline",
            event,
            maxlen=100_000,  # Bounded stream, ~10MB
        )
    except Exception:
        # Metrics loss is acceptable, pipeline execution is not
        pass
```

---

## 4. Aggregation Consumer

### 4.1 Consumer Architecture

**Component:** `MetricsAggregationConsumer`
**Location:** `smartmemory/pipeline/metrics_consumer.py`

**Responsibilities:**
1. Read metric events from Redis Stream `smartmemory:metrics:pipeline`
2. Group events into 5-minute time buckets by workspace
3. Compute aggregations (counts, percentiles, distributions, histograms)
4. Write pre-aggregated results to Redis
5. Acknowledge processed events to prevent reprocessing

**Consumer group configuration:**
- Consumer group name: `metrics_aggregator`
- Consumer name: `consumer-{hostname}-{pid}`
- Read batch size: 100 events
- Block time: 5000ms (5 seconds)

### 4.2 Time Bucketing Logic

**Time bucket calculation:**

```python
def get_time_bucket(timestamp: float, bucket_size_minutes: int = 5) -> str:
    """
    Round timestamp down to nearest bucket boundary.

    Example:
        timestamp = 1738852823 (2026-02-06T14:33:43Z)
        bucket_size = 5 minutes
        result = "2026-02-06T14:30:00Z"
    """
    bucket_seconds = bucket_size_minutes * 60
    bucket_timestamp = int(timestamp // bucket_seconds) * bucket_seconds
    return datetime.fromtimestamp(bucket_timestamp, tz=timezone.utc).isoformat()
```

**Bucket window:**
- Events are grouped into non-overlapping 5-minute windows
- Window closes after 5 minutes of inactivity (no new events for that bucket)
- Once closed, bucket is finalized and written to Redis

### 4.3 Aggregation Logic

**Per-stage aggregation:**

```python
class StageMetrics:
    """Accumulates metrics for a single stage within a time bucket."""

    def __init__(self):
        self.durations = []        # List of durations for percentile calculation
        self.error_count = 0       # Count of errors
        self.retry_count = 0       # Total retries across all executions
        self.count = 0             # Total stage executions
        self.error_types = {}      # Counter for error types

    def add_event(self, event):
        """Process a single event and update aggregates."""
        self.count += 1

        if event['data'].get('duration_ms'):
            self.durations.append(event['data']['duration_ms'])

        if event['event_type'] == 'stage_error':
            self.error_count += 1
            error_type = event['data'].get('error_type', 'Unknown')
            self.error_types[error_type] = self.error_types.get(error_type, 0) + 1

        if event['data'].get('retry_count'):
            self.retry_count += event['data']['retry_count']

    def compute_percentiles(self):
        """Calculate p50, p95, p99 latency percentiles."""
        if not self.durations:
            return None
        return {
            'p50': float(np.percentile(self.durations, 50)),
            'p95': float(np.percentile(self.durations, 95)),
            'p99': float(np.percentile(self.durations, 99))
        }
```

**Extraction quality aggregation:**

```python
class ExtractionQualityAggregator:
    """Aggregates extraction quality metrics across memories."""

    def __init__(self):
        self.entity_counts = []          # List of entity counts per memory
        self.relation_counts = []        # List of relation counts per memory
        self.confidence_bins = {         # Histogram bins
            "[0.0-0.5)": 0,
            "[0.5-0.7)": 0,
            "[0.7-0.8)": 0,
            "[0.8-0.9)": 0,
            "[0.9-1.0]": 0
        }
        self.entity_types = {}           # Counter for entity types
        self.provisional_count = 0       # Entities with provisional types
        self.confirmed_count = 0         # Entities with confirmed types
        self.ruler_only = 0              # Entities found only by ruler
        self.llm_only = 0                # Entities found only by LLM
        self.both = 0                    # Entities found by both

    def add_extraction_event(self, event):
        """Process extraction attribution events."""
        data = event['data']

        # Count entity by type
        entity_type = data.get('entity_type', 'other')
        self.entity_types[entity_type] = self.entity_types.get(entity_type, 0) + 1

        # Track type status
        if data.get('type_status') == 'provisional':
            self.provisional_count += 1
        else:
            self.confirmed_count += 1

        # Track extractor attribution
        extractors = data.get('extractors', [])
        if 'entity_ruler' in extractors and 'llm_extract' in extractors:
            self.both += 1
        elif 'entity_ruler' in extractors:
            self.ruler_only += 1
        elif 'llm_extract' in extractors:
            self.llm_only += 1

        # Track confidence distribution
        confidence = data.get('confidence', 0.0)
        if confidence < 0.5:
            self.confidence_bins["[0.0-0.5)"] += 1
        elif confidence < 0.7:
            self.confidence_bins["[0.5-0.7)"] += 1
        elif confidence < 0.8:
            self.confidence_bins["[0.7-0.8)"] += 1
        elif confidence < 0.9:
            self.confidence_bins["[0.8-0.9)"] += 1
        else:
            self.confidence_bins["[0.9-1.0]"] += 1
```

### 4.4 Storage Keys and TTL

**Pipeline metrics:**
- Key pattern: `metrics:pipeline:{workspace_id}:{timestamp_bucket}`
- Type: Redis Hash
- TTL: 7 days
- Example: `metrics:pipeline:ws_abc123:2026-02-06T14:30:00Z`

**Ontology current state:**
- Key pattern: `metrics:ontology:{workspace_id}`
- Type: Redis Hash
- TTL: None (updated in place, always reflects current state)

**Ontology promotion history:**
- Key pattern: `metrics:ontology:promotions:{workspace_id}`
- Type: Redis Sorted Set (score = timestamp of promotion event)
- TTL: 30 days

**Extraction quality metrics:**
- Key pattern: `metrics:extraction:{workspace_id}:{timestamp_bucket}`
- Type: Redis Hash
- TTL: 7 days

**Convergence history:**
- Key pattern: `metrics:ontology:convergence:{workspace_id}`
- Type: Redis Sorted Set (score = timestamp, value = JSON with coverage metrics)
- TTL: 90 days (long-term trend analysis)

### 4.5 Deployment

**Development mode:**
- Run as background thread within service process
- Auto-start when `smart-memory-service` starts
- Single consumer instance

**Production mode:**
- Run as separate process/container
- Docker Compose service: `metrics-consumer`
- Kubernetes Deployment with replica count = 1 initially
- Consumer groups prevent duplicate processing if scaled horizontally

**Scaling considerations:**
- Single consumer sufficient for <10,000 memories/day
- Consumer groups allow horizontal scaling if needed
- Monitor lag metric: `events_in_stream - acknowledged_events`
- Alert if lag > 10,000 events (indicates consumer falling behind)

---

## 5. Insights Dashboard Integration

### 5.1 API Endpoints

**New routes in smart-memory-service:**

```
GET /api/metrics/pipeline
GET /api/metrics/ontology
GET /api/metrics/extraction
```

**Query parameters:**

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `workspace_id` | string | Workspace scope (from auth) | Required |
| `start` | ISO8601 | Start of time range | 24 hours ago |
| `end` | ISO8601 | End of time range | Now |

**Response schema for `/api/metrics/pipeline`:**

```json
{
  "workspace_id": "ws_abc123",
  "start": "2026-02-06T14:00:00Z",
  "end": "2026-02-06T15:00:00Z",
  "buckets": [
    {
      "timestamp_bucket": "2026-02-06T14:00:00Z",
      "total_memories_processed": 142,
      "throughput_per_min": 28.4,
      "stages": {
        "classify": { /* StageMetrics */ },
        "entity_ruler": { /* StageMetrics */ },
        "llm_extract": { /* StageMetrics */ }
      },
      "bottleneck_stage": "llm_extract",
      "total_errors": 3,
      "total_retries": 5
    }
  ]
}
```

**Response schema for `/api/metrics/ontology`:**

```json
{
  "workspace_id": "ws_abc123",
  "timestamp": "2026-02-06T14:35:22Z",
  "type_counts": {
    "seed": 14,
    "provisional": 127,
    "confirmed": 43,
    "total": 184
  },
  "entity_ruler": {
    "pattern_count": 43,
    "new_patterns_last_5min": 2,
    "coverage_rate": 0.689
  },
  "promotion_events_last_5min": {
    "provisional_to_confirmed": 2,
    "rejected": 0
  },
  "convergence": {
    "ruler_coverage_7d_ago": 0.42,
    "ruler_coverage_now": 0.689,
    "trend": "improving"
  }
}
```

**Response schema for `/api/metrics/extraction`:**

```json
{
  "workspace_id": "ws_abc123",
  "start": "2026-02-06T14:00:00Z",
  "end": "2026-02-06T15:00:00Z",
  "buckets": [
    {
      "timestamp_bucket": "2026-02-06T14:00:00Z",
      "entity_stats": { /* mean, median, p95, by_type */ },
      "relation_stats": { /* mean, median, p95 */ },
      "confidence_distribution": { /* histogram */ },
      "type_usage": { /* provisional vs confirmed ratios */ },
      "extractor_attribution": { /* ruler vs LLM counts */ }
    }
  ]
}
```

### 5.2 Dashboard Sections

The Insights dashboard will have three new sections, matching the three metric categories.

#### Section 1: Pipeline Performance Dashboard

**Location:** `smart-memory-insights/web/src/pages/PipelineDashboard.jsx` (new component)

**Visualizations:**

1. **Line chart: Per-stage latency over time**
   - X-axis: Time (5-minute buckets)
   - Y-axis: Duration (ms, log scale)
   - Lines: p50, p95, p99 for each stage
   - Highlight bottleneck stage in red

2. **Bar chart: Current throughput**
   - X-axis: Stage name
   - Y-axis: Memories/minute
   - Shows current 5-minute bucket throughput

3. **Table: Stage-by-stage breakdown**
   - Columns: Stage | Count | p50 | p95 | p99 | Errors | Retries
   - Sortable by any column
   - Error cells highlighted in red if error_rate > 1%

4. **Alert badge: Bottleneck detection**
   - Displays stage name with highest latency contribution
   - "llm_extract is the bottleneck (82% of total time)"

5. **Line chart: Queue depth over time** (event-bus mode only)
   - X-axis: Time
   - Y-axis: Events in queue
   - Separate line per stage
   - Alert if queue depth > 1000

#### Section 2: Ontology Growth Dashboard

**Location:** `smart-memory-insights/web/src/pages/OntologyDashboard.jsx` (new component)

**Visualizations:**

1. **Stacked area chart: Type registry growth**
   - X-axis: Memory count (or time)
   - Y-axis: Type count
   - Three areas: seed (constant), provisional (growing), confirmed (growing)
   - Shows ontology learning over time

2. **Line chart: EntityRuler pattern count**
   - X-axis: Memory count (or time)
   - Y-axis: Pattern count
   - Single line showing cumulative pattern growth
   - Inflection point indicates convergence

3. **Progress bar: Coverage rate**
   - Current ruler coverage: 68.9%
   - Target: 95%
   - Color: green if >90%, yellow if 70-90%, red if <70%

4. **Line chart: Convergence curve**
   - X-axis: Memory count
   - Y-axis: Ruler coverage rate (%)
   - Shows ruler quality approaching LLM quality
   - Horizontal line at 95% target

5. **Table: Recent promotion events**
   - Columns: Type | Prev Status | New Status | Reason | Frequency | Confidence
   - Shows last 20 promotions
   - "ProductName | provisional | confirmed | confidence_threshold | 8 | 0.92"

#### Section 3: Extraction Quality Dashboard

**Location:** `smart-memory-insights/web/src/pages/ExtractionDashboard.jsx` (new component)

**Visualizations:**

1. **Histogram: Entity count distribution per memory**
   - X-axis: Entity count bins (0-5, 6-10, 11-15, 16-20, 21+)
   - Y-axis: Number of memories
   - Shows typical extraction density

2. **Histogram: Relation count distribution per memory**
   - X-axis: Relation count bins (0-2, 3-5, 6-10, 11+)
   - Y-axis: Number of memories

3. **Bar chart: Confidence distribution**
   - X-axis: Confidence bins ([0.0-0.5), [0.5-0.7), [0.7-0.8), [0.8-0.9), [0.9-1.0])
   - Y-axis: Entity count
   - Color gradient from red (low confidence) to green (high confidence)

4. **Pie chart: Provisional vs confirmed type usage**
   - Two slices: provisional (31%), confirmed (69%)
   - High provisional ratio indicates active learning

5. **Stacked bar chart: Extractor attribution**
   - Three segments: ruler-only, LLM-only, both
   - X-axis: Time buckets
   - Shows ruler learning over time (ruler-only should grow)

6. **Table: Top entity types by frequency**
   - Columns: Type | Count | % of Total
   - Sorted by count descending
   - "Person | 312 | 21.4%"

### 5.3 Replacing Stub Functions

**Current stubs in `smartMemoryClient.js` (lines 283-289):**

```javascript
async getExtractionStats() {
  return this.apiCall('/api/extraction/stats');
}

async getExtractionOperations() {
  return this.apiCall('/api/extraction/operations');
}
```

**Implementation strategy:**

1. **Backend routes** (new file: `memory_service/api/routes/metrics.py`):
   - Implement `/api/metrics/extraction` → reads `metrics:extraction:*` Redis keys
   - Implement `/api/metrics/pipeline` → reads `metrics:pipeline:*` Redis keys
   - Implement `/api/metrics/ontology` → reads `metrics:ontology:*` Redis keys

2. **Client mapping** (update `smartMemoryClient.js`):
   - `getExtractionStats()` → calls `/api/metrics/extraction`
   - `getExtractionOperations()` → calls `/api/metrics/pipeline`
   - Keep function names for backward compatibility
   - No UI code changes needed (already calls these functions)

3. **Migration path**:
   - Deploy backend routes first (will return empty arrays until metrics consumer runs)
   - Deploy metrics consumer
   - Wait 5 minutes for first bucket aggregation
   - Dashboard automatically populates with real data

---

## 6. Current State

### 6.1 Existing Components

**IngestionObserver (331 LOC):**
- **Location:** `smartmemory/memory/ingestion/observer.py`
- **Status:** Active, currently emits events at ingestion stage transitions
- **Capabilities:**
  - `emit_event(event_type, data, component)` — Generic event emission to Redis Streams
  - `track_stage(stage_name)` — Context manager for automatic stage timing
  - `emit_extraction_results()` — Extraction entity/relation counts
  - `emit_performance_metrics()` — Performance summary events
  - Uses `EventSpooler` to write to Redis Streams

**Reuse strategy:**
- Adapt `IngestionObserver` to emit new metric event schema defined in Section 3
- Keep `track_stage()` context manager pattern (already in use)
- Add new metric emission methods:
  - `emit_ontology_promotion()`
  - `emit_extraction_attribution()`
  - `emit_stage_metrics()` (generalized version of existing methods)
- Rename class to `PipelineMetricsEmitter` for clarity
- Update all pipeline stages to use new emitter

**EventSpooler:**
- **Location:** `smartmemory/observability/event_spooler.py`
- **Status:** Already writes to Redis Streams
- **Used by:** IngestionObserver
- **Changes needed:** None (works as-is)

**SmartMemoryClient.js:**
- **Location:** `smart-memory-insights/web/src/api/smartMemoryClient.js`
- **Lines 283-289:** Stub implementations of `getExtractionStats()` and `getExtractionOperations()`
- **Status:** Already has API call infrastructure
- **Changes needed:** Only backend route updates, no client-side changes

### 6.2 Redis Streams

**Current usage:**
- Stream: `smartmemory:events`
- Purpose: Real-time event notifications for Insights UI (WebSocket subscriptions)
- Consumer: Insights dashboard (WebSocket listeners)

**New usage:**
- Stream: `smartmemory:metrics:pipeline`
- Purpose: Metric event log for aggregation
- Consumer: `MetricsAggregationConsumer` (consumer group)

**Why separate streams:**
- Events stream: real-time display, no aggregation, WebSocket broadcast
- Metrics stream: batch aggregation, consumer groups, no real-time requirement
- Decouples concerns, allows different retention policies (events: 1 day, metrics: 7 days)

### 6.3 Implementation Gaps

**Components to create:**
1. **MetricsAggregationConsumer** (~200 LOC)
   - File: `smartmemory/pipeline/metrics_consumer.py`
   - Reads from Redis Stream, aggregates, writes to Redis

2. **Metric emission from new pipeline stages** (~100 LOC)
   - Update `entity_ruler` stage to emit ruler hit metrics
   - Update `ontology_constrain` stage to emit promotion events and attribution
   - Update all stages to emit standardized completion events

3. **API routes for `/api/metrics/*`** (~150 LOC)
   - File: `memory_service/api/routes/metrics.py`
   - Three endpoints: pipeline, ontology, extraction

4. **Dashboard React components** (~800 LOC total)
   - `PipelineDashboard.jsx` (~300 LOC)
   - `OntologyDashboard.jsx` (~300 LOC)
   - `ExtractionDashboard.jsx` (~200 LOC)

5. **Ontology graph queries** (~50 LOC)
   - Query type counts by status from ontology graph
   - Query EntityRuler pattern counts

**Components to modify:**
1. **IngestionObserver → PipelineMetricsEmitter** (rename + extend)
   - Add new event types
   - Align with Section 3 event schema

2. **Pipeline stages** (minimal changes)
   - Call `emit_metric()` after execute()
   - Pass stage-specific data

3. **Backend routes** (add new file)
   - `memory_service/api/routes/metrics.py`

---

## 7. Performance Considerations

### 7.1 Pipeline Overhead

**Metric emission cost:**
- Event serialization: ~0.1ms per event (JSON encoding)
- Redis Stream write (xadd): ~0.5ms per event (async, non-blocking)
- Total per stage: <1ms
- For 10-stage pipeline: <10ms total overhead (~1% of typical 1-second pipeline)

**Mitigation strategies:**
- Async emission (fire-and-forget, never blocks pipeline)
- Batch writes if multiple events per stage (not needed at current scale)
- Graceful degradation if Redis unavailable (emit fails silently)

### 7.2 Aggregation Consumer Performance

**Processing capacity:**
- Target: 1,000 events/second
- Typical load: 100-200 events/second
  - Calculation: 10-20 memories/sec × 10 stages/pipeline = 100-200 events/sec
- Headroom: 5-10x capacity

**Resource usage:**
- Memory: <100MB (in-memory bucket aggregation, ~100 buckets × 1MB each)
- CPU: <5% (mostly NumPy percentile calculations)
- Redis writes: ~200 writes/hour for 10 workspaces (1 write per bucket per workspace)

**Bottleneck analysis:**
- Not network: Redis writes are <1ms each
- Not CPU: Percentile calculation is fast (NumPy vectorized)
- Potential bottleneck: Large number of concurrent workspaces (100+)
  - Solution: Horizontal scaling via consumer groups

### 7.3 Dashboard Query Performance

**Query performance:**
- Single Redis hash read: <1ms
- Typical dashboard query: 12 buckets (1 hour of data at 5-min granularity)
- Total: 12 hash reads × 1ms = <15ms
- Dashboard loads 3 metric types: 3 × 15ms = <50ms total
- User experience: sub-100ms page load

**Caching:**
- Browser caches API responses for 30 seconds (HTTP Cache-Control header)
- Dashboard auto-refreshes every 5 minutes (aligns with bucket size)
- No server-side caching needed (Redis is already fast)

---

## 8. Future Enhancements

**Not in Phase 6 scope, but architecture supports:**

1. **Alerting**: Threshold-based alerts on metric anomalies
   - High error rate: >5% for any stage over 1 hour
   - Slow stage: p95 latency > 2× baseline
   - Low ruler coverage: <50% (ontology regression)
   - Emit to separate Redis Stream: `smartmemory:alerts`

2. **Anomaly detection**: ML-based anomaly detection on time-series metrics
   - Sudden throughput drop (>50% decrease)
   - Latency spike (>3σ above mean)
   - Unusual extraction patterns (entity count outliers)

3. **Cost tracking**: LLM API cost tracking per workspace
   - Token usage by stage (llm_extract, reasoning validation)
   - Cost per memory (average and trend)
   - Budget alerts (daily/monthly spend thresholds)

4. **Custom metrics**: User-defined metrics via plugin system
   - Domain-specific extraction quality (e.g., medical entity accuracy)
   - Business KPIs (memories per user, retention metrics)

5. **Metric export**: Export to external observability platforms
   - Prometheus exporter (scrape endpoint at `/metrics`)
   - OpenTelemetry integration (OTLP traces + metrics)
   - Datadog/NewRelic/Grafana Cloud connectors

---

## See Also

- [Pipeline Architecture](pipeline-architecture.md) — Stage execution model, transport modes
- [Extraction Stages](extraction-stages.md) — Stage-specific metrics emitted
- [Service & API](service-api.md) — API endpoint specifications
- [Implementation Plan](../2026-02-05-implementation-plan.md) — Phase 6 implementation details
