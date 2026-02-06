# Phase 5: Service + API — Session-Ready Implementation Plan

**Date:** 2026-02-06
**Prerequisite:** Phases 1-4 COMPLETE
**Estimated scope:** ~800-1000 LOC new, ~200 LOC modified

---

## Goal

Expose the v2 pipeline through REST API with named config management, pattern admin, ontology status endpoints, full event-bus transport (Redis Streams with per-stage consumer groups) for async execution, and retry-with-undo for stage failures.

**Decisions (from user input):**
- Pipeline configs stored in **FalkorDB** (not Redis — no persistent storage in Redis)
- EventBusTransport: **full implementation** with per-stage consumer groups
- Studio prompt UI (5.6): **REMOVED** — already fully functional (PromptsPanel, PromptsConfigPanel, UnifiedPromptService, all working)

---

## Step 1: Pipeline Config CRUD Routes (5.1)

**File:** `smart-memory-service/memory_service/api/routes/pipeline.py` — EDIT (266 lines, append)

Add named pipeline config management endpoints to the existing pipeline routes file.

### New Request/Response Models

```python
class PipelineConfigCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: str = ""
    config: dict = Field(..., description="PipelineConfigBundle as dict")

class PipelineConfigResponse(BaseModel):
    name: str
    description: str
    config: dict
    workspace_id: str
    created_at: str
    updated_at: str
```

### Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `GET /memory/pipeline/configs` | List named configs for workspace | Returns `{configs: [...], count: N}` |
| `POST /memory/pipeline/configs` | Save named config | Validates config shape, stores to FalkorDB |
| `GET /memory/pipeline/configs/{name}` | Load config by name | Returns full config or 404 |
| `PUT /memory/pipeline/configs/{name}` | Update existing config | Merge or replace |
| `DELETE /memory/pipeline/configs/{name}` | Remove config | Soft-delete or hard-delete |

### Storage

Pipeline configs stored as nodes in **FalkorDB** ontology graph (not Redis — Redis is not used for persistent storage):

```cypher
CREATE (c:PipelineConfig {
    name: $name,
    description: $description,
    config_json: $config_json,
    workspace_id: $workspace_id,
    created_at: $now,
    updated_at: $now
})
```

Query: `MATCH (c:PipelineConfig {workspace_id: $ws}) RETURN c`

Use `SecureSmartMemory` scope for workspace isolation. Add methods to `OntologyGraph`:
- `save_pipeline_config(name, config_json, workspace_id) -> bool`
- `get_pipeline_config(name, workspace_id) -> dict | None`
- `list_pipeline_configs(workspace_id) -> list[dict]`
- `delete_pipeline_config(name, workspace_id) -> bool`

### Dependencies

```python
from service_common.auth.core import get_scope_provider
from service_common.security import create_secure_smart_memory
```

---

## Step 2: Pattern Admin Routes (5.2)

**File:** `smart-memory-service/memory_service/api/routes/ontology.py` — EDIT (776 lines, append)

Add pattern management endpoints to the existing ontology routes file.

### Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `GET /memory/ontology/patterns` | List patterns with stats | Query PatternManager/OntologyGraph, return `{patterns: [{name, type, source, confidence, is_global}], stats: {seed: N, promoted: N, tenant: N}}` |
| `POST /memory/ontology/patterns` | Manual add/override | Create EntityPattern node, notify PatternManager reload |
| `DELETE /memory/ontology/patterns/{pattern_id}` | Remove pattern | Delete EntityPattern node, notify reload |

### Implementation Notes

- Instantiate `OntologyGraph` from `SecureSmartMemory._graph` (ontology graph)
- Call `ontology_graph.get_entity_patterns(workspace_id)` for listing
- Call `ontology_graph.get_pattern_stats()` for stats
- Call `ontology_graph.add_entity_pattern(...)` for creation
- After mutation, call `PatternManager.notify_reload(workspace_id)` if Redis available

### Request Models

```python
class PatternCreateRequest(BaseModel):
    name: str = Field(..., min_length=2)
    entity_type: str = Field(...)
    confidence: float = Field(0.9, ge=0.0, le=1.0)
    is_global: bool = False

class PatternResponse(BaseModel):
    name: str
    entity_type: str
    confidence: float
    source: str  # seed, promoted, llm_discovery
    is_global: bool
    workspace_id: str | None
```

---

## Step 3: Ontology Status Routes (5.3)

**File:** `smart-memory-service/memory_service/api/routes/ontology.py` — EDIT (append)

### Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `GET /memory/ontology/status` | Convergence metrics | Pattern counts by layer, type status distribution (seed/provisional/confirmed), promotion/rejection rates |
| `POST /memory/ontology/import` | Import OntologyIR | Seed/migrate ontology from JSON |
| `GET /memory/ontology/export` | Export as OntologyIR | Dump full ontology for backup/migration |

### Status Response Shape

```python
class OntologyStatusResponse(BaseModel):
    pattern_stats: dict  # {seed: N, promoted: N, llm_discovery: N}
    type_counts: dict    # {seed: N, provisional: N, confirmed: N}
    total_entity_types: int
    total_patterns: int
    workspace_id: str
```

### Import/Export

Use existing `OntologyIR` from `smartmemory/ontology/`. Import creates types + patterns. Export serializes entire ontology graph.

---

## Step 4: EventBusTransport (5.4)

**Create:** `smart-memory/smartmemory/pipeline/transport/event_bus.py` (~250 LOC)

This is the major infrastructure piece. Implements the same interface as in-process execution but serializes PipelineState between stages via Redis Streams.

### Architecture

```
Producer (API request) → serialize PipelineState → publish to stage stream
                                                          ↓
                                              Stage consumer reads from stream
                                              → deserialize PipelineState
                                              → execute stage
                                              → serialize result
                                              → publish to next stage stream
                                                          ↓
                                              ... until all stages complete
```

### Key Classes

```python
class EventBusTransport:
    """Redis Streams transport for async pipeline execution."""

    def __init__(self, redis_client=None, stream_prefix="smartmemory:pipeline"):
        self._redis = redis_client
        self._prefix = stream_prefix

    def submit(self, state: PipelineState, config: PipelineConfigBundle) -> str:
        """Submit pipeline for async execution. Returns run_id."""
        # Serialize state + config → publish to first stage stream
        # Return run_id for polling

    def get_status(self, run_id: str) -> dict:
        """Poll execution status. Returns {stage, status, error}."""

    def get_result(self, run_id: str) -> PipelineState | None:
        """Get completed pipeline result."""
```

```python
class StageConsumer:
    """Consumes from a stage's Redis Stream, executes, publishes to next."""

    def __init__(self, stage_name: str, stage: StageCommand, transport: EventBusTransport):
        ...

    def run(self, max_iterations: int | None = None) -> int:
        """Process messages. Returns count processed."""
```

### Stream naming

- `smartmemory:pipeline:{workspace_id}:classify` — classify stage input
- `smartmemory:pipeline:{workspace_id}:coreference` — coreference stage input
- etc.
- `smartmemory:pipeline:{workspace_id}:results` — completed results

### Serialization

PipelineState is a dataclass — serialize with `dataclasses.asdict()` + JSON. Deserialize with `PipelineState(**data)`. Handle datetime fields with ISO format.

### Integration

**File:** `smart-memory-service/memory_service/api/routes/ingest.py` — EDIT

Add `async` mode parameter to ingest endpoint:

```python
@router.post("/ingest")
def ingest_memory(request: IngestRequest, mode: str = Query("sync", enum=["sync", "async"])):
    if mode == "async":
        transport = EventBusTransport()
        run_id = transport.submit(initial_state, config)
        return {"run_id": run_id, "status": "queued"}
    else:
        # existing sync path
```

---

## Step 5: Retry with Undo (5.5)

**File:** `smart-memory/smartmemory/pipeline/runner.py` — EDIT

Add retry policy to PipelineRunner.

### RetryPolicy

```python
@dataclass
class StageRetryPolicy:
    max_retries: int = 2
    backoff_seconds: float = 1.0
    backoff_multiplier: float = 2.0
    fallback: str | None = None  # stage name to fall back to, or "skip"
```

### Runner Changes

In `PipelineRunner.run()`, wrap each stage execution:

```python
for stage in stages:
    retry_count = 0
    policy = config.retry_policies.get(stage.name, default_policy)
    while True:
        try:
            state = stage.execute(state, config)
            break
        except Exception as e:
            retry_count += 1
            if retry_count > policy.max_retries:
                # Undo partial work
                state = stage.undo(state)
                if policy.fallback == "skip":
                    break
                raise
            time.sleep(policy.backoff_seconds * (policy.backoff_multiplier ** (retry_count - 1)))
```

### Config Extension

**File:** `smart-memory/smartmemory/pipeline/config.py` — EDIT

```python
@dataclass
class PipelineConfigBundle:
    # ... existing fields ...
    retry_policies: dict[str, StageRetryPolicy] = field(default_factory=dict)
```

---

## Step 6: Tests

### Service Tests

**Create:** `smart-memory-service/tests/test_pipeline_config_routes.py` (~8 tests)
- CRUD operations for named configs
- Workspace isolation
- Config validation

**Create:** `smart-memory-service/tests/test_pattern_admin_routes.py` (~6 tests)
- Pattern listing, creation, deletion
- Stats endpoint

**Create:** `smart-memory-service/tests/test_ontology_status_routes.py` (~4 tests)
- Status endpoint, import/export roundtrip

### Core Tests

**Create:** `smart-memory/tests/unit/pipeline_v2/test_event_bus_transport.py` (~10 tests)
- Submit + poll pattern
- Serialization roundtrip
- Stage consumer processing
- Error handling + DLQ

**Create:** `smart-memory/tests/unit/pipeline_v2/test_retry_undo.py` (~8 tests)
- Retry succeeds on transient failure
- Undo called on permanent failure
- Fallback to skip
- Backoff timing

---

## Step 7: Documentation + Report

- Update CHANGELOG.md with Phase 5 entries
- Update implementation plan to mark Phase 5 COMPLETE
- Write `docs/plans/reports/phase-5-service-api-report.md`
- Update CLAUDE.md if API surface changes

---

## Execution Order

```
Step 1 (Config CRUD) → Step 2 (Pattern Admin) → Step 3 (Ontology Status)
    ↓ (can parallelize Steps 2-3)
Step 4 (EventBusTransport) → Step 5 (Retry with Undo) → Step 6 (Tests) → Step 7 (Docs)
```

---

## Verification

```bash
# Service tests
cd /Users/ruze/reg/my/SmartMemory/smart-memory-service
pytest tests/test_pipeline_config_routes.py tests/test_pattern_admin_routes.py tests/test_ontology_status_routes.py -v

# Core tests
cd /Users/ruze/reg/my/SmartMemory/smart-memory
PYTHONPATH=. pytest tests/unit/pipeline_v2/test_event_bus_transport.py tests/unit/pipeline_v2/test_retry_undo.py -v

# Lint
cd /Users/ruze/reg/my/SmartMemory/smart-memory
ruff check --fix smartmemory/ && ruff format smartmemory/

cd /Users/ruze/reg/my/SmartMemory/smart-memory-service
ruff check --fix memory_service/ && ruff format memory_service/
```

---

## Key Files Reference

### Existing files to read first
- `smart-memory-service/memory_service/api/routes/pipeline.py` — existing pipeline routes
- `smart-memory-service/memory_service/api/routes/ontology.py` — existing ontology routes (776 lines)
- `smart-memory-service/memory_service/api/routes/ingest.py` — existing ingest route
- `smart-memory-service/memory_service/api/base_router.py` — router factory
- `smart-memory-service/memory_service/api/scope.py` — scope policies
- `smart-memory-service/service.py` — route registration
- `smart-memory/smartmemory/pipeline/runner.py` — PipelineRunner
- `smart-memory/smartmemory/pipeline/config.py` — PipelineConfigBundle
- `smart-memory/smartmemory/pipeline/state.py` — PipelineState
- `smart-memory/smartmemory/graph/ontology_graph.py` — OntologyGraph (Phase 4 methods)
- `smart-memory/smartmemory/ontology/pattern_manager.py` — PatternManager (Phase 4)
- `smart-memory/smartmemory/observability/events.py` — RedisStreamQueue patterns

### New files to create
- `smart-memory/smartmemory/pipeline/transport/event_bus.py`
- `smart-memory/smartmemory/pipeline/transport/__init__.py`
- `smart-memory-service/tests/test_pipeline_config_routes.py`
- `smart-memory-service/tests/test_pattern_admin_routes.py`
- `smart-memory-service/tests/test_ontology_status_routes.py`
- `smart-memory/tests/unit/pipeline_v2/test_event_bus_transport.py`
- `smart-memory/tests/unit/pipeline_v2/test_retry_undo.py`
