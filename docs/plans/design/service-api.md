# Service & API Design

**Date:** 2026-02-06
**Version:** 1.0
**Status:** Design specification
**Related:** [Pipeline Architecture](pipeline-architecture.md), [Implementation Plan](../2026-02-05-implementation-plan.md)

---

## Overview

This document specifies the REST API and service layer for SmartMemory's ontology-grounded pipeline. The service exposes pipeline configuration, execution, and management endpoints with full multi-tenant isolation.

**Key architectural decisions:**

1. **PipelineConfig replaces Ontology CRUD** - Configuration management for the unified pipeline, not separate ontology management
2. **Breakpoint execution** - Studio and API clients can run partial pipelines with inspect/modify/resume
3. **Event-bus transport** - Redis Streams enable async execution with per-stage horizontal scaling
4. **Three-layer prompt resolution** - Default prompts → MongoDB tenant overrides → runtime PipelineConfig values
5. **Msgpack serialization** - Safe, efficient serialization for PipelineState checkpoints and event-bus messages

---

## 1. Route Inventory

See Appendix A for the complete route summary table with all 14 endpoints.

### 1.1 PipelineConfig Management

Replaces the existing ontology CRUD routes. Named configurations are workspace-scoped.

**Routes:**
- GET `/memory/pipeline/configs` - List named configs
- POST `/memory/pipeline/configs` - Save named config
- GET `/memory/pipeline/configs/{name}` - Load config by name
- PUT `/memory/pipeline/configs/{name}` - Update config
- DELETE `/memory/pipeline/configs/{name}` - Delete config

### 1.2 Pattern Admin

Direct pattern management for manual corrections.

**Routes:**
- GET `/memory/ontology/patterns` - List patterns with stats
- POST `/memory/ontology/patterns` - Manual pattern add/override
- DELETE `/memory/ontology/patterns/{id}` - Remove pattern

### 1.3 Ontology Status

Read-only endpoints for convergence metrics.

**Routes:**
- GET `/memory/ontology/status` - Convergence metrics, pattern counts
- POST `/memory/ontology/import` - Import OntologyIR (seeding, migration)
- GET `/memory/ontology/export` - Export as OntologyIR

### 1.4 Pipeline Execution

Execute the pipeline with breakpoint control.

**Routes:**
- POST `/memory/pipeline/run` - Full pipeline run
- POST `/memory/pipeline/run-to` - Run to breakpoint
- POST `/memory/pipeline/run-from` - Resume from checkpoint

**Checkpoint storage uses msgpack:**

```python
# Redis key: checkpoint:{workspace_id}:{checkpoint_id}
# TTL: 3600 seconds (1 hour)
# Value: serialized PipelineState (msgpack format)

import msgpack

def save_checkpoint(workspace_id: str, state: PipelineState, breakpoint_stage: str) -> str:
    checkpoint_id = str(uuid.uuid4())
    redis_key = f"checkpoint:{workspace_id}:{checkpoint_id}"
    redis_client.setex(redis_key, 3600, msgpack.packb(state.to_dict()))
    return checkpoint_id
```

---

## 2. Routes Being Replaced

### Ontology CRUD → PipelineConfig Management

**Deleted routes:**
- `POST /memory/ontology/registries` → `POST /memory/pipeline/configs`
- `GET /memory/ontology/registries` → `GET /memory/pipeline/configs`
- `GET /memory/ontology/registries/{id}` → `GET /memory/pipeline/configs/{name}`
- `PUT /memory/ontology/registries/{id}` → `PUT /memory/pipeline/configs/{name}`
- `DELETE /memory/ontology/registries/{id}` → `DELETE /memory/pipeline/configs/{name}`

**Rationale:** Ontology is persistent knowledge accumulated by pipeline execution, not a standalone resource to manage.

### Ontology Inference → Pipeline Stages

**Deleted:**
- `POST /memory/ontology/inference` → Use `POST /memory/pipeline/run-to` with `breakpoint_stage="store"`

### Studio Integration → Breakpoint Execution

**Deleted:**
- Studio extraction preview → `POST /memory/pipeline/run-to`
- Studio parameter tuning → `run-to` + modify + `run-from`
- Studio grid search → Client-side iteration

---

## 3. Event-Bus Transport

### Architecture

**In-process (dev):** Sequential function calls in same process

**Event-bus (production):** Redis Streams with per-stage consumer groups

### Redis Streams Schema

**Streams:**
- `stage:{stage_name}` - Per-stage input queue
- `results:{workspace_id}:{memory_id}` - Final results (TTL: 1 hour)
- `dlq:{stage_name}` - Dead-letter queue

**Serialization:** msgpack format for safe, efficient serialization

### Worker Deployment

Same Docker image, different entrypoint per stage:

```yaml
worker-llm-extract:
  image: smartmemory-service:latest
  command: ["python", "-m", "smartmemory.pipeline.worker", "--stage", "llm_extract"]
  deploy:
    replicas: 4  # Scale for throughput
```

---

## 4. Prompt Management

### Three-Layer Resolution

1. **prompts.json** (defaults) → 2. **MongoDB** (workspace overrides) → 3. **PipelineConfig** (runtime)

### Existing Infrastructure

**Built:**
- PromptProvider abstraction
- Studio API endpoints (full CRUD)

**Not built:**
- Studio React UI components

**Migration needed:**
- Move hardcoded prompts from `llm_single.py` and `reasoning.py` to `prompts.json`

---

## 5. Multi-Tenancy

All routes use `SecureSmartMemory` for automatic tenant isolation:

```python
scope_provider = get_scope_provider(request)
smart_memory = create_secure_smart_memory(scope_provider)
```

**Scoping:**
- PipelineConfig: Workspace
- Ontology patterns: Workspace
- Ontology graph: Workspace (separate graph `ws_{id}_ontology`)
- Memory items: Workspace + User
- Checkpoints: Workspace
- Prompt overrides: Workspace

---

## 6. Worker Deployment Model

### Development Mode

In-process sequential runner (no workers needed)

### Production Sync Mode

In-process with retry/fallback (no event bus)

### Production Async Mode

Event-bus runner with per-stage consumer groups. Horizontal scaling by adding replicas to bottleneck stages.

**Bottleneck identification:** Monitor `redis-cli XLEN stage:{stage_name}` queue depth

---

## 7. Implementation Checklist

**Phase 5 deliverables:**

- [ ] 5.1: PipelineConfig management routes
- [ ] 5.2: Pattern admin routes
- [ ] 5.3: Ontology status routes
- [ ] 5.4: EventBusTransport (Redis Streams)
- [ ] 5.5: Retry with undo logic
- [ ] 5.6: Finish Studio prompt editing UI

**Dependencies:** Phase 3 complete (store, link, enrich, evolve stages)

**Acceptance criteria:**
- Named configs saveable/loadable per workspace
- Event-bus transport runs full pipeline asynchronously
- Stage failures retry with undo, then fallback
- Pattern admin allows manual corrections
- Studio can run breakpoint execution via API

---

## Appendix A: Route Summary Table

| Method | Endpoint | Description | Auth | Scope |
|--------|----------|-------------|------|-------|
| GET | `/memory/pipeline/configs` | List named configs | Required | Workspace |
| POST | `/memory/pipeline/configs` | Save named config | Required | Workspace |
| GET | `/memory/pipeline/configs/{name}` | Load config by name | Required | Workspace |
| PUT | `/memory/pipeline/configs/{name}` | Update config | Required | Workspace |
| DELETE | `/memory/pipeline/configs/{name}` | Delete config | Required | Workspace |
| GET | `/memory/ontology/patterns` | List patterns with stats | Required | Workspace |
| POST | `/memory/ontology/patterns` | Manual pattern add | Required | Workspace |
| DELETE | `/memory/ontology/patterns/{id}` | Remove pattern | Required | Workspace |
| GET | `/memory/ontology/status` | Convergence metrics | Required | Workspace |
| POST | `/memory/ontology/import` | Import OntologyIR | Required | Workspace |
| GET | `/memory/ontology/export` | Export as OntologyIR | Required | Workspace |
| POST | `/memory/pipeline/run` | Full pipeline run | Required | Workspace |
| POST | `/memory/pipeline/run-to` | Run to breakpoint | Required | Workspace |
| POST | `/memory/pipeline/run-from` | Resume from checkpoint | Required | Workspace |

**Total:** 14 endpoints (11 new, 3 kept from old ontology routes)

**Deleted:** ~10 old ontology CRUD endpoints + 3 Studio integration endpoints

---

## Appendix B: Config Storage Schema

PipelineConfig stored in FalkorDB:

```cypher
CREATE (c:PipelineConfig {
  id: "cfg_abc123",
  name: "high-precision",
  workspace_id: "ws_123",
  mode: "async",
  config_json: "<serialized as JSON>",
  created_at: datetime("2026-02-06T10:30:00Z"),
  updated_at: datetime("2026-02-06T10:30:00Z")
})

CREATE INDEX FOR (c:PipelineConfig) ON (c.workspace_id, c.name)
```

---

For complete API examples, convergence score calculation, and detailed deployment configurations, see the full implementation plan and related design documents
