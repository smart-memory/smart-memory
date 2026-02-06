# Pipeline Architecture Design

**Date:** 2026-02-05
**Status:** Design Complete
**Part of:** Ontology-Grounded Extraction Implementation Plan

---

## 1. Overview

This document specifies the unified pipeline architecture for SmartMemory's ontology-grounded extraction system. The pipeline replaces three existing orchestrators (`MemoryIngestionFlow`, `FastIngestionFlow`, `ExtractorPipeline`) with a single command-based architecture that supports breakpoint execution, undo, serializable state, and orthogonal transport mechanisms.

### Design Goals

1. **One pipeline** - Unified architecture for all ingestion paths (dev, production sync, production async, Studio preview)
2. **Debugger semantics** - Run to breakpoint, inspect state, modify parameters, resume
3. **Pure functions** - Stages are deterministic transforms of (state, config) with no hidden dependencies
4. **Transport-agnostic** - Same stages work for in-process function calls or event-bus message passing
5. **Undo support** - Preview mode discards computed state, production mode cleans up database writes

### Architecture Overview

```
                        PipelineConfig (per-workspace, tunable)
                              |
                              v
┌──────────────────────────────────────────────────────────────────────────┐
│                     Pipeline (linear, StageCommands)                      │
│                                                                          │
│  text ──> classify ──> coreference ──> simplify ──> entity_ruler         │
│               |             |               |               |            │
│               v             v               v               v            │
│          PipelineState flows through, accumulating at each stage          │
│               |             |               |               |            │
│               v             v               v               v            │
│  ──> llm_extract ──> ontology_constrain ──> store ──> link ──> enrich   │
│                                                                    |     │
│                                                              ──> evolve  │
│                                                                          │
│  Each stage: StageCommand.execute(state, config) -> state               │
│              StageCommand.undo(state) -> state                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 2. StageCommand Protocol

Every pipeline stage implements the `StageCommand` protocol:

```python
from typing import Protocol
from smartmemory.pipeline.state import PipelineState
from smartmemory.pipeline.config import PipelineConfig

class StageCommand(Protocol):
    """Protocol for pipeline stages that can execute and undo."""

    name: str  # Stage identifier (e.g., "classify", "entity_ruler")

    def execute(self, state: PipelineState, config: PipelineConfig) -> PipelineState:
        """Execute the stage, returning updated state.

        Args:
            state: Current pipeline state (immutable input)
            config: Pipeline configuration (stage reads its subtree)

        Returns:
            New state with stage results accumulated

        Note:
            This is a pure function. All inputs and outputs are explicit.
            Stages do not modify state in place.
        """
        ...

    def undo(self, state: PipelineState) -> PipelineState:
        """Undo the effects of this stage.

        Args:
            state: State after this stage executed

        Returns:
            State with this stage's effects removed

        Note:
            Undo behavior depends on execution mode:
            - Preview mode: discard computed state (cheap)
            - Production mode: clean up database writes (expensive)

            The stage reads state.mode to determine which path to take.
        """
        ...
```

### Undo Modes

Stages implement two undo behaviors based on `state.mode`:

**Preview mode** (`mode="preview"`):
- Stage computed results but made no persistent changes
- Undo is cheap: pop stage from `stage_history`, remove computed fields
- Example: `entity_ruler` undo removes `ruler_entities` from state

**Production mode** (`mode="sync"` or `mode="async"`):
- Stage wrote to databases (FalkorDB, Redis, MongoDB)
- Undo is expensive: query what was written, delete it
- Example: `store` undo deletes the created memory node, entity nodes, embeddings

```python
def undo(self, state: PipelineState) -> PipelineState:
    if state.mode == "preview":
        # Cheap undo: just remove computed state
        return dataclasses.replace(
            state,
            ruler_entities=None,
            stage_history=[s for s in state.stage_history if s != self.name]
        )
    else:
        # Expensive undo: clean up database writes
        if state.item_id:
            self.graph.delete_memory(state.item_id)
        return dataclasses.replace(
            state,
            item_id=None,
            stage_history=[s for s in state.stage_history if s != self.name]
        )
```

### Stage Independence

Stages are pure functions:
- No hidden state (singletons, class variables)
- No global configuration lookups
- All inputs explicit: `(state, config) -> state`
- Stages don't know about transport (in-process vs event-bus)
- Stages don't know about breakpoints or undo (runner orchestrates)

---

## 3. PipelineState

The `PipelineState` dataclass is the single source of truth flowing through the pipeline:

```python
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime

@dataclass
class PipelineState:
    """Immutable state passed through pipeline stages."""

    # Execution context
    mode: str  # "sync" | "async" | "preview"
    workspace_id: str
    user_id: Optional[str] = None
    team_id: Optional[str] = None

    # Input (set at pipeline start)
    text: str
    raw_metadata: dict = field(default_factory=dict)
    memory_type: Optional[str] = None  # User-specified or None (classifier decides)

    # Pre-processing stages
    resolved_text: Optional[str] = None  # After coreference resolution
    simplified_text: Optional[str] = None  # After simplification

    # Extraction stages
    ruler_entities: Optional[list[dict]] = None  # EntityRuler results
    llm_entities: Optional[list[dict]] = None  # LLM extraction results
    llm_relations: Optional[list[dict]] = None  # LLM extraction results

    # Constraint stage
    entities: list[dict] = field(default_factory=list)  # Validated entities
    relations: list[dict] = field(default_factory=list)  # Validated relations
    rejected: list[dict] = field(default_factory=list)  # Failed validation
    promotion_candidates: list[dict] = field(default_factory=list)  # New types

    # Storage stage
    item_id: Optional[str] = None  # Created memory ID
    entity_ids: dict[str, str] = field(default_factory=dict)  # entity_name -> node_id

    # Post-processing stages
    links: list[dict] = field(default_factory=list)  # Cross-reference links
    enrichments: dict = field(default_factory=dict)  # Wikidata, sentiment, etc.
    evolutions: list[dict] = field(default_factory=list)  # Memory evolutions

    # Pipeline metadata (for debugging and metrics)
    stage_history: list[str] = field(default_factory=list)  # ["classify", "entity_ruler", ...]
    stage_timings: dict[str, float] = field(default_factory=dict)  # stage_name -> seconds
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None

    def to_dict(self) -> dict:
        """Serialize state for checkpointing or event-bus transport."""
        return {
            "mode": self.mode,
            "workspace_id": self.workspace_id,
            "user_id": self.user_id,
            "team_id": self.team_id,
            "text": self.text,
            "raw_metadata": self.raw_metadata,
            "memory_type": self.memory_type,
            "resolved_text": self.resolved_text,
            "simplified_text": self.simplified_text,
            "ruler_entities": self.ruler_entities,
            "llm_entities": self.llm_entities,
            "llm_relations": self.llm_relations,
            "entities": self.entities,
            "relations": self.relations,
            "rejected": self.rejected,
            "promotion_candidates": self.promotion_candidates,
            "item_id": self.item_id,
            "entity_ids": self.entity_ids,
            "links": self.links,
            "enrichments": self.enrichments,
            "evolutions": self.evolutions,
            "stage_history": self.stage_history,
            "stage_timings": self.stage_timings,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PipelineState":
        """Deserialize state from checkpoint or event-bus message."""
        started_at = datetime.fromisoformat(data["started_at"]) if data.get("started_at") else datetime.utcnow()
        completed_at = datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None

        return cls(
            mode=data["mode"],
            workspace_id=data["workspace_id"],
            user_id=data.get("user_id"),
            team_id=data.get("team_id"),
            text=data["text"],
            raw_metadata=data.get("raw_metadata", {}),
            memory_type=data.get("memory_type"),
            resolved_text=data.get("resolved_text"),
            simplified_text=data.get("simplified_text"),
            ruler_entities=data.get("ruler_entities"),
            llm_entities=data.get("llm_entities"),
            llm_relations=data.get("llm_relations"),
            entities=data.get("entities", []),
            relations=data.get("relations", []),
            rejected=data.get("rejected", []),
            promotion_candidates=data.get("promotion_candidates", []),
            item_id=data.get("item_id"),
            entity_ids=data.get("entity_ids", {}),
            links=data.get("links", []),
            enrichments=data.get("enrichments", {}),
            evolutions=data.get("evolutions", []),
            stage_history=data.get("stage_history", []),
            stage_timings=data.get("stage_timings", {}),
            started_at=started_at,
            completed_at=completed_at,
        )
```

### State Accumulation

State is immutable. Each stage returns a new state with its results added:

```python
# entity_ruler stage
new_state = dataclasses.replace(
    state,
    ruler_entities=extracted_entities,
    stage_history=state.stage_history + ["entity_ruler"],
    stage_timings={**state.stage_timings, "entity_ruler": 0.004}
)
```

### Checkpointing

State is fully serializable. Checkpoints enable:
- Recovery after crashes (production async mode)
- Grid search / parameter tuning (run to breakpoint once, replay from checkpoint N times)
- Debugging (inspect state at any stage)

---

## 4. PipelineConfig

Configuration is a hierarchy of nested Pydantic dataclasses. Each stage reads its subtree:

```python
from pydantic import BaseModel, Field
from typing import Optional

class RetryConfig(BaseModel):
    """Per-stage retry policy."""
    max_retries: int = 3
    backoff_multiplier: float = 2.0
    on_failure: str = "retry"  # "retry" | "skip" | "abort" | "fallback"
    fallback_to: Optional[str] = None  # Stage name to fallback to

class ClassifyConfig(BaseModel):
    """Memory type classification configuration."""
    model: str = "gpt-4o-mini"
    fallback_type: str = "semantic"
    confidence_threshold: float = 0.7

class CoreferenceConfig(BaseModel):
    """Coreference resolution configuration."""
    enabled: bool = True
    model: str = "en_coreference_web_trf"

class SimplifyConfig(BaseModel):
    """Text simplification configuration."""
    clause_splitting: bool = True
    relative_clause_extraction: bool = True
    passive_to_active: bool = True
    appositive_extraction: bool = True

class EntityRulerConfig(BaseModel):
    """EntityRuler configuration."""
    enabled: bool = True
    pattern_sources: list[str] = ["seed", "learned_global", "learned_tenant"]
    min_confidence: float = 0.8

class LLMExtractConfig(BaseModel):
    """LLM-based extraction configuration."""
    model: str = "groq/llama-3.3-70b-versatile"
    prompt: str = "extraction"  # Resolved via PromptProvider
    temperature: float = 0.1
    max_entities: int = 50

class PromotionConfig(BaseModel):
    """Ontology promotion gates."""
    reasoning_validation: bool = True
    min_frequency: int = 1
    min_confidence: float = 0.8
    human_review: bool = False

class ConstrainConfig(BaseModel):
    """Ontology constraint configuration."""
    promotion: PromotionConfig = Field(default_factory=PromotionConfig)
    domain_range_validation: bool = True

class ExtractionConfig(BaseModel):
    """Composite extraction stage configuration."""
    entity_ruler: EntityRulerConfig = Field(default_factory=EntityRulerConfig)
    llm_extract: LLMExtractConfig = Field(default_factory=LLMExtractConfig)
    ontology_constrain: ConstrainConfig = Field(default_factory=ConstrainConfig)
    enrichment_tier: Optional[str] = "groq"  # "groq" | "gemma" | None
    self_learning_enabled: bool = True

class StoreConfig(BaseModel):
    """Storage configuration."""
    embed_model: str = "text-embedding-3-small"
    vector_dims: int = 1536
    dedup_threshold: float = 0.95

class LinkConfig(BaseModel):
    """Linking configuration."""
    similarity_threshold: float = 0.7
    max_links: int = 10
    cross_type_linking: bool = True

class WikidataConfig(BaseModel):
    """Wikidata enrichment configuration."""
    enabled: bool = True
    max_facts: int = 5

class SentimentConfig(BaseModel):
    """Sentiment enrichment configuration."""
    enabled: bool = True

class TemporalConfig(BaseModel):
    """Temporal enrichment configuration."""
    enabled: bool = True

class TopicConfig(BaseModel):
    """Topic enrichment configuration."""
    enabled: bool = False

class EnrichConfig(BaseModel):
    """Enrichment configuration."""
    wikidata: WikidataConfig = Field(default_factory=WikidataConfig)
    sentiment: SentimentConfig = Field(default_factory=SentimentConfig)
    temporal: TemporalConfig = Field(default_factory=TemporalConfig)
    topic: TopicConfig = Field(default_factory=TopicConfig)

class EvolveConfig(BaseModel):
    """Evolution configuration."""
    enabled_evolvers: list[str] = [
        "WorkingToEpisodicEvolver",
        "EpisodicToSemanticEvolver",
        "EpisodicDecayEvolver",
    ]
    decay_rates: dict[str, float] = {"episodic": 0.95, "working": 0.85}
    synthesis_threshold: float = 0.8

class PipelineConfig(BaseModel):
    """Complete pipeline configuration."""

    # Metadata
    name: str = "default"
    workspace_id: str
    mode: str = "sync"  # "sync" | "async" | "preview"

    # Retry policy
    retry: RetryConfig = Field(default_factory=RetryConfig)

    # Cross-cutting parameters (auto-inferred or manually tunable)
    domain_vocabulary: Optional[str] = None  # e.g., "medical", "legal", "software"
    relation_depth: int = 3  # Max depth for relation traversal
    temporal_sensitivity: float = 0.5  # 0=ignore time, 1=strict temporal
    contradiction_tolerance: float = 0.3  # 0=strict, 1=permissive
    confidence_requirement: float = 0.7  # Min confidence for extraction
    scope: str = "WORKSPACE"  # "TENANT" | "WORKSPACE" | "USER"

    # Stage configurations
    classify: ClassifyConfig = Field(default_factory=ClassifyConfig)
    coreference: CoreferenceConfig = Field(default_factory=CoreferenceConfig)
    simplify: SimplifyConfig = Field(default_factory=SimplifyConfig)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    store: StoreConfig = Field(default_factory=StoreConfig)
    link: LinkConfig = Field(default_factory=LinkConfig)
    enrich: EnrichConfig = Field(default_factory=EnrichConfig)
    evolve: EvolveConfig = Field(default_factory=EvolveConfig)
```

### Configuration Storage

Configs are stored per workspace in FalkorDB as JSON:

```cypher
CREATE (:PipelineConfig {
    name: "default",
    workspace_id: "ws_123",
    config: '{"classify": {"model": "gpt-4o-mini"}, ...}'
})
```

Named configs support multiple parameter sets per workspace:
- `default` - Standard ingestion
- `high-precision` - Low recall, high confidence
- `bulk-import` - Fast, minimal enrichment

### Prompt Resolution

The `PipelineConfig` contains resolved prompt values at runtime. Before pipeline execution, the `PromptProvider` resolves prompts using three-layer precedence:

```python
# Three-layer resolution
1. MongoDB per-tenant override (if exists)
2. prompts.json default (if no override)
3. Inline value in config (if manually specified)

# Example
config.extraction.llm_extract.prompt = "extraction"  # Key in prompts.json
resolved_prompt = prompt_provider.get("extraction", workspace_id)
config.extraction.llm_extract.prompt = resolved_prompt  # Now contains full text
```

### OntologyConfig

The ontology configuration is loaded from the ontology graph at pipeline start (not stored in PipelineConfig):

```python
@dataclass
class OntologyConfig:
    """Ontology state loaded from graph."""
    seed_types: list[str]  # ["Person", "Organization", ...]
    provisional_types: list[str]  # Auto-created, not yet confirmed
    confirmed_types: list[str]  # Passed promotion gates
    type_pair_priors: dict[tuple[str, str], float]  # (type1, type2) -> probability
    entity_patterns: dict[str, list[dict]]  # type -> [patterns]
    mutual_exclusions: list[tuple[str, str]]  # [(type1, type2), ...]
```

This is loaded once per pipeline run and used by stages that need ontology context (`entity_ruler`, `ontology_constrain`).

---

## 5. PipelineRunner

The runner orchestrates stage execution with two orthogonal axes: breakpoints and transport.

```python
from typing import Optional

class PipelineRunner:
    """Orchestrates pipeline execution with breakpoints and transport."""

    def __init__(
        self,
        stages: list[StageCommand],
        transport: Transport,
        config_loader: ConfigLoader,
    ):
        self.stages = {s.name: s for s in stages}
        self.stage_order = [s.name for s in stages]
        self.transport = transport
        self.config_loader = config_loader

    def run(
        self,
        text: str,
        config: PipelineConfig,
        metadata: Optional[dict] = None,
    ) -> PipelineState:
        """Execute full pipeline from start to finish.

        Args:
            text: Input text to process
            config: Pipeline configuration
            metadata: Optional metadata to attach

        Returns:
            Final pipeline state after all stages
        """
        state = PipelineState(
            mode=config.mode,
            workspace_id=config.workspace_id,
            text=text,
            raw_metadata=metadata or {},
        )

        return self.run_from(state, config, start_from=None, stop_after=None)

    def run_to(
        self,
        text: str,
        config: PipelineConfig,
        stop_after: str,
        metadata: Optional[dict] = None,
    ) -> PipelineState:
        """Execute pipeline until specified stage completes (breakpoint).

        Args:
            text: Input text to process
            config: Pipeline configuration
            stop_after: Stage name to stop after
            metadata: Optional metadata to attach

        Returns:
            Pipeline state after the breakpoint stage

        Example:
            # Run extraction stages only
            state = runner.run_to(text, config, stop_after="ontology_constrain")
        """
        state = PipelineState(
            mode=config.mode,
            workspace_id=config.workspace_id,
            text=text,
            raw_metadata=metadata or {},
        )

        return self.run_from(state, config, start_from=None, stop_after=stop_after)

    def run_from(
        self,
        state: PipelineState,
        config: PipelineConfig,
        start_from: Optional[str] = None,
        stop_after: Optional[str] = None,
    ) -> PipelineState:
        """Resume pipeline from checkpoint state.

        Args:
            state: Checkpoint state (from previous run_to)
            config: Pipeline configuration (may differ from checkpoint)
            start_from: Stage to start from (None = start at first incomplete stage)
            stop_after: Stage to stop after (None = run to end)

        Returns:
            Pipeline state after execution

        Example:
            # Run to extraction, modify config, re-run extraction + storage
            state1 = runner.run_to(text, config1, stop_after="llm_extract")
            config2 = dataclasses.replace(config1, store=new_store_config)
            state2 = runner.run_from(state1, config2, start_from="store")
        """
        # Determine start stage
        if start_from:
            start_idx = self.stage_order.index(start_from)
        else:
            # Resume from first incomplete stage
            completed = set(state.stage_history)
            start_idx = next(
                (i for i, name in enumerate(self.stage_order) if name not in completed),
                len(self.stage_order)  # All complete
            )

        # Determine end stage
        if stop_after:
            stop_idx = self.stage_order.index(stop_after) + 1
        else:
            stop_idx = len(self.stage_order)

        # Execute stages
        for stage_name in self.stage_order[start_idx:stop_idx]:
            stage = self.stages[stage_name]
            state = self._execute_stage(stage, state, config)

        return state

    def undo_to(self, state: PipelineState, target: str) -> PipelineState:
        """Undo stages back to target stage (rollback).

        Args:
            state: Current state
            target: Stage to roll back to

        Returns:
            State with stages after target undone

        Example:
            # Run full pipeline, then undo storage and linking to re-run
            state = runner.run(text, config)
            state = runner.undo_to(state, target="ontology_constrain")
            state = runner.run_from(state, new_config, start_from="store")
        """
        target_idx = self.stage_order.index(target)
        completed = state.stage_history

        # Undo in reverse order
        for stage_name in reversed(completed[target_idx + 1:]):
            stage = self.stages[stage_name]
            state = stage.undo(state)

        return state

    def _execute_stage(
        self,
        stage: StageCommand,
        state: PipelineState,
        config: PipelineConfig,
    ) -> PipelineState:
        """Execute a single stage with retry and error handling."""
        retry_config = config.retry
        attempts = 0
        last_error = None

        while attempts <= retry_config.max_retries:
            try:
                start = time.time()
                new_state = self.transport.execute(stage, state, config)
                elapsed = time.time() - start

                # Record timing
                new_state = dataclasses.replace(
                    new_state,
                    stage_timings={**new_state.stage_timings, stage.name: elapsed}
                )

                return new_state

            except Exception as e:
                last_error = e
                attempts += 1

                if attempts > retry_config.max_retries:
                    # Exhausted retries
                    if retry_config.on_failure == "skip":
                        logger.warning(f"Stage {stage.name} failed, skipping: {e}")
                        return state
                    elif retry_config.on_failure == "fallback" and retry_config.fallback_to:
                        logger.warning(f"Stage {stage.name} failed, falling back to {retry_config.fallback_to}")
                        fallback_stage = self.stages[retry_config.fallback_to]
                        return self._execute_stage(fallback_stage, state, config)
                    else:
                        raise

                # Undo partial work before retry
                state = stage.undo(state)

                # Backoff
                backoff = retry_config.backoff_multiplier ** (attempts - 1)
                time.sleep(backoff)
```

### Execution Modes

The same runner supports different execution patterns via config:

| Use Case | Breakpoints | Transport | Effect |
|----------|-------------|-----------|--------|
| Dev/CI | none | in-process | Full pipeline, function calls |
| Production sync | none | in-process | Full pipeline with retry |
| Production async | none | event-bus | Each stage is a Redis Stream consumer |
| Studio/tuning | user-set | in-process | `run_to()`, `run_from()`, undo, iterate |
| Grid search | set + replay | in-process | `run_to()` once, `run_from()` N times |

---

## 6. Transport

Transport abstractions decouple stage execution from communication mechanism.

### Transport Protocol

```python
class Transport(Protocol):
    """Protocol for stage execution transport."""

    def execute(
        self,
        stage: StageCommand,
        state: PipelineState,
        config: PipelineConfig,
    ) -> PipelineState:
        """Execute a stage, returning updated state.

        How execution happens is transport-specific:
        - InProcessTransport: function call
        - EventBusTransport: serialize -> publish -> consume -> deserialize
        """
        ...
```

### InProcessTransport

Direct function calls. Stages execute in the same process:

```python
class InProcessTransport:
    """Execute stages as function calls in the same process."""

    def execute(
        self,
        stage: StageCommand,
        state: PipelineState,
        config: PipelineConfig,
    ) -> PipelineState:
        return stage.execute(state, config)
```

### EventBusTransport

Redis Streams message passing. Each stage is a consumer group:

```python
class EventBusTransport:
    """Execute stages via Redis Streams event bus."""

    def __init__(self, redis_client: Redis):
        self.redis = redis_client
        self.stream_prefix = "pipeline"

    def execute(
        self,
        stage: StageCommand,
        state: PipelineState,
        config: PipelineConfig,
    ) -> PipelineState:
        # Serialize state and config
        message = {
            "state": state.to_dict(),
            "config": config.model_dump(),
        }

        # Publish to stage-specific stream
        stream = f"{self.stream_prefix}:{stage.name}"
        msg_id = self.redis.xadd(stream, message)

        # Wait for result (blocking read with timeout)
        result_stream = f"{self.stream_prefix}:{stage.name}:results"
        results = self.redis.xread(
            {result_stream: msg_id},
            block=30000,  # 30s timeout
        )

        if not results:
            raise TimeoutError(f"Stage {stage.name} did not complete within timeout")

        # Deserialize result
        result_data = results[0][1][0][1]
        return PipelineState.from_dict(result_data["state"])
```

### Consumer Group Pattern

In async mode, each stage has a dedicated consumer group that processes messages from its stream:

```python
# Consumer process for entity_ruler stage
def entity_ruler_consumer():
    stream = "pipeline:entity_ruler"
    consumer_group = "entity_ruler_workers"

    while True:
        # Read from stream
        messages = redis.xreadgroup(
            consumer_group,
            "worker-1",
            {stream: ">"},
            count=1,
            block=5000,
        )

        if not messages:
            continue

        stream_name, msg_list = messages[0]
        msg_id, msg_data = msg_list[0]

        # Deserialize
        state = PipelineState.from_dict(msg_data["state"])
        config = PipelineConfig.model_validate(msg_data["config"])

        # Execute stage
        stage = EntityRulerStage()
        new_state = stage.execute(state, config)

        # Publish result
        result_stream = "pipeline:entity_ruler:results"
        redis.xadd(result_stream, {"state": new_state.to_dict()})

        # Acknowledge
        redis.xack(stream, consumer_group, msg_id)
```

Horizontal scaling: run multiple consumer processes, Redis distributes messages across workers.

---

## 7. Retry Policy

Retry behavior is configured per-stage in `PipelineConfig.retry`:

```python
class RetryConfig(BaseModel):
    max_retries: int = 3
    backoff_multiplier: float = 2.0
    on_failure: str = "retry"  # "retry" | "skip" | "abort" | "fallback"
    fallback_to: Optional[str] = None
```

### Retry Flow

1. Stage execution fails with exception
2. Runner calls `stage.undo(state)` to clean up partial work
3. Wait for backoff period (2^attempt seconds)
4. Retry execution
5. After max retries, apply failure policy:
   - `retry`: already tried, now abort
   - `skip`: log warning, continue to next stage
   - `abort`: propagate exception, pipeline fails
   - `fallback`: execute fallback stage instead

### Fallback Example

```python
# Config: if llm_extract fails, fall back to entity_ruler only
config.retry.on_failure = "fallback"
config.retry.fallback_to = "entity_ruler"

# Execution: llm_extract times out after 3 retries
# -> undo llm_extract partial state
# -> execute entity_ruler instead
# -> continue pipeline with ruler results
```

---

## 8. What Gets Replaced

This architecture replaces and consolidates existing code:

| Existing Code | Lines | Replacement |
|---------------|-------|-------------|
| `MemoryIngestionFlow` | 473 | `Pipeline.run(text, config)` |
| `FastIngestionFlow` | 502 | Deleted (async is config flag) |
| `ExtractorPipeline` (Studio) | 491 | Deleted (Studio calls core) |
| Duplicated normalization | ~130 | Consolidated in stages |
| **Total** | **~1,600 LOC** | **Single pipeline** |

### Migration Path

1. Wrap existing pipeline components as `StageCommand` implementations
2. Replace `SmartMemory.ingest()` to delegate to `Pipeline.run()`
3. Replace Studio preview with `Pipeline.run_to()`
4. Delete old orchestrators after validation

---

## 9. Usage Examples

### Basic Ingestion

```python
from smartmemory.pipeline import Pipeline, PipelineConfig

# Load config for workspace
config = config_loader.load("default", workspace_id="ws_123")

# Run full pipeline
pipeline = Pipeline(stages, InProcessTransport())
state = pipeline.run("Claude Code is a CLI tool", config)

# Result
print(state.item_id)  # "mem_abc123"
print(state.entities)  # [{"text": "Claude Code", "type": "Tool"}, ...]
```

### Breakpoint Execution (Studio)

```python
# Run extraction stages only
state = pipeline.run_to(text, config, stop_after="ontology_constrain")

# Inspect intermediate state
print(state.ruler_entities)  # EntityRuler found 3 entities
print(state.llm_entities)  # LLM found 5 entities
print(state.promotion_candidates)  # 2 new types proposed

# User modifies config in Studio UI
config.extraction.llm_extract.model = "gpt-4o-mini"

# Resume from storage stage
state = pipeline.run_from(state, config, start_from="store")
```

### Grid Search (Parameter Tuning)

```python
# Run to extraction once
state = pipeline.run_to(text, config, stop_after="ontology_constrain")

# Try N different storage configs
for store_config in configs:
    config_variant = dataclasses.replace(config, store=store_config)
    final_state = pipeline.run_from(state, config_variant, start_from="store")

    # Evaluate results
    evaluate(final_state)

    # Undo for next iteration
    state = pipeline.undo_to(final_state, target="ontology_constrain")
```

### Async Production Mode

```python
# Config for event-bus transport
config = PipelineConfig(
    workspace_id="ws_123",
    mode="async",
    name="production",
)

# Run pipeline (each stage publishes to Redis Stream)
pipeline = Pipeline(stages, EventBusTransport(redis))
state = pipeline.run(text, config)  # Returns immediately with queued state

# Consumer processes handle execution
# Results collected in Redis
```

---

## 10. Stage Order

The pipeline executes stages in this order:

1. `classify` - Memory type classification
2. `coreference` - Pronoun resolution
3. `simplify` - Text simplification (clause splitting, passive->active, etc.)
4. `entity_ruler` - Fast pattern-based entity extraction (spaCy + EntityRuler)
5. `llm_extract` - LLM-based entity and relation extraction
6. `ontology_constrain` - Validate against ontology, create provisional types, identify promotion candidates
7. `store` - Write memory, entities, relations to FalkorDB, compute embeddings
8. `link` - Cross-reference linking to related memories
9. `enrich` - Wikidata, sentiment, temporal, topic enrichments
10. `evolve` - Memory evolution (decay, synthesis, reinforcement)

Each stage is independently testable and configurable via its subtree in `PipelineConfig`.

---

## 11. Metrics Emission

Stages emit metrics events to Redis Streams for observability:

```python
def execute(self, state: PipelineState, config: PipelineConfig) -> PipelineState:
    start = time.time()

    # Stage logic
    result = self._do_work(state, config)

    elapsed = time.time() - start

    # Emit metric event
    self.redis.xadd("pipeline:metrics", {
        "event": "stage_completed",
        "stage": self.name,
        "workspace_id": state.workspace_id,
        "latency_ms": elapsed * 1000,
        "entity_count": len(result.entities),
        "timestamp": datetime.utcnow().isoformat(),
    })

    return result
```

Metrics consumer aggregates these into time-bucketed data for the Insights dashboard.

---

## 12. Summary

This pipeline architecture provides:

1. **Unified orchestration** - One pipeline for all use cases
2. **Debugger semantics** - Run to breakpoint, inspect, modify, resume
3. **Pure functions** - Stages are deterministic transforms of (state, config)
4. **Transport-agnostic** - Same stages work in-process or async
5. **Undo support** - Cheap preview mode, expensive production cleanup
6. **Retry with fallback** - Per-stage error handling
7. **Serializable state** - Checkpointing and event-bus transport
8. **Metrics-ready** - Built-in observability hooks

Implementation proceeds in Phase 1 (Pipeline Foundation) of the master plan.
