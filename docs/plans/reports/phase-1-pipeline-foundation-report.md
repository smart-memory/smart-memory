# Phase 1: Pipeline Foundation — Implementation Report

**Date completed:** 2026-02-05
**Phase:** 1 of 8
**Status:** COMPLETE

---

## Summary

Replaced three orchestrators (`MemoryIngestionFlow`, `FastIngestionFlow`, `ExtractorPipeline`) with a unified pipeline built on the StageCommand protocol. Established the core abstractions (PipelineState, PipelineConfig, PipelineRunner) that all subsequent phases build on. Created a separate ontology FalkorDB graph with 14 seed entity types.

## Deliverables

| # | Deliverable | Files | Status |
|---|------------|-------|--------|
| 1.1 | `StageCommand` protocol (execute/undo) | `smartmemory/pipeline/protocol.py` | Done |
| 1.2 | `PipelineState` dataclass (serializable, checkpointable) | `smartmemory/pipeline/state.py` | Done |
| 1.3 | `PipelineConfig` hierarchy (nested dataclasses, per-workspace) | `smartmemory/pipeline/config.py` | Done |
| 1.4 | `PipelineRunner` with `InProcessTransport` | `smartmemory/pipeline/runner.py` | Done |
| 1.5 | `run()`, `run_to()`, `run_from()`, `undo_to()` API | `smartmemory/pipeline/runner.py` | Done |
| 1.6 | Separate ontology FalkorDB graph (`ws_{id}_ontology`) | `smartmemory/graph/ontology_graph.py` | Done |
| 1.7 | Seed 14 entity types (three-tier status: seed/provisional/confirmed) | `smartmemory/graph/ontology_graph.py` | Done |
| 1.8 | Wrap existing stages as StageCommands (8 stages) | `smartmemory/pipeline/stages/` | Done |
| 1.9 | `SmartMemory.ingest()` delegates to `Pipeline.run()` | `smartmemory/smart_memory.py` | Done |
| 1.10 | Delete `FastIngestionFlow` (502 LOC) | `smartmemory/memory/ingestion/` | Done |

## Files Created

```
smartmemory/pipeline/__init__.py
smartmemory/pipeline/protocol.py          # StageCommand protocol
smartmemory/pipeline/state.py             # PipelineState dataclass
smartmemory/pipeline/config.py            # PipelineConfig hierarchy
smartmemory/pipeline/runner.py            # PipelineRunner + InProcessTransport
smartmemory/pipeline/stages/__init__.py   # Stage exports
smartmemory/pipeline/stages/classify.py   # ClassifyStage
smartmemory/pipeline/stages/coreference.py # CoreferenceStageCommand
smartmemory/pipeline/stages/extract.py    # ExtractStage (legacy wrapper, replaced in Phase 2)
smartmemory/pipeline/stages/store.py      # StoreStage
smartmemory/pipeline/stages/link.py       # LinkStage
smartmemory/pipeline/stages/enrich.py     # EnrichStage
smartmemory/pipeline/stages/ground.py     # GroundStage
smartmemory/pipeline/stages/evolve.py     # EvolveStage
smartmemory/graph/ontology_graph.py       # OntologyGraph (separate FalkorDB graph)
tests/unit/pipeline_v2/                   # All Phase 1 tests
```

## Files Deleted

```
smartmemory/memory/ingestion/fast_ingestion_flow.py  # 502 LOC, unused
```

## Architecture Decisions

1. **StageCommand protocol** over abstract base class — duck typing, testable with mocks
2. **Dataclass-based state** with `dataclasses.replace()` — immutable-by-convention, serializable
3. **Separate ontology FalkorDB graph** (`ws_{id}_ontology`) — no bleed risk, no label filtering needed
4. **Three-tier type status** (seed/provisional/confirmed) — gradual trust building
5. **InProcessTransport first** — event-bus transport deferred to Phase 5

## Test Coverage

- 109 unit tests passing
- Covers: state serialization, config defaults, runner execution modes (run/run_to/run_from/undo_to), all 8 stage wrappers, ontology graph operations

## Pipeline at End of Phase 1

```
classify → coreference → extract → store → link → enrich → ground → evolve
(8 stages, extract is a legacy wrapper around ExtractionPipeline)
```

## Notes

- `ExtractStage` was a thin wrapper calling `ExtractionPipeline.extract_semantics()`. Replaced by 4 native stages in Phase 2.
- `PipelineConfig` uses `DataclassModelMixin` (not Pydantic) matching the existing codebase pattern.
- All configs extend `MemoryBaseModel` for consistent serialization.
