# Phase 3: Storage & Post-Processing — Implementation Report

**Date**: 2026-02-06
**Status**: COMPLETE

## Summary

Phase 3 adds pipeline metrics emission via Redis Streams, deduplicates normalization logic, exposes a public pipeline factory for Studio, rewires Studio's extraction preview to use v2 `PipelineRunner.run_to()`, and marks legacy `ExtractorPipeline` for deprecation.

## Deliverables

| # | Deliverable | Status | Notes |
|---|------------|--------|-------|
| 3.1 | `store` StageCommand | Already existed (Phase 1) | No changes needed |
| 3.2 | `link` StageCommand | Already existed (Phase 1) | No changes needed |
| 3.3 | `enrich` StageCommand | Already existed (Phase 1) | No changes needed |
| 3.4 | `evolve` StageCommand | Already existed (Phase 1) | No changes needed |
| 3.5 | Pipeline metrics emission | COMPLETE | `PipelineMetricsEmitter` + runner integration |
| 3.6 | ExtractorPipeline deprecation | COMPLETE (scoped) | Deprecation warning added; Studio extraction preview rewired to v2; full deletion deferred to Phase 7 |
| 3.7 | Normalization deduplication | COMPLETE | `_sanitize_relation_type()` replaced with canonical `sanitize_relation_type()` |

## Files Created

| File | Purpose |
|------|---------|
| `smartmemory/pipeline/metrics.py` | `PipelineMetricsEmitter` — fire-and-forget Redis Streams metrics |
| `tests/unit/pipeline_v2/test_metrics.py` | 8 tests covering emitter behavior, error resilience, runner integration |
| `docs/plans/reports/phase-3-storage-post-processing-report.md` | This report |

## Files Modified

| File | Changes |
|------|---------|
| `smartmemory/memory/pipeline/enrichment.py` | Removed `_sanitize_relation_type()` method; added import for canonical `sanitize_relation_type()`; replaced 3 call sites |
| `smartmemory/memory/pipeline/storage.py` | Added import for canonical `sanitize_relation_type()`; replaced `hasattr` guard pattern on line 282 |
| `smartmemory/pipeline/runner.py` | Added `metrics_emitter` parameter to `__init__`; emit `on_stage_complete()` after timing in `_execute_stage()`; emit on error path; emit `on_pipeline_complete()` in `_run_stages()` |
| `smartmemory/pipeline/__init__.py` | Added `PipelineMetricsEmitter` to exports |
| `smartmemory/smart_memory.py` | Added `create_pipeline_runner()` public method; wired `PipelineMetricsEmitter` into `_create_pipeline_runner()` |
| `smartmemory/memory/pipeline/extractor.py` | Added deprecation warning to `ExtractorPipeline.__init__()` and docstring |
| `smart-memory-studio/server/.../pipeline_registry.py` | Added `_v2_runners` cache and `get_v2_runner()` function |
| `smart-memory-studio/server/.../extraction.py` | Added `run_extraction_v2()` using `PipelineRunner.run_to()` |
| `smart-memory-studio/server/.../transaction.py` | Rewired `preview_extraction()` to use v2 with fallback to v1; added `logging` import |
| `tests/unit/pipeline_v2/test_integration.py` | Added `test_metrics_emitter_called_for_all_11_stages` |
| `CHANGELOG.md` | Added Phase 3 entries under `[Unreleased]` |

## Architecture Decisions

1. **Metrics are fire-and-forget**: `PipelineMetricsEmitter` wraps all emissions in try/except to ensure metrics never break pipeline execution. EventSpooler is lazy-initialized to avoid Redis connection overhead when metrics aren't consumed.

2. **v2 extraction preview with v1 fallback**: `preview_extraction()` tries v2 first. If it fails (missing text, import error, etc.), it falls back to v1 `ExtractorPipeline`. This enables gradual migration.

3. **Public factory caches runner**: `create_pipeline_runner()` returns a cached `PipelineRunner` instance. The runner itself is stateless — all mutable state flows through `PipelineState` objects.

4. **Normalization uses canonical function**: The weak `_sanitize_relation_type()` (space→underscore only) is replaced with `sanitize_relation_type()` from `ingestion/utils.py` which handles regex cleanup, uppercase conversion, FalkorDB-safe prefix, and 50-char limit.

## Test Coverage

- `test_metrics.py`: 8 tests
  - `test_on_stage_complete_emits_event` — verifies event schema
  - `test_on_stage_complete_with_error` — verifies error status
  - `test_on_stage_complete_tracks_timing` — verifies timing accumulation
  - `test_on_pipeline_complete_emits_summary` — verifies summary event
  - `test_emitter_handles_redis_failure_gracefully` — error resilience
  - `test_emitter_with_none_spooler_is_silent` — Redis unavailable
  - `test_runner_calls_metrics_emitter_for_all_stages` — runner integration (3 stages)
  - `test_runner_calls_metrics_on_error_path` — error path metrics
- `test_integration.py`: 1 new test
  - `test_metrics_emitter_called_for_all_11_stages` — full pipeline metrics integration

## Deferred Items

| Item | Reason | Target |
|------|--------|--------|
| Full `ExtractorPipeline` deletion (491 LOC) | ~15 Studio files import from old pipeline | Phase 7 |
| Full Studio service layer migration | Too many files, too risky for Phase 3 | Phase 7 |
| Pre-aggregated metrics consumer | Depends on Insights dashboard (Phase 6) | Phase 6 |

## Corrections from Plan

1. **transaction.py line 97**: Plan said to change `pipeline.run_extraction(typed_cfg)` directly. Instead, implemented a try/except v2-first-then-v1-fallback pattern for safer migration.
2. **`PipelineConfig.preview()` workspace_id**: Plan used `scope_provider.workspace_id` but the actual accessor is `scope_provider.user.tenant_id`.
3. **`PipelineMetricsEmitter.on_stage_complete()` signature**: Plan included `config` parameter; dropped it since config is not needed for metrics and simplifies the interface.
