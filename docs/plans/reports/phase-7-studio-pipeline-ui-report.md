# Phase 7: Studio Pipeline UI — Implementation Report

**Date:** 2026-02-06
**Status:** COMPLETE (with noted follow-ups)
**Session plan:** `gentle-exploring-quiche.md`

---

## Summary

Phase 7 delivers the Studio Pipeline UI: a Learning dashboard for self-learning ontology observability, a PipelineConfig editor with save/load profiles, a breakpoint-aware debug runner for step-through pipeline execution, and prompt UI polish. The implementation adds 8 backend endpoints (learning API), 5 dashboard components, 5 custom hooks, 2 pipeline components, and a service client. The Learning page is fully wired with live data; PipelineConfigEditor and BreakpointRunner are built as standalone components pending MemoryLab integration.

---

## Deliverables

| # | Deliverable | Status | Notes |
|---|-------------|--------|-------|
| 7.1 | PipelineConfig editor | DONE | `PipelineConfigEditor.jsx` with 8 stage sections, save/load named profiles via Profiles API |
| 7.2 | Breakpoint execution UI | DONE | `BreakpointRunner.jsx` with run-to, resume, undo, reset. Client-side stage chaining (Option B) |
| 7.3 | Learning page (ontology viewer) | DONE | Full dashboard: stats cards, promotion queue, type registry, pattern browser, activity feed |
| 7.4 | Prompt UI polish | DONE | Added copy-to-clipboard, character count, last-modified timestamp to PromptsPanel |
| 7.5 | Benchmarking workflow | DEFERRED | Deferred to Phase 8 per user decision |
| 7.6 | Grid search / parameter tuning | DEFERRED | Deferred to Phase 8 per user decision |

---

## Files Created

| File | Purpose | ~LOC |
|------|---------|------|
| `smart-memory-studio/server/memory_studio/api/routes/learning.py` | 8 API endpoints for learning dashboard (stats, convergence, promotions, approve/reject, patterns, types, activity) | ~335 |
| `smart-memory-studio/web/src/pages/Learning.jsx` | Learning dashboard page with 5-section layout | ~75 |
| `smart-memory-studio/web/src/components/learning/LearningStatsCards.jsx` | Stats row: total types, patterns, queue depth, avg confidence | ~80 |
| `smart-memory-studio/web/src/components/learning/PromotionQueue.jsx` | Pending promotions with approve/reject, gate details | ~130 |
| `smart-memory-studio/web/src/components/learning/TypeRegistry.jsx` | Searchable/sortable table of entity types | ~120 |
| `smart-memory-studio/web/src/components/learning/PatternBrowser.jsx` | Filterable pattern list with layer badges | ~120 |
| `smart-memory-studio/web/src/components/learning/ActivityFeed.jsx` | Timeline of learning events | ~80 |
| `smart-memory-studio/web/src/components/pipeline/PipelineConfigEditor.jsx` | Form-based config editor, 8 stage sections, profile save/load | ~240 |
| `smart-memory-studio/web/src/components/pipeline/StageConfigSection.jsx` | Collapsible stage config with type-appropriate field rendering | ~100 |
| `smart-memory-studio/web/src/components/pipeline/ConfigField.jsx` | Smart field renderer (switch, number, text, select) | ~80 |
| `smart-memory-studio/web/src/components/pipeline/BreakpointRunner.jsx` | Debug runner with step-through execution, state inspection | ~200 |
| `smart-memory-studio/web/src/components/pipeline/PipelineStepper.jsx` | Horizontal stage visualization with status indicators | ~90 |
| `smart-memory-studio/web/src/components/pipeline/StateInspector.jsx` | Tabbed view of PipelineState (entities, relations, timings) | ~100 |
| `smart-memory-studio/web/src/services/LearningService.js` | API client for `/learning/*` endpoints | ~55 |
| `smart-memory-studio/web/src/hooks/learning/useLearningStats.js` | Stats hook with 30s auto-refresh | ~30 |
| `smart-memory-studio/web/src/hooks/learning/useLearningPromotions.js` | Promotions hook with approve/reject mutations | ~40 |
| `smart-memory-studio/web/src/hooks/learning/useLearningPatterns.js` | Patterns hook with layer/search filtering | ~30 |
| `smart-memory-studio/web/src/hooks/learning/useLearningTypes.js` | Types hook with search/sort params | ~30 |
| `smart-memory-studio/web/src/hooks/learning/useLearningActivity.js` | Activity hook with 30s auto-refresh | ~30 |

## Files Modified

| File | Changes |
|------|---------|
| `smart-memory-studio/server/memory_studio/api/routes/__init__.py` | Registered learning router |
| `smart-memory-studio/web/src/pages/index.jsx` | Added `/Learning` route |
| `smart-memory-studio/web/src/pages/Layout.jsx` | Added "Learning" nav item with `TrendingUp` icon |
| `smart-memory-studio/web/src/components/shared/panels/PromptsPanel.jsx` | Added copy-to-clipboard button, character count, last-modified timestamp, improved empty state |
| `smart-memory/CHANGELOG.md` | Phase 7 entries under Unreleased/Added |

## Files Deleted

None.

---

## Architecture Decisions

1. **Learning page as standalone dashboard** (not OntologyLab tab): The Learning page is a full-page dashboard focused on observability, while OntologyLab remains a sandbox for editing. Avoids overloading the lab layout.

2. **Client-side stage chaining for breakpoints** (Option B): BreakpointRunner chains individual stage previews client-side rather than a server-side `run-to` endpoint. Simpler, more flexible for MVP; server-side can be added later for performance.

3. **Raw useState hooks over React Query**: Learning hooks use raw `useState`/`useEffect` instead of the `@tanstack/react-query` pattern used elsewhere in Studio (e.g., `useProfileQueries.js`). Noted as a consistency issue for follow-up.

4. **PipelineConfigEditor groups stages at config level**: Instead of mapping 1:1 to the 11 pipeline stages, the editor groups by config sections (8 groups: classify, coreference, simplify, extraction, store, link, enrich, evolve). The `extraction` group encompasses entity_ruler, llm_extract, promotion, and constrain sub-configs.

5. **Activity feed from current state**: The MVP activity endpoint reconstructs events from current graph state (confirmed types = promotions, provisional = discoveries, patterns = additions). No event log stored yet — real time-series deferred.

---

## Review Findings

A triple code review identified issues in three categories:

### Critical (fixed in this phase)
| # | Issue | Resolution |
|---|-------|------------|
| C2 | Memory leak: no AbortController in learning hooks | Fixed: added AbortController to all 5 hooks |
| C3 | N+1 queries: `get_frequency()` called per-type in 4 endpoints | Fixed: added `get_all_frequencies()` batch method |
| C4 | Private API: `og._get_backend()` in reject_promotion | Fixed: added public `reject()` to OntologyGraph |
| C5 | Null crash: confidence values can be None | Fixed: safe defaults in backend |

### Deferred to follow-up
| # | Issue | Notes |
|---|-------|-------|
| C1 | PipelineConfigEditor + BreakpointRunner not wired into MemoryLab | Requires MemoryLab panel integration — significant scope |
| I1 | Hooks use raw useState instead of React Query | Consistency issue, not a bug |
| I2 | `'rejected'` not in VALID_STATUSES | Fixed alongside C4 |
| I4 | No debounce on search inputs | Planning standard requires 300ms debounce |
| I5 | No pagination | Planning standard requires PAGE_SIZE=25 |
| I9 | Activity events lack timestamps | MVP reconstructs from state, not event log |
| I10 | Convergence endpoint unused | Endpoint exists but no frontend consumer |
| M1 | Deep links between Learning ↔ OntologyLab | Nice-to-have cross-linking |

---

## Test Coverage

No new automated tests added in this phase. Phase 8 (Hardening) includes test requirements:
- 8.1: Pipeline integration tests
- 8.3: Self-learning tests (promotion flow, convergence)

**Frontend builds:** `smart-memory-studio/web` builds successfully (4.72s, no errors).
**Backend lint:** `ruff check` and `ruff format` pass clean.

---

## Corrections from Plan

| Plan Item | Actual |
|-----------|--------|
| Plan specified 5 custom hooks in `hooks/learning/` | All 5 implemented as planned |
| Plan specified PipelineConfigEditor as MemoryLab integration | Built as standalone component; MemoryLab integration deferred |
| Plan specified "Run All" button in BreakpointRunner | Not implemented; handleReset provides full reset instead |
| Plan specified 11 stage sections in config editor | Implemented as 8 grouped sections matching config structure |
| Plan specified deep links (Step 6) | Not implemented; deferred to follow-up |

---

## Deferred Items

- **MemoryLab integration** (C1): PipelineConfigEditor and BreakpointRunner need to be wired into MemoryConfigurationPanel. This was the plan's Step 6 integration work.
- **React Query migration** (I1): Learning hooks should use `@tanstack/react-query` for consistency with `useProfileQueries.js`.
- **Debounce + pagination**: Planning standards require 300ms debounce on text inputs and PAGE_SIZE=25 pagination.
- **Convergence visualization**: `/learning/convergence` endpoint exists but has no frontend consumer.
- **Event log**: Activity feed reconstructs from state; proper LearningEvent node for time-series history.
- **Deep links**: Learning page → OntologyLab type highlighting, and reverse.
