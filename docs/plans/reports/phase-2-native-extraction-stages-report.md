# Phase 2: Native Extraction Stages — Implementation Report

**Date completed:** 2026-02-05
**Phase:** 2 of 8
**Status:** COMPLETE

---

## Summary

Replaced the single legacy `ExtractStage` wrapper with 4 native stages: `SimplifyStage`, `EntityRulerStage`, `LLMExtractStage`, `OntologyConstrainStage`. Pipeline grew from 8 stages to 11 stages. The legacy `ExtractionPipeline` is no longer called by the pipeline. Self-learning promotion flow (Redis Streams) and Studio frontend changes were deferred.

## Deliverables

| # | Deliverable | Files | Status |
|---|------------|-------|--------|
| 2.1 | `PipelineState` field rename: `simplified_text` → `simplified_sentences: List[str]` | `smartmemory/pipeline/state.py` | Done |
| 2.2 | `PipelineConfig` extensions (5 config dataclasses) | `smartmemory/pipeline/config.py` | Done |
| 2.3 | `SimplifyStage` — spaCy dep-parse text simplification | `smartmemory/pipeline/stages/simplify.py` | Done |
| 2.4 | `EntityRulerStage` — spaCy NER with label mapping | `smartmemory/pipeline/stages/entity_ruler.py` | Done |
| 2.5 | `LLMExtractStage` — wraps `LLMSingleExtractor` | `smartmemory/pipeline/stages/llm_extract.py` | Done |
| 2.6 | `OntologyConstrainStage` — entity merge, type validation, relation filtering | `smartmemory/pipeline/stages/ontology_constrain.py` | Done |
| 2.7 | Delete `ExtractStage` + rewire pipeline (8 → 11 stages) | `smartmemory/pipeline/stages/`, `smart_memory.py` | Done |
| 2.8 | Studio backend updates (models + pipeline_info) | `smart-memory-studio/server/` | Done |
| 2.9 | Prompt migration to `prompts.json` | — | Deferred to Phase 3 |

## Files Created

```
smartmemory/pipeline/stages/simplify.py           # SimplifyStage (150 LOC)
smartmemory/pipeline/stages/entity_ruler.py       # EntityRulerStage (131 LOC)
smartmemory/pipeline/stages/llm_extract.py        # LLMExtractStage (~120 LOC)
smartmemory/pipeline/stages/ontology_constrain.py # OntologyConstrainStage (~200 LOC)
tests/unit/pipeline_v2/stages/test_simplify.py    # 9 tests
tests/unit/pipeline_v2/stages/test_entity_ruler.py # 7 tests
tests/unit/pipeline_v2/stages/test_llm_extract.py  # 8 tests
tests/unit/pipeline_v2/stages/test_ontology_constrain.py # 12 tests
```

## Files Modified

```
smartmemory/pipeline/state.py               # simplified_text → simplified_sentences
smartmemory/pipeline/config.py              # Extended 5 config dataclasses
smartmemory/pipeline/stages/__init__.py     # New exports, removed ExtractStage
smartmemory/smart_memory.py                 # Rewired 8 → 11 stages
tests/unit/pipeline_v2/test_state.py        # Field rename
tests/unit/pipeline_v2/test_config.py       # 17 new config tests
tests/unit/pipeline_v2/test_integration.py  # Rewrote for 11-stage pipeline
smart-memory-studio/server/memory_studio/api/routes/pipeline_info.py  # 4 new stage descriptions
smart-memory-studio/server/memory_studio/api/routes/pipeline.py       # TODO for substage preview
smart-memory-studio/server/memory_studio/models/pipeline.py           # 3 new request dataclasses
smart-memory/CHANGELOG.md                   # Phase 2 entries
```

## Files Deleted

```
smartmemory/pipeline/stages/extract.py          # Legacy wrapper, replaced by 4 native stages
tests/unit/pipeline_v2/stages/test_extract.py   # Replaced by 4 native test files
```

## Config Changes

| Config | Changes |
|--------|---------|
| `SimplifyConfig` | Rewritten: `enabled=True`, `split_clauses=True`, `extract_relative=True`, `passive_to_active=True`, `extract_appositives=True`, `min_token_count=4` |
| `EntityRulerConfig` | Added: `pattern_sources=["builtin"]`, `min_confidence=0.85`, `spacy_model="en_core_web_sm"` |
| `LLMExtractConfig` | Added: `max_relations=30` |
| `ConstrainConfig` | Added: `domain_range_validation=True` |
| `PromotionConfig` | Added: `reasoning_validation=False`, `min_frequency=2`, `min_confidence=0.7` |

## Stage Details

### SimplifyStage

Splits complex sentences into simpler atomic statements using spaCy dependency parsing. Four configurable transforms:

- **split_clauses** — splits on coordinating conjunctions (`cc` dep)
- **extract_relative** — extracts `relcl` dependencies into standalone sentences
- **passive_to_active** — no-op placeholder (full rewriting needs dedicated rewriter)
- **extract_appositives** — extracts `appos` dependencies into "X is Y" form

Short text bypass: texts with fewer than `min_token_count` tokens skip spaCy processing.

### EntityRulerStage

Rule-based entity extraction using spaCy NER with label mapping to SmartMemory's type system:

- `PERSON` → `person`, `ORG/NORP` → `organization`, `GPE/LOC/FAC` → `location`
- `DATE/TIME` → `temporal`, `EVENT` → `event`, `PRODUCT` → `product`
- Others → `concept`

Fixed confidence of 0.9 (spaCy NER doesn't produce per-entity scores). Deduplicates by `(name.lower(), entity_type)`.

### LLMExtractStage

Wraps `LLMSingleExtractor` from the plugin system. Returns `MemoryItem` entities and `{source_id, target_id, relation_type}` relation dicts. Truncates to `max_entities` and `max_relations`.

### OntologyConstrainStage

The critical merge + validation stage:

1. **Merge** ruler + LLM entities by name (case-insensitive). Higher confidence wins. Ruler type preferred when ruler found the entity.
2. **Validate** entity types against `OntologyGraph.get_type_status()`:
   - seed/confirmed/provisional → accept
   - unknown + high confidence → `add_provisional()`, add to `promotion_candidates`
   - unknown + low confidence → rejected
3. **Filter relations** — keep only those with both endpoints in accepted entities
4. **Apply limits** — truncate to `max_entities` / `max_relations`
5. **Auto-promote** if `require_approval=False` in PromotionConfig

## Pipeline at End of Phase 2

```
classify → coreference → simplify → entity_ruler → llm_extract → ontology_constrain → store → link → enrich → ground → evolve
(11 stages)
```

## Test Coverage

- 188 unit tests passing (up from 109 after Phase 1)
- New tests: 36 (9 simplify + 7 entity_ruler + 8 llm_extract + 12 ontology_constrain)
- Updated tests: 17 config tests + integration test rewrite

## Deferred Items

- **Prompt migration** (2.9) — hardcoded prompts in `llm_single.py` and `reasoning.py` not yet moved to `prompts.json`. Deferred to Phase 3.
- **Self-learning promotion flow** — Redis Streams-based promotion deferred to Phase 4.
- **Studio frontend** — only backend metadata/models updated. Frontend deferred to Phase 7.
- **passive_to_active transform** — no-op placeholder. Full passive-to-active rewriting requires a dedicated syntactic rewriter.

## Corrections from Plan

| Plan Assumption | Actual Implementation | Resolution |
|----------------|----------------------|------------|
| `SimplifyConfig.model` field | Not needed (spaCy model is on EntityRulerConfig) | Removed from SimplifyConfig |
| `PipelineConfig` uses Pydantic | Uses `DataclassModelMixin` (MemoryBaseModel) | Followed existing pattern |
| EntityRuler adds spaCy EntityRuler pipe | Uses native spaCy NER (no pipe addition needed) | Simpler approach, same output |
| `LLMSingleExtractor.extract()` returns specific format | Returns `{'entities': List[MemoryItem], 'relations': List[dict]}` | Matched actual API |
