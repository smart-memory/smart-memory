"""
Fast ingestion flow for agentic memory.
Stores items immediately (<500ms) and queues expensive operations for background processing.
"""

import logging
import time
from datetime import datetime, timezone
from typing import Dict, Any

from smartmemory.memory.context_types import IngestionContext
from smartmemory.memory.ingestion.flow import MemoryIngestionFlow
from smartmemory.observability.events import RedisStreamQueue
from smartmemory.observability.instrumentation import emit_after, emit_ctx


class FastIngestionFlow(MemoryIngestionFlow):
    """Fast ingestion flow with background processing."""

    def __init__(self, memory, linking, enrichment, background_mode: str = "redis", **kwargs):
        super().__init__(memory, linking, enrichment, **kwargs)
        self.logger = logging.getLogger(__name__)
        self.mode = (background_mode or "redis").lower()
        # Redis Streams job queues (namespace-aware, config-driven) only when using redis mode
        if self.mode == "redis":
            self._enrich_queue = RedisStreamQueue.for_enrich()
            self._ground_queue = RedisStreamQueue.for_ground()
        else:
            self._enrich_queue = None
            self._ground_queue = None

    def _payload_ingest_fast(self, ctx: IngestionContext):
        try:
            d = ctx or {}
            return {
                'entities_extracted': len(d.get('entities', []) or []),
                'background_queued': bool(d.get('background_queued', False)),
                'ingestion_time_ms': int((d.get('ingestion_time') or 0) * 1000),
            }
        except Exception:
            return {}

    @emit_after(
        "performance_metrics",
        component="ingestion",
        operation="ingest_run_fast",
        payload_fn=lambda self, args, kwargs, result: self._payload_ingest_fast(result),
        measure_time=True,
    )
    def run_fast(self, item, context: IngestionContext = None, adapter_name=None, converter_name=None, extractor_name=None) -> IngestionContext:
        """
        Fast ingestion - store immediately, process later.
        Target: <500ms ingestion time.
        """
        start_time = time.time()
        context = context or IngestionContext()

        # FAST PATH - Minimal processing for immediate storage

        # 1. Adapt input and classify (fast)
        item = self.to_memory_item(item, adapter_name)
        types = self.classify_item(item)
        context['item'] = item
        context['classified_types'] = types

        # 2. Quick semantic extraction (cached/lightweight only)
        entities = []
        relations = []
        ontology_extraction = {}

        if 'semantic' in types:
            try:
                # Use lightweight extraction or cached results
                extraction = self._extract_semantics_fast(item, extractor_name)
                if extraction:
                    entities = extraction.get('entities', [])
                    relations = extraction.get('relations', [])
                    ontology_extraction = {
                        'entities': entities,
                        'relations': relations,
                    }
            except Exception as e:
                # Fail fast on extraction errors in fast path
                raise

        context['entities'] = entities
        context['relations'] = relations
        context['ontology_extraction'] = ontology_extraction

        # 3. Immediate creation via CRUD (dual-node by default)
        if ontology_extraction and len(entities) > 0:
            add_result = self.memory.add_basic(
                item,
                ontology_extraction=ontology_extraction
            )
            # Normalize return shape
            created_entity_ids = []
            if isinstance(add_result, dict):
                item_id = add_result.get('memory_node_id')
                created_entity_ids = add_result.get('entity_node_ids', []) or []
            else:
                item_id = add_result
            # Extract entity IDs
            entity_ids = {}
            for i, entity in enumerate(entities):
                # Robustly resolve a display name for the entity from various shapes
                name = None
                try:
                    # Dict-shaped entity
                    if isinstance(entity, dict):
                        name = entity.get('name') or entity.get('content') or entity.get('text')
                    else:
                        # MemoryItem or similar object with attributes
                        if hasattr(entity, 'metadata') and isinstance(getattr(entity, 'metadata', None), dict):
                            name = entity.metadata.get('name')
                        if not name and hasattr(entity, 'content'):
                            name = getattr(entity, 'content')
                        if not name and hasattr(entity, 'name'):
                            name = getattr(entity, 'name')
                except Exception:
                    # Fall through to default naming
                    pass
                if not name or not isinstance(name, str) or name.strip() == "":
                    name = f"entity_{i}"
                # Use actual created ID when available, else fallback by convention
                real_id = created_entity_ids[i] if i < len(created_entity_ids) else f"{item_id}_entity_{i}"
                entity_ids[name] = real_id
        else:
            add_result = self.memory.add_basic(item)
            if isinstance(add_result, dict):
                item_id = add_result.get('memory_node_id')
            else:
                item_id = add_result
            entity_ids = {}

        item.item_id = item_id
        item.update_status('created', notes='Item ingested (fast path)')
        context['entity_ids'] = entity_ids

        # 4. Basic vector store (fast - can be async later)
        try:
            self._save_to_vector_store_fast(context)
        except Exception as e:
            # Fail fast on vector store errors
            raise

        ingestion_time = time.time() - start_time
        context['ingestion_time'] = ingestion_time
        context['background_queued'] = False

        # 5. Background processing according to mode
        if self.mode == "redis":
            try:
                # Enqueue enrichment job
                enrich_payload = {
                    'job_type': 'enrich',
                    'item_id': item_id,
                    'types': types,
                    'entities': entities,
                    'entity_ids': entity_ids,
                }
                if self._enrich_queue:
                    msg_id = self._enrich_queue.enqueue(enrich_payload)
                    try:
                        emit_ctx(
                            "background_process",
                            component="background",
                            operation="enqueue_enrich",
                            data={
                                "item_id": item_id,
                                "message_id": msg_id,
                                "stream": getattr(self._enrich_queue, "stream_name", None),
                                "has_entities": bool(entities),
                                "types": types,
                            },
                        )
                    except Exception:
                        pass
                # Soft-state tagging on node for observability
                try:
                    self.memory.update_properties(
                        item_id,
                        {
                            'processing_state': 'queued_enrich',
                            'last_enqueued_at': datetime.now(timezone.utc).isoformat()
                        }
                    )
                except Exception:
                    pass

                # Optionally enqueue grounding job if we already have entities
                if entities and self._ground_queue:
                    ground_payload = {
                        'job_type': 'ground',
                        'item_id': item_id,
                        'entities': entities,
                        # 'provenance_candidates': []  # optional, worker may derive
                    }
                    g_msg_id = self._ground_queue.enqueue(ground_payload)
                    try:
                        emit_ctx(
                            "background_process",
                            component="background",
                            operation="enqueue_ground",
                            data={
                                "item_id": item_id,
                                "message_id": g_msg_id,
                                "stream": getattr(self._ground_queue, "stream_name", None),
                                "entities_count": len(entities),
                            },
                        )
                    except Exception:
                        pass
                context['background_queued'] = True
            except Exception as e:
                # Fail fast on background job enqueue failures
                raise

        # Local in-process background only when mode=local
        if not context.get('background_queued') and self.mode == "local" and self.background_processor:
            try:
                # Use synchronous background processing instead of async
                if hasattr(self.background_processor, 'enqueue_enrichment'):
                    # Call synchronous version directly
                    self.background_processor.enqueue_enrichment(
                        memory_id=item_id,
                        context={
                            'item': item,
                            'types': types,
                            'entities': entities,
                            'entity_ids': entity_ids
                        },
                        priority=1
                    )
                    try:
                        emit_ctx(
                            "background_process",
                            component="background",
                            operation="local_enqueue_enrich",
                            data={
                                "item_id": item_id,
                                "has_entities": bool(entities),
                                "types": types,
                            },
                        )
                    except Exception:
                        pass
                    if entities and hasattr(self.background_processor, 'enqueue_grounding'):
                        # Call synchronous version directly
                        self.background_processor.enqueue_grounding(
                            memory_id=item_id,
                            entities=entities,
                            priority=2
                        )
                        try:
                            emit_ctx(
                                "background_process",
                                component="background",
                                operation="local_enqueue_ground",
                                data={
                                    "item_id": item_id,
                                    "entities_count": len(entities),
                                },
                            )
                        except Exception:
                            pass
                    context['background_queued'] = True
                else:
                    # If local mode is configured but loop isn't running, consider it a failure
                    raise RuntimeError("Background processor loop not running; cannot enqueue tasks")
            except Exception as e:
                # Fail fast on local background queuing failures
                raise

        self.logger.info(f"Fast ingestion completed in {ingestion_time:.3f}s for item {item_id}")
        return context

    def _extract_semantics_fast(self, item, extractor_name=None):
        """Fast semantic extraction - use cached results or lightweight extraction."""
        # Try cached extraction first
        cached_result = self._get_cached_extraction(item)
        if cached_result:
            return cached_result

        # Use lightweight extraction (no LLM calls)
        if extractor_name == 'ontology':
            # Skip expensive LLM extraction in fast path
            return None

        # Use configured fast default extractor (LLM-enabled by default)
        try:
            try:
                from smartmemory.utils import get_config
                fast_default_cfg = (get_config('extractor') or {}).get('fast_default')
            except Exception:
                fast_default_cfg = None
            # If not explicitly configured, align with canonical default selection (typically 'llm')
            try:
                selected_default = fast_default_cfg or self._select_default_extractor()
            except Exception:
                selected_default = fast_default_cfg or 'llm'
            return self.extract_semantics(item, extractor_name or selected_default)
        except Exception as e:
            # Fail fast if fast extraction fails
            raise

    def _get_cached_extraction(self, item):
        """Check for cached extraction results."""
        try:
            # Import cache lazily to avoid hard dependency if Redis is down
            from smartmemory.utils.cache import get_cache
            cache = get_cache()
        except Exception as e:
            # Cache unavailable; proceed without cached extraction
            if hasattr(self, 'logger'):
                self.logger.debug(f"Fast extraction cache unavailable: {e}")
            return None

        # Robustly obtain the textual content to hash
        try:
            content = item.content if hasattr(item, 'content') else str(item)
        except Exception:
            content = str(item)

        try:
            cached = cache.get_entity_extraction(content)
            if cached:
                if hasattr(self, 'logger'):
                    snippet = content[:50].replace("\n", " ")
                    self.logger.debug(f"Fast path: cache hit for extraction on '{snippet}...' ")
                return cached
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.debug(f"Fast path: cache lookup failed: {e}")
        return None

    def _save_to_vector_store_fast(self, context):
        """Fast vector store save using existing encapsulated method."""
        try:
            # Use the existing encapsulated vector storage method from parent class
            # This ensures proper API usage and maintains architectural consistency
            self._save_to_vector_and_graph(context)
        except Exception as e:
            raise Exception(f"Vector store save failed: {e}")

    @emit_after(
        "background_process",
        component="background",
        operation="enrich_process",
        payload_fn=lambda self, args, kwargs, result: {
            "item_id": (args[1] if len(args) > 1 else kwargs.get("memory_id")),
            "entities_in": len(((args[2] if len(args) > 2 else kwargs.get("context") or {}) or {}).get("entities", [])),
        },
        measure_time=True,
    )
    def _background_enrichment(self, memory_id: str, context: Dict[str, Any]):
        """Background enrichment processor."""
        try:
            # Ensure the memory item exists via public API
            try:
                existing_item = self.memory.get(memory_id)
            except Exception:
                existing_item = None
            if not existing_item:
                self.logger.warning(f"Memory node {memory_id} not found for enrichment")
                return

            # Run full enrichment (LLM calls, Wikipedia, etc.)
            enrichment_context = {
                'item': context['item'],
                'node_ids': context.get('entity_ids') or {},
                'entities': context.get('entities', [])
            }

            enrichment_result = self.enrichment.enrich(enrichment_context)

            # Update nodes with enrichment results
            if enrichment_result:
                self._apply_enrichment_results(memory_id, enrichment_result)

            return enrichment_result

        except Exception as e:
            self.logger.error(f"Background enrichment failed for {memory_id}: {e}")
            raise

    @emit_after(
        "background_process",
        component="background",
        operation="ground_process",
        payload_fn=lambda self, args, kwargs, result: {
            "item_id": (args[1] if len(args) > 1 else kwargs.get("memory_id")),
            "entities_count": len((args[2] if len(args) > 2 else kwargs.get("entities", [])) or []),
        },
        measure_time=True,
    )
    def _background_grounding(self, memory_id: str, entities: list):
        """Background grounding processor."""
        try:
            # Run grounding for entities
            grounding_context = {
                'memory_id': memory_id,
                'entities': entities,
                'entity_ids': {}  # Will be populated
            }

            # Use public SmartMemory grounding wrapper
            self.memory.ground_context(grounding_context)

            return {'grounded_entities': len(entities)}

        except Exception as e:
            self.logger.error(f"Background grounding failed for {memory_id}: {e}")
            raise

    @emit_after(
        "background_process",
        component="background",
        operation="evolution_process",
        payload_fn=lambda self, args, kwargs, result: {
            "memory_type": (args[1] if len(args) > 1 else kwargs.get("memory_type")),
            "items_count": len((args[2] if len(args) > 2 else kwargs.get("items", [])) or []),
        },
        measure_time=True,
    )
    def _background_evolution(self, memory_type: str, items: list):
        """Background evolution processor."""
        try:
            # Use public SmartMemory evolution wrapper
            evolution_result = self.memory.run_evolution_cycle()

            return evolution_result

        except Exception as e:
            self.logger.error(f"Background evolution failed for {memory_type}: {e}")
            raise

    def _apply_enrichment_results(self, memory_id: str, enrichment_result: Dict[str, Any]):
        """Apply enrichment results to memory and entity nodes."""
        try:
            # Backward-compatibility: map legacy keys (summary, tags) into properties if not already present
            try:
                legacy_props = {}
                if isinstance(enrichment_result, dict):
                    if 'summary' in enrichment_result:
                        legacy_props.setdefault('summary', enrichment_result.get('summary'))
                    if 'tags' in enrichment_result:
                        # merge with any existing properties.tags later
                        legacy_props.setdefault('tags', enrichment_result.get('tags') or [])
                if legacy_props:
                    if 'properties' not in enrichment_result or not isinstance(enrichment_result.get('properties'), dict):
                        enrichment_result['properties'] = {}
                    # Merge tags specially (dedupe)
                    if 'tags' in legacy_props:
                        existing = enrichment_result['properties'].get('tags') or []
                        merged = list(dict.fromkeys(list(existing) + list(legacy_props['tags'])))
                        enrichment_result['properties']['tags'] = merged
                    # Set summary if not already provided by a plugin
                    if 'summary' in legacy_props and 'summary' not in enrichment_result['properties']:
                        enrichment_result['properties']['summary'] = legacy_props['summary']
            except Exception:
                pass
            # Update memory node properties
            if 'properties' in enrichment_result:
                # Update via SmartMemory public API (config-driven write mode inside CRUD)
                try:
                    self.memory.update_properties(memory_id, enrichment_result['properties'])
                except Exception as e:
                    self.logger.error(f"CRUD update failed for {memory_id}: {e}")

            # Create new relations if specified
            if 'relations' in enrichment_result:
                for rel in enrichment_result['relations']:
                    self.memory.add_edge(
                        source_id=rel['source_id'],
                        target_id=rel['target_id'],
                        relation_type=rel['relation_type'],
                        properties=rel.get('properties') or {}
                    )

            # Observability: log sentiment/topic metrics if enabled
            try:
                from smartmemory.utils import get_config
                obs_cfg = (get_config('observability') or {})
                if (obs_cfg.get('enrichment_metrics') or {}).get('enabled', False):
                    props = enrichment_result.get('properties') or {}
                    sentiment = props.get('sentiment') if isinstance(props, dict) else None
                    topics = props.get('topics') if isinstance(props, dict) else None
                    metrics = {
                        'memory_id': memory_id,
                        'sentiment_label': (sentiment or {}).get('label') if isinstance(sentiment, dict) else None,
                        'sentiment_score': (sentiment or {}).get('score') if isinstance(sentiment, dict) else None,
                        'dominant_topic': props.get('dominant_topic') if isinstance(props, dict) else None,
                        'topics_count': len(topics) if isinstance(topics, list) else 0,
                    }
                    self.logger.info(f"enrichment_metrics: {metrics}")
            except Exception:
                # Metrics are best-effort
                pass

        except Exception as e:
            self.logger.error(f"Failed to apply enrichment results for {memory_id}: {e}")
