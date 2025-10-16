"""
Streamlined ingestion flow orchestrator.

This module provides a clean orchestration layer that delegates to specialized pipeline modules:
- ExtractionPipeline: Handles all extraction logic
- StoragePipeline: Handles storage and graph operations  
- EnrichmentPipeline: Handles enrichment operations
- IngestionRegistry: Manages component registration
- IngestionObserver: Handles observability

The main flow is reduced to ~200 lines of pure orchestration.
"""
import logging
import time
from typing import TYPE_CHECKING

from smartmemory.memory.context_types import IngestionContext
from smartmemory.memory.ingestion import utils as ingestion_utils
from smartmemory.memory.ingestion.enrichment import EnrichmentPipeline
from smartmemory.memory.ingestion.extraction import ExtractionPipeline
from smartmemory.memory.ingestion.observer import IngestionObserver
from smartmemory.memory.ingestion.registry import IngestionRegistry
from smartmemory.memory.ingestion.storage import StoragePipeline
from smartmemory.memory.pipeline.config import (
    ClassificationConfig, LinkingConfig, StorageConfig,
    GroundingConfig, EnrichmentConfig, ExtractionConfig,
    InputAdapterConfig, PipelineConfigBundle
)
from smartmemory.models.memory_item import MemoryItem
from smartmemory.observability.instrumentation import emit_after

if TYPE_CHECKING:
    pass


class MemoryIngestionFlow:
    """
    Streamlined ingestion flow orchestrator.
    
    Delegates complex operations to specialized pipeline modules while maintaining
    a clean orchestration interface for the main ingestion workflow.
    """

    def __init__(self, memory: "SmartMemory", linking, enrichment, adapters=None, converters=None, extractors=None, enrichers=None):
        """Initialize the ingestion flow with pipeline modules."""
        self._logger = logging.getLogger(__name__)
        self.memory = memory
        self.linking = linking
        self.enrichment = enrichment
        self.adapters = adapters or {}
        self.converters = converters or {}

        # Initialize core modules
        self.registry = IngestionRegistry()
        self.observer = IngestionObserver()

        # Initialize pipeline modules
        self.extraction_pipeline = ExtractionPipeline(self.registry, self.observer)
        self.storage_pipeline = StoragePipeline(self.memory, self.observer)
        self.enrichment_pipeline = EnrichmentPipeline(self.memory, self.enrichment, self.observer)

        # Register additional extractors and enrichers if provided
        if extractors:
            for name, fn in extractors.items():
                self.registry.register_extractor(name, fn)
        if enrichers:
            for name, fn in enrichers.items():
                self.registry.register_enricher(name, fn)

    # Registry methods delegate to the registry module
    def register_adapter(self, name: str, adapter_fn):
        """Register a new input adapter by name."""
        self.registry.register_adapter(name, adapter_fn)

    def register_converter(self, name: str, converter_fn):
        """Register a new type converter by name."""
        self.registry.register_converter(name, converter_fn)

    def register_extractor(self, name: str, extractor_fn):
        """Register a new entity/relation extractor by name."""
        self.registry.register_extractor(name, extractor_fn)

    def register_enricher(self, name: str, enricher_fn):
        """Register a new enrichment routine by name."""
        self.registry.register_enricher(name, enricher_fn)

    def to_memory_item(self, item, adapter_name=None):
        """Convert input to MemoryItem if needed, normalize universal metadata."""
        from datetime import datetime

        if not isinstance(item, MemoryItem):
            item = MemoryItem(**item) if isinstance(item, dict) else MemoryItem(content=str(item))

        # Normalize metadata
        now = datetime.now()
        if not hasattr(item, 'metadata') or item.metadata is None:
            item.metadata = {}
        item.metadata.setdefault('created_at', now)
        item.metadata['updated_at'] = now
        item.metadata.setdefault('status', 'created')
        item.memory_type = 'semantic'
        return item

    def _resolve_configurations(self, pipeline_config, classification_config, linking_config,
                                storage_config, grounding_config, enrichment_config,
                                extraction_config, input_adapter_config, adapter_name,
                                converter_name, extractor_name, enricher_names) -> PipelineConfigBundle:
        """Resolve configuration parameters with backward compatibility."""
        bundle = pipeline_config or PipelineConfigBundle()

        # Override with individual configs if provided
        if input_adapter_config:
            bundle.input_adapter = input_adapter_config
        elif adapter_name and bundle.input_adapter:
            bundle.input_adapter.adapter_name = adapter_name

        if classification_config:
            bundle.classification = classification_config

        if extraction_config:
            bundle.extraction = extraction_config
        elif extractor_name and bundle.extraction:
            bundle.extraction.extractor_name = extractor_name

        if storage_config:
            bundle.storage = storage_config

        if linking_config:
            bundle.linking = linking_config

        if enrichment_config:
            bundle.enrichment = enrichment_config
        elif enricher_names and bundle.enrichment:
            bundle.enrichment.enricher_names = enricher_names

        if grounding_config:
            bundle.grounding = grounding_config

        return bundle

    @emit_after(
        "performance_metrics",
        component="ingestion",
        operation="ingest_run",
        payload_fn=lambda self, args, kwargs, result: ingestion_utils.extract_payload_for_instrumentation(result),
        measure_time=True,
    )
    def run(self, item, context: IngestionContext = None,
            # Legacy parameters for backward compatibility
            adapter_name=None, converter_name=None, extractor_name=None, enricher_names=None,
            # New configuration parameters
            pipeline_config: PipelineConfigBundle = None,
            classification_config: ClassificationConfig = None,
            linking_config: LinkingConfig = None,
            storage_config: StorageConfig = None,
            grounding_config: GroundingConfig = None,
            enrichment_config: EnrichmentConfig = None,
            extraction_config: ExtractionConfig = None,
            input_adapter_config: InputAdapterConfig = None) -> IngestionContext:
        """
        Execute the streamlined ingestion flow with modular pipeline delegation.
        
        This orchestrator coordinates the pipeline stages while delegating complex
        operations to specialized pipeline modules.
        """
        context = context or IngestionContext()
        context['start_time'] = time.time()

        # Resolve configuration parameters
        config_bundle = self._resolve_configurations(
            pipeline_config, classification_config, linking_config, storage_config,
            grounding_config, enrichment_config, extraction_config, input_adapter_config,
            adapter_name, converter_name, extractor_name, enricher_names
        )

        # Stage 1: Input adaptation and classification
        adapter_config = config_bundle.input_adapter
        item = self.to_memory_item(item, adapter_config.adapter_name if adapter_config else adapter_name)

        classification_conf = config_bundle.classification
        types = self.classify_item(item, classification_conf)
        context['item'] = item
        context['classified_types'] = types

        # Emit ingestion start event
        extraction_conf = config_bundle.extraction
        actual_extractor = extraction_conf.extractor_name if extraction_conf else extractor_name
        actual_adapter = adapter_config.adapter_name if adapter_config else adapter_name

        self.observer.emit_ingestion_start(
            item_id=item.item_id,
            content_length=len(item.content),
            extractor=actual_extractor or 'default',
            adapter=actual_adapter or 'default'
        )

        # Stage 2: Semantic extraction (delegated to extraction pipeline)
        self.observer.emit_event('extraction_start', {
            'item_id': item.item_id,
            'extractor': extractor_name or 'default'
        })

        try:
            extraction = self.extraction_pipeline.extract_semantics(item, actual_extractor, config_bundle.extraction)
            # Standardize on 'entities' for internal flow (Path A); accept 'nodes' for back-compat
            entities = extraction.get('entities') or extraction.get('nodes') or []
            relations = extraction.get('relations', [])

            # Emit extraction results
            self.observer.emit_extraction_results(
                item_id=item.item_id,
                entities_count=len(entities),
                relations_count=len(relations),
                extractor=extractor_name or 'default'
            )

            # Build ontology_extraction payload
            ontology_extraction = None
            if entities or relations:
                ontology_extraction = {
                    'entities': entities,  # Use normalized entities (MemoryItems)
                    'relations': relations,
                }

        except Exception as e:
            self.observer.emit_error(
                item_id=item.item_id,
                error=str(e),
                error_type=type(e).__name__,
                stage='extraction'
            )
            raise

        context['entities'] = entities
        context['relations'] = relations
        context['ontology_extraction'] = ontology_extraction

        # Stage 3: Item storage and entity creation
        add_result = self.memory._crud.add(item, ontology_extraction=ontology_extraction)
        entity_ids = self._process_add_result(add_result, entities, item)
        context['entity_ids'] = entity_ids

        # Stage 4: Relationship processing (delegated to storage pipeline)
        if relations and len(relations) > 0:
            try:
                self.storage_pipeline.process_extracted_relations(context, item.item_id, relations)
                print(f"✅ Processed {len(relations)} relationships for item: {item.item_id}")
            except Exception as e:
                print(f"⚠️  Failed to process relationships: {e}")
                raise

        # Stage 5: Linking
        self.linking.link_new_item(context)
        context['links'] = context.get('links') or {}

        # Stage 6: Vector and graph storage (delegated to storage pipeline)
        self.storage_pipeline.save_to_vector_and_graph(context)

        # Stage 7: Enrichment (delegated to enrichment pipeline)
        context['node_ids'] = context.get('entity_ids') or {}
        enrichment_result = self.enrichment_pipeline.run_enrichment(context)
        context['enrichment_result'] = enrichment_result

        # Stage 7.5: Ground entities to Wikipedia (populate provenance_candidates)
        try:
            import logging
            logger = logging.getLogger(__name__)
            from smartmemory.plugins.grounders import WikipediaGrounder
            grounder = WikipediaGrounder()
            entities_list = context.get('entities', [])
            entity_ids = context.get('entity_ids', {})
            
            if entities_list:
                # Update entity MemoryItems with their IDs so grounder can create edges
                logger.info(f"Entity IDs mapping: {entity_ids}")
                for entity in entities_list:
                    if hasattr(entity, 'metadata') and entity.metadata:
                        entity_name = entity.metadata.get('name')
                        if entity_name and entity_name in entity_ids:
                            entity.item_id = entity_ids[entity_name]
                            logger.info(f"Set entity '{entity_name}' item_id to: {entity.item_id}")
                
                # Pass MemoryItems directly - grounder extracts what it needs
                provenance_candidates = grounder.ground(item, entities_list, self.memory._graph)
                context['provenance_candidates'] = provenance_candidates
                logger.info(f"✅ Grounded {len(provenance_candidates)} entities to Wikipedia")
            else:
                context['provenance_candidates'] = []
        except Exception as e:
            logger.warning(f"Wikipedia grounding failed: {e}")
            context['provenance_candidates'] = []

        # Stage 8: Grounding (create GROUNDED_IN edges)
        provenance_candidates = context.get('provenance_candidates', [])
        if provenance_candidates:
            self.memory._grounding.ground(context)

        # Emit completion events
        self._emit_completion_events(context, extractor_name, adapter_name)

        return context

    def classify_item(self, item, classification_config: ClassificationConfig = None) -> list[str]:
        """Classify the item for routing using configurable rules and indicators."""
        config = classification_config or ClassificationConfig()
        types = set()

        # Extract metadata
        t = item.metadata.get("type") if hasattr(item, "metadata") else None
        tags = item.metadata.get("tags", []) if hasattr(item, "metadata") else []
        content = getattr(item, 'content', '')

        # Always add core types
        types.add("semantic")
        types.add("zettel")

        # Add explicit type if present
        if t:
            types.add(t)

        # Content-based classification if enabled
        if config.content_analysis_enabled and content:
            content_lower = content.lower()

            for memory_type, indicators in config.content_indicators.items():
                matches = sum(1 for indicator in indicators if indicator.lower() in content_lower)
                if matches > 0:
                    confidence = min(matches / len(indicators), 1.0)
                    if confidence >= config.inferred_confidence:
                        types.add(memory_type)

        # Tag/metadata-based classification
        for memory_type, indicators in config.content_indicators.items():
            indicator_set = {ind.lower() for ind in indicators}

            if t and t.lower() in indicator_set:
                types.add(memory_type)

            if any(tag.lower() in indicator_set for tag in tags):
                types.add(memory_type)

        return list(types)

    def _process_add_result(self, add_result, entities, item):
        """Process the add result and create entity ID mappings."""
        entity_ids = {}

        if isinstance(add_result, dict):
            item_id = add_result.get('memory_node_id')
            created_entity_ids = add_result.get('entity_node_ids', []) or []

            for i, entity in enumerate(entities):
                # Extract name from either MemoryItem or dict
                if hasattr(entity, 'metadata') and entity.metadata and 'name' in entity.metadata:
                    entity_name = entity.metadata['name']
                elif isinstance(entity, dict):
                    metadata = entity.get('metadata', {})
                    entity_name = metadata.get('name') or entity.get('name', f'entity_{i}')
                else:
                    entity_name = f'entity_{i}'

                real_id = created_entity_ids[i] if i < len(created_entity_ids) else f"{item_id}_entity_{i}"
                entity_ids[entity_name] = real_id
        else:
            # Legacy return (string item_id)
            item_id = add_result
            for i, entity in enumerate(entities):
                # Extract name from either MemoryItem or dict
                if hasattr(entity, 'metadata') and entity.metadata and 'name' in entity.metadata:
                    entity_name = entity.metadata['name']
                elif isinstance(entity, dict):
                    metadata = entity.get('metadata', {})
                    entity_name = metadata.get('name') or entity.get('name', f'entity_{i}')
                else:
                    entity_name = f'entity_{i}'

                entity_ids[entity_name] = f"{item_id}_entity_{i}"

        item.item_id = item_id
        item.update_status('created', notes='Item ingested')
        return entity_ids

    def _emit_completion_events(self, context, extractor_name, adapter_name):
        """Emit completion events and metrics."""
        end_time = time.time()
        total_duration_ms = (end_time - context.get('start_time', end_time)) * 1000

        self.observer.emit_ingestion_complete(
            item_id=context.get('item_id'),
            entities_extracted=len(context.get('entities', [])),
            relations_extracted=len(context.get('relations', [])),
            total_duration_ms=total_duration_ms,
            extractor=context.get('extractor_name', extractor_name or 'default'),
            adapter=context.get('adapter_name', adapter_name or 'default')
        )

        self.observer.emit_performance_metrics(context, total_duration_ms)
        self.observer.emit_graph_statistics(self.memory)
