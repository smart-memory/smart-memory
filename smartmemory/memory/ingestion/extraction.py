"""
Extraction pipeline module for ingestion flow.

This module handles all entity and relation extraction logic, including:
- Extractor selection and fallback mechanisms
- Ontology-aware extraction
- Legacy extractor support
- Entity conversion to MemoryItem objects
"""
import logging
from typing import Dict, List, Any, Optional

from smartmemory.memory.pipeline.config import ExtractionConfig
from smartmemory.models.memory_item import MemoryItem

logger = logging.getLogger(__name__)


class ExtractionPipeline:
    """
    Handles all extraction logic for the ingestion pipeline.
    
    Supports both modern extractors (returning dict format) and legacy extractors
    (returning tuple format), with automatic fallback mechanisms.
    """

    def __init__(self, registry, observer):
        """
        Initialize extraction pipeline.
        
        Args:
            registry: IngestionRegistry instance for extractor management
            observer: IngestionObserver instance for event emission
        """
        self.registry = registry
        self.observer = observer

    def extract_semantics(self, item: MemoryItem, extractor_name: Optional[str] = None,
                          extraction_config: Optional[ExtractionConfig] = None) -> Dict[str, Any]:
        """
        Extract entities and relations from the item using the specified extractor.
        
        Args:
            item: MemoryItem to extract from
            extractor_name: Specific extractor to use (legacy parameter)
            extraction_config: Configuration for extraction behavior
            
        Returns:
            Dict with 'entities' and 'relations' keys
        """
        # Resolve extraction configuration
        config = extraction_config or ExtractionConfig()

        # Use config extractor_name if available, then parameter, then default
        final_extractor_name = config.extractor_name or extractor_name
        if not final_extractor_name:
            final_extractor_name = self.registry.select_default_extractor()

        extractor = self.registry.get_extractor(final_extractor_name)

        # Determine fallback order up-front if fallback is enabled
        fallback_order = []
        if config.enable_fallback_chain:
            fallback_order = self.registry.get_fallback_order(primary=final_extractor_name)

        # If the configured extractor isn't registered, pick the first available fallback
        if extractor is None:
            if fallback_order:
                final_extractor_name = fallback_order[0]
                extractor = self.registry.get_extractor(final_extractor_name)
                fallback_order = self.registry.get_fallback_order(primary=final_extractor_name)
            if extractor is None:
                raise ValueError(f"Extractor '{final_extractor_name}' not registered.")

        # Handle ontology-aware extractor differently
        if final_extractor_name == 'ontology' and config.ontology_extraction:
            return self._extract_with_ontology(item, extractor, fallback_order, config)
        else:
            return self._extract_with_legacy(item, extractor, fallback_order, config)

    def _extract_with_ontology(self, item: MemoryItem, extractor, fallback_order: List[str],
                               config: ExtractionConfig) -> Dict[str, Any]:
        """Handle ontology-aware extraction with fallback support."""
        try:
            user_id = getattr(item, 'user_id', None) or item.metadata.get('user_id')
            result = extractor.extract_entities_and_relations(item.content, user_id)

            # Convert OntologyNode objects to MemoryItem objects
            entities = []
            for node in result['entities']:
                memory_item = node.to_memory_item()
                entities.append(memory_item)

            relations = result['relations']

            # If ontology returns nothing, fall back to configured extractors
            if not entities and not relations and config.enable_fallback_chain:
                return self._try_fallback_extractors(item, fallback_order, config)

            return {
                'entities': entities,
                'relations': relations
            }
        except Exception:
            # On failure, fall back to configured extractors
            return self._try_fallback_extractors(item, fallback_order, config)

    def _extract_with_legacy(self, item: MemoryItem, extractor, fallback_order: List[str],
                             config: ExtractionConfig) -> Dict[str, Any]:
        """Handle legacy extractor with fallback support."""
        try:
            # Call extract method with content
            content = item.content if hasattr(item, 'content') else str(item)
            result = extractor.extract(content)
        except Exception as e:
            logger.error(f"Extractor failed: {e}", exc_info=True)
            result = {}

        # Convert extractor output to consistent format
        if isinstance(result, dict):
            entities = result.get('entities', [])
            relations = result.get('relations', [])
        elif isinstance(result, tuple) and len(result) == 3:
            # Legacy tuple format: (item, entities, relations)
            _item, entities, relations = result
        else:
            entities, relations = [], []
            logger.warning(f"Unexpected extractor result format: {type(result)}")

        # Convert legacy entities to MemoryItem objects
        converted_entities = self._convert_entities_to_memory_items(entities)

        # If no results, try fallback extractors
        if not converted_entities and not relations and config.enable_fallback_chain:
            logger.debug("No extraction results, trying fallback extractors")
            return self._try_fallback_extractors(item, fallback_order, config)

        return {
            'entities': converted_entities,
            'relations': relations
        }

    def _try_fallback_extractors(self, item: MemoryItem, fallback_order: List[str],
                                 config: ExtractionConfig) -> Dict[str, Any]:
        """Try fallback extractors in order until one succeeds."""
        attempts = 0

        for fb_name in fallback_order:
            if attempts >= config.max_extraction_attempts:
                break

            fb = self.registry.get_extractor(fb_name)
            if not fb:
                continue

            try:
                fb_res = fb(item)
                attempts += 1
            except Exception:
                attempts += 1
                continue

            # Normalize fallback outputs
            if isinstance(fb_res, dict):
                fb_entities = fb_res.get('entities', [])
                fb_relations = fb_res.get('relations', [])
            elif isinstance(fb_res, tuple) and len(fb_res) == 3:
                _it, fb_entities, fb_relations = fb_res
            else:
                fb_entities, fb_relations = [], []

            # Convert entities to MemoryItem objects
            converted_entities = []
            if config.legacy_extractor_support:
                converted_entities = self._convert_entities_to_memory_items(fb_entities)

            if converted_entities or fb_relations:
                return {
                    'entities': converted_entities,
                    'relations': fb_relations
                }

        # Nothing worked - return empty results
        return {
            'entities': [],
            'relations': []
        }

    def _convert_entities_to_memory_items(self, entities: List[Any]) -> List[MemoryItem]:
        """Convert various entity formats to MemoryItem objects."""
        converted_entities = []

        for i, entity in enumerate(entities):
            if isinstance(entity, MemoryItem):
                # Already a MemoryItem, use as-is
                converted_entities.append(entity)
            elif isinstance(entity, dict):
                # Convert dict entity to MemoryItem
                entity_content = entity.get('name', entity.get('text', f'entity_{i}'))
                entity_type = entity.get('type', entity.get('label', 'entity'))
                metadata = {
                    'name': entity.get('name', entity_content),
                    'confidence': entity.get('confidence', 1.0),
                    'source': 'legacy_extractor'
                }
                # Add any additional properties from the entity dict
                for key, value in entity.items():
                    if key not in ['name', 'text', 'type', 'label', 'confidence']:
                        metadata[key] = value

                memory_item = MemoryItem(
                    content=entity_content,
                    memory_type=entity_type,
                    metadata=metadata
                )
                converted_entities.append(memory_item)
            elif isinstance(entity, str):
                # Convert string entity to MemoryItem
                memory_item = MemoryItem(
                    content=entity,
                    memory_type='entity',
                    metadata={
                        'name': entity,
                        'confidence': 1.0,
                        'source': 'legacy_extractor'
                    }
                )
                converted_entities.append(memory_item)
            else:
                # Convert other types to string then MemoryItem
                entity_str = str(entity)
                memory_item = MemoryItem(
                    content=entity_str,
                    memory_type='entity',
                    metadata={
                        'name': entity_str,
                        'confidence': 1.0,
                        'source': 'legacy_extractor'
                    }
                )
                converted_entities.append(memory_item)

        return converted_entities
