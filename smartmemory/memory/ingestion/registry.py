"""
Registry module for ingestion pipeline stages.

This module handles registration and management of extractors, enrichers, adapters,
and converters for the ingestion pipeline. It consolidates all registry logic and
provides fallback mechanisms for extractor selection.
"""
import logging
from typing import Dict, List, Optional, Callable

from smartmemory.observability.instrumentation import emit_after
from smartmemory.utils import get_config

logger = logging.getLogger(__name__)


class IngestionRegistry:
    """
    Centralized registry for ingestion pipeline stages.
    
    Manages registration of extractors, enrichers, adapters, and converters
    with automatic fallback mechanisms and performance instrumentation.
    """

    def __init__(self):
        # Store extractor classes for lazy loading
        self.extractor_registry: Dict[str, type] = {}  # Class types
        self.extractor_instances: Dict[str, any] = {}  # Instantiated extractors (cache)
        
        self.enricher_registry: Dict[str, Callable] = {}
        self.adapter_registry: Dict[str, Callable] = {}
        self.converter_registry: Dict[str, Callable] = {}
        self._register_defaults()

    def _register_defaults(self):
        """Register default extractor classes for lazy loading."""
        # Register classes instead of instances - instantiated only when first requested
        
        # Import classes (lightweight, no model loading)
        try:
            from smartmemory.plugins.extractors import SpacyExtractor, LLMExtractor, RebelExtractor, RelikExtractor
            from smartmemory.extraction.extractor import OntologyExtractor
            
            # Register extractor classes
            self.register_extractor_class('spacy', SpacyExtractor)
            self.register_extractor_class('llm', LLMExtractor)
            self.register_extractor_class('gpt4o_triple', LLMExtractor)  # Alias
            self.register_extractor_class('rebel', RebelExtractor)
            self.register_extractor_class('relik', RelikExtractor)
            self.register_extractor_class('ontology', OntologyExtractor)
        except ImportError as e:
            logger.warning(f"Failed to import some extractors: {e}")

    def register_adapter(self, name: str, adapter_fn: Callable):
        """Register a new input adapter by name."""
        self.adapter_registry[name] = adapter_fn

    def register_converter(self, name: str, converter_fn: Callable):
        """Register a new type converter by name."""
        self.converter_registry[name] = converter_fn

    def register_extractor_class(self, name: str, extractor_class: type):
        """Register an extractor class for lazy instantiation."""
        self.extractor_registry[name] = extractor_class
        logger.debug(f"Registered extractor class: {name}")
    
    def register_extractor(self, name: str, extractor_fn: Callable):
        """Register a new entity/relation extractor by name with performance instrumentation."""
        # Wrap extractor to emit performance metrics on each call without changing behavior
        try:
            def _payload_extractor(result):
                try:
                    if isinstance(result, dict):
                        ents = result.get('entities', []) or []
                        rels = result.get('relations', []) or []
                        return {
                            'entities_count': len(ents),
                            'relations_count': len(rels),
                            'extractor_name': name
                        }
                    # Legacy tuple format: (item, entities, relations)
                    elif isinstance(result, tuple) and len(result) >= 3:
                        _, entities, relations = result[:3]
                        return {
                            'entities_count': len(entities) if entities else 0,
                            'relations_count': len(relations) if relations else 0,
                            'extractor_name': name
                        }
                    return {'extractor_name': name}
                except Exception:
                    return {}

            wrapped = emit_after(
                "performance_metrics",
                component="extractor",
                operation=f"extractor:{name}",
                payload_fn=lambda self, args, kwargs, result: _payload_extractor(result),
                measure_time=True,
            )(extractor_fn)
            self.extractor_instances[name] = wrapped
        except Exception:
            # Fallback: register as-is
            self.extractor_instances[name] = extractor_fn

    def register_enricher(self, name: str, enricher_fn: Callable):
        """Register a new enrichment routine by name."""
        self.enricher_registry[name] = enricher_fn

    def get_fallback_order(self, primary: Optional[str] = None) -> List[str]:
        """
        Return a config-driven extractor fallback order, filtered to registered extractors.
        Defaults to ['llm', 'relik', 'gliner', 'spacy'] and removes the primary extractor if provided.
        """
        try:
            cfg = get_config('extractor') or {}
        except Exception:
            cfg = {}

        order = cfg.get('fallback_order')
        if not order:
            order = ['llm', 'relik', 'gliner', 'spacy']

        # Remove duplicates while preserving order
        seen = set()
        deduped = []
        for name in order:
            if name not in seen:
                seen.add(name)
                deduped.append(name)

        # Remove primary if provided
        if primary:
            deduped = [n for n in deduped if n != primary]

        # Keep only registered extractors (factories or instances)
        return [n for n in deduped if n in self.extractor_registry or n in self.extractor_instances]

    def select_default_extractor(self) -> Optional[str]:
        """
        Select the default extractor name using config and availability.
        Prefers extractor['default'] if registered, else the first available from the fallback order,
        else 'ontology' if registered, else any registered extractor.
        """
        import logging
        logger = logging.getLogger(__name__)

        try:
            cfg = get_config('extractor') or {}
        except Exception as e:
            logger.warning(f"Failed to load extractor config: {e}")
            cfg = {}

        default = cfg.get('default', 'llm')

        if default and (default in self.extractor_registry or default in self.extractor_instances):
            return default
        else:
            available = list(self.extractor_registry.keys())
            logger.warning(f"Configured default extractor '{default}' not registered. Available: {available}")

        # Try fallback order
        for name in self.get_fallback_order(primary=None):
            if name in self.extractor_registry or name in self.extractor_instances:
                return name

        # As a last resort, consider ontology or any registered
        if 'ontology' in self.extractor_registry:
            return 'ontology'

        return next(iter(self.extractor_registry.keys()), None)

    def get_extractor(self, name: str) -> Optional[Callable]:
        """Get an extractor by name, instantiating lazily if needed."""
        # Check if already instantiated
        if name in self.extractor_instances:
            return self.extractor_instances[name]
        
        # Check if we have a class registered
        extractor_class = self.extractor_registry.get(name)
        if extractor_class:
            try:
                logger.debug(f"Lazy loading extractor: {name}")
                instance = extractor_class()  # Instantiate the class
                self.extractor_instances[name] = instance
                return instance
            except Exception as e:
                logger.error(f"Failed to instantiate extractor '{name}': {e}")
                return None
        
        return None

    def get_enricher(self, name: str) -> Optional[Callable]:
        """Get an enricher by name."""
        return self.enricher_registry.get(name)

    def get_adapter(self, name: str) -> Optional[Callable]:
        """Get an adapter by name."""
        return self.adapter_registry.get(name)

    def get_converter(self, name: str) -> Optional[Callable]:
        """Get a converter by name."""
        return self.converter_registry.get(name)

    def list_extractors(self) -> List[str]:
        """List all registered extractor names (factories + instances)."""
        return list(set(self.extractor_registry.keys()) | set(self.extractor_instances.keys()))

    def list_enrichers(self) -> List[str]:
        """List all registered enricher names."""
        return list(self.enricher_registry.keys())

    def list_adapters(self) -> List[str]:
        """List all registered adapter names."""
        return list(self.adapter_registry.keys())

    def list_converters(self) -> List[str]:
        """List all registered converter names."""
        return list(self.converter_registry.keys())

    def is_extractor_registered(self, name: str) -> bool:
        """Check if an extractor is registered (factory or instance)."""
        return name in self.extractor_registry or name in self.extractor_instances

    def is_enricher_registered(self, name: str) -> bool:
        """Check if an enricher is registered."""
        return name in self.enricher_registry
