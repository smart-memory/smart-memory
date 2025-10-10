"""
Registry module for ingestion pipeline stages.

This module handles registration and management of extractors, enrichers, adapters,
and converters for the ingestion pipeline. It consolidates all registry logic and
provides fallback mechanisms for extractor selection.
"""
from typing import Dict, List, Optional, Callable

from smartmemory.observability.instrumentation import emit_after
from smartmemory.utils import get_config


class IngestionRegistry:
    """
    Centralized registry for ingestion pipeline stages.
    
    Manages registration of extractors, enrichers, adapters, and converters
    with automatic fallback mechanisms and performance instrumentation.
    """

    def __init__(self):
        self.extractor_registry: Dict[str, Callable] = {}
        self.enricher_registry: Dict[str, Callable] = {}
        self.adapter_registry: Dict[str, Callable] = {}
        self.converter_registry: Dict[str, Callable] = {}
        self._register_defaults()

    def _register_defaults(self):
        """Register default extractors with graceful fallbacks for missing dependencies."""
        # Register default extractors (e.g., 'spacy') which now loads config internally.
        # Import each extractor in isolation so one missing module doesn't block others.

        # spaCy
        try:
            from smartmemory.plugins import make_spacy_extractor
            self.register_extractor('spacy', make_spacy_extractor())
        except Exception:
            pass

        # GLiNER
        try:
            from smartmemory.plugins import make_gliner_extractor
            self.register_extractor('gliner', make_gliner_extractor())
        except Exception:
            pass

        # ReLiK
        try:
            from smartmemory.plugins.extractors.relik import make_relik_extractor
            self.register_extractor('relik', make_relik_extractor())
        except Exception:
            pass

        # Lightweight LLM extractor wrapper (if present)
        try:
            from smartmemory.plugins import make_llm_extractor
            self.register_extractor('llm', make_llm_extractor())
        except Exception:
            pass

        # Backward-compat: alias 'gpt4o_triple' to the consolidated 'llm' extractor
        try:
            from smartmemory.plugins import make_llm_extractor
            self.register_extractor('gpt4o_triple', make_llm_extractor())
        except Exception:
            pass

        # Ontology-aware extractor (LLM). Only register when API key configured and import succeeds.
        try:
            from smartmemory.extraction.extractor import create_ontology_aware_extractor
            try:
                llm_cfg = get_config('extractor').get('llm') or {}
                if llm_cfg.get('openai_api_key'):
                    self.register_extractor('ontology', create_ontology_aware_extractor())
            except Exception:
                # Be resilient to config import/validation errors; extractor is optional
                pass
        except Exception:
            # Ontology extractor module unavailable; ignore
            pass

    def register_adapter(self, name: str, adapter_fn: Callable):
        """Register a new input adapter by name."""
        self.adapter_registry[name] = adapter_fn

    def register_converter(self, name: str, converter_fn: Callable):
        """Register a new type converter by name."""
        self.converter_registry[name] = converter_fn

    def register_extractor(self, name: str, extractor_fn: Callable):
        """Register a new entity/relation extractor by name with performance instrumentation."""
        # Wrap extractor to emit performance metrics on each call without changing behavior
        try:
            def _payload_extractor(result):
                try:
                    if isinstance(result, dict):
                        ents = result.get('entities', []) or []
                        trips = result.get('triples', []) or []
                        return {
                            'entities_count': len(ents),
                            'triples_count': len(trips),
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
            self.extractor_registry[name] = wrapped
        except Exception:
            # Fallback: register as-is
            self.extractor_registry[name] = extractor_fn

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

        # Keep only registered extractors
        return [n for n in deduped if n in self.extractor_registry]

    def select_default_extractor(self) -> Optional[str]:
        """
        Select the default extractor name using config and availability.
        Prefers extractor['default'] if registered, else the first available from the fallback order,
        else 'ontology' if registered, else any registered extractor.
        """
        try:
            cfg = get_config('extractor') or {}
        except Exception:
            cfg = {}

        default = cfg.get('default', 'llm')
        if default and default in self.extractor_registry:
            return default

        # Try fallback order
        for name in self.get_fallback_order(primary=None):
            if name in self.extractor_registry:
                return name

        # As a last resort, consider ontology or any registered
        if 'ontology' in self.extractor_registry:
            return 'ontology'

        return next(iter(self.extractor_registry.keys()), None)

    def get_extractor(self, name: str) -> Optional[Callable]:
        """Get an extractor by name."""
        return self.extractor_registry.get(name)

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
        """List all registered extractor names."""
        return list(self.extractor_registry.keys())

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
        """Check if an extractor is registered."""
        return name in self.extractor_registry

    def is_enricher_registered(self, name: str) -> bool:
        """Check if an enricher is registered."""
        return name in self.enricher_registry
