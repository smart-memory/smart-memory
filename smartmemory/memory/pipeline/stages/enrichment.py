from smartmemory.observability.instrumentation import emit_after
from smartmemory.plugins.manager import get_plugin_manager


def _make_payload_fn(plugin_name: str):
    def _pf(self_or_none, args, kwargs, result):
        try:
            props_count = 0
            rels_count = 0
            if isinstance(result, dict):
                props = result.get('properties') or {}
                if isinstance(props, dict):
                    props_count = len(props)
                rels = result.get('relations') or []
                if isinstance(rels, list):
                    rels_count = len(rels)
            return {
                "plugin": plugin_name,
                "properties_added": props_count,
                "relationships_added": rels_count,
            }
        except Exception:
            return {"plugin": plugin_name}

    return _pf


class Enrichment:
    """
    Handles memory enrichment logic using the plugin system.
    
    All enrichers are loaded from the PluginRegistry, which discovers them automatically
    from built-in plugins and external plugins via entry points.
    """

    def __init__(self, graph):
        self.graph = graph
        
        # Get plugin manager and registry
        plugin_manager = get_plugin_manager()
        self.plugin_registry = plugin_manager.registry
        
        # Build enricher registry from plugin system
        self.enricher_registry = {}
        for enricher_name in self.plugin_registry.list_plugins('enricher'):
            enricher_class = self.plugin_registry.get_enricher(enricher_name)
            if enricher_class:
                # Create callable that instantiates and calls enrich
                def make_enricher_callable(cls):
                    def enricher_fn(item, node_ids=None):
                        instance = cls()
                        return instance.enrich(item, node_ids)
                    return enricher_fn
                self.enricher_registry[enricher_name] = make_enricher_callable(enricher_class)
        
        # Default pipeline: all enrichers, in registry order
        self._enricher_pipeline = list(self.enricher_registry.keys())
        
        # Wrap registry callables to emit performance metrics automatically
        for _name, _callable in list(self.enricher_registry.items()):
            try:
                wrapped = emit_after(
                    "performance_metrics",
                    component="enricher",
                    operation_fn=lambda s, a, k, r, n=_name: f"enricher:{n}",
                    payload_fn=_make_payload_fn(_name),
                    measure_time=True,
                )(_callable)
                self.enricher_registry[_name] = wrapped
            except Exception:
                # If wrapping fails, keep original callable
                self.enricher_registry[_name] = _callable

    def register_enricher(self, name, enricher_fn):
        """Register a new enricher by name."""
        self.enricher_registry[name] = enricher_fn

    def enrich(self, context, enricher_names=None):
        """
        Call all enrichers in the pipeline (in order). If enricher_names is None, use all enrichers from the registry.
        Merges results from all enrichers.
        """
        pipeline = enricher_names or self._enricher_pipeline
        result = {}
        for enricher_name in pipeline:
            enricher = self.enricher_registry.get(enricher_name)
            if enricher is None:
                raise ValueError(f"Enricher '{enricher_name}' not registered.")
            enricher_result = enricher(context["item"], context["node_ids"])
            if enricher_result:
                # Merge 'properties' deeply so multiple enrichers can contribute
                if 'properties' in enricher_result:
                    props = enricher_result.get('properties') or {}
                    if props:
                        if 'properties' not in result or not isinstance(result.get('properties'), dict):
                            result['properties'] = {}
                        result['properties'].update(props)
                # Merge 'relations' as a concatenated list
                if 'relations' in enricher_result:
                    rels = enricher_result.get('relations') or []
                    if rels:
                        if 'relations' not in result or not isinstance(result.get('relations'), list):
                            result['relations'] = []
                        result['relations'].extend(rels)
                # Merge any other top-level keys via last-wins (legacy compatibility)
                for k, v in enricher_result.items():
                    if k in ('properties', 'relations'):
                        continue
                    result[k] = v
        return result
