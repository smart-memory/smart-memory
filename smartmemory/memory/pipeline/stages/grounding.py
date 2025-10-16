from smartmemory.observability.instrumentation import emit_after
from smartmemory.plugins.manager import get_plugin_manager


def _ground_payload(self_or_none, args, kwargs, result):
    try:
        ctx = args[0] if args else kwargs.get('context')
        item = ctx.get('item') if isinstance(ctx, dict) else None
        item_id = getattr(item, 'item_id', None)
        prov = ctx.get('provenance_candidates') if isinstance(ctx, dict) else None
        prov_count = len(prov) if isinstance(prov, list) else (1 if prov else 0)
        return {
            'item_id': item_id,
            'provenance_count': prov_count,
        }
    except Exception:
        return {}


class Grounding:
    """
    Handles grounding/provenance logic using the plugin system.
    
    Grounders are loaded from the PluginRegistry, which discovers them automatically
    from built-in plugins and external plugins via entry points.
    """

    def __init__(self, graph):
        """Initialize grounding component with graph backend."""
        self.graph = graph
        
        # Get plugin manager and registry
        plugin_manager = get_plugin_manager()
        self.plugin_registry = plugin_manager.registry
        
        # Get default grounder (wikipedia_grounder)
        self.default_grounder_class = self.plugin_registry.get_grounder('wikipedia_grounder')
        if self.default_grounder_class:
            self.default_grounder = self.default_grounder_class()
        else:
            self.default_grounder = None

    @emit_after(
        "background_process",
        component="grounding",
        operation="ground",
        payload_fn=_ground_payload,
        measure_time=True,
    )
    def ground(self, context):
        """
        Ground a memory item using the plugin system.
        
        Uses registered grounder plugins to link memory items to external
        knowledge sources for provenance and validation.
        """
        item = context.get('item')
        if not item or not hasattr(item, 'item_id'):
            return
        
        item_id = item.item_id
        source_url = context.get('source_url')
        validation = context.get('validation')
        
        # Get entities from node
        node = self.graph.get_node(item_id)
        if not node:
            return
        
        # Extract entities from MemoryItem or dict
        if hasattr(node, 'metadata'):
            # MemoryItem object
            entities = node.metadata.get('semantic_entities', [])
        elif isinstance(node, dict):
            # Dict
            entities = node.get('semantic_entities') or (node.get('metadata') or {}).get('semantic_entities', [])
        else:
            entities = []
        if not entities:
            return
        
        # Use default grounder (Wikipedia) if available
        if self.default_grounder:
            try:
                provenance_candidates = self.default_grounder.ground(item, entities, self.graph)
                
                # Update node with provenance information
                if provenance_candidates:
                    # Get Wikipedia data for provenance URL
                    from smartmemory.plugins.enrichers import WikipediaEnricher
                    wiki = WikipediaEnricher()
                    wiki_data = wiki.enrich(node, {'semantic_entities': entities})
                    
                    provenance_url = None
                    if wiki_data.get('wikipedia_data'):
                        urls = [v.get('url') for v in wiki_data['wikipedia_data'].values() if v.get('url')]
                        provenance_url = urls[0] if urls else None
                    
                    node['provenance'] = {
                        'type': 'wikipedia',
                        'entities': entities,
                        'wikipedia_data': wiki_data.get('wikipedia_data') or {},
                        'source_url': provenance_url or source_url,
                        'validation': validation,
                    }
                    self.graph.add_node(item_id=item_id, properties=node)
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Error grounding item {item_id}: {e}")
        
        # Fallback: use provided provenance_candidates if grounder not available
        elif context.get('provenance_candidates'):
            provenance_candidates = context.get('provenance_candidates')
            for prov_id in provenance_candidates:
                self.graph.add_edge(item_id, prov_id, edge_type="GROUNDED_IN", properties={})
