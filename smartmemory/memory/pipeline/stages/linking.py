class Linking:
    """
    Internal helper for SmartMemory. Handles memory linking logic (semantic memory/graph-based linking).
    This class is NOT meant to be used directly—always access linking via SmartMemory's API.
    """

    def __init__(self, graph):
        """Initialize linking component with graph backend."""
        self.graph = graph

    def link_new_item(self, context):
        """
        Handle semantic relationships from ontology extraction.
        No longer creates generic 'RELATED' relationships since we have semantic ones.
        """
        node_ids = context.get('entity_ids') or {}
        item = context.get('item')

        # Skip automatic RELATED linking - we now have semantic relationships from ontology extraction
        # Handle semantic_entities and semantic_relations if present
        semantic_entities = node_ids.get('semantic_entities', [])
        semantic_relations = node_ids.get('semantic_relations', [])

        # Add entity nodes (but don't create generic HAS_ENTITY links since we have semantic relationships)
        for entity in semantic_entities:
            if isinstance(entity, dict):
                eid = entity.get('id') or entity.get('name')
                if eid:
                    self.graph.add_node(item_id=eid, properties=entity)

        # Add semantic relation edges between entities
        for rel in semantic_relations:
            if isinstance(rel, dict):
                src = rel.get('source')
                tgt = rel.get('target')
                typ = rel.get('type', 'RELATED')
                if src and tgt:
                    self.graph.add_edge(src, tgt, edge_type=typ, properties={})

    def link(self, source_id: str, target_id: str, link_type: str = "RELATED", memory_type: str = "semantic") -> str:
        # Only semantic memory supports linking
        if memory_type == "semantic":
            self.graph.add_edge(source_id, target_id, edge_type=link_type, properties={})
            return f"Linked {source_id} to {target_id} as {link_type}"
        else:
            raise NotImplementedError(f"Linking not supported for memory_type: {memory_type}")

    def get_links(self, item_id: str, memory_type: str = "semantic") -> list:
        """
        Return a list of (source, predicate, target) triples for all edges involving item_id.
        """
        if memory_type != "semantic":
            return []
        # Assumes self.graph has .edges attribute (for InMemoryGraphBackend) or similar API
        triples = []
        # Try to access edges directly; fallback to backend if needed
        edges = getattr(getattr(self.graph, 'backend', self.graph), 'edges', None)
        if edges is None:
            # Not in-memory backend; try to use a method if available
            if hasattr(self.graph, 'get_edges_for_node'):
                edges = self.graph.get_edges_for_node(item_id)
            else:
                raise NotImplementedError("Triple extraction not supported for this graph backend.")
        for e in edges:
            if e["source"] == item_id:
                triples.append((e["source"], e["type"], e["target"]))
            elif e["target"] == item_id:
                triples.append((e["target"], e["type"], e["source"]))
        return triples
