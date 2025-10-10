from typing import List


class MemoryReasoner:
    """
    Multi-store reasoning and retrieval utilities for agentic memory systems.
    Aggregates and traverses across semantic, episodic, procedural, working, and optional note/graph stores.
    """

    def __init__(self, semantic, episodic, procedural, working, zettel=None):
        self.semantic = semantic
        self.episodic = episodic
        self.procedural = procedural
        self.working = working
        self.zettel = zettel

    def get_entity(self, entity_id: str):
        # Example: retrieve entity from any store that supports it
        if hasattr(self.semantic, 'get_entity'):
            entity = self.semantic.get_entity(entity_id)
            if entity:
                return entity
        # Could extend to other stores if needed
        return None

    def get_facts(self, entity_id: str) -> List:
        # Example: aggregate facts about entity from all stores
        facts = []
        if hasattr(self.semantic, 'get_facts'):
            facts.extend(self.semantic.get_facts(entity_id))
        # Extend to episodic, procedural, etc. if needed
        return facts

    def shortest_path(self, start_id: str, end_id: str):
        # Example: traverse semantic graph for shortest path
        if hasattr(self.semantic, 'shortest_path'):
            return self.semantic.shortest_path(start_id, end_id)
        return None

    def extract_subgraph(self, tag: str):
        if hasattr(self.semantic, 'extract_subgraph'):
            return self.semantic.extract_subgraph(tag)
        return None

    def relevance_score(self, key: str, target_keys: list):
        if hasattr(self.semantic, 'relevance_score'):
            return self.semantic.relevance_score(key, target_keys)
        return None

    # Add more cross-store or graph/entity reasoning methods as needed.
