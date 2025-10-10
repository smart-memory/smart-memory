"""
Wikipedia Grounder Plugin

Grounds memory items to Wikipedia articles for provenance and validation.
"""

from typing import Optional, List
from smartmemory.plugins.base import GrounderPlugin, PluginMetadata


class WikipediaGrounder(GrounderPlugin):
    """
    Grounds entities to Wikipedia articles for provenance.
    
    This grounder links memory items to Wikipedia articles based on
    recognized entities, providing external validation and context.
    """
    
    @classmethod
    def metadata(cls) -> PluginMetadata:
        """Return plugin metadata for discovery."""
        return PluginMetadata(
            name="wikipedia_grounder",
            version="1.0.0",
            author="SmartMemory Team",
            description="Grounds entities to Wikipedia articles for provenance",
            plugin_type="grounder",
            dependencies=["wikipedia>=1.4.0"],
            min_smartmemory_version="0.1.0",
            tags=["grounding", "wikipedia", "provenance", "validation"]
        )
    
    def ground(self, item, entities: list, graph) -> list:
        """
        Ground memory item to Wikipedia articles.
        
        Args:
            item: The memory item to ground
            entities: List of entities to ground
            graph: The graph backend for creating edges
        
        Returns:
            list: List of provenance candidate node IDs
        """
        if not entities:
            return []
        
        provenance_candidates = []
        
        try:
            # Use WikipediaEnricher to get data
            from smartmemory.plugins.enrichers import WikipediaEnricher
            
            wiki = WikipediaEnricher()
            wiki_data = wiki.enrich(item, {'semantic_entities': entities})
            
            # Create provenance nodes and edges
            for entity, data in wiki_data.get('wikipedia_data', {}).items():
                if data.get('exists'):
                    wiki_id = f"wikipedia:{entity.replace(' ', '_').lower()}"
                    
                    # Add Wikipedia node to graph
                    node_properties = {
                        'entity': entity,
                        'summary': data.get('summary', '')[:300],
                        'categories': data.get('categories', []),
                        'url': data.get('url'),
                        'type': 'wikipedia_article'
                    }
                    
                    graph.add_node(item_id=wiki_id, properties=node_properties)
                    
                    # Add edge from item to Wikipedia node
                    if hasattr(item, 'item_id'):
                        graph.add_edge(
                            item.item_id,
                            wiki_id,
                            edge_type="GROUNDED_IN",
                            properties={}
                        )
                    
                    provenance_candidates.append(wiki_id)
        
        except Exception as e:
            # Log error but don't fail the grounding process
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error grounding to Wikipedia: {e}")
        
        return provenance_candidates
