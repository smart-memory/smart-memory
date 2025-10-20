"""
Wikipedia Grounder Plugin

Grounds memory items to Wikipedia articles for provenance and validation.
"""
import logging

from smartmemory.plugins.base import GrounderPlugin, PluginMetadata

logger = logging.getLogger(__name__)


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
            tags=["grounding", "wikipedia", "provenance", "validation"],
            requires_network=True
        )

    def ground(self, item, entities, graph) -> list:
        """
        Ground entities to Wikipedia articles.
        
        Args:
            item: The memory item (for context)
            entities: List of entity MemoryItems to ground
            graph: The graph backend for creating edges
        
        Returns:
            list: List of provenance candidate node IDs (Wikipedia nodes created)
        """
        if not entities:
            return []
        
        provenance_candidates = []

        try:
            # Extract entity names from MemoryItems for WikipediaEnricher
            entity_names = []
            entity_map = {}  # Map entity name -> MemoryItem
            
            for entity in entities:
                if hasattr(entity, 'metadata') and entity.metadata:
                    name = entity.metadata.get('name')
                    if name:
                        entity_names.append(name)
                        entity_map[name] = entity
            
            if not entity_names:
                return []
            
            # Use WikipediaEnricher to get data
            from smartmemory.plugins.enrichers import WikipediaEnricher
            wiki = WikipediaEnricher()
            wiki_data = wiki.enrich(item, {'semantic_entities': entity_names})

            logger.info(f"WikipediaGrounder: wiki_data has {len(wiki_data.get('wikipedia_data', {}))} entries")
            for entity_name, data in wiki_data.get('wikipedia_data', {}).items():
                logger.info(f"WikipediaGrounder: Processing entity '{entity_name}', exists={data.get('exists')}")
                if data.get('exists'):
                    wiki_id = f"wikipedia:{entity_name.replace(' ', '_').lower()}"
                    
                    # Check if Wikipedia node already exists (shared across all users)
                    existing_node = graph.get_node(wiki_id)
                    if not existing_node:
                        # Create new Wikipedia node
                        node_properties = {
                            'entity': entity_name,
                            'summary': data.get('summary', '')[:300],
                            'categories': data.get('categories', []),
                            'url': data.get('url'),
                            'type': 'wikipedia_article',
                            'node_category': 'grounding'
                        }
                        
                        graph.add_node(item_id=wiki_id, properties=node_properties, is_global=True)
                        logger.info(f"✅ Created Wikipedia node: {wiki_id}")
                    else:
                        logger.info(f"♻️  Reusing existing Wikipedia node: {wiki_id}")
                    
                    # Add edge from ENTITY to Wikipedia node (not from memory)
                    # Get entity MemoryItem and extract its node ID
                    entity_item = entity_map.get(entity_name)
                    if entity_item and hasattr(entity_item, 'item_id'):
                        graph.add_edge(
                            entity_item.item_id,
                            wiki_id,
                            edge_type="GROUNDED_IN",
                            properties={}
                        )
                        logger.info(f"✅ Created GROUNDED_IN edge: {entity_item.item_id} -> {wiki_id}")
                    else:
                        logger.warning(f"⚠️  No entity MemoryItem found for '{entity_name}', skipping edge creation")
                    
                    provenance_candidates.append(wiki_id)

        except Exception as e:
            # Log error but don't fail the grounding process
            logger.error(f"Error grounding to Wikipedia: {e}", exc_info=True)
        return provenance_candidates
