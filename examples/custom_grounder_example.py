"""
Example: Creating a Custom Grounder Plugin

This example demonstrates how to create a custom grounder plugin that
links memories to external knowledge sources for provenance and validation.
"""

from typing import Optional, List, Dict, Any
from smartmemory.plugins.base import GrounderPlugin, PluginMetadata


class DBpediaGrounder(GrounderPlugin):
    """
    Custom grounder that links entities to DBpedia knowledge base.
    
    This grounder queries DBpedia SPARQL endpoint to find matching
    entities and creates provenance links.
    """
    
    @classmethod
    def metadata(cls) -> PluginMetadata:
        """Return plugin metadata for discovery and registration."""
        return PluginMetadata(
            name="dbpedia_grounder",
            version="1.0.0",
            author="Your Name",
            description="Grounds entities to DBpedia knowledge base",
            plugin_type="grounder",
            dependencies=["SPARQLWrapper>=2.0.0"],
            min_smartmemory_version="0.1.0",
            tags=["dbpedia", "sparql", "knowledge-base"]
        )
    
    def __init__(self):
        """Initialize the DBpedia grounder."""
        self.endpoint = "http://dbpedia.org/sparql"
        self._sparql = None
    
    def _load_sparql(self):
        """Lazy load SPARQL wrapper."""
        if self._sparql is None:
            try:
                from SPARQLWrapper import SPARQLWrapper, JSON
                self._sparql = SPARQLWrapper(self.endpoint)
                self._sparql.setReturnFormat(JSON)
            except ImportError:
                self._sparql = None
    
    def _query_dbpedia(self, entity: str) -> Optional[str]:
        """
        Query DBpedia for an entity.
        
        Args:
            entity: Entity name to search for
        
        Returns:
            DBpedia URI if found, None otherwise
        """
        if not self._sparql:
            return None
        
        # Simple SPARQL query to find entity
        query = f"""
        SELECT DISTINCT ?resource WHERE {{
            ?resource rdfs:label "{entity}"@en .
        }} LIMIT 1
        """
        
        try:
            self._sparql.setQuery(query)
            results = self._sparql.query().convert()
            
            if results["results"]["bindings"]:
                return results["results"]["bindings"][0]["resource"]["value"]
        except Exception:
            pass
        
        return None
    
    def ground(self, item, entities: List[str], graph) -> List[str]:
        """
        Ground entities to DBpedia.
        
        Args:
            item: The memory item being grounded
            entities: List of entity names to ground
            graph: The graph backend to add provenance nodes
        
        Returns:
            List of provenance node IDs created
        """
        self._load_sparql()
        provenance_candidates = []
        
        for entity in entities:
            # Query DBpedia
            dbpedia_uri = self._query_dbpedia(entity)
            
            if dbpedia_uri:
                # Create provenance node
                node_id = f"dbpedia:{entity.replace(' ', '_')}"
                
                try:
                    graph.add_node(
                        item_id=node_id,
                        properties={
                            'entity': entity,
                            'uri': dbpedia_uri,
                            'type': 'dbpedia_resource',
                            'source': 'dbpedia'
                        }
                    )
                    
                    # Create edge from memory item to provenance
                    graph.add_edge(
                        item.item_id,
                        node_id,
                        edge_type="GROUNDED_IN",
                        properties={'confidence': 0.9}
                    )
                    
                    provenance_candidates.append(node_id)
                except Exception:
                    pass
        
        return provenance_candidates


class CustomAPIGrounder(GrounderPlugin):
    """
    Custom grounder that links to a custom API or knowledge base.
    
    This is a template for creating grounders that connect to
    your own APIs or knowledge sources.
    """
    
    @classmethod
    def metadata(cls) -> PluginMetadata:
        return PluginMetadata(
            name="custom_api_grounder",
            version="1.0.0",
            author="Your Name",
            description="Grounds entities to custom API",
            plugin_type="grounder",
            tags=["api", "custom"]
        )
    
    def __init__(self, api_url: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize with API configuration.
        
        Args:
            api_url: Base URL for the API
            api_key: API key for authentication
        """
        self.api_url = api_url or "https://api.example.com"
        self.api_key = api_key
    
    def _query_api(self, entity: str) -> Optional[Dict[str, Any]]:
        """
        Query your custom API for entity information.
        
        Args:
            entity: Entity to look up
        
        Returns:
            API response data or None
        """
        # Example implementation (replace with your actual API call)
        try:
            import requests
            
            headers = {}
            if self.api_key:
                headers['Authorization'] = f'Bearer {self.api_key}'
            
            response = requests.get(
                f"{self.api_url}/entities/{entity}",
                headers=headers,
                timeout=5
            )
            
            if response.status_code == 200:
                return response.json()
        except Exception:
            pass
        
        return None
    
    def ground(self, item, entities: List[str], graph) -> List[str]:
        """
        Ground entities to custom API.
        
        Args:
            item: The memory item being grounded
            entities: List of entity names to ground
            graph: The graph backend to add provenance nodes
        
        Returns:
            List of provenance node IDs created
        """
        provenance_candidates = []
        
        for entity in entities:
            # Query your API
            api_data = self._query_api(entity)
            
            if api_data:
                # Create provenance node
                node_id = f"custom:{entity.replace(' ', '_')}"
                
                try:
                    graph.add_node(
                        item_id=node_id,
                        properties={
                            'entity': entity,
                            'data': api_data,
                            'type': 'custom_resource',
                            'source': 'custom_api'
                        }
                    )
                    
                    # Create edge
                    graph.add_edge(
                        item.item_id,
                        node_id,
                        edge_type="GROUNDED_IN",
                        properties={'source': 'custom_api'}
                    )
                    
                    provenance_candidates.append(node_id)
                except Exception:
                    pass
        
        return provenance_candidates


# Example usage
if __name__ == "__main__":
    from smartmemory.models.memory_item import MemoryItem
    
    # Create a memory item
    item = MemoryItem(
        content="Albert Einstein developed the theory of relativity",
        memory_type="semantic",
        item_id="test_item_123"
    )
    
    # Example entities to ground
    entities = ["Albert Einstein", "theory of relativity"]
    
    # Create grounder
    grounder = DBpediaGrounder()
    
    print("DBpedia Grounder Example:")
    print(f"  Name: {grounder.metadata().name}")
    print(f"  Description: {grounder.metadata().description}")
    print(f"  Entities to ground: {entities}")
    print()
    
    # Note: In real usage, you would pass an actual graph backend
    # from smartmemory.graph.smart_graph import SmartGraph
    # graph = SmartGraph()
    # provenance = grounder.ground(item, entities, graph)
    
    print("Custom API Grounder Example:")
    api_grounder = CustomAPIGrounder(
        api_url="https://api.example.com",
        api_key="your_api_key_here"
    )
    print(f"  Name: {api_grounder.metadata().name}")
    print(f"  API URL: {api_grounder.api_url}")
    print()
    
    print("Note: Run these grounders with a SmartGraph instance to create provenance links!")
