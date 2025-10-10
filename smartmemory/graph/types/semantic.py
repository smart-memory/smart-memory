import logging
from typing import Optional, Any, List

from smartmemory.graph.base import BaseMemoryGraph
from smartmemory.graph.types.interfaces import MemoryGraphInterface
from smartmemory.models.memory_item import MemoryItem
from smartmemory.stores.converters import SemanticConverter
from smartmemory.stores.mixins import (
    GraphErrorHandlingMixin, GraphLoggingMixin,
    GraphValidationMixin, GraphPerformanceMixin
)

logger = logging.getLogger(__name__)


class SemanticMemoryGraph(
    BaseMemoryGraph,
    MemoryGraphInterface,
    GraphErrorHandlingMixin,
    GraphLoggingMixin,
    GraphValidationMixin,
    GraphPerformanceMixin
):
    """
    Semantic memory using a graph database (Neo4j/Graphiti) backend.
    
    Implements hybrid architecture with:
    - Standard interface (MemoryGraphInterface)
    - Optional mixins for common functionality
    - Type-specific converter for semantic processing
    """

    def __init__(self, **kwargs):
        # Initialize mixins first
        super().__init__(**kwargs)

        # Initialize converter for semantic-specific transformations
        self.converter = SemanticConverter()

    def add(self, item: MemoryItem, key: Optional[str] = None):
        """Add a semantic memory item using converter and mixins."""
        with self.performance_context("semantic_add"):
            # Use converter to prepare item for semantic storage
            graph_data = self.converter.to_graph_data(item)

            # Validate using mixin
            if not self.validate_item(item):
                return self.handle_error(f"Validation failed for item {key or item.item_id}")

            try:
                node = self.graph.add_node(
                    item_id=graph_data.node_id,
                    properties=graph_data.node_properties,
                    memory_type="semantic"
                )

                # Add extracted relations using converter
                for relation in graph_data.relations:
                    self._add_edge_safe(
                        source_id=relation.source_id,
                        target_id=relation.target_id,
                        edge_type=relation.relation_type,
                        properties=relation.properties,
                        memory_type="semantic"
                    )

                item.metadata["_node"] = node
                self.log_operation("add", f"Added semantic item {key or item.item_id}")
                return item

            except Exception as e:
                return self.handle_error(f"Failed to add semantic item {key or item.item_id}: {e}")

    def add_relation(self, source_id: str, target_id: str, relation: str, score: float = 1.0):
        try:
            self._add_edge_safe(source_id, target_id, edge_type=relation.upper(),
                                properties={"score": score}, memory_type="semantic")
            logger.info(f"Added relation {relation} between {source_id} and {target_id}.")
        except Exception as e:
            logger.error(f"Failed to add relation {relation} between {source_id} and {target_id}: {e}")

    def get_neighbors(self, item_id: str) -> list:
        """Return IDs of neighboring nodes (outbound and inbound)."""
        try:
            return self.graph.get_neighbors(item_id)
        except Exception as e:
            logger.error(f"Failed to get neighbors for {item_id}: {e}")
            return []

    def get_entity(self, entity_id: str) -> Any:
        """Get a node by ID (alias for get)."""
        return self.get(entity_id)

    def list_entities(self, limit: int = 20) -> list:
        """List up to 'limit' entity nodes."""
        try:
            # This method needs to be refactored to use the actual graphiti_core API
            # For now, it's left as is
            nodes = self.graph.get_nodes(label="Episode", limit=limit)
            return [{"key": n["name"], "content": n.get("content")}
                    for n in nodes]
        except Exception as e:
            logger.error(f"Failed to list entities: {e}")
            return []

    def get_facts(self, entity_id: str) -> list:
        """Return facts (edges/relations) about a node."""
        try:
            # This method needs to be refactored to use the actual graphiti_core API
            # For now, it's left as is
            edges = self.graph.get_edges(source_id=entity_id)
            return [{
                "relation": e.get("label"),
                "target": e.get("target_id"),
                "score": e.get("score")
            } for e in edges]
        except Exception as e:
            logger.error(f"Failed to get facts for {entity_id}: {e}")
            return []

    def get_node(self, key: str) -> Any:
        """Get a node directly from the graph backend."""
        try:
            return self.graph.get_node(key)
        except Exception as e:
            logger.error(f"Failed to get graph node {key}: {e}")
            return None

    # Inherited get() method from BaseMemoryGraph provides consistent implementation

    def search(self, query: str, top_k: int = 5):
        return self.graph.search(query, top_k=top_k)

    def hybrid_search(
            self,
            query: str,
            top_k: int = 5,
            graph_expand: int = 1,
            edge_types: Optional[List[str]] = None,
            direction: str = "both",
            filter_type: Optional[str] = None,
            filter_property: Optional[dict] = None,
            context_limit: int = 20,
            path_pattern: Optional[List[str]] = None,
            return_metadata: bool = False,
    ) -> List[MemoryItem]:
        """
        Advanced hybrid retrieval: vector search + configurable graph expansion.
        - edge_types: restrict expansion to these edge types (None=all)
        - direction: 'out', 'in', or 'both'
        - filter_type: filter returned nodes by type
        - filter_property: filter by node property dict
        - context_limit: max number of nodes to return
        - path_pattern: list of edge types for path-based expansion
        - return_metadata: if True, include node/edge metadata
        """
        try:
            # Step 1: Vector search
            hits = self.vector_store.search(query, top_k=top_k)
            hit_ids = [h["id"] for h in hits]
            results = set(hit_ids)
            # Step 2: Graph expansion
            for item_id in hit_ids:
                if path_pattern:
                    # Path-based expansion (e.g., ["CAUSES", "RESULTS_IN"])
                    nodes = self.graph.traverse_path(item_id, path_pattern)
                else:
                    nodes = self.graph.traverse(
                        item_id,
                        relation=edge_types,
                        depth=graph_expand,
                        direction=direction,
                    )
                for n in nodes:
                    if filter_type and n.get("type") != filter_type:
                        continue
                    if filter_property and not all(n.get(k) == v for k, v in filter_property.items()):
                        continue
                    results.add(n["name"])
            # Step 3: Gather and deduplicate MemoryItems
            from smartmemory.models.memory_item import MemoryItem
            items = []
            for item_id in results:
                episodes = self.graph.retrieve_episodes([item_id])
                if episodes:
                    ep = episodes[0]
                    if return_metadata:
                        items.append(MemoryItem(item_id=item_id, content=ep.get("content"), metadata=ep))
                    else:
                        items.append(MemoryItem(item_id=item_id, content=ep.get("content")))
            # Step 4: Context window management
            items = items[:context_limit]
            return items
        except Exception as e:
            logger.error(f"Failed advanced hybrid search for query '{query}': {e}")
            return []

    # Inherited remove() method from BaseMemoryGraph provides consistent implementation

    def get_related_facts(self, key: str, depth: int = 1):
        # Traverse related facts up to a certain depth
        try:
            # This method needs to be refactored to use the actual graphiti_core API
            # For now, it's left as is
            nodes = self.graph.traverse(key, relation="RELATED", depth=depth)
            return [{"key": n["name"], "content": n.get("content")}
                    for n in nodes]
        except Exception as e:
            logger.error(f"Failed to traverse from semantic fact {key}: {e}")
            return []

    def shortest_path(self, start_key: str, end_key: str):
        # Find shortest path between two facts
        try:
            # This method needs to be refactored to use the actual graphiti_core API
            # For now, it's left as is
            path = self.graph.shortest_path(start_key, end_key)
            return [n["name"] for n in path]
        except Exception as e:
            logger.error(f"Failed to find shortest path from {start_key} to {end_key}: {e}")
            return []

    def extract_subgraph(self, tag: str):
        # Extract all facts with a given tag and their links
        try:
            # This method needs to be refactored to use the actual graphiti_core API
            # For now, it's left as is
            subgraph = self.graph.extract_subgraph(tag)
            return [{"from": e["from_key"], "to": e["to_key"]} for e in subgraph]
        except Exception as e:
            logger.error(f"Failed to extract subgraph for tag {tag}: {e}")
            return []

    def embeddings_search(self, embedding, top_k: int = 5):
        """
        Search for top_k facts most similar to the given embedding. Requires 'embedding' property on nodes.
        Falls back to Python/numpy if DB does not support vector search.
        """
        import numpy as np
        # Try to fetch all embeddings and compute cosine similarity
        try:
            # This method needs to be refactored to use the actual graphiti_core API
            # For now, it's left as is
            nodes = self.graph.get_nodes(label="Episode")
            scored = []
            for n in nodes:
                node_emb = np.array(n.get("embedding"))
                query_emb = np.array(embedding)
                sim = float(np.dot(node_emb, query_emb) / (np.linalg.norm(node_emb) * np.linalg.norm(query_emb) + 1e-8))
                scored.append({"key": n["name"], "content": n.get("content"), "score": sim})
            scored.sort(key=lambda x: x["score"], reverse=True)
            return scored[:top_k]
        except Exception as e:
            logger.error(f"Failed to run embeddings search: {e}")
            return []

    def relevance_score(self, key: str, target_keys: list):
        # Example: score based on path length or number of connections
        scores = {}
        for target in target_keys:
            path = self.shortest_path(key, target)
            if path:
                scores[target] = 1 / len(path)
            else:
                scores[target] = 0
        return scores

    def store_entities_relations(self, extraction: dict, item: Any) -> None:
        """
        Store extracted entities and relations into the graph DB.
        Args:
            extraction (dict): Extracted entities and relations.
            item (Any): The memory item to associate entities/relations with.
        """
        entities = extraction.get("entities", [])
        relations = extraction.get("relations", [])
        for ent in entities:
            ent_name = ent["name"] if isinstance(ent, dict) else ent
            ent_valid_start = ent.get("valid_start_time") if isinstance(ent, dict) else None
            node_data = {
                "name": ent_name,
                "valid_start_time": ent_valid_start or item.valid_start_time,
                "transaction_time": item.transaction_time.isoformat()
            }
            # This method needs to be refactored to use the actual graphiti_core API
            # For now, it's left as is
            self.graph.add_node(item_id=ent_name, properties=node_data, memory_type="entity")
        for rel in relations:
            src = rel["source"] if isinstance(rel, dict) else rel
            tgt = rel["target"] if isinstance(rel, dict) else rel
            rel_valid_start = rel.get("valid_start_time") if isinstance(rel, dict) else None
            # This method needs to be refactored to use the actual graphiti_core API
            # For now, it's left as is
            # Use actual relation type from extraction, don't default to generic "RELATED"
            rel_type = rel.get("type", "RELATED") if isinstance(rel, dict) else "RELATED"
            # Skip creating RELATED relationships - we want semantic ones only
            if rel_type != "RELATED":
                self._add_edge_safe(src, tgt, edge_type=rel_type,
                                    properties={
                                        "valid_start_time": (rel_valid_start or item.valid_start_time).isoformat(),
                                        "transaction_time": item.transaction_time.isoformat()
                                    },
                                    memory_type="relation")

    def delete_links(self, note_id: str) -> bool:
        """Remove all relationships (edges) to and from the node with the given note_id."""
        try:
            # Graphiti API: remove all edges for a node
            self.graph.remove_edges(note_id)
            logger.info(f"Removed all links for {note_id} from graph DB.")
            return True
        except Exception as e:
            logger.error(f"Failed to remove links for {note_id}: {e}")
            return False

    def add_concept(self, name: str, description: str = None, type_: str = None):
        """
        Add a concept as an EntityNode with label 'Concept'.
        """
        try:
            concept_node = self.graph.add_node(
                item_id=name,
                properties={"name": name, "description": description, "type": type_, "label": "Concept"},
                memory_type="concept"
            )
            return concept_node
        except Exception as e:
            logger.error(f"Failed to add concept '{name}': {e}")
            return None

    def get_concepts(self):
        """Retrieve all concepts."""
        try:
            return self.graph.search_nodes({"label": "Concept"})
        except Exception as e:
            logger.error(f"Failed to retrieve concepts: {e}")
            return []

    # Removed redundant clear() method; now inherited from BaseMemoryStore

    # SemanticMemoryGraph always manages its graphiti connection; no dynamic store removal needed.
    # All usages of self.graphiti now use MemGraphiti.  # Added comment
