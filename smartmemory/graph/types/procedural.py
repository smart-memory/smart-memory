import logging

from smartmemory.graph.base import BaseMemoryGraph
from smartmemory.graph.core import Triple
from smartmemory.graph.types.interfaces import MemoryGraphInterface
from smartmemory.models.memory_item import MemoryItem
from smartmemory.stores.converters import ProceduralConverter
from smartmemory.stores.mixins import (
    GraphErrorHandlingMixin, GraphLoggingMixin,
    GraphValidationMixin, GraphPerformanceMixin
)

logger = logging.getLogger(__name__)


class ProceduralMemoryGraph(
    BaseMemoryGraph,
    MemoryGraphInterface,
    GraphErrorHandlingMixin,
    GraphLoggingMixin,
    GraphValidationMixin,
    GraphPerformanceMixin
):
    """
    Procedural memory using a graph database backend (Graphiti/Neo4j).
    Stores procedures as subgraphs: nodes = steps/conditionals, edges = transitions.
    
    Implements hybrid architecture with:
    - Standard interface (MemoryGraphInterface)
    - Optional mixins for common functionality
    - Type-specific converter for procedural processing
    """

    def __init__(self, *args, **kwargs):
        # Initialize mixins first
        super().__init__(*args, **kwargs)

        # Initialize converter for procedural-specific transformations
        self.converter = ProceduralConverter()

    def add(self, procedure_data, **kwargs):
        """Add a procedure using converter and mixins."""
        with self.performance_context("procedural_add"):
            # Convert dict to MemoryItem if needed, then use converter
            if isinstance(procedure_data, dict):
                memory_item = MemoryItem(
                    content=procedure_data.get("description", procedure_data.get("name", "")),
                    metadata=procedure_data,
                    item_id=procedure_data.get("item_id")
                )
            else:
                memory_item = procedure_data

            # Use converter to prepare item for procedural storage
            graph_data = self.converter.to_graph_data(memory_item)

            # Validate using mixin
            if not self.validate_item(memory_item):
                return self.handle_error(f"Validation failed for procedure {graph_data.node_id}")

            try:
                # Add main procedure node
                proc_node = self.graph.add_node(
                    item_id=graph_data.node_id,
                    properties=graph_data.node_properties,
                    memory_type="procedure"
                )

                # Add extracted relations using converter
                for relation in graph_data.relations:
                    self._add_edge_safe(
                        source_id=relation.source_id,
                        target_id=relation.target_id,
                        edge_type=relation.relation_type,
                        properties=relation.properties,
                        memory_type="procedure"
                    )

                self.log_operation("add", f"Added procedural item {graph_data.node_id}")
                return proc_node

            except Exception as e:
                return self.handle_error(f"Failed to add procedural item {graph_data.node_id}: {e}")

    # Deprecate add_procedure in favor of add
    add_procedure = add

    def get(self, key: str):
        """Retrieve a procedure by id."""
        try:
            proc = self.graph.get_node(key)
            if proc is None:
                return None

            steps = self.graph.get_neighbors(key, edge_type="HAS_STEP")
            return {"procedure": proc, "steps": steps}
        except Exception as e:
            logger.error(f"Failed to get procedure {key}: {e}")
            return None

    def search(self, query: str, top_k: int = 5, **kwargs):
        """Search procedures using enhanced semantic similarity."""
        try:
            # Get all procedures
            procs = self.graph.search_nodes({"label": "Procedure"})

            if not procs:
                return []

            # Use enhanced similarity framework for semantic search
            try:
                from smartmemory.similarity import EnhancedSimilarityFramework
                from smartmemory.models.memory_item import MemoryItem

                # Create query item for similarity comparison
                query_item = MemoryItem(content=query, metadata={'type': 'query'})

                # Initialize similarity framework
                similarity_framework = EnhancedSimilarityFramework(graph_store=self.graph)

                # Calculate similarities and rank procedures
                scored_procedures = []
                for proc in procs:
                    # Convert procedure to MemoryItem for similarity calculation
                    proc_content = proc["properties"].get('name', '') + ' ' + proc["properties"].get('description', '') + ' ' + proc["properties"].get('procedure_body', '')
                    proc_item = MemoryItem(
                        content=proc_content,
                        metadata=proc["properties"],
                        item_id=proc["properties"].get('item_id', '')
                    )

                    # Calculate similarity score
                    similarity_score = similarity_framework.calculate_similarity(query_item, proc_item)
                    scored_procedures.append((proc, similarity_score))

                # Sort by similarity score (descending) and return top_k
                scored_procedures.sort(key=lambda x: x[1], reverse=True)
                return [proc for proc, score in scored_procedures[:top_k] if score > 0.1]  # Filter very low scores

            except ImportError:
                # Fallback to enhanced keyword matching if similarity framework unavailable
                logger.warning("Enhanced similarity framework not available, using enhanced keyword matching")
                return self._enhanced_keyword_search(query, procs, top_k)

        except Exception as e:
            logger.error(f"Failed to search procedures: {e}")
            return []

    def _enhanced_keyword_search(self, query: str, procedures: list, top_k: int = 5) -> list:
        """Enhanced keyword search with fuzzy matching as fallback."""
        query_words = set(query.lower().split())
        scored_procedures = []

        for proc in procedures:
            name = proc["properties"].get('name', '').lower()
            desc = proc["properties"].get('description', '').lower()
            body = proc["properties"].get('procedure_body', '').lower()
            content_words = set((name + ' ' + desc + ' ' + body).split())

            # Calculate word overlap score
            if query_words and content_words:
                intersection = len(query_words & content_words)
                union = len(query_words | content_words)
                jaccard_score = intersection / union if union > 0 else 0.0

                # Add partial word matching bonus
                partial_matches = 0
                for q_word in query_words:
                    for c_word in content_words:
                        if q_word in c_word or c_word in q_word:
                            partial_matches += 1
                            break

                partial_score = partial_matches / len(query_words) if query_words else 0.0
                final_score = max(jaccard_score, partial_score * 0.7)

                if final_score > 0.1:  # Only include reasonable matches
                    scored_procedures.append((proc, final_score))

        # Sort by score and return top_k
        scored_procedures.sort(key=lambda x: x[1], reverse=True)
        return [proc for proc, score in scored_procedures[:top_k]]

    # Inherited remove() method from BaseMemoryGraph provides consistent implementation

    def add_relation(self, source_id: str, target_id: str, relation: str, score: float = 1.0, source_name: str = None, target_name: str = None, tags: str = "", item=None):
        try:
            group_id = getattr(item, "group_id", None) or tags or "default"
            properties = {"score": score, "group_id": group_id}
            triple = Triple(subject=source_id, predicate=relation.upper(), object=target_id)
            self.graph.add_triple(triple, properties)
        except Exception as e:
            logger.error(f"Failed to add relation {relation} between {source_id} and {target_id}: {e}")

    def get_procedure_steps(self, proc_id: str):
        """Get steps for a specific procedure."""
        try:
            procedure_data = self.get(proc_id)
            if procedure_data and "steps" in procedure_data:
                return procedure_data["steps"]

            # Fallback: get steps from graph relationships
            steps = self.graph.get_neighbors(proc_id, edge_type="HAS_STEP")
            return [{"step": n["name"], "content": n.get("content")} for n in steps]
        except Exception as e:
            logger.error(f"Failed to get steps for procedure {proc_id}: {e}")
            return []

    def get_related_procedures(self, proc_id: str, depth: int = 1):
        """Get procedures related to the given procedure using graph traversal."""
        try:
            # Use graph traversal to find related procedures
            if hasattr(self.graph, 'traverse'):
                nodes = self.graph.traverse(proc_id, relation="RELATED", depth=depth)
                return nodes
            else:
                # Fallback: get neighbors with RELATED edge type
                neighbors = self.graph.get_neighbors(proc_id, edge_type="RELATED")
                return neighbors
        except Exception as e:
            logger.error(f"Failed to get related procedures for {proc_id}: {e}")
            return []

# Removed redundant clear() method; now inherited from BaseMemoryStore
