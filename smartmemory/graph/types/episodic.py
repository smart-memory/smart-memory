import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from smartmemory.graph.base import BaseMemoryGraph
from smartmemory.graph.types.interfaces import MemoryGraphInterface
from smartmemory.models.memory_item import MemoryItem
from smartmemory.stores.converters.episodic_converter import EpisodicConverter
from smartmemory.stores.mixins import (
    GraphErrorHandlingMixin, GraphLoggingMixin,
    GraphValidationMixin, GraphPerformanceMixin
)

logger = logging.getLogger(__name__)


# ---
# Episodic Memory Graph Schema (Dynamic & Extensible)
#
# Node Types:
#   - Episode: title, description, timestamp, participants, location, outcome, tags
#   - Action (optional): description, order, actor, result
#   - Entity (optional): name, type
# Edge Types (dynamic):
#   - HAS_ACTION: (Episode) -> (Action)
#   - INVOLVES_ENTITY: (Episode|Action) -> (Entity)
#   - NEXT_ACTION: (Action) -> (Action)
#   - Any new relation type as needed (e.g., RESULTED_IN, CAUSED_BY, ANNOTATED_BY, etc.)
#
# The LLM/agentic system can introduce new relation types at any time.
# ---

class EpisodicMemoryGraph(
    BaseMemoryGraph,
    MemoryGraphInterface,
    GraphErrorHandlingMixin,
    GraphLoggingMixin,
    GraphValidationMixin,
    GraphPerformanceMixin
):
    """
    Episodic memory using a graph database (Graphiti/Neo4j) backend.
    Stores episodes (events/experiences) with optional actions, entities, and dynamic relations.
    
    Implements hybrid architecture with:
    - Standard interface (MemoryGraphInterface)
    - Optional mixins for common functionality
    - Type-specific converter for episodic processing
    """

    def __init__(self, *args, **kwargs):
        # Initialize mixins first
        super().__init__(*args, **kwargs)

        # Initialize converter for episodic-specific transformations
        self.converter = EpisodicConverter()

    def add_episode(self,
                    episode_data: Dict[str, Any],
                    actions: Optional[List[Dict[str, Any]]] = None,
                    entities: Optional[List[Dict[str, Any]]] = None,
                    relations: Optional[List[Dict[str, Any]]] = None):
        """Add an episode with optional actions, entities, and dynamic relations."""
        try:
            node = self.graph.add_node(
                item_id=episode_data.get("item_id") or episode_data.get("name"),
                properties=episode_data,
                memory_type="episode"
            )
            if actions:
                for action in actions:
                    action_node = self.graph.add_node(
                        item_id=action.get("item_id") or action.get("description"),
                        properties=action,
                        memory_type="action"
                    )
                    self.graph.add_edge(node["item_id"], action_node["item_id"], edge_type="HAS_ACTION", memory_type="episode")
            if entities:
                for entity in entities:
                    entity_node = self.graph.add_node(
                        item_id=entity.get("item_id") or entity.get("name"),
                        properties=entity,
                        memory_type="entity"
                    )
                    self.graph.add_edge(node["item_id"], entity_node["item_id"], edge_type="INVOLVES_ENTITY", memory_type="episode")
            if relations:
                for rel in relations:
                    self._add_edge_safe(
                        rel["source"], rel["target"], edge_type=rel.get("type", "RELATED"),
                        properties=rel.get("properties") or {}, memory_type="episode"
                    )
            return node
        except Exception as e:
            logger.error(f"Failed to add episode: {e}")
            return None

    def add_dynamic_relation(self, source_id: Any, target_id: Any, relation_type: str, properties: Optional[Dict[str, Any]] = None):
        """Add a dynamic relation (edge) of any type between two nodes."""
        try:
            self._add_edge_safe(source_id, target_id, edge_type=relation_type,
                                properties=properties or {}, memory_type="dynamic")
        except Exception as e:
            logger.error(f"Failed to add dynamic relation {relation_type}: {e}")

    def get_episode(self, episode_id: Any) -> Optional[dict]:
        """Retrieve an episode node and its directly connected actions, entities, and all outgoing relations."""
        try:
            episode = self.graph.get_node(episode_id)
            if episode is None:
                return None

            actions = self.graph.get_neighbors(episode_id, edge_type="HAS_ACTION")
            entities = self.graph.get_neighbors(episode_id, edge_type="INVOLVES_ENTITY")
            outgoing = self.graph.get_neighbors(episode_id)
            return {
                "episode": episode,
                "actions": actions,
                "entities": entities,
                "relations": outgoing
            }
        except Exception as e:
            logger.error(f"Failed to get episode {episode_id}: {e}")
            return None

    def query_by_relation(self, source_id: Any, relation_type: str):
        """Retrieve all target nodes connected by a given relation type."""
        try:
            return self.graph.get_neighbors(source_id, edge_type=relation_type)
        except Exception as e:
            logger.error(f"Failed to query relation {relation_type} from {source_id}: {e}")
            return []

    def find_episodes_by_property(self, key: str, value: Any):
        """Find episodes by property (e.g., timestamp, participant, tag)."""
        try:
            return self.graph.search_nodes({"label": "Episode", key: value})
        except Exception as e:
            logger.error(f"Failed to find episodes by {key}={value}: {e}")
            return []

    def multi_hop_traversal(self, start_id: Any, relation_types: list, max_depth: int = 3) -> set:
        """Traverse the graph from a starting node across multiple relation types up to a given depth."""
        try:
            visited = set()
            frontier = [(start_id, 0)]
            while frontier:
                item_id, depth = frontier.pop()
                if item_id in visited or depth > max_depth:
                    continue
                visited.add(item_id)
                for rel in relation_types:
                    neighbors = self.graph.get_neighbors(item_id, edge_type=rel)
                    for n in neighbors:
                        frontier.append((n["item_id"], depth + 1))
            return visited
        except Exception as e:
            logger.error(f"Failed multi-hop traversal: {e}")
            return set()

    def temporal_query(self, after: Optional[datetime] = None, before: Optional[datetime] = None):
        """Retrieve episodes within a specific time range."""
        try:
            episodes = self.graph.search_nodes({"label": "Episode"})
            results = []
            for ep in episodes:
                ts = ep["properties"].get("timestamp")
                if not ts:
                    continue
                ts_dt = ts if isinstance(ts, datetime) else datetime.fromisoformat(ts)
                if (after and ts_dt < after) or (before and ts_dt > before):
                    continue
                results.append(ep)
            return results
        except Exception as e:
            logger.error(f"Failed temporal query: {e}")
            return []

    def pattern_search(self, action_sequence: list) -> list:
        """Find episodes containing a specific ordered sequence of actions (by description)."""
        try:
            matched_episodes = []
            episodes = self.graph.search_nodes({"label": "Episode"})
            for ep in episodes:
                actions = self.graph.get_neighbors(ep["item_id"], edge_type="HAS_ACTION")
                action_descs = [a["properties"].get("description", "") for a in sorted(actions, key=lambda x: x["properties"].get("order", 0))]
                seq_str = "|||".join(action_descs)
                pat_str = "|||".join(action_sequence)
                if pat_str in seq_str:
                    matched_episodes.append(ep)
            return matched_episodes
        except Exception as e:
            logger.error(f"Failed pattern search: {e}")
            return []

    def causal_outcome_query(self, outcome_value: str, relation_type: str = "RESULTED_IN") -> list:
        """Find episodes that caused or resulted in a specific outcome (using a dynamic relation)."""
        try:
            episodes = self.graph.search_nodes({"label": "Episode"})
            matched = []
            for ep in episodes:
                outgoing = self.graph.get_neighbors(ep["item_id"], edge_type=relation_type)
                for rel in outgoing:
                    target = self.graph.get_node(rel["item_id"])
                    if target and target["properties"].get("outcome") == outcome_value:
                        matched.append(ep)
            return matched
        except Exception as e:
            logger.error(f"Failed causal/outcome query: {e}")
            return []

    def add(self, item, key: str = None, **kwargs):
        """Add an episode using converter and mixins."""
        with self.performance_context("episodic_add"):
            # Use converter to prepare item for episodic storage
            if isinstance(item, MemoryItem):
                graph_data = self.converter.to_graph_data(item)
            else:
                # Convert dict to MemoryItem first, then use converter
                memory_item = MemoryItem(
                    content=item.get("description", ""),
                    metadata=item,
                    item_id=item.get("item_id", key)
                )
                graph_data = self.converter.to_graph_data(memory_item)

            # Validate using mixin
            if not self.validate_item(item if isinstance(item, MemoryItem) else memory_item):
                return self.handle_error(f"Validation failed for item {key or 'unknown'}")

            try:
                # Add main episode node
                node = self.graph.add_node(
                    item_id=graph_data.node_id,
                    properties=graph_data.node_properties,
                    memory_type="episode"
                )

                # Add extracted relations using converter
                for relation in graph_data.relations:
                    self._add_edge_safe(
                        source_id=relation.source_id,
                        target_id=relation.target_id,
                        edge_type=relation.relation_type,
                        properties=relation.properties,
                        memory_type="episode"
                    )

                self.log_operation("add", f"Added episodic item {key or graph_data.node_id}")
                return node

            except Exception as e:
                return self.handle_error(f"Failed to add episodic item {key or 'unknown'}: {e}")

    def get(self, key: str):
        """Retrieve an episode by id."""
        return self.get_episode(key)

    def search(self, query: str, top_k: int = 5, **kwargs):
        """Search episodes using enhanced semantic similarity."""
        try:
            # Get all episodes
            episodes = self.graph.search_nodes({"label": "Episode"})

            if not episodes:
                return []

            # Use enhanced similarity framework for semantic search
            try:
                from smartmemory.similarity import EnhancedSimilarityFramework
                from smartmemory.models.memory_item import MemoryItem

                # Create query item for similarity comparison
                query_item = MemoryItem(content=query, metadata={'type': 'query'})

                # Initialize similarity framework
                similarity_framework = EnhancedSimilarityFramework(graph_store=self.graph)

                # Calculate similarities and rank episodes
                scored_episodes = []
                for ep in episodes:
                    # Convert episode to MemoryItem for similarity calculation
                    ep_content = ep["properties"].get('description', '') + ' ' + ep["properties"].get('title', '')
                    ep_item = MemoryItem(
                        content=ep_content,
                        metadata=ep["properties"],
                        item_id=ep["properties"].get('item_id', '')
                    )

                    # Calculate similarity score
                    similarity_score = similarity_framework.calculate_similarity(query_item, ep_item)
                    scored_episodes.append((ep, similarity_score))

                # Sort by similarity score (descending) and return top_k
                scored_episodes.sort(key=lambda x: x[1], reverse=True)
                return [ep for ep, score in scored_episodes[:top_k] if score > 0.1]  # Filter very low scores

            except ImportError:
                # Fallback to enhanced keyword matching if similarity framework unavailable
                logger.warning("Enhanced similarity framework not available, using enhanced keyword matching")
                return self._enhanced_keyword_search(query, episodes, top_k)

        except Exception as e:
            logger.error(f"Failed to search episodes: {e}")
            return []

    def _enhanced_keyword_search(self, query: str, episodes: list, top_k: int = 5) -> list:
        """Enhanced keyword search with fuzzy matching as fallback."""
        query_words = set(query.lower().split())
        scored_episodes = []

        for ep in episodes:
            desc = ep["properties"].get('description', '').lower()
            title = ep["properties"].get('title', '').lower()
            content_words = set((desc + ' ' + title).split())

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
                    scored_episodes.append((ep, final_score))

        # Sort by score and return top_k
        scored_episodes.sort(key=lambda x: x[1], reverse=True)
        return [ep for ep, score in scored_episodes[:top_k]]

    # Inherited remove() method from BaseMemoryGraph provides consistent implementation

    # Removed redundant clear() method; now inherited from BaseMemoryStore

    # The LLM/agentic layer can call add_dynamic_relation at runtime to introduce new edge types as needed.
    # Retrieval methods (get_episode, query_by_relation, find_episodes_by_property) support flexible, dynamic querying.
