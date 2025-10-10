import logging
from typing import Optional, Any, Dict

from smartmemory.graph.base import BaseMemoryGraph
from smartmemory.graph.types.interfaces import MemoryGraphInterface
from smartmemory.models.memory_item import MemoryItem
from smartmemory.stores.converters.zettel_converter import ZettelConverter
from smartmemory.stores.mixins import (
    GraphErrorHandlingMixin, GraphLoggingMixin,
    GraphValidationMixin, GraphPerformanceMixin
)

logger = logging.getLogger(__name__)


class ZettelMemoryGraph(
    BaseMemoryGraph,
    MemoryGraphInterface,
    GraphErrorHandlingMixin,
    GraphLoggingMixin,
    GraphValidationMixin,
    GraphPerformanceMixin
):
    """
    Store layer for ZettelMemory. Handles all graph operations for notes, tags, links, and dynamic relations.
    
    Implements hybrid architecture with:
    - Standard interface (MemoryGraphInterface)
    - Optional mixins for common functionality
    - Type-specific converter for Zettelkasten processing
    """

    def __init__(self, *args, **kwargs):
        # Initialize mixins first
        super().__init__(*args, **kwargs)

        # Initialize converter for Zettelkasten-specific transformations
        self.converter = ZettelConverter()

    def add(self, item, key: str = None, **kwargs):
        """
        Add a note using converter and mixins.
        """
        with self.performance_context("zettel_add"):
            # Convert dict to MemoryItem if needed, then use converter
            if isinstance(item, dict):
                memory_item = MemoryItem(
                    content=item.get('content', item.get('zettel_body', '')),
                    metadata=item,
                    item_id=item.get('item_id', key)
                )
            else:
                memory_item = item

            # Use converter to prepare item for Zettelkasten storage
            graph_data = self.converter.to_graph_data(memory_item)

            # Validate using mixin
            if not self.validate_item(memory_item):
                return self.handle_error(f"Validation failed for note {key or graph_data.node_id}")

            try:
                # Add main note node
                note_node = self.graph.add_node(
                    item_id=graph_data.node_id,
                    properties=graph_data.node_properties,
                    memory_type="zettel"
                )

                # Add extracted relations using converter
                for relation in graph_data.relations:
                    # Create target nodes for tags and concepts if they don't exist
                    if relation.relation_type == "TAGGED_WITH":
                        tag_name = relation.properties.get('tag_name', relation.target_id.replace('tag_', ''))
                        tag_node = self.graph.add_node(
                            item_id=relation.target_id,
                            properties={"name": tag_name, "label": "Tag"},
                            memory_type="tag"
                        )
                    elif relation.relation_type == "MENTIONS":
                        concept_name = relation.properties.get('concept_name', relation.target_id.replace('concept_', ''))
                        concept_node = self.graph.add_node(
                            item_id=relation.target_id,
                            properties={
                                "name": concept_name,
                                "title": concept_name,
                                "content": f"Concept: {concept_name}",
                                "label": "Concept"
                            },
                            memory_type="concept"
                        )
                    elif relation.relation_type in ["LINKS_TO", "LINKS_TO_BACK"]:
                        # CRITICAL FIX: Create placeholder nodes for wikilink targets that don't exist yet
                        target_exists = self.graph.get_node(relation.target_id) is not None
                        if not target_exists:
                            # Create placeholder note node for the wikilink target
                            placeholder_node = self.graph.add_node(
                                item_id=relation.target_id,
                                properties={
                                    "item_id": relation.target_id,
                                    "content": f"# {relation.target_id}\n\nPlaceholder note created for wikilink reference.",
                                    "title": relation.target_id,
                                    "name": relation.target_id,
                                    "label": "Note",
                                    "placeholder": True,
                                    "created_by_wikilink": True
                                },
                                memory_type="zettel"
                            )

                    # Add the relation
                    self._add_edge_safe(
                        source_id=relation.source_id,
                        target_id=relation.target_id,
                        edge_type=relation.relation_type,
                        properties=relation.properties,
                        memory_type="zettel"
                    )

                self.log_operation("add", f"Added Zettelkasten note {graph_data.node_id}")
                return note_node

            except Exception as e:
                return self.handle_error(f"Failed to add Zettelkasten note {graph_data.node_id}: {e}")

    # Inherited get() method from BaseMemoryGraph provides consistent implementation

    # Inherited remove() method from BaseMemoryGraph provides consistent implementation

    def search(self, query: str, top_k: int = 5, **kwargs) -> list:
        """
        Search for notes by query string. Returns a list of dicts.
        """
        try:
            # SmartGraph does not implement search yet; stub with find_nodes by content substring
            notes = self.graph.search_nodes({"label": "Note"})
            filtered = [n for n in notes if query.lower() in n["properties"].get("content", "").lower()]
            return [{"key": n.get("item_id"), "content": n["properties"].get("content")} for n in filtered[:top_k]]
        except Exception as e:
            logger.error(f"Failed to search notes for query '{query}': {e}")
            return []

    def add_dynamic_relation(self, source_id: Any, target_id: Any, relation_type: str, properties: Optional[Dict[str, Any]] = None):
        try:
            self._add_edge_safe(source_id, target_id, edge_type=relation_type,
                                properties=properties or {}, memory_type="dynamic")
        except Exception as e:
            logger.error(f"Failed to add dynamic relation {relation_type}: {e}")

    def find_notes_by_tag(self, tag_name: str) -> list:
        try:
            tag_nodes = self.graph.search_nodes({"name": tag_name, "label": "Tag"})
            if not tag_nodes:
                return []
            tag_id = tag_nodes[0].get("item_id") if isinstance(tag_nodes[0], dict) else tag_nodes[0]
            return self.graph.get_neighbors(tag_id, edge_type="TAGGED_WITH")
        except Exception as e:
            logger.error(f"Failed to find notes by tag '{tag_name}': {e}")
            return []

    def find_notes_by_property(self, key: str, value: Any) -> list:
        try:
            return self.graph.search_nodes({"label": "Note", key: value})
        except Exception as e:
            logger.error(f"Failed to find notes by {key}={value}: {e}")
            return []

    def notes_linked_to(self, note_id: Any) -> list:
        try:
            return self.graph.get_neighbors(note_id, edge_type="LINKED_TO")
        except Exception as e:
            logger.error(f"Failed to find notes linked to {note_id}: {e}")
            return []

    def notes_mentioning(self, entity_id: Any) -> list:
        try:
            return self.graph.get_neighbors(entity_id, edge_type="MENTIONS")
        except Exception as e:
            logger.error(f"Failed to find notes mentioning {entity_id}: {e}")
            return []

    def query_by_dynamic_relation(self, source_id: Any, relation_type: str) -> list:
        try:
            return self.graph.get_neighbors(source_id, edge_type=relation_type)
        except Exception as e:
            logger.error(f"Failed to query relation {relation_type} from {source_id}: {e}")
            return []
