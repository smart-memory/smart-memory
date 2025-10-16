"""
Node Type Abstraction Layer for Dual-Node Architecture

This module provides clean separation between memory nodes (for system processing)
and entity nodes (for domain modeling) to enable flexible infrastructure changes.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Any, Union

from smartmemory.models.memory_item import MemoryItem


class NodeCategory(Enum):
    """Categories of nodes in the dual-node architecture."""
    MEMORY = "memory"  # System processing nodes (Semantic, Episodic, etc.)
    ENTITY = "entity"  # Domain modeling nodes (Person, Organization, etc.)


class MemoryNodeType(Enum):
    """Memory node types for system processing."""
    SEMANTIC = "semantic"
    EPISODIC = "episodic"
    PROCEDURAL = "procedural"
    WORKING = "working"
    ZETTEL = "zettel"


class EntityNodeType(Enum):
    """Entity node types for domain modeling."""
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    CONCEPT = "concept"
    EVENT = "event"
    TOOL = "tool"
    SKILL = "skill"


@dataclass
class MemoryNodeSpec:
    """Specification for creating a memory node."""
    item_id: str
    node_type: MemoryNodeType
    properties: Dict[str, Any]
    content: str
    metadata: Dict[str, Any]


@dataclass
class EntityNodeSpec:
    """Specification for creating an entity node."""
    entity_type: Union[EntityNodeType, str]  # Accept both enum and string
    properties: Dict[str, Any]
    relations: List[Dict[str, Any]]  # Relations to other entities


@dataclass
class DualNodeSpec:
    """Complete specification for dual-node creation."""
    memory_node: MemoryNodeSpec
    entity_nodes: List[EntityNodeSpec]


class NodeTypeProcessor:
    """Processes different node types with segregated logic."""

    def __init__(self, graph_backend):
        self.graph = graph_backend

    def create_dual_node_structure(self, spec: DualNodeSpec) -> Dict[str, Any]:
        """
        Create a complete dual-node structure from specification.
        
        Args:
            spec: DualNodeSpec containing memory and entity node specifications
            
        Returns:
            Dict with creation results and node IDs
        """
        # Prepare memory node properties
        memory_properties = self._prepare_memory_properties(spec.memory_node)

        # Prepare entity nodes
        entity_nodes = self._prepare_entity_nodes(spec.entity_nodes)

        # Create dual-node structure atomically
        result = self.graph.add_dual_node(
            item_id=spec.memory_node.item_id,
            memory_properties=memory_properties,
            memory_type=spec.memory_node.node_type.value,
            entity_nodes=entity_nodes
        )

        return result

    def _prepare_memory_properties(self, memory_spec: MemoryNodeSpec) -> Dict[str, Any]:
        """Prepare properties for memory node creation."""
        properties = dict(memory_spec.properties)
        properties.update({
            'content': memory_spec.content,
            'node_category': NodeCategory.MEMORY.value,
            'memory_type': memory_spec.node_type.value,
        })

        # Merge metadata
        if memory_spec.metadata:
            for key, value in memory_spec.metadata.items():
                # Avoid overwriting system properties
                if key not in ['node_category', 'memory_type']:
                    properties[key] = value

        return properties

    def _prepare_entity_nodes(self, entity_specs: List[EntityNodeSpec]) -> List[Dict[str, Any]]:
        """Prepare entity node specifications for creation."""
        entity_nodes = []

        for spec in entity_specs:
            entity_node = {
                'entity_type': spec.entity_type if isinstance(spec.entity_type, str) else spec.entity_type.value,
                'properties': dict(spec.properties),
                'relations': spec.relations
            }

            # Add node category to properties
            entity_node['properties']['node_category'] = NodeCategory.ENTITY.value
            entity_node['properties']['entity_type'] = spec.entity_type if isinstance(spec.entity_type, str) else spec.entity_type.value

            entity_nodes.append(entity_node)

        return entity_nodes

    def extract_dual_node_spec_from_memory_item(
            self,
            item: MemoryItem,
            ontology_extraction_result: Dict[str, Any] = None
    ) -> DualNodeSpec:
        """
        Convert a MemoryItem and ontology extraction into a DualNodeSpec.
        
        Args:
            item: MemoryItem to convert
            ontology_extraction_result: Result from ontology extractor
            
        Returns:
            DualNodeSpec ready for dual-node creation
        """
        # Create memory node spec
        memory_type = self._determine_memory_type(item)

        # Preserve user_id and other critical properties
        properties = {}
        metadata = item.metadata or {}

        # Extract user_id from multiple sources for robust preservation
        user_id = None
        if hasattr(item, 'user_id') and item.user_id:
            user_id = item.user_id
        elif metadata.get('user_id'):
            user_id = metadata.get('user_id')

        # Preserve user_id in both properties and metadata for reliable retrieval
        if user_id:
            properties['user_id'] = user_id
            metadata['user_id'] = user_id

        memory_spec = MemoryNodeSpec(
            item_id=item.item_id,
            node_type=memory_type,
            properties=properties,
            content=item.content or "",
            metadata=metadata
        )

        # Create entity node specs from ontology extraction
        entity_specs = []
        if ontology_extraction_result:
            entities = ontology_extraction_result.get('entities', [])
            relations = ontology_extraction_result.get('relations', [])

            # Convert ontology extraction to dual-node spec with deduplication
            seen_entities = set()  # Track (name, type) pairs to avoid duplicates

            # Create entity_id_to_index mapping for relationship processing
            # Accept both MemoryItem objects and dicts for flexibility
            entity_id_to_index = {}
            for i, entity in enumerate(entities):
                # Duck typing: support both MemoryItem and dict
                if isinstance(entity, MemoryItem):
                    entity_id = entity.item_id
                elif isinstance(entity, dict):
                    entity_id = entity.get('item_id') or entity.get('id')
                else:
                    entity_id = None

                if entity_id:
                    entity_id_to_index[entity_id] = i

            for i, entity in enumerate(entities):
                # Duck typing: extract properties from either MemoryItem or dict
                if isinstance(entity, MemoryItem):
                    entity_name = entity.metadata.get('name', f'entity_{i}') if entity.metadata else f'entity_{i}'
                    entity_metadata = entity.metadata or {}
                    entity_memory_type = getattr(entity, 'memory_type', None)
                    entity_id = entity.item_id
                elif isinstance(entity, dict):
                    entity_metadata = entity.get('metadata', {}) or {}
                    entity_name = entity_metadata.get('name') or entity.get('name', f'entity_{i}')
                    entity_memory_type = entity.get('memory_type')
                    entity_id = entity.get('item_id') or entity.get('id')
                else:
                    # Fallback for other types
                    entity_name = f'entity_{i}'
                    entity_metadata = {}
                    entity_memory_type = None
                    entity_id = None

                # Prefer explicit entity_type in metadata; fallback to memory_type; default to 'entity'
                if 'entity_type' in entity_metadata and entity_metadata['entity_type']:
                    entity_type_str = str(entity_metadata['entity_type']).lower()
                elif entity_memory_type:
                    entity_type_str = str(entity_memory_type).lower()
                else:
                    entity_type_str = 'entity'

                # Extract entity properties
                properties = {
                    'name': entity_metadata.get('name', '') or entity_name,
                    'confidence': entity_metadata.get('confidence', 1.0),
                    'source': 'ontology_extraction'
                }

                # Add entity-specific properties from metadata
                for key, value in entity_metadata.items():
                    if key not in ['name', 'confidence', 'source']:
                        properties[key] = value

                # Find relationships for this entity
                entity_relationships = []
                if entity_id:
                    for relation in relations:
                        source_id = relation.get('source_id')
                        target_id = relation.get('target_id')

                        if source_id == entity_id and target_id in entity_id_to_index:
                            entity_relationships.append({
                                'target_index': entity_id_to_index[target_id],
                                'relation_type': relation.get('relation_type', 'RELATED')
                            })

                entity_spec = EntityNodeSpec(
                    entity_type=entity_type_str,
                    properties=properties,
                    relations=entity_relationships
                )
                entity_specs.append(entity_spec)

        return DualNodeSpec(
            memory_node=memory_spec,
            entity_nodes=entity_specs
        )

    def _determine_memory_type(self, item: MemoryItem) -> MemoryNodeType:
        """Determine the memory type for a MemoryItem."""
        # Check item's type field first
        if hasattr(item, 'type') and item.memory_type:
            try:
                return MemoryNodeType(item.memory_type.lower())
            except ValueError:
                pass

        # Check metadata
        if item.metadata and 'memory_type' in item.metadata:
            try:
                return MemoryNodeType(item.metadata['memory_type'].lower())
            except ValueError:
                pass

        # Default to semantic
        return MemoryNodeType.SEMANTIC

    def query_memory_nodes(self, memory_type: MemoryNodeType = None, **filters) -> List[Dict[str, Any]]:
        """Query memory nodes with optional filtering using supported backend API (no raw Cypher)."""
        # Build property-based query to preserve encapsulation and avoid label-dependent Cypher
        query: Dict[str, Any] = {
            'node_category': NodeCategory.MEMORY.value,
        }
        if memory_type:
            query['memory_type'] = memory_type.value
        if filters:
            query.update(filters)
        try:
            return self.graph.backend.search_nodes(query)
        except Exception:
            return []

    def query_entity_nodes(self, entity_type: EntityNodeType = None, **filters) -> List[Dict[str, Any]]:
        """Query entity nodes with optional filtering using supported backend API (no raw Cypher)."""
        query: Dict[str, Any] = {
            'node_category': NodeCategory.ENTITY.value,
        }
        if entity_type:
            query['entity_type'] = entity_type.value
        if filters:
            query.update(filters)
        try:
            return self.graph.backend.search_nodes(query)
        except Exception:
            return []
