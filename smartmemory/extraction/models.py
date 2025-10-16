# models/ontology_response.py
from dataclasses import dataclass, field
from typing import List, Union, Dict, Any

from smartmemory.models.base import MemoryBaseModel
from smartmemory.models.ontology import OntologyNode


@dataclass
class PropertyKV(MemoryBaseModel):
    key: str = field(default="", metadata={"description": "Property key"})
    value: Union[str, float, int, bool, None] = field(default=None, metadata={"description": "Property value"})


@dataclass
class Entity(MemoryBaseModel):
    name: str = field(default="", metadata={"description": "The name of the entity"})
    type: str = field(default="", metadata={"description": "The type of the entity"})
    properties: List[PropertyKV] = field(default_factory=list, metadata={"description": "Entity properties"})


@dataclass
class Relationship(MemoryBaseModel):
    source: str = field(default="", metadata={"description": "The name of the source entity"})
    type: str = field(default="", metadata={"description": "The type of the relationship"})
    target: str = field(default="", metadata={"description": "The name of the target entity"})
    confidence: float = field(default=0.0, metadata={"description": "Relationship confidence score"})


@dataclass
class OntologyExtractionResponse(MemoryBaseModel):
    entities: List[Entity] = field(default_factory=list, metadata={"description": "Extracted entities"})
    relations: List[Relationship] = field(default_factory=list, metadata={"description": "Extracted relations"})


@dataclass
# Flexible ontology node that can represent any LLM-extracted entity type
class GenericOntologyNode(OntologyNode):
    """Flexible ontology node that can represent any LLM-extracted entity type without rigid constraints."""

    dynamic_node_type: str = ""
    # Bucket for arbitrary properties extracted by the LLM
    attributes: Dict[str, Any] = field(default_factory=dict)

    @property
    def node_type(self) -> str:
        """Return the LLM-extracted node type."""
        return self.dynamic_node_type

    def get_searchable_content(self) -> str:
        """Return content for search indexing."""
        content_parts = [self.name]
        if hasattr(self, 'description') and self.description:
            content_parts.append(self.description)

        # Add any other string properties for searchability
        for key, value in self.__dict__.items():
            if key not in ['item_id', 'name', 'description', 'dynamic_node_type', 'attributes'] and isinstance(value, str):
                content_parts.append(value)

        # Include string values from attributes as well
        try:
            for k, v in (self.attributes or {}).items():
                if isinstance(v, str):
                    content_parts.append(v)
        except Exception:
            pass

        return ' '.join(content_parts)

    def to_memory_item(self) -> 'MemoryItem':
        """Convert to MemoryItem for storage with proper metadata handling."""
        from smartmemory.models.memory_item import MemoryItem

        # Build metadata from dataclass, include dynamic node_type
        metadata = self.to_dict()
        metadata.pop('item_id', None)
        metadata['node_type'] = self.dynamic_node_type

        kwargs = dict(
            content=self.get_searchable_content(),
            type=self.dynamic_node_type,
            metadata=metadata,
        )
        if self.item_id:
            kwargs['item_id'] = self.item_id
        return MemoryItem(**kwargs)
