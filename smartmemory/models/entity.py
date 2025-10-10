from dataclasses import dataclass, field
from typing import Optional

from smartmemory.models.base import MemoryBaseModel
from smartmemory.models.memory_item import MemoryItem


@dataclass
class Entity(MemoryBaseModel):
    """An entity node in the memory graph."""
    name: str = ""
    item_id: Optional[str] = None
    entity_type: str = ""  # Changed from 'type' to 'entity_type' for clarity
    properties: dict = field(default_factory=dict)

    @classmethod
    def from_memory_item(cls, item: MemoryItem) -> "Entity":
        return cls(
            item_id=item.item_id,
            name=item.content or item.metadata.get("name", ""),
            entity_type=item.metadata.get("entity_type", item.metadata.get("type", "")),
            properties=item.metadata.get("properties") or {}
        )

    def to_memory_item(self) -> MemoryItem:
        # Generate item_id if not set to avoid validation error
        import uuid
        item_id = self.item_id or str(uuid.uuid4())

        return MemoryItem(
            item_id=item_id,
            content=self.name,
            memory_type="Entity",
            metadata={
                "entity_type": self.entity_type,
                "properties": self.properties
            }
        )
