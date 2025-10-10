from dataclasses import dataclass, field
from typing import Optional, List

from smartmemory.models.base import MemoryBaseModel
from smartmemory.models.memory_item import MemoryItem


@dataclass
class Tag(MemoryBaseModel):
    """A tag node in the memory graph.
    Fields:
        item_id: Unique identifier
        name: Canonical label for the tag
        memory_types: List of memory types (e.g., 'semantic', 'procedural')
    """
    memory_types: List[str] = field(default_factory=list)
    """A tag node in the memory graph."""
    name: str = ""
    item_id: Optional[str] = None

    @classmethod
    def from_memory_item(cls, item: MemoryItem) -> "Tag":
        return cls(
            item_id=item.item_id,
            name=item.content or item.metadata.get("name", ""),
            memory_types=item.metadata.get("memory_types", [])
        )

    def to_memory_item(self) -> MemoryItem:
        meta = dict(getattr(self, "metadata", {}))
        meta.setdefault("memory_types", self.memory_types)
        meta.setdefault("name", self.name)
        kwargs = dict(content=self.name, type="Tag", metadata=meta)
        if self.item_id:
            kwargs["item_id"] = self.item_id
        return MemoryItem(**kwargs)
