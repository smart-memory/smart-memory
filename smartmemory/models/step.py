from dataclasses import dataclass
from typing import Optional

from smartmemory.models.base import MemoryBaseModel
from smartmemory.models.memory_item import MemoryItem


@dataclass
class Step(MemoryBaseModel):
    """A step node in the memory graph."""
    name: str = ""
    item_id: Optional[str] = None
    order: int = 0
    description: str = ""

    @classmethod
    def from_memory_item(cls, item: MemoryItem) -> "Step":
        return cls(
            item_id=item.item_id,
            name=item.content or item.metadata.get("name", ""),
            order=item.metadata.get("order", 0),
            description=item.metadata.get("description", "")
        )

    def to_memory_item(self) -> MemoryItem:
        kwargs = dict(
            content=self.name,
            type="step",
            metadata={"order": self.order, "description": self.description},
        )
        if self.item_id:
            kwargs["item_id"] = self.item_id
        return MemoryItem(**kwargs)
