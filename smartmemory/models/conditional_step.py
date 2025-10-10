from dataclasses import dataclass
from typing import Optional

from smartmemory.models.base import MemoryBaseModel
from smartmemory.models.memory_item import MemoryItem


@dataclass
class ConditionalStep(MemoryBaseModel):
    """A conditional step node in the memory graph."""
    name: str = ""
    item_id: Optional[str] = None
    condition: str = ""

    @classmethod
    def from_memory_item(cls, item: MemoryItem) -> "ConditionalStep":
        return cls(
            item_id=item.item_id,
            name=item.content or item.metadata.get("name", ""),
            condition=item.metadata.get("condition", "")
        )

    def to_memory_item(self) -> MemoryItem:
        kwargs = dict(
            content=self.name,
            type="conditional_step",
            metadata={"condition": self.condition},
        )
        if self.item_id:
            kwargs["item_id"] = self.item_id
        return MemoryItem(**kwargs)
