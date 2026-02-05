"""Base class for contradiction detection strategies."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from smartmemory.models.memory_item import MemoryItem
from smartmemory.reasoning.challenger import Conflict


@dataclass
class DetectionContext:
    """Context passed to each detector in the cascade."""

    new_assertion: str
    existing_item: MemoryItem
    existing_fact: str
    extra: Dict[str, Any] = field(default_factory=dict)


class ContradictionDetector(ABC):
    """Strategy interface for a single contradiction detection method."""

    name: str = "base"

    @abstractmethod
    def detect(self, ctx: DetectionContext) -> Optional[Conflict]:
        """Return a Conflict if a contradiction is found, else None."""
