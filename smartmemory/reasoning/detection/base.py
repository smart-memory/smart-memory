"""Base class for contradiction detection strategies."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from smartmemory.models.memory_item import MemoryItem
from smartmemory.reasoning.models import Conflict


@dataclass
class DetectionContext:
    """Context passed to each detector in the cascade."""

    new_assertion: str
    existing_item: MemoryItem
    existing_fact: str
    extra: Dict[str, Any] = field(default_factory=dict)


class ContradictionDetector(ABC):
    """Strategy interface for a single contradiction detection method.

    Subclasses MUST set ``name`` as a class attribute (used in cascade logging).
    """

    name: str  # No default -- subclasses must define

    @abstractmethod
    def detect(self, ctx: DetectionContext) -> Optional[Conflict]:
        """Return a Conflict if a contradiction is found, else None."""
