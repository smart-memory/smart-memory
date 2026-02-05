"""Base class for conflict resolution strategies."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from smartmemory.reasoning.models import Conflict


class ConflictResolver(ABC):
    """Strategy interface for resolving a detected conflict.

    Subclasses MUST set ``name`` as a class attribute (used in cascade logging).
    """

    name: str  # No default -- subclasses must define

    @abstractmethod
    def resolve(self, conflict: Conflict) -> Optional[Dict[str, Any]]:
        """Attempt to resolve a conflict. Return result dict or None if unable."""
