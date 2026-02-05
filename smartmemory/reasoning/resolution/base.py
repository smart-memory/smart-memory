"""Base class for conflict resolution strategies."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from smartmemory.reasoning.challenger import Conflict


class ConflictResolver(ABC):
    """Strategy interface for resolving a detected conflict."""

    name: str = "base"

    @abstractmethod
    def resolve(self, conflict: Conflict) -> Optional[Dict[str, Any]]:
        """Attempt to resolve a conflict. Return result dict or None if unable."""
