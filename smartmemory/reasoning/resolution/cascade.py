"""Orchestrator that runs resolvers in order, returns first successful resolution."""

import logging
from typing import Any, Dict, List, Optional

from smartmemory.reasoning.challenger import Conflict

from .base import ConflictResolver

logger = logging.getLogger(__name__)


class ResolutionCascade:
    """Run a sequence of resolvers; return the first successful resolution."""

    def __init__(self, resolvers: List[ConflictResolver]):
        self.resolvers = resolvers

    def resolve(self, conflict: Conflict) -> Optional[Dict[str, Any]]:
        for resolver in self.resolvers:
            result = resolver.resolve(conflict)
            if result and result.get("auto_resolved"):
                return result
        return None
