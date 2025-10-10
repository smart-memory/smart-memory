from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from smartmemory.models.base import MemoryBaseModel, StageRequest
from smartmemory.plugins.base import EvolverPlugin, PluginMetadata
from .base import Evolver


@dataclass
class WorkingToProceduralConfig(MemoryBaseModel):
    k: int = 5  # minimum pattern count to promote


@dataclass
class WorkingToProceduralRequest(StageRequest):
    k: int = 5
    context: Dict[str, Any] = field(default_factory=dict)
    run_id: Optional[str] = None


class WorkingToProceduralEvolver(Evolver, EvolverPlugin):
    """
    Evolves repeated skill/tool patterns in working memory to procedural memory (macro creation).
    """

    @classmethod
    def metadata(cls) -> PluginMetadata:
        """Return plugin metadata for discovery."""
        return PluginMetadata(
            name="working_to_procedural",
            version="1.0.0",
            author="SmartMemory Team",
            description="Promotes repeated skill patterns from working to procedural memory",
            plugin_type="evolver",
            dependencies=[],
            min_smartmemory_version="0.1.0"
        )

    def evolve(self, memory, logger=None):
        # Example: promote repeated tool usage to procedural macro
        cfg = getattr(self, "config")
        if not hasattr(cfg, "k"):
            raise TypeError(
                "WorkingToProceduralEvolver requires a typed config with a 'k' attribute. "
                "Please provide WorkingToProceduralConfig or a compatible typed config."
            )
        k = int(getattr(cfg, "k"))
        patterns = memory.working.detect_skill_patterns(min_count=k)
        for pattern in patterns:
            memory.procedural.add_macro(pattern)
            if logger:
                logger.info(f"Promoted working skill pattern to procedural: {pattern}")
