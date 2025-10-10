from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from smartmemory.models.base import MemoryBaseModel, StageRequest
from smartmemory.plugins.base import EvolverPlugin, PluginMetadata
from .base import Evolver


@dataclass
class ZettelPruneConfig(MemoryBaseModel):
    # No knobs yet; keep placeholder to enforce typed config presence
    pass


@dataclass
class ZettelPruneRequest(StageRequest):
    # No params yet; placeholder for convention-based request
    context: Dict[str, Any] = field(default_factory=dict)
    run_id: Optional[str] = None


class ZettelPruneEvolver(Evolver, EvolverPlugin):
    """
    Prunes/merges low-quality or duplicate zettels for graph health.
    """

    @classmethod
    def metadata(cls) -> PluginMetadata:
        """Return plugin metadata for discovery."""
        return PluginMetadata(
            name="zettel_prune",
            version="1.0.0",
            author="SmartMemory Team",
            description="Prunes and merges low-quality or duplicate zettels",
            plugin_type="evolver",
            dependencies=[],
            min_smartmemory_version="0.1.0"
        )

    def evolve(self, memory, logger=None):
        # Require typed config presence (even if empty) to ensure convention compliance
        cfg = getattr(self, "config")
        if not isinstance(cfg, MemoryBaseModel):
            raise TypeError("ZettelPruneEvolver requires a typed config (ZettelPruneConfig).")
        zettels = memory.zettel.get_low_quality_or_duplicates()
        for z in zettels:
            memory.zettel.prune_or_merge(z)
            if logger:
                logger.info(f"Pruned/merged zettel: {z}")
