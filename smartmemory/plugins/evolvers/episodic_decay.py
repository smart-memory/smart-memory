from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from smartmemory.models.base import MemoryBaseModel, StageRequest
from smartmemory.plugins.base import EvolverPlugin, PluginMetadata
from .base import Evolver


@dataclass
class EpisodicDecayConfig(MemoryBaseModel):
    half_life: int = 30  # days


@dataclass
class EpisodicDecayRequest(StageRequest):
    half_life: int = 30
    context: Dict[str, Any] = field(default_factory=dict)
    run_id: Optional[str] = None


class EpisodicDecayEvolver(Evolver, EvolverPlugin):
    """
    Archives or deletes stale episodic events based on age or relevance.
    """

    @classmethod
    def metadata(cls) -> PluginMetadata:
        """Return plugin metadata for discovery."""
        return PluginMetadata(
            name="episodic_decay",
            version="1.0.0",
            author="SmartMemory Team",
            description="Archives stale episodic events based on age",
            plugin_type="evolver",
            dependencies=[],
            min_smartmemory_version="0.1.0"
        )

    def evolve(self, memory, logger=None):
        cfg = getattr(self, "config")
        if not hasattr(cfg, "half_life"):
            raise TypeError(
                "EpisodicDecayEvolver requires a typed config with 'half_life'. "
                "Provide EpisodicDecayConfig or a compatible typed config."
            )
        half_life = int(getattr(cfg, "half_life"))
        stale_events = memory.episodic.get_stale_events(half_life=half_life)
        for event in stale_events:
            memory.episodic.archive(event)
            if logger:
                logger.info(f"Archived stale episodic event: {event}")
