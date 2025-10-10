from typing import Any, Dict


class Evolver:
    """
    Base class for all memory evolvers.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

    def evolve(self, memory, logger=None):
        """
        Apply evolution logic to the memory system.
        Should be overridden by subclasses.
        """
        raise NotImplementedError("Evolver subclasses must implement evolve()")
