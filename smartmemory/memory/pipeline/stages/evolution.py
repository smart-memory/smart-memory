"""
Evolution orchestrator component for SmartMemory.
Handles all evolution logic and coordination between memory types.
"""
import logging
from typing import List, Optional, Dict, Any

from smartmemory.evolution.flow import EvolutionFlow
from smartmemory.evolution.registry import EVOLVER_REGISTRY

logger = logging.getLogger(__name__)


class EvolutionOrchestrator:
    """
    Orchestrates evolution processes between memory types.
    Eliminates mixed abstractions by centralizing evolution logic.
    """

    def __init__(self, smart_memory):
        """
        Initialize with reference to the smartmemory instance for evolution operations.
        
        Args:
            smart_memory: The SmartMemory instance to operate on
        """
        self.smart_memory = smart_memory
        # Optional dynamic DAG workflow (set at runtime by Studio)
        self._workflow: Optional[EvolutionFlow] = None

    def should_evolve_working_to_episodic(self) -> bool:
        """
        Determine if working memory should be evolved to episodic memory.
        Simple heuristic: evolve every few items to populate episodic memory.
        """
        try:
            # Get working memory items count using unified interface
            working_items = self.smart_memory.search("*", memory_type="working", top_k=100)
            # Evolve every 3-5 items to ensure episodic memory gets populated
            return len(working_items) >= 3
        except Exception as e:
            logger.debug(f"Evolution check failed: {e}")
            return False

    def should_evolve_working_to_procedural(self) -> bool:
        """
        Determine if working memory should be evolved to procedural memory.
        Simple heuristic: evolve when we have enough working memory items.
        """
        try:
            working_items = self.smart_memory.search("*", memory_type="working", top_k=100)
            # Evolve every 5-7 items to build procedural knowledge
            return len(working_items) >= 5
        except Exception as e:
            logger.debug(f"Evolution check failed: {e}")
            return False

    def run_evolution_cycle(self):
        """
        Run evolution cycle to populate episodic and procedural memory from working memory.
        This is the key to populating memory stores that depend on evolution.
        """
        # If a dynamic workflow is installed, delegate to it
        if getattr(self, "_workflow", None):
            try:
                self.run_workflow()
                return
            except Exception as e:
                logger.error(f"Dynamic workflow execution failed, falling back: {e}")

        # Check if we should evolve working memory to episodic
        if self.should_evolve_working_to_episodic():
            self.commit_working_to_episodic()
            logger.info("Evolved working memory to episodic memory")

        # Check if we should evolve working memory to procedural
        if self.should_evolve_working_to_procedural():
            self.commit_working_to_procedural()
            logger.info("Evolved working memory to procedural memory")

    # ------------------------------
    # Dynamic DAG workflow support
    # ------------------------------
    def set_workflow(self, dag: "EvolutionFlow") -> None:
        """Install a dynamic DAG to drive evolution (built at runtime)."""
        if not isinstance(dag, EvolutionFlow):
            raise TypeError("dag must be an EvolutionFlow")
        dag.validate_acyclic()
        self._workflow = dag

    def get_workflow(self) -> Optional["EvolutionFlow"]:
        return getattr(self, "_workflow", None)

    def run_workflow(self, context: Optional[Dict[str, Any]] = None, stop_on_error: bool = False) -> Dict[str, Any]:
        """
        Execute the currently installed DAG. Conditions are callables evaluated at runtime.
        Returns a summary dict: {"executed": [node_ids], "skipped": [node_ids], "errors": {node_id: str}}
        """
        dag = getattr(self, "_workflow", None)
        if not dag:
            raise RuntimeError("No workflow set. Call set_workflow() first.")
        dag.validate_acyclic()
        order = dag.topological_order()
        executed: List[str] = []
        skipped: List[str] = []
        errors: Dict[str, str] = {}

        for node_id in order:
            node = dag.nodes[node_id]
            try:
                should_run = True
                if node.condition is not None:
                    should_run = bool(node.condition(self.smart_memory, context))
                if not should_run:
                    skipped.append(node_id)
                    continue

                # Resolve via registry key (preferred) or dotted path fallback
                evolver_cls = EVOLVER_REGISTRY.try_resolve(node.evolver_path)
                if evolver_cls is None:
                    raise ImportError(f"Could not resolve evolver '{node.evolver_path}' from registry or dotted path")
                evolver = evolver_cls(config=getattr(self.smart_memory, 'config', {}))

                # Allow extra params for evolve
                params = node.params or {}
                result = evolver.evolve(self.smart_memory, logger=logger, **params)
                logger.info(f"Workflow node '{node_id}' executed: {type(evolver).__name__} -> {type(result).__name__ if result is not None else 'None'}")
                executed.append(node_id)
            except Exception as e:
                logger.exception(f"Workflow node '{node_id}' failed: {e}")
                errors[node_id] = str(e)
                if stop_on_error:
                    break

        return {"executed": executed, "skipped": skipped, "errors": errors}

    def commit_working_to_episodic(self, remove_from_source: bool = True) -> List[str]:
        """
        Promote (evolve) working memory buffer to episodic memory.
        """
        try:
            # Try to import and use the proper evolver
            from smartmemory.plugins.evolvers.working_to_episodic import WorkingToEpisodicEvolver
            config = getattr(self.smart_memory, 'config', {}) if hasattr(self.smart_memory, 'config') else {}
            evolver = WorkingToEpisodicEvolver(config=config)
            result = evolver.evolve(self.smart_memory, logger=logger)
            return result
        except ImportError as e:
            logger.warning(f"WorkingToEpisodicEvolver not available: {e}")
            # Fallback: manually create episodic items from working memory
            return self._fallback_working_to_episodic()

    def commit_working_to_procedural(self, remove_from_source: bool = True) -> List[str]:
        """
        Promote (evolve) working memory buffer to procedural memory.
        """
        try:
            # Try to import and use the proper evolver
            from smartmemory.plugins.evolvers.working_to_procedural import WorkingToProceduralEvolver
            config = getattr(self.smart_memory, 'config', {}) if hasattr(self.smart_memory, 'config') else {}
            evolver = WorkingToProceduralEvolver(config=config)
            result = evolver.evolve(self.smart_memory, logger=logger)
            return result
        except ImportError as e:
            logger.warning(f"WorkingToProceduralEvolver not available: {e}")
            # Fallback: manually create procedural items from working memory
            return self._fallback_working_to_procedural()

    def _fallback_working_to_episodic(self) -> List[str]:
        """
        Fallback method to evolve working memory to episodic memory.
        Creates episodic memories from working memory items.
        """
        try:
            # Get working memory items using SmartMemory's search interface
            working_items = self.smart_memory.search("*", memory_type="working", top_k=100)
            if not working_items:
                return []

            episodic_items = []
            for item in working_items:
                # Create episodic version of working memory item
                from smartmemory.models.memory_item import MemoryItem
                episodic_item = MemoryItem(
                    content=f"Learning session: {item.content}",
                    memory_type="episodic",
                    metadata={
                        **getattr(item, 'metadata', {}),
                        'session_type': 'learning',
                        'evolved_from': 'working',
                        'timestamp': getattr(item, 'metadata', {}).get('start_date', '')
                    }
                )
                # Add to episodic memory
                item_id = self.smart_memory.add(episodic_item)
                episodic_items.append(item_id)
                logger.info(f"Evolved working memory item to episodic: {item_id}")

            return episodic_items
        except Exception as e:
            logger.error(f"Fallback working to episodic evolution failed: {e}")
            raise

    def _fallback_working_to_procedural(self) -> List[str]:
        """
        Fallback method to evolve working memory to procedural memory.
        Creates procedural memories (skills/strategies) from working memory items.
        """
        try:
            # Get working memory items using SmartMemory's search interface
            working_items = self.smart_memory.search("*", memory_type="working", top_k=100)
            if not working_items:
                return []

            procedural_items = []
            for item in working_items:
                # Create procedural version focusing on skills/strategies
                topic = getattr(item, 'metadata', {}).get('topic', 'unknown')
                from smartmemory.models.memory_item import MemoryItem
                procedural_item = MemoryItem(
                    content=f"Teaching strategy for {topic}: Progressive learning approach",
                    memory_type="procedural",
                    metadata={
                        'skill_type': 'teaching_strategy',
                        'topic': topic,
                        'effectiveness': 0.8,
                        'evolved_from': 'working',
                        'week': getattr(item, 'metadata', {}).get('week', 0)
                    }
                )
                # Add to procedural memory
                item_id = self.smart_memory.add(procedural_item)
                procedural_items.append(item_id)
                logger.info(f"Evolved working memory item to procedural: {item_id}")

            return procedural_items
        except Exception as e:
            logger.error(f"Fallback working to procedural evolution failed: {e}")
            raise

# ------------------------------
# DAG primitives
# ------------------------------
