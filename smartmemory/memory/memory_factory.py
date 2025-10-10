"""
MemoryFactory: Unified factory for creating memory instances with consistent configuration.

Provides centralized memory creation, configuration validation, and dependency injection.
"""

import logging
from enum import Enum
from typing import Optional, Dict, Any, Type

from smartmemory.configuration import MemoryConfig
from smartmemory.memory.base import MemoryBase

logger = logging.getLogger(__name__)


class MemoryType(Enum):
    """Enumeration of supported memory types."""
    SEMANTIC = "semantic"
    EPISODIC = "episodic"
    PROCEDURAL = "procedural"
    WORKING = "working"
    ZETTEL = "zettel"


class MemoryFactory:
    """
    Unified factory for creating memory instances.
    
    Provides consistent initialization patterns, configuration validation,
    and dependency injection for all memory types.
    """

    # Registry of memory type to class mappings
    _memory_registry: Dict[MemoryType, Type[MemoryBase]] = {}

    @classmethod
    def register_memory_type(cls, memory_type: MemoryType, memory_class: Type[MemoryBase]):
        """Register a memory type with its corresponding class."""
        cls._memory_registry[memory_type] = memory_class
        logger.debug(f"Registered memory type {memory_type.value} -> {memory_class.__name__}")

    @classmethod
    def create_memory(
            cls,
            memory_type: MemoryType,
            config: Optional[MemoryConfig] = None,
            **kwargs
    ) -> MemoryBase:
        """
        Create a memory instance of the specified type.
        
        Args:
            memory_type: The type of memory to create
            config: Optional configuration object
            **kwargs: Additional arguments passed to memory constructor
            
        Returns:
            Configured memory instance
            
        Raises:
            ValueError: If memory type is not registered
            RuntimeError: If memory creation fails
        """
        if memory_type not in cls._memory_registry:
            available_types = [t.value for t in cls._memory_registry.keys()]
            raise ValueError(f"Unknown memory type: {memory_type.value}. Available: {available_types}")

        memory_class = cls._memory_registry[memory_type]

        try:
            # Validate configuration if provided
            if config:
                cls._validate_config(memory_type, config)

            # Create memory instance with appropriate initialization pattern
            memory_instance = cls._create_memory_instance(memory_class, memory_type, config, **kwargs)

            logger.info(f"Created {memory_type.value} memory instance: {memory_class.__name__}")
            return memory_instance

        except Exception as e:
            logger.error(f"Failed to create {memory_type.value} memory: {e}")
            raise RuntimeError(f"Memory creation failed for {memory_type.value}: {e}")

    @classmethod
    def create_semantic_memory(cls, config: Optional[MemoryConfig] = None, **kwargs) -> MemoryBase:
        """Create a semantic memory instance."""
        return cls.create_memory(MemoryType.SEMANTIC, config, **kwargs)

    @classmethod
    def create_episodic_memory(cls, config: Optional[MemoryConfig] = None, **kwargs) -> MemoryBase:
        """Create an episodic memory instance."""
        return cls.create_memory(MemoryType.EPISODIC, config, **kwargs)

    @classmethod
    def create_procedural_memory(cls, config: Optional[MemoryConfig] = None, **kwargs) -> MemoryBase:
        """Create a procedural memory instance."""
        return cls.create_memory(MemoryType.PROCEDURAL, config, **kwargs)

    @classmethod
    def create_working_memory(cls, config: Optional[MemoryConfig] = None, **kwargs) -> MemoryBase:
        """Create a working memory instance."""
        return cls.create_memory(MemoryType.WORKING, config, **kwargs)

    @classmethod
    def create_zettel_memory(cls, config: Optional[MemoryConfig] = None, **kwargs) -> MemoryBase:
        """Create a zettel memory instance."""
        return cls.create_memory(MemoryType.ZETTEL, config, **kwargs)

    @classmethod
    def create_all_memory_types(cls, config: Optional[MemoryConfig] = None, **kwargs) -> Dict[MemoryType, MemoryBase]:
        """
        Create instances of all registered memory types.
        
        Returns:
            Dictionary mapping memory types to their instances
        """
        memory_instances = {}

        for memory_type in cls._memory_registry.keys():
            try:
                memory_instances[memory_type] = cls.create_memory(memory_type, config, **kwargs)
            except Exception as e:
                logger.error(f"Failed to create {memory_type.value} memory in batch creation: {e}")
                # Continue with other types rather than failing completely

        logger.info(f"Created {len(memory_instances)} memory instances in batch")
        return memory_instances

    @classmethod
    def get_registered_types(cls) -> list[MemoryType]:
        """Get list of all registered memory types."""
        return list(cls._memory_registry.keys())

    @classmethod
    def is_type_registered(cls, memory_type: MemoryType) -> bool:
        """Check if a memory type is registered."""
        return memory_type in cls._memory_registry

    @classmethod
    def _create_memory_instance(cls, memory_class: Type[MemoryBase], memory_type: MemoryType, config: Optional[MemoryConfig], **kwargs) -> MemoryBase:
        """
        Create memory instance with appropriate initialization pattern.
        
        Different memory types may require different initialization patterns.
        """
        # Try different initialization patterns based on memory class requirements
        try:
            # First try with config parameter
            if config:
                return memory_class(config=config, **kwargs)
            else:
                return memory_class(**kwargs)
        except TypeError as e:
            if "config" in str(e):
                # If config parameter not accepted, try without it
                try:
                    return memory_class(**kwargs)
                except Exception as e2:
                    logger.error(f"Failed both initialization patterns for {memory_type.value}: {e}, {e2}")
                    raise e2
            else:
                raise e

    @classmethod
    def _validate_config(cls, memory_type: MemoryType, config: Any):
        """
        Validate configuration for a specific memory type.
        
        Args:
            memory_type: The memory type to validate for
            config: Configuration to validate (flexible type)
            
        Raises:
            ValueError: If configuration is invalid
        """
        # Be flexible about config types - accept MemoryConfig, SimpleNamespace, dict, etc.
        if config is None:
            return  # None config is acceptable

        # Basic validation - just check that it's some kind of config object
        if not hasattr(config, '__dict__') and not isinstance(config, dict):
            raise ValueError(f"Invalid config for {memory_type.value}: must be an object with attributes")

        # Optional validation for common config attributes
        if hasattr(config, 'graph') or hasattr(config, 'vector_store'):
            logger.debug(f"Configuration for {memory_type.value} has expected attributes")
        else:
            logger.debug(f"Configuration for {memory_type.value} may be minimal - this is acceptable")


# Auto-register memory types when module is imported
def _register_default_memory_types():
    """Register default memory types with the factory."""
    try:
        from smartmemory.memory.types.semantic_memory import SemanticMemory
        MemoryFactory.register_memory_type(MemoryType.SEMANTIC, SemanticMemory)
    except ImportError as e:
        logger.warning(f"Could not register SemanticMemory: {e}")

    try:
        from smartmemory.memory.types.episodic_memory import EpisodicMemory
        MemoryFactory.register_memory_type(MemoryType.EPISODIC, EpisodicMemory)
    except ImportError as e:
        logger.warning(f"Could not register EpisodicMemory: {e}")

    try:
        from smartmemory.memory.types.procedural_memory import ProceduralMemory
        MemoryFactory.register_memory_type(MemoryType.PROCEDURAL, ProceduralMemory)
    except ImportError as e:
        logger.warning(f"Could not register ProceduralMemory: {e}")

    try:
        from smartmemory.memory.types.working_memory import WorkingMemory
        MemoryFactory.register_memory_type(MemoryType.WORKING, WorkingMemory)
    except ImportError as e:
        logger.warning(f"Could not register WorkingMemory: {e}")

    try:
        from smartmemory.memory.types.zettel_memory import ZettelMemory
        MemoryFactory.register_memory_type(MemoryType.ZETTEL, ZettelMemory)
    except ImportError as e:
        logger.warning(f"Could not register ZettelMemory: {e}")


# Register default types on module import
_register_default_memory_types()
