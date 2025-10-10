"""
StoreFactory: Unified factory for creating store instances with consistent configuration.

Provides centralized store creation, configuration validation, and dependency injection.
"""

import logging
from enum import Enum
from typing import Optional, Dict, Any, Type

from smartmemory.configuration import MemoryConfig

logger = logging.getLogger(__name__)


class StoreType(Enum):
    """Enumeration of supported store types."""
    GRAPH = "graph"
    VECTOR = "vector"
    SEMANTIC_GRAPH = "semantic_graph"
    EPISODIC_GRAPH = "episodic_graph"
    PROCEDURAL_GRAPH = "procedural_graph"
    ZETTEL_GRAPH = "zettel_graph"


class StoreFactory:
    """
    Unified factory for creating store instances.
    
    Provides consistent initialization patterns, configuration validation,
    and dependency injection for all store types.
    """

    # Registry of store type to class mappings
    _store_registry: Dict[StoreType, Type] = {}

    @classmethod
    def register_store_type(cls, store_type: StoreType, store_class: Type):
        """Register a store type with its corresponding class."""
        cls._store_registry[store_type] = store_class
        logger.debug(f"Registered store type {store_type.value} -> {store_class.__name__}")

    @classmethod
    def create_store(
            cls,
            store_type: StoreType,
            config: Optional[MemoryConfig] = None,
            **kwargs
    ) -> Any:
        """
        Create a store instance of the specified type.
        
        Args:
            store_type: The type of store to create
            config: Optional configuration object
            **kwargs: Additional arguments passed to store constructor
            
        Returns:
            Configured store instance
            
        Raises:
            ValueError: If store type is not registered
            RuntimeError: If store creation fails
        """
        if store_type not in cls._store_registry:
            available_types = [t.value for t in cls._store_registry.keys()]
            raise ValueError(f"Unknown store type: {store_type.value}. Available: {available_types}")

        store_class = cls._store_registry[store_type]

        try:
            # Validate configuration if provided
            if config:
                cls._validate_config(store_type, config)

            # Create store instance with consistent initialization
            store_instance = cls._create_store_instance(store_class, store_type, config, **kwargs)

            logger.info(f"Created {store_type.value} store instance: {store_class.__name__}")
            return store_instance

        except Exception as e:
            logger.error(f"Failed to create {store_type.value} store: {e}")
            raise RuntimeError(f"Store creation failed for {store_type.value}: {e}")

    @classmethod
    def create_graph_store(cls, config: Optional[MemoryConfig] = None, **kwargs) -> Any:
        """Create a graph store instance."""
        return cls.create_store(StoreType.GRAPH, config, **kwargs)

    @classmethod
    def create_vector_store(cls, config: Optional[MemoryConfig] = None, **kwargs) -> Any:
        """Create a vector store instance."""
        return cls.create_store(StoreType.VECTOR, config, **kwargs)

    @classmethod
    def create_semantic_graph(cls, config: Optional[MemoryConfig] = None, **kwargs) -> Any:
        """Create a semantic memory graph store instance."""
        return cls.create_store(StoreType.SEMANTIC_GRAPH, config, **kwargs)

    @classmethod
    def create_episodic_graph(cls, config: Optional[MemoryConfig] = None, **kwargs) -> Any:
        """Create an episodic memory graph store instance."""
        return cls.create_store(StoreType.EPISODIC_GRAPH, config, **kwargs)

    @classmethod
    def create_procedural_graph(cls, config: Optional[MemoryConfig] = None, **kwargs) -> Any:
        """Create a procedural memory graph store instance."""
        return cls.create_store(StoreType.PROCEDURAL_GRAPH, config, **kwargs)

    @classmethod
    def create_zettel_graph(cls, config: Optional[MemoryConfig] = None, **kwargs) -> Any:
        """Create a zettel memory graph store instance."""
        return cls.create_store(StoreType.ZETTEL_GRAPH, config, **kwargs)

    @classmethod
    def create_all_graph_stores(cls, config: Optional[MemoryConfig] = None, **kwargs) -> Dict[StoreType, Any]:
        """
        Create instances of all registered graph store types.
        
        Returns:
            Dictionary mapping store types to their instances
        """
        graph_store_types = [
            StoreType.SEMANTIC_GRAPH,
            StoreType.EPISODIC_GRAPH,
            StoreType.PROCEDURAL_GRAPH,
            StoreType.ZETTEL_GRAPH
        ]

        store_instances = {}

        for store_type in graph_store_types:
            if store_type in cls._store_registry:
                try:
                    store_instances[store_type] = cls.create_store(store_type, config, **kwargs)
                except Exception as e:
                    logger.error(f"Failed to create {store_type.value} store in batch creation: {e}")
                    # Continue with other types rather than failing completely

        logger.info(f"Created {len(store_instances)} graph store instances in batch")
        return store_instances

    @classmethod
    def get_registered_types(cls) -> list[StoreType]:
        """Get list of all registered store types."""
        return list(cls._store_registry.keys())

    @classmethod
    def is_type_registered(cls, store_type: StoreType) -> bool:
        """Check if a store type is registered."""
        return store_type in cls._store_registry

    @classmethod
    def _create_store_instance(cls, store_class: Type, store_type: StoreType, config: Optional[MemoryConfig], **kwargs) -> Any:
        """
        Create store instance with appropriate initialization pattern.
        
        Different store types may require different initialization patterns.
        """
        # Handle graph stores that need config
        if store_type in [StoreType.SEMANTIC_GRAPH, StoreType.EPISODIC_GRAPH,
                          StoreType.PROCEDURAL_GRAPH, StoreType.ZETTEL_GRAPH]:
            if config:
                return store_class(config=config, **kwargs)
            else:
                return store_class(**kwargs)

        # Handle generic graph store
        elif store_type == StoreType.GRAPH:
            if config and hasattr(config, 'graph'):
                return store_class(config.graph, **kwargs)
            else:
                return store_class(**kwargs)

        # Handle vector store
        elif store_type == StoreType.VECTOR:
            if config and hasattr(config, 'vector_store'):
                return store_class(config.vector_store, **kwargs)
            else:
                return store_class(**kwargs)

        # Default initialization
        else:
            if config:
                return store_class(config=config, **kwargs)
            else:
                return store_class(**kwargs)

    @classmethod
    def _validate_config(cls, store_type: StoreType, config: Any):
        """
        Validate configuration for a specific store type.
        
        Args:
            store_type: The store type to validate for
            config: Configuration to validate (flexible type)
            
        Raises:
            ValueError: If configuration is invalid
        """
        # Be flexible about config types - accept MemoryConfig, SimpleNamespace, dict, etc.
        if config is None:
            return  # None config is acceptable

        # Basic validation - just check that it's some kind of config object
        if not hasattr(config, '__dict__') and not isinstance(config, dict):
            raise ValueError(f"Invalid config for {store_type.value}: must be an object with attributes")

        # Optional validation for store-specific attributes
        if store_type == StoreType.GRAPH or 'graph' in store_type.value:
            if hasattr(config, 'graph'):
                logger.debug(f"Graph configuration found for {store_type.value}")
            else:
                logger.debug(f"No graph config for {store_type.value} - using defaults")

        if store_type == StoreType.VECTOR:
            if hasattr(config, 'vector_store'):
                logger.debug(f"Vector store configuration found for {store_type.value}")
            else:
                logger.debug(f"No vector store config for {store_type.value} - using defaults")


# Auto-register store types when module is imported
def _register_default_store_types():
    """Register default store types with the factory."""
    try:
        from smartmemory.graph.smartgraph import SmartGraph
        StoreFactory.register_store_type(StoreType.GRAPH, SmartGraph)
    except ImportError as e:
        logger.warning(f"Could not register SmartGraph: {e}")

    try:
        from smartmemory.stores.vector.vector_store import VectorStore
        StoreFactory.register_store_type(StoreType.VECTOR, VectorStore)
    except ImportError as e:
        logger.warning(f"Could not register VectorStore: {e}")

    try:
        from smartmemory.graph.types.semantic import SemanticMemoryGraph
        StoreFactory.register_store_type(StoreType.SEMANTIC_GRAPH, SemanticMemoryGraph)
    except ImportError as e:
        logger.warning(f"Could not register SemanticMemoryGraph: {e}")

    try:
        from smartmemory.graph.types.episodic import EpisodicMemoryGraph
        StoreFactory.register_store_type(StoreType.EPISODIC_GRAPH, EpisodicMemoryGraph)
    except ImportError as e:
        logger.warning(f"Could not register EpisodicMemoryGraph: {e}")

    try:
        from smartmemory.graph.types.procedural import ProceduralMemoryGraph
        StoreFactory.register_store_type(StoreType.PROCEDURAL_GRAPH, ProceduralMemoryGraph)
    except ImportError as e:
        logger.warning(f"Could not register ProceduralMemoryGraph: {e}")

    try:
        from smartmemory.graph.types.zettel import ZettelMemoryGraph
        StoreFactory.register_store_type(StoreType.ZETTEL_GRAPH, ZettelMemoryGraph)
    except ImportError as e:
        logger.warning(f"Could not register ZettelMemoryGraph: {e}")


# Register default types on module import
_register_default_store_types()
