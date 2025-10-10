# Plugin module for agentic memory (extractors, enrichers, etc.)

# Export plugin system components
from smartmemory.plugins.base import (
    PluginBase,
    PluginMetadata,
    EnricherPlugin,
    EvolverPlugin,
    ExtractorPlugin,
    EmbeddingProviderPlugin,
    GrounderPlugin,
    validate_plugin_class,
    check_version_compatibility,
)

from .registry import (
    PluginRegistry,
    get_plugin_registry,
    reset_plugin_registry,
)

from .manager import (
    PluginManager,
    get_plugin_manager,
    reset_plugin_manager,
)

__all__ = [
    # Base classes
    'PluginBase',
    'PluginMetadata',
    'EnricherPlugin',
    'EvolverPlugin',
    'ExtractorPlugin',
    'EmbeddingProviderPlugin',
    'GrounderPlugin',
    
    # Registry
    'PluginRegistry',
    'get_plugin_registry',
    'reset_plugin_registry',
    
    # Manager
    'PluginManager',
    'get_plugin_manager',
    'reset_plugin_manager',
    
    # Utilities
    'validate_plugin_class',
    'check_version_compatibility',
]
