"""
Pipeline utility functions for reducing code duplication.
Leverages existing ConfigDict infrastructure instead of rebuilding.
"""
from typing import Dict, Any

from smartmemory.configuration.models import ConfigDict
from smartmemory.memory.pipeline.components import ComponentResult


def create_error_result(stage: str, error: Exception, **additional_metadata) -> ComponentResult:
    """
    Create standardized error ComponentResult for pipeline stages.
    
    Args:
        stage: Component stage name (e.g., 'linking_engine', 'storage_engine')
        error: The exception that occurred
        **additional_metadata: Additional metadata fields to include
        
    Returns:
        ComponentResult with standardized error format
    """
    metadata = {
        'stage': stage,
        'error_type': type(error).__name__,
        **additional_metadata
    }

    return ComponentResult(
        success=False,
        data={'error': str(error)},
        metadata=metadata
    )


def wrap_config(config: Dict[str, Any], component_name: str) -> ConfigDict:
    """
    Wrap raw config dict in ConfigDict for fail-fast access.
    
    Args:
        config: Raw configuration dictionary
        component_name: Name of component for error context
        
    Returns:
        ConfigDict with proper error context
        
    Raises:
        TypeError: If config is not a dictionary
    """
    if not isinstance(config, dict):
        raise TypeError(f"{component_name} config must be a dictionary, got {type(config).__name__}")

    return ConfigDict(config, f"{component_name}.config")


def validate_required_config(config_dict: ConfigDict, *required_keys: str) -> None:
    """
    Validate that required configuration keys exist using ConfigDict.require().
    
    Args:
        config_dict: ConfigDict instance
        *required_keys: Required configuration keys
        
    Raises:
        KeyError: If any required keys are missing (from ConfigDict.require())
    """
    config_dict.require(*required_keys)
