"""
ConfigDict - Fail-fast configuration dictionary with clear error messages.

This module provides a dictionary subclass that gives clear error messages
when required configuration keys are missing, replacing complex defensive
patterns with simple direct access.
"""

from typing import Any, Dict


class ConfigDict(dict):
    """
    A dictionary subclass that provides clear error messages for missing keys
    and supports both bracket notation and property syntax for configuration access.
    
    Instead of complex patterns like:
        redis_cfg = (cfg.get("cache") or {} or {}).get("redis") or {}
        host = str(redis_cfg.get("host") or os.getenv('REDIS_HOST', 'localhost'))
    
    Use simple direct access:
        host = config["cache"]["redis"]["host"]  # Crashes with clear error if missing
        host = config.cache.redis.host          # Property syntax also works
        timeout = config.cache.redis.get("timeout", 30)  # Optional with explicit default
    """

    def __init__(self, data: Dict[str, Any], path: str = "config"):
        """
        Initialize ConfigDict with data and path context for error messages.
        
        Args:
            data: The configuration dictionary data
            path: The path context for error messages (e.g., "config.cache.redis")
        """
        super().__init__(data)
        self._path = path

    def __getitem__(self, key: str) -> Any:
        """
        Get an item with clear error message if key is missing.
        
        Args:
            key: The configuration key to retrieve
            
        Returns:
            The configuration value, wrapped in ConfigDict if it's a dict
            
        Raises:
            KeyError: With clear message indicating the missing config path
        """
        if key not in self:
            raise KeyError(
                f"Required configuration key '{self._path}.{key}' is missing. "
                f"Please add this key to your config.json file."
            )

        value = super().__getitem__(key)

        # If the value is a dictionary, wrap it in ConfigDict for nested access
        if isinstance(value, dict):
            return ConfigDict(value, f"{self._path}.{key}")

        return value

    def __getattr__(self, key: str) -> Any:
        """
        Support property syntax for configuration access.
        
        This allows config.cache.redis.host instead of config["cache"]["redis"]["host"]
        
        Args:
            key: The configuration key to retrieve
            
        Returns:
            The configuration value, wrapped in ConfigDict if it's a dict
            
        Raises:
            AttributeError: With clear message indicating the missing config path
        """
        if key.startswith('_'):
            # Don't interfere with internal attributes
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")

        try:
            return self[key]
        except KeyError:
            raise AttributeError(
                f"Required configuration key '{self._path}.{key}' is missing. "
                f"Please add this key to your config.json file."
            )

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get an optional configuration value with explicit default.
        
        This method is for optional configuration keys where you want to provide
        an explicit default value.
        
        Args:
            key: The configuration key to retrieve
            default: Default value if key is missing
            
        Returns:
            The configuration value or default, wrapped in ConfigDict if it's a dict
        """
        value = super().get(key, default)

        # If the value is a dictionary, wrap it in ConfigDict for nested access
        if isinstance(value, dict):
            return ConfigDict(value, f"{self._path}.{key}")

        return value

    def require(self, *keys: str) -> None:
        """
        Validate that required keys exist, failing fast with clear errors.
        
        This method can be used at startup to validate configuration
        before the application begins processing.
        
        Args:
            *keys: Required configuration keys to validate
            
        Raises:
            KeyError: If any required key is missing
        """
        missing_keys = []
        for key in keys:
            if key not in self:
                missing_keys.append(f"{self._path}.{key}")

        if missing_keys:
            raise KeyError(
                f"Required configuration keys are missing: {', '.join(missing_keys)}. "
                f"Please add these keys to your config.json file."
            )

    def __repr__(self) -> str:
        """String representation showing the path context."""
        return f"ConfigDict({self._path}): {dict(self)}"


def create_config_dict(data: Dict[str, Any], path: str = "config") -> ConfigDict:
    """
    Factory function to create a ConfigDict from configuration data.
    
    Args:
        data: The configuration dictionary data
        path: The path context for error messages
        
    Returns:
        A ConfigDict instance
    """
    return ConfigDict(data, path)
