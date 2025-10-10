"""
Configuration Validator

Consolidates configuration validation from:
- models.py (ConfigDict with fail-fast validation)
- config.py (various __post_init__ validations)
"""

import logging
from typing import Any, Dict, List, Optional, Set, Union

logger = logging.getLogger(__name__)


class ConfigValidator:
    """Unified configuration validation with clear error messages"""

    @staticmethod
    def validate_required_keys(config: Dict[str, Any], required_keys: List[str], path: str = "config") -> None:
        """Validate that required keys exist, failing fast with clear errors.
        
        Args:
            config: Configuration dictionary to validate
            required_keys: List of required keys
            path: Path context for error messages
            
        Raises:
            KeyError: If any required key is missing
        """
        missing_keys = []
        for key in required_keys:
            if key not in config:
                missing_keys.append(f"{path}.{key}")

        if missing_keys:
            raise KeyError(
                f"Required configuration keys are missing: {', '.join(missing_keys)}. "
                f"Please add these keys to your config.json file."
            )

    @staticmethod
    def validate_range(value: Union[int, float], min_val: Union[int, float], max_val: Union[int, float],
                       key_path: str) -> None:
        """Validate that a numeric value is within specified range.
        
        Args:
            value: Value to validate
            min_val: Minimum allowed value (inclusive)
            max_val: Maximum allowed value (inclusive)
            key_path: Configuration key path for error messages
            
        Raises:
            ValueError: If value is outside the valid range
        """
        if not min_val <= value <= max_val:
            raise ValueError(
                f"Configuration key '{key_path}' must be between {min_val} and {max_val}, "
                f"got {value}"
            )

    @staticmethod
    def validate_choice(value: Any, valid_choices: Set[Any], key_path: str, allow_custom: bool = False) -> None:
        """Validate that a value is one of the allowed choices.
        
        Args:
            value: Value to validate
            valid_choices: Set of valid choices
            key_path: Configuration key path for error messages
            allow_custom: Whether to allow custom values not in valid_choices
            
        Raises:
            ValueError: If value is not in valid choices and allow_custom is False
        """
        if value not in valid_choices:
            if not allow_custom:
                raise ValueError(
                    f"Configuration key '{key_path}' must be one of {valid_choices}, "
                    f"got '{value}'"
                )
            else:
                logger.warning(
                    f"Configuration key '{key_path}' has custom value '{value}'. "
                    f"Valid choices are: {valid_choices}"
                )

    @staticmethod
    def validate_storage_config(config: Dict[str, Any], path: str = "storage") -> None:
        """Validate storage configuration section.
        
        Args:
            config: Storage configuration dictionary
            path: Path context for error messages
        """
        if "storage_strategy" in config:
            ConfigValidator.validate_choice(
                config["storage_strategy"],
                {"dual_node", "single_node", "memory_only"},
                f"{path}.storage_strategy"
            )

    @staticmethod
    def validate_grounding_config(config: Dict[str, Any], path: str = "grounding") -> None:
        """Validate grounding configuration section.
        
        Args:
            config: Grounding configuration dictionary
            path: Path context for error messages
        """
        if "confidence_threshold" in config:
            ConfigValidator.validate_range(
                config["confidence_threshold"],
                0.0, 1.0,
                f"{path}.confidence_threshold"
            )

        if "grounding_strategy" in config:
            ConfigValidator.validate_choice(
                config["grounding_strategy"],
                {"wikipedia", "custom", "hybrid", "none"},
                f"{path}.grounding_strategy"
            )

    @staticmethod
    def validate_linking_config(config: Dict[str, Any], path: str = "linking") -> None:
        """Validate linking configuration section.
        
        Args:
            config: Linking configuration dictionary
            path: Path context for error messages
        """
        if "similarity_threshold" in config:
            ConfigValidator.validate_range(
                config["similarity_threshold"],
                0.0, 1.0,
                f"{path}.similarity_threshold"
            )

    @staticmethod
    def validate_extraction_config(config: Dict[str, Any], path: str = "extraction") -> None:
        """Validate extraction configuration section.
        
        Args:
            config: Extraction configuration dictionary
            path: Path context for error messages
        """
        if "extractor_name" in config and config["extractor_name"] is not None:
            ConfigValidator.validate_choice(
                config["extractor_name"],
                {"llm", "spacy", "gliner", "relik", "ontology"},
                f"{path}.extractor_name",
                allow_custom=True
            )

    @staticmethod
    def validate_classification_config(config: Dict[str, Any], path: str = "classification") -> None:
        """Validate classification configuration section.
        
        Args:
            config: Classification configuration dictionary
            path: Path context for error messages
        """
        # Set default content indicators if none provided
        if "content_indicators" not in config or not config["content_indicators"]:
            config["content_indicators"] = {
                'episodic': ['event', 'happened', 'occurred', 'experience'],
                'semantic': ['definition', 'concept', 'knowledge', 'fact'],
                'procedural': ['how to', 'steps', 'process', 'method'],
                'zettel': ['note', 'idea', 'thought', 'reminder']
            }

    @staticmethod
    def validate_input_adapter_config(config: Dict[str, Any], path: str = "input_adapter") -> None:
        """Validate input adapter configuration section.
        
        Args:
            config: Input adapter configuration dictionary
            path: Path context for error messages
        """
        if "adapter_name" in config:
            ConfigValidator.validate_choice(
                config["adapter_name"],
                {"default", "text", "dict", "json", "file"},
                f"{path}.adapter_name",
                allow_custom=True
            )

    @staticmethod
    def validate_pipeline_config(config: Dict[str, Any], path: str = "pipeline") -> None:
        """Validate entire pipeline configuration bundle.
        
        Args:
            config: Pipeline configuration dictionary
            path: Path context for error messages
        """
        # Validate individual pipeline stage configurations
        if "input_adapter" in config:
            ConfigValidator.validate_input_adapter_config(
                config["input_adapter"],
                f"{path}.input_adapter"
            )

        if "classification" in config:
            ConfigValidator.validate_classification_config(
                config["classification"],
                f"{path}.classification"
            )

        if "extraction" in config:
            ConfigValidator.validate_extraction_config(
                config["extraction"],
                f"{path}.extraction"
            )

        if "storage" in config:
            ConfigValidator.validate_storage_config(
                config["storage"],
                f"{path}.storage"
            )

        if "linking" in config:
            ConfigValidator.validate_linking_config(
                config["linking"],
                f"{path}.linking"
            )

        if "grounding" in config:
            ConfigValidator.validate_grounding_config(
                config["grounding"],
                f"{path}.grounding"
            )

    @staticmethod
    def validate_connection_config(config: Dict[str, Any], service_name: str,
                                   required_keys: Optional[List[str]] = None) -> None:
        """Validate service connection configuration.
        
        Args:
            config: Service configuration dictionary
            service_name: Name of the service (for error messages)
            required_keys: Optional list of required keys
            
        Raises:
            KeyError: If required keys are missing
        """
        if required_keys:
            ConfigValidator.validate_required_keys(
                config,
                required_keys,
                f"config.{service_name}"
            )

        # Common connection validations
        if "host" in config and not config["host"]:
            raise ValueError(f"Configuration key 'config.{service_name}.host' cannot be empty")

        if "port" in config:
            try:
                port = int(config["port"])
                if not 1 <= port <= 65535:
                    raise ValueError(
                        f"Configuration key 'config.{service_name}.port' must be between 1 and 65535, "
                        f"got {port}"
                    )
            except (ValueError, TypeError):
                raise ValueError(
                    f"Configuration key 'config.{service_name}.port' must be a valid integer, "
                    f"got {config['port']}"
                )


class ValidatedConfigDict(dict):
    """
    A dictionary subclass that provides clear error messages for missing keys
    and supports both bracket notation and property syntax for configuration access.
    
    Consolidates functionality from models.py ConfigDict.
    """

    def __init__(self, data: Dict[str, Any], path: str = "config"):
        """
        Initialize ValidatedConfigDict with data and path context for error messages.
        
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
            The configuration value, wrapped in ValidatedConfigDict if it's a dict
            
        Raises:
            KeyError: With clear message indicating the missing config path
        """
        if key not in self:
            raise KeyError(
                f"Required configuration key '{self._path}.{key}' is missing. "
                f"Please add this key to your config.json file."
            )

        value = super().__getitem__(key)

        # If the value is a dictionary, wrap it in ValidatedConfigDict for nested access
        if isinstance(value, dict):
            return ValidatedConfigDict(value, f"{self._path}.{key}")

        return value

    def __getattr__(self, key: str) -> Any:
        """
        Support property syntax for configuration access.
        
        This allows config.cache.redis.host instead of config["cache"]["redis"]["host"]
        
        Args:
            key: The configuration key to retrieve
            
        Returns:
            The configuration value, wrapped in ValidatedConfigDict if it's a dict
            
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
            The configuration value or default, wrapped in ValidatedConfigDict if it's a dict
        """
        value = super().get(key, default)

        # If the value is a dictionary, wrap it in ValidatedConfigDict for nested access
        if isinstance(value, dict):
            return ValidatedConfigDict(value, f"{self._path}.{key}")

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
        ConfigValidator.validate_required_keys(dict(self), list(keys), self._path)

    def __repr__(self) -> str:
        """String representation showing the path context."""
        return f"ValidatedConfigDict({self._path}): {dict(self)}"
