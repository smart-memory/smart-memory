"""
Unified Configuration Class

Consolidates all configuration classes into a single unified interface:
- MemoryConfig (from config_loader.py)
- MemoryConfig (from __init__.py)
- MemoryConfig (from service_common/__init__.py)
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from smartmemory.models.base import MemoryBaseModel
from .manager import ConfigManager
from .validator import ValidatedConfigDict

logger = logging.getLogger(__name__)


@dataclass
class MemoryConfig(MemoryBaseModel):
    """
    Unified configuration class that consolidates all scattered config implementations.
    
    Provides backward compatibility with:
    - MemoryConfig (config_loader.py)
    - MemoryConfig (__init__.py)
    - MemoryConfig (service_common/__init__.py)
    """

    # Configuration path as a dataclass field
    config_path: Optional[str] = field(default=None)

    # Internal fields (will be set in __post_init__)
    _manager: Optional[ConfigManager] = field(default=None, init=False)
    _config: Dict[str, Any] = field(default_factory=dict, init=False)

    # Configuration section fields
    semantic: Dict[str, Any] = field(default_factory=dict, init=False)
    episodic: Dict[str, Any] = field(default_factory=dict, init=False)
    procedural: Dict[str, Any] = field(default_factory=dict, init=False)
    working: Dict[str, Any] = field(default_factory=dict, init=False)
    zettel: Dict[str, Any] = field(default_factory=dict, init=False)
    vector: Dict[str, Any] = field(default_factory=dict, init=False)
    cache: Dict[str, Any] = field(default_factory=dict, init=False)
    graph_db: Dict[str, Any] = field(default_factory=dict, init=False)
    extractor: Dict[str, Any] = field(default_factory=dict, init=False)
    enricher: Dict[str, Any] = field(default_factory=dict, init=False)
    adapter: Dict[str, Any] = field(default_factory=dict, init=False)
    converter: Dict[str, Any] = field(default_factory=dict, init=False)
    observability: Dict[str, Any] = field(default_factory=dict, init=False)
    background: Dict[str, Any] = field(default_factory=dict, init=False)
    ontology: Dict[str, Any] = field(default_factory=dict, init=False)
    auth: Dict[str, Any] = field(default_factory=dict, init=False)
    mongodb: Dict[str, Any] = field(default_factory=dict, init=False)
    analytics: Dict[str, Any] = field(default_factory=dict, init=False)
    active_namespace: Optional[str] = field(default=None, init=False)

    def __post_init__(self):
        """Initialize unified configuration after dataclass initialization"""
        # Call parent __post_init__ if it exists
        super().__post_init__() if hasattr(super(), '__post_init__') else None

        self._manager = ConfigManager(self.config_path)
        self._config = self._manager.get_config()

        # Initialize all section attributes for backward compatibility
        self._initialize_sections()

        # Validate configuration
        try:
            self._manager.validate_config()
        except Exception as e:
            logger.warning(f"Configuration validation failed: {e}")

    def _initialize_sections(self) -> None:
        """Initialize configuration section attributes for backward compatibility"""
        # Core memory sections (from MemoryConfig)
        self.semantic = self._config.get("semantic") or {}
        self.episodic = self._config.get("episodic") or {}
        self.procedural = self._config.get("procedural") or {}
        self.working = self._config.get("working") or {}
        self.zettel = self._config.get("zettel") or {}

        # Infrastructure sections
        self.vector = self._config.get("vector") or {}
        self.cache = self._config.get("cache") or {}
        self.graph_db = self._config.get("graph_db") or {}

        # Pipeline sections
        self.extractor = self._config.get("extractor") or {}
        self.enricher = self._config.get("enricher") or {}
        self.adapter = self._config.get("adapter") or {}
        self.converter = self._config.get("converter") or {}

        # System sections
        self.observability = self._config.get("observability") or {}
        self.background = self._config.get("background") or {}

        # Legacy ontology section (from MemoryConfig)
        self.ontology = self._config.get("ontology") or {}

        # Platform sections (from MemoryConfig)
        self.auth = self._config.get("auth") or {}
        self.mongodb = self._config.get("mongodb") or {}
        self.analytics = self._config.get("analytics") or {}

        # Active namespace for introspection
        self.active_namespace = self._manager.active_namespace

    def reload_if_stale(self, force: bool = False) -> None:
        """Reload configuration if file has changed
        
        Args:
            force: Force reload even if file hasn't changed
        """
        self._manager.reload_if_stale(force)
        self._config = self._manager.get_config()
        self._initialize_sections()

    def get_store_config(self, store_name: str) -> Dict[str, Any]:
        """Get store configuration, falling back to graph_db if store config is empty
        
        Args:
            store_name: Name of the store
            
        Returns:
            Store configuration dictionary
        """
        return self._manager.get_store_config(store_name)

    def get_section(self, section_name: str, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get a specific configuration section
        
        Args:
            section_name: Name of the configuration section
            default: Default value if section doesn't exist
            
        Returns:
            Configuration section dictionary
        """
        return self._manager.get_section(section_name, default)

    def get_validated_config(self) -> ValidatedConfigDict:
        """Get configuration wrapped for fail-fast access
        
        Returns:
            ValidatedConfigDict instance with clear error messages
        """
        return self._manager.get_validated_config()

    def validate(self) -> None:
        """Validate the current configuration
        
        Raises:
            KeyError: If required keys are missing
            ValueError: If configuration values are invalid
        """
        self._manager.validate_config()

    @property
    def resolved_config_path(self) -> str:
        """Get the resolved configuration file path"""
        return self._manager.config_path

    # Dictionary-like access for backward compatibility
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with default
        
        Args:
            key: Configuration key
            default: Default value if key doesn't exist
            
        Returns:
            Configuration value or default
        """
        return self._config.get(key, default)

    def __getitem__(self, key: str) -> Any:
        """Dictionary-style access to configuration
        
        Args:
            key: Configuration key
            
        Returns:
            Configuration value
            
        Raises:
            KeyError: If key doesn't exist
        """
        return self._config[key]

    def __contains__(self, key: str) -> bool:
        """Check if configuration key exists
        
        Args:
            key: Configuration key to check
            
        Returns:
            True if key exists, False otherwise
        """
        return key in self._config

    def keys(self):
        """Get configuration keys"""
        return self._config.keys()

    def values(self):
        """Get configuration values"""
        return self._config.values()

    def items(self):
        """Get configuration items"""
        return self._config.items()

    def __repr__(self) -> str:
        """String representation of configuration"""
        return f"MemoryConfig(path={self.config_path}, namespace={self.active_namespace})"


# --- Convenience helpers (concise API) ---

def get_validated_config() -> ValidatedConfigDict:
    """Return the full validated configuration with attribute access and fail-fast semantics.

    Usage:
        cfg = get_validated_config()
        mongo = cfg["mongodb"]  # or cfg.mongodb
    """
    return ConfigManager().get_validated_config()


def get_config(section: str) -> ValidatedConfigDict:
    """Return a validated configuration section directly.

    Example:
        mongo = get_config("mongodb")
        mongo.require("uri", "database")
        uri = mongo.uri
    """
    cfg = get_validated_config()
    return cfg[section]


# Back-compat alias (deprecated): use get_config()
def get_section(section: str) -> ValidatedConfigDict:  # pragma: no cover
    return get_config(section)


# --- One-liner: resolve overrides against config ---
class _SectionOverlay:
    """Lightweight overlay that prefers overrides, falls back to base config.

    Provides attribute and item access like the underlying ValidatedConfigDict.
    """

    def __init__(self, base: ValidatedConfigDict, overrides: Dict[str, Any]):
        self._base = base
        self._over = {k: v for k, v in overrides.items() if v is not None}

    def __getattr__(self, name: str) -> Any:
        if name in self._over:
            return self._over[name]
        return getattr(self._base, name)

    def __getitem__(self, key: str) -> Any:
        if key in self._over:
            return self._over[key]
        return self._base[key]

    def get(self, key: str, default: Any = None) -> Any:
        if key in self._over:
            return self._over[key]
        return self._base.get(key, default)


def resolve_config(section: str, required: tuple[str, ...] = (), **overrides: Any) -> _SectionOverlay:
    """Return a section view where provided overrides take precedence over config.

    - required: keys that must be present either via overrides or config.
    - Overrides with value None are ignored (treated as not provided).
    """
    base = get_config(section)
    missing = tuple(k for k in required if k not in overrides or overrides[k] is None)
    if missing:
        base.require(*missing)
    return _SectionOverlay(base, overrides)
