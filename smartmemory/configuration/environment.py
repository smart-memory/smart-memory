"""
Environment Variable Handler

Consolidates environment variable handling and expansion from:
- config_loader.py (_env_override_leaf_keys, _expand_env_string, _expand_env_in_obj)
"""

import json
import logging
import os
import re
from typing import Any, Dict

logger = logging.getLogger(__name__)

# Environment variable pattern for ${VAR} and ${VAR:-default}
_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")


class EnvironmentHandler:
    """Unified environment variable handling and expansion"""

    @staticmethod
    def load_dotenv():
        """Load .env file if available (non-fatal if dotenv not installed)"""
        try:
            from dotenv import load_dotenv
            load_dotenv()  # Does not override existing env by default
            logger.debug("Loaded .env file")
        except ImportError:
            logger.debug("python-dotenv not available, skipping .env file")
        except Exception as e:
            logger.warning(f"Failed to load .env file: {e}")

    @staticmethod
    def expand_env_string(s: str) -> Any:
        """Expand ${VAR} and ${VAR:-default} in a string.
        
        If the entire string is a single placeholder, attempt to auto-cast
        to int/float/bool/null or JSON (for objects/arrays).
        
        Args:
            s: String to expand
            
        Returns:
            Expanded value with appropriate type casting
        """
        if not isinstance(s, str):
            return s

        # Detect if the whole string is exactly one placeholder
        whole_match = re.fullmatch(_ENV_PATTERN, s)

        def repl(m: re.Match) -> str:
            name = m.group(1)
            default = m.group(2)
            return os.environ.get(name, default or "")

        expanded = _ENV_PATTERN.sub(repl, s)

        if whole_match:
            # Attempt simple auto-casting when the value is entirely the placeholder
            v = expanded.strip()
            # Try JSON literals
            if v.lower() in {"true", "false"}:
                return v.lower() == "true"
            if v.lower() in {"null", "none"}:
                return None
            # Try int / float
            try:
                if v.isdigit() or (v.startswith("-") and v[1:].isdigit()):
                    return int(v)
                return float(v)
            except ValueError:
                pass
            # Try object/array JSON
            if (v.startswith("{") and v.endswith("}")) or (v.startswith("[") and v.endswith("]")):
                try:
                    return json.loads(v)
                except Exception:
                    pass
        return expanded

    @classmethod
    def expand_env_in_obj(cls, obj: Any) -> Any:
        """Recursively expand environment variables in strings within dict/list structures.
        
        Args:
            obj: Object to expand (dict, list, str, or other)
            
        Returns:
            Object with environment variables expanded
        """
        if isinstance(obj, dict):
            return {k: cls.expand_env_in_obj(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [cls.expand_env_in_obj(v) for v in obj]
        if isinstance(obj, str):
            return cls.expand_env_string(obj)
        return obj

    @staticmethod
    def override_leaf_keys(d: Dict[str, Any], prefix: str = None) -> Dict[str, Any]:
        """Recursively override leaf keys in dict d with env vars of form PREFIX_KEY.
        
        This preserves the legacy leaf-key environment override behavior.
        
        Args:
            d: Dictionary to process
            prefix: Environment variable prefix
            
        Returns:
            Dictionary with leaf keys overridden by environment variables
        """
        out = {}
        for k, v in d.items():
            env_key = f"{prefix}_{k}" if prefix else k
            if isinstance(v, dict):
                out[k] = EnvironmentHandler.override_leaf_keys(v, env_key.upper())
            else:
                env_val = os.environ.get(env_key.upper())
                out[k] = env_val if env_val is not None else v
        return out

    @classmethod
    def process_config_dict(cls, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Process configuration dictionary with full environment handling.
        
        Args:
            config_dict: Raw configuration dictionary
            
        Returns:
            Processed configuration with environment expansion and overrides
        """
        # First expand ${VAR} and ${VAR:-default} inside string values
        config_expanded = cls.expand_env_in_obj(config_dict)

        # Then apply legacy leaf-key env overrides (UPPER_CASE paths)
        processed = cls.override_leaf_keys(config_expanded)

        return processed

    @staticmethod
    def resolve_config_path(path_candidate: str) -> str:
        """Find an existing config file based on candidate and fallbacks.
        
        Args:
            path_candidate: Primary config file path
            
        Returns:
            Absolute path to config file (may not exist)
        """
        # Expand env vars and user tilde in the provided path
        primary = os.path.expanduser(os.path.expandvars(path_candidate))

        if os.path.exists(primary):
            return os.path.abspath(primary)

        basename = os.path.basename(primary)
        current_path = os.path.join(os.getcwd(), basename)
        if os.path.exists(current_path):
            return os.path.abspath(current_path)

        parent_path = os.path.join(os.path.dirname(os.getcwd()), basename)
        if os.path.exists(parent_path):
            return os.path.abspath(parent_path)

        logger.warning(f"Could not find config file at {primary}, {current_path}, or {parent_path}. Using empty config.")
        # Return absolute path even if missing to keep env/export consistent
        return os.path.abspath(primary)

    @staticmethod
    def set_config_path_env(resolved_path: str):
        """Set the resolved config path in environment for process-wide consistency.
        
        Args:
            resolved_path: Absolute path to config file
        """
        os.environ['SMARTMEMORY_CONFIG'] = resolved_path
        logger.debug(f"Set SMARTMEMORY_CONFIG={resolved_path}")

    @staticmethod
    def get_namespace() -> str:
        """Get active namespace from environment or config.
        
        Returns:
            Active namespace name or None
        """
        return os.environ.get("TEST_NAMESPACE")
