"""
Core prompts loader - pure file-based JSON loading with environment variable expansion.
No MongoDB, no workspace overrides, no multi-tenancy. Just clean file loading.
"""
import json
import logging
import os
import re
import threading
from typing import Any, Dict

logger = logging.getLogger(__name__)

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")


def _expand_env_string(s: str) -> Any:
    """Expand ${VAR} and ${VAR:-default} in a string.
    If the entire string is a single placeholder, attempt to auto-cast
    to int/float/bool/null or JSON (for objects/arrays).
    """
    if not isinstance(s, str):
        return s

    whole_match = re.fullmatch(_ENV_PATTERN, s)

    def repl(m: re.Match) -> str:
        name = m.group(1)
        default = m.group(2)
        return os.environ.get(name, default or "")

    expanded = _ENV_PATTERN.sub(repl, s)

    if whole_match:
        v = expanded.strip()
        if v.lower() in {"true", "false"}:
            return v.lower() == "true"
        if v.lower() in {"null", "none"}:
            return None
        try:
            if v.isdigit() or (v.startswith("-") and v[1:].isdigit()):
                return int(v)
            return float(v)
        except ValueError:
            pass
        if (v.startswith("{") and v.endswith("}")) or (v.startswith("[") and v.endswith("]")):
            try:
                return json.loads(v)
            except Exception:
                pass
    return expanded


def _expand_env_in_obj(obj: Any) -> Any:
    """Recursively expand environment variables in strings within dict/list structures."""
    if isinstance(obj, dict):
        return {k: _expand_env_in_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env_in_obj(v) for v in obj]
    if isinstance(obj, str):
        return _expand_env_string(obj)
    return obj


class PromptsConfig:
    """Loader for prompts configuration stored in a separate JSON file.

    Resolution order for the file path:
    - Explicit prompts_path arg
    - $SMARTMEMORY_PROMPTS environment variable
    - 'prompts.json' in current working directory
    - 'prompts.json' in parent directory
    """

    def __init__(self, prompts_path: str | None = None):
        candidate = prompts_path if prompts_path is not None else os.environ.get("SMARTMEMORY_PROMPTS", "prompts.json")
        self._prompts_path = self._resolve_prompts_path(candidate)
        # Export resolved absolute path for process-wide consistency
        os.environ["SMARTMEMORY_PROMPTS"] = self._prompts_path

        self._lock = threading.RLock()
        self._last_mtime = 0.0
        self._prompts: Dict[str, Any] = {}

        self._load_prompts()

    def _resolve_prompts_path(self, path_candidate: str) -> str:
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
        # Also check the directory of this module (smartmemory)
        module_dir = os.path.dirname(os.path.abspath(__file__))
        module_path = os.path.join(module_dir, basename)
        if os.path.exists(module_path):
            return os.path.abspath(module_path)
        logger.warning(f"Could not find prompts file at {primary}, {current_path}, or {parent_path}. Using empty prompts.")
        return os.path.abspath(primary)

    def _load_prompts(self):
        with self._lock:
            prompts_dict: Dict[str, Any] = {}
            try:
                if os.path.exists(self._prompts_path):
                    with open(self._prompts_path, "r", encoding="utf-8") as f:
                        prompts_dict = json.load(f)
                    logger.info(f"Loaded prompts from: {self._prompts_path}")
                    try:
                        self._last_mtime = os.path.getmtime(self._prompts_path)
                    except Exception:
                        self._last_mtime = 0.0
                else:
                    logger.warning(f"Prompts file not found: {self._prompts_path}. Using empty prompts.")
            except Exception as e:
                logger.error(f"Error loading prompts from {self._prompts_path}: {e}")

            self._prompts = _expand_env_in_obj(prompts_dict) or {}

    def reload_if_stale(self, force: bool = False):
        try:
            mtime = os.path.getmtime(self._prompts_path) if os.path.exists(self._prompts_path) else 0.0
        except Exception:
            mtime = 0.0
        if force or (mtime and mtime > self._last_mtime):
            logger.info("Prompts file changed; reloading prompts configuration")
            self._load_prompts()

    @property
    def prompts(self) -> Dict[str, Any]:
        return self._prompts


# --- Global helpers ---
_GLOBAL_PROMPTS_CONFIG: PromptsConfig | None = None


def get_prompts_config() -> PromptsConfig:
    """Get a singleton PromptsConfig instance to avoid repeated file IO.
    Respects $SMARTMEMORY_PROMPTS if set.
    """
    global _GLOBAL_PROMPTS_CONFIG
    if _GLOBAL_PROMPTS_CONFIG is None:
        _GLOBAL_PROMPTS_CONFIG = PromptsConfig()
    else:
        # Hot-reload if file changed
        try:
            _GLOBAL_PROMPTS_CONFIG.reload_if_stale()
        except Exception:
            pass
    return _GLOBAL_PROMPTS_CONFIG


def get_prompts() -> Dict[str, Any]:
    """Return the full prompts dictionary."""
    return get_prompts_config().prompts


def get_prompt_value(path: str | list[str], default: Any = None) -> Any:
    """Retrieve a nested prompt template value by dot-path or list path.
    Example: get_prompt_value("extractor.ontology.system_template")
    """
    parts = path.split(".") if isinstance(path, str) else list(path)
    node: Any = get_prompts()
    try:
        for key in parts:
            if not isinstance(node, dict):
                return default
            node = node.get(key)
        return node if node is not None else default
    except Exception:
        return default


def apply_placeholders(template: str, mapping: Dict[str, str]) -> str:
    """Replace placeholder keys with their values.
    
    Supports multiple formats for compatibility:
    - {{KEY}} - Jinja2/Mustache style (preferred)
    - <KEY> - XML/angle bracket style
    - {KEY} - Simple brace (use cautiously with JSON)
    
    Note: We use simple string replacement instead of str.format() to avoid
    conflicts with JSON braces in prompts that contain example outputs.
    """
    if not isinstance(template, str):
        return template
    
    out = template
    for k, v in (mapping or {}).items():
        v_str = str(v)
        # Replace in order of specificity (most specific first)
        out = out.replace(f"{{{{{k}}}}}", v_str)  # {{KEY}}
        out = out.replace(f"<{k}>", v_str)        # <KEY>
        # Only replace single braces if not already replaced by double braces
        if f"{{{{{k}}}}}" not in template:
            out = out.replace(f"{{{k}}}", v_str)  # {KEY}
    
    return out
