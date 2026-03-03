"""DIST-LITE-5: Config file, env var overlay, OS keychain integration.

Single canonical config path: ~/.config/smartmemory/config.toml (XDG on Linux/macOS,
%APPDATA%\\smartmemory\\config.toml on Windows). Written by `smartmemory setup`.

API keys are stored in the OS keychain via `keyring`, never in the config file.
The config only stores `api_key_set = true` as a sentinel. The SMARTMEMORY_API_KEY
env var bypasses the keychain entirely — the correct path for CI/Docker.
"""
from __future__ import annotations

import os
import tomllib
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import tomli_w


_VALID_MODES = frozenset({"local", "remote"})


class UnconfiguredError(RuntimeError):
    """Raised by storage.get_memory() when no config exists and auto-migration fails.

    Typed (not bare RuntimeError) so local_api.py can catch it specifically
    and return HTTP 503 rather than letting FastAPI surface it as 500.
    """


@dataclass
class SmartMemoryConfig:
    mode: Optional[str] = None    # None = unconfigured; "local" | "remote"
    api_url: str = "https://api.smartmemory.ai"
    api_key_set: bool = False      # sentinel — actual key in OS keychain, never here
    team_id: str = ""
    coreference: bool = False
    llm_provider: str = "none"
    llm_model: str = ""
    data_dir: str = "~/.smartmemory"


def config_path() -> Path:
    """XDG-aware config path.

    Linux/macOS: ~/.config/smartmemory/config.toml (respects XDG_CONFIG_HOME)
    Windows:     %APPDATA%\\smartmemory\\config.toml
    """
    if os.name == "nt":
        base = Path(os.environ.get("APPDATA", Path.home()))
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "smartmemory" / "config.toml"


def load_config() -> SmartMemoryConfig:
    """Load config from file, apply env var overrides.

    Missing file → SmartMemoryConfig(mode=None) (unconfigured).
    Invalid TOML → warns and returns unconfigured (never raises).
    Env vars are applied after file parse and always win.
    """
    path = config_path()
    cfg = SmartMemoryConfig()

    if path.exists():
        try:
            raw = tomllib.loads(path.read_text(encoding="utf-8"))
            sm = raw.get("smartmemory", {})
            local = raw.get("local", {})
            remote = raw.get("remote", {})
            file_mode = sm.get("mode")
            if file_mode is not None and file_mode not in _VALID_MODES:
                warnings.warn(
                    f"Invalid mode {file_mode!r} in config file {path} "
                    f"(expected one of {sorted(_VALID_MODES)}) — treating as unconfigured.",
                    stacklevel=2,
                )
                file_mode = None
            cfg.mode = file_mode
            cfg.api_url = remote.get("api_url", cfg.api_url)
            cfg.api_key_set = remote.get("api_key_set", False)
            cfg.team_id = remote.get("team_id", "")
            cfg.coreference = local.get("coreference", False)
            cfg.llm_provider = local.get("llm_provider", "none")
            cfg.llm_model = local.get("llm_model", "")
            cfg.data_dir = local.get("data_dir", "~/.smartmemory")
        except Exception as exc:
            warnings.warn(
                f"Failed to parse config at {path}: {exc} — treating as unconfigured.",
                stacklevel=2,
            )

    # Env vars always win — applied after file parse
    if mode := os.environ.get("SMARTMEMORY_MODE"):
        if mode not in _VALID_MODES:
            raise ValueError(
                f"Invalid SMARTMEMORY_MODE={mode!r}. "
                f"Expected one of: {', '.join(sorted(_VALID_MODES))}"
            )
        cfg.mode = mode
    if url := os.environ.get("SMARTMEMORY_API_URL"):
        cfg.api_url = url
    if os.environ.get("SMARTMEMORY_API_KEY"):
        cfg.api_key_set = True  # presence satisfies the configured check
    if tid := os.environ.get("SMARTMEMORY_TEAM_ID"):
        cfg.team_id = tid
    if d := os.environ.get("SMARTMEMORY_DATA_DIR"):
        cfg.data_dir = d
    if p := os.environ.get("SMARTMEMORY_LLM_PROVIDER"):
        cfg.llm_provider = p

    return cfg


def save_config(cfg: SmartMemoryConfig) -> None:
    """Write config to file. Creates directory with mode 0o700 (owner-read-only)."""
    path = config_path()
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    data: dict = {
        "local": {
            "coreference": cfg.coreference,
            "llm_provider": cfg.llm_provider,
            "llm_model": cfg.llm_model,
            "data_dir": cfg.data_dir,
        },
        "remote": {
            "api_url": cfg.api_url,
            "api_key_set": cfg.api_key_set,
            "team_id": cfg.team_id,
        },
    }
    # mode must be present at top level when set; omit the [smartmemory] table entirely
    # when unconfigured so load_config() correctly returns mode=None
    if cfg.mode is not None:
        data = {"smartmemory": {"mode": cfg.mode}, **data}
    path.write_text(tomli_w.dumps(data), encoding="utf-8")


def is_configured() -> bool:
    """True if mode is set via env var or config file."""
    if os.environ.get("SMARTMEMORY_MODE"):
        return True
    return load_config().mode is not None


def get_api_key() -> str:
    """Return API key. Env var always wins over keychain. Warns if neither set.

    Never raises — headless Linux (Docker, CI) has no usable keychain and
    keyring raises or returns None silently depending on version.
    """
    if key := os.environ.get("SMARTMEMORY_API_KEY"):
        return key
    try:
        import keyring
        key = keyring.get_password("smartmemory", "api_key") or ""
    except Exception:
        key = ""
    if not key:
        warnings.warn(
            "No API key found. Set SMARTMEMORY_API_KEY env var "
            "or run: smartmemory setup --mode remote --api-key sk_...",
            stacklevel=2,
        )
    return key


def set_api_key(key: str) -> None:
    """Store API key in OS keychain. Warns (does not raise) if keychain unavailable.

    Headless Linux environments (Docker, SSH, CI) commonly have no usable
    Secret Service backend. Callers should tell users to set SMARTMEMORY_API_KEY
    as the env var fallback if this warns.
    """
    try:
        import keyring
        keyring.set_password("smartmemory", "api_key", key)
    except Exception:
        warnings.warn(
            "OS keychain unavailable — API key not persisted. "
            "Set SMARTMEMORY_API_KEY env var to avoid re-entering on next run.",
            stacklevel=2,
        )


def _detect_and_migrate() -> bool:
    """Auto-write local config when no config exists.

    Called by storage.get_memory() before raising UnconfiguredError.
    Since all local deps (smartmemory-core, usearch, filelock) are now
    hard dependencies of the smartmemory package, we default to local mode
    on first run — the user has everything they need.

    Returns True if migration succeeded (local config written).
    """
    save_config(SmartMemoryConfig(mode="local"))
    return True
