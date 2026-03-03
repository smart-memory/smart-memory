"""DIST-LITE-5: Storage — dual-mode dispatch (local SQLite or remote hosted API).

get_memory() returns either a local SmartMemory instance or a RemoteMemory instance
depending on config. All callers (server.py MCP tools, local_api.py viewer) go through
this module and duck-type on the result — each operation has an explicit branch
because the return types differ (MemoryItem vs dict, sort_by support, etc.).

Local deps (smartmemory-core, filelock) are hard dependencies — always available.
"""
from __future__ import annotations

import atexit
import logging
import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING

from filelock import FileLock

if TYPE_CHECKING:
    from smartmemory import SmartMemory
    from smartmemory_pkg.remote_backend import RemoteMemory

from smartmemory_pkg.config import (
    SmartMemoryConfig,
    UnconfiguredError,
    _detect_and_migrate,
    is_configured,
    load_config,
)

log = logging.getLogger(__name__)

_memory: "SmartMemory | None" = None
_data_path: Path | None = None  # resolved once on first init; reused by ingest()
_init_lock = threading.Lock()

_remote_memory: "RemoteMemory | None" = None
_remote_init_lock = threading.Lock()

WRITE_LOCK_TIMEOUT = 5.0  # seconds


# --- Directory & lock resolution -------------------------------------------------


def _resolve_data_dir(explicit: str | None = None) -> Path:
    """Plugin-layer env var adapter. Core factory does not read env vars."""
    raw = explicit or os.environ.get("SMARTMEMORY_DATA_DIR")
    base = Path(raw) if raw else Path.home() / ".smartmemory"
    return base


def _get_lock_file(data_path: Path):
    return FileLock(str(data_path / ".write.lock"), timeout=WRITE_LOCK_TIMEOUT)


# --- Singleton lifecycle ---------------------------------------------------------


def _get_local_memory(data_dir: str | None = None) -> "SmartMemory":
    """Return the local SmartMemory singleton, initialising on first call.

    Thread-safe via double-checked locking. Registers atexit shutdown on first init.
    """
    global _memory, _data_path
    if _memory is not None:
        return _memory
    with _init_lock:
        if _memory is not None:  # double-checked
            return _memory
        from smartmemory.tools.factory import create_lite_memory
        from smartmemory_pkg.event_sink import get_event_sink
        from smartmemory_pkg.patterns import LitePatternManager

        data_path = _resolve_data_dir(data_dir)
        data_path.mkdir(parents=True, exist_ok=True)
        _data_path = data_path  # cache so ingest() uses the same path as the singleton
        pattern_manager = LitePatternManager(data_path)
        _memory = create_lite_memory(
            data_dir=str(data_path),
            entity_ruler_patterns=pattern_manager,
            event_sink=get_event_sink(),    # DIST-LITE-3
        )
        atexit.register(_shutdown)
        return _memory


def _get_remote_memory(cfg: SmartMemoryConfig) -> "RemoteMemory":
    """Return the RemoteMemory singleton. Double-checked locking, same pattern as local."""
    global _remote_memory
    if _remote_memory is not None:
        return _remote_memory
    with _remote_init_lock:
        if _remote_memory is not None:
            return _remote_memory
        from smartmemory_pkg.remote_backend import RemoteMemory
        _remote_memory = RemoteMemory(api_url=cfg.api_url, team_id=cfg.team_id)
        return _remote_memory


def get_memory(data_dir: str | None = None):
    """Return the active memory backend (local or remote).

    Raises UnconfiguredError if no config exists and auto-migration fails.
    Auto-migration: if [local] deps are importable but no config exists, writes
    a local config automatically (upgrade path for existing installations).
    """
    if not is_configured():
        if not _detect_and_migrate():
            raise UnconfiguredError(
                "SmartMemory is not configured. Run: smartmemory setup"
            )
    cfg = load_config()
    if cfg.mode == "remote":
        return _get_remote_memory(cfg)
    return _get_local_memory(data_dir)


def _shutdown() -> None:
    """Flush state to disk on clean exit. Called by atexit.

    Verified API paths (confirmed against core source):
      UsearchVectorBackend._save()  — private flush, no public save()
      SQLiteBackend.close()         — via memory._graph.backend.close() (factory.py:79)
    """
    global _memory
    if _memory is None:
        return
    try:
        if hasattr(_memory, "_vector_backend") and _memory._vector_backend is not None:
            _memory._vector_backend._save()
        if hasattr(_memory, "_graph") and hasattr(_memory._graph, "backend"):
            _memory._graph.backend.close()
    except Exception as exc:
        log.warning(
            "SmartMemory shutdown error — session data may not be fully persisted: %s",
            exc,
        )
    finally:
        _memory = None


# --- Helpers -------------------------------------------------------------------


def _normalize_ingest_result(result) -> str:
    """Normalize SmartMemory.ingest() return value to a plain item_id string.

    SmartMemory.ingest() returns Union[str, Dict[str, Any]]:
      str  — when sync=True (default in Lite mode): item_id directly
      dict — when sync=False: {"item_id": str, "queued": bool}
    Lite mode defaults to sync=True, but we normalize both cases defensively.
    """
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        return result.get("item_id") or str(result)
    return str(result)


# --- Operations ------------------------------------------------------------------


def ingest(content: str, memory_type: str = "episodic") -> str:
    """Ingest content into the active backend.

    Remote mode: delegates to RemoteMemory.ingest() — no file lock needed.
    Local mode: acquires filelock before calling SmartMemory.ingest() because
      usearch and entity_patterns.jsonl require cross-process coordination.
      On Timeout: raises filelock.Timeout — caller surfaces as error string.
    """
    mem = get_memory()
    from smartmemory_pkg.remote_backend import RemoteMemory
    if isinstance(mem, RemoteMemory):
        return mem.ingest(content, memory_type)
    # Local path — acquire write lock for cross-process coordination
    data_path = _data_path if _data_path is not None else _resolve_data_dir()
    lock = _get_lock_file(data_path)
    with lock:
        result = mem.ingest(content, memory_type=memory_type)
    return _normalize_ingest_result(result)


def search(query: str, top_k: int = 5) -> list[dict]:
    """Search memories by semantic similarity. Returns list[dict] in both modes."""
    mem = get_memory()
    from smartmemory_pkg.remote_backend import RemoteMemory
    if isinstance(mem, RemoteMemory):
        return mem.search(query, top_k)  # already returns list[dict]
    results = mem.search(query, top_k=top_k)
    return [r.to_dict() for r in results]  # MemoryItem objects need .to_dict()


def recall(cwd: str | None = None, top_k: int = 10) -> str:
    """Recall recent and relevant memories. Remote delegates to RemoteMemory.recall().

    Local uses sort_by="recency" which is not supported in the remote search API,
    so recall() cannot be unified — explicit branch required.
    """
    mem = get_memory()
    from smartmemory_pkg.remote_backend import RemoteMemory
    if isinstance(mem, RemoteMemory):
        return mem.recall(cwd, top_k)
    # Local path unchanged
    recent = mem.search("", top_k=top_k // 2, sort_by="recency")
    semantic = mem.search(cwd or "", top_k=top_k // 2) if cwd else []
    seen: set[str] = set()
    items = []
    for r in recent + semantic:
        if r.item_id not in seen:
            seen.add(r.item_id)
            items.append(r)
    if not items:
        return ""
    lines = ["## SmartMemory Context\n"]
    for item in items[:top_k]:
        lines.append(f"- [{item.memory_type}] {item.content[:200]}")
    return "\n".join(lines)


def get(item_id: str) -> dict:
    """Get a single memory by item_id. Returns dict in both modes.

    Local returns MemoryItem (needs .to_dict()); remote returns dict | None directly.
    """
    mem = get_memory()
    from smartmemory_pkg.remote_backend import RemoteMemory
    if isinstance(mem, RemoteMemory):
        return mem.get(item_id) or {}  # already a dict
    item = mem.get(item_id)
    return item.to_dict() if item else {}
