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
    from smartmemory_app.remote_backend import RemoteMemory

from smartmemory_app.config import (
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
    Pins the embedding provider from config before creating the memory instance
    so that env var changes (e.g. OPENAI_API_KEY appearing) don't silently switch
    the provider and cause dimension mismatches.
    """
    global _memory, _data_path
    if _memory is not None:
        return _memory
    with _init_lock:
        if _memory is not None:  # double-checked
            return _memory
        from smartmemory.ontology.pattern_manager import PatternManager
        from smartmemory.tools.factory import create_lite_memory
        from smartmemory_app.event_sink import get_event_sink
        from smartmemory_app.patterns import JSONLPatternStore

        # Pin embedding provider from config before core reads env
        cfg = load_config()
        if not os.environ.get("SMARTMEMORY_EMBEDDING_PROVIDER"):
            os.environ["SMARTMEMORY_EMBEDDING_PROVIDER"] = cfg.embedding_provider

        # DIST-FULL-LOCAL-1 Phase 2b: apply coreference config to pipeline profile
        from smartmemory.pipeline.config import PipelineConfig

        profile = PipelineConfig.default()
        if not cfg.coreference:
            profile.coreference.enabled = False

        data_path = _resolve_data_dir(data_dir)
        data_path.mkdir(parents=True, exist_ok=True)
        _data_path = data_path  # cache so ingest() uses the same path as the singleton
        pattern_manager = PatternManager(store=JSONLPatternStore(data_path))
        _memory = create_lite_memory(
            data_dir=str(data_path),
            entity_ruler_patterns=pattern_manager,
            pipeline_profile=profile,
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
        from smartmemory_app.remote_backend import RemoteMemory
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


def _split_recall_counts(top_k: int) -> tuple[int, int]:
    """Split recall budget between recency and semantic passes.

    Keep at least one slot for the recency pass so ``top_k=1`` still returns the
    most recent memory instead of issuing two zero-length searches.
    """
    requested = max(1, top_k)
    recent_k = max(1, (requested + 1) // 2)
    semantic_k = max(0, requested - recent_k)
    return recent_k, semantic_k


# --- Operations ------------------------------------------------------------------


def ingest(
    content: str,
    memory_type: str = "episodic",
    sync: bool = True,
    properties: dict[str, str] | None = None,
):
    """Ingest content into the active backend.

    Args:
        content: Text to ingest.
        memory_type: Memory type (episodic, semantic, etc.).
        sync: If True (default), run full pipeline synchronously and return item_id string.
              If False, run Tier 1 only (spaCy + EntityRuler) and return dict with
              item_id + entity_ids for background Tier 2 enrichment.
        properties: Optional user-supplied key-value properties stored in metadata.

    Remote mode: delegates to RemoteMemory.ingest() — no file lock needed.
    Local mode: acquires filelock before calling SmartMemory.ingest() because
      usearch and entity_patterns.jsonl require cross-process coordination.
      On Timeout: raises filelock.Timeout — caller surfaces as error string.
    """
    mem = get_memory()
    from smartmemory_app.remote_backend import RemoteMemory
    if isinstance(mem, RemoteMemory):
        return mem.ingest(content, memory_type)  # TODO: pass properties to remote API
    # Reserved keys that user properties must not overwrite
    _RESERVED = frozenset({"memory_type", "node_category", "item_id", "content",
                           "embedding", "created_at", "valid_from", "valid_to"})
    ctx: dict = {"memory_type": memory_type}
    if properties:
        # Flatten user properties into context so they become top-level node
        # properties (metadata keys merge into graph node properties dict).
        for k, v in properties.items():
            if k not in _RESERVED:
                ctx[k] = v
    # Local path — acquire write lock for cross-process coordination
    data_path = _data_path if _data_path is not None else _resolve_data_dir()
    lock = _get_lock_file(data_path)
    with lock:
        result = mem.ingest(content, context=ctx, sync=sync)
    if not sync and isinstance(result, dict):
        return result  # Return full dict with entity_ids for async enrichment
    return _normalize_ingest_result(result)


def _list_all_memories(mem) -> list[dict]:
    """Return all memory nodes (excluding entity/relation/Version nodes).

    Used by wildcard search (`*`). Filters to user memory items only —
    enrichment creates entity/relation nodes that should not appear in
    user-facing search results.
    """
    backend = mem._graph.backend
    snapshot = backend.serialize()
    _EXCLUDED_TYPES = {"Version", "entity", "relation", "pattern"}
    items = []
    for raw in snapshot.get("nodes", []):
        mt = raw.get("memory_type", "")
        if mt in _EXCLUDED_TYPES:
            continue
        props = raw.get("properties", {})
        nc = props.get("node_category", "")
        if nc in ("entity", "relation"):
            continue
        # Include user metadata so property filters work on wildcard results
        metadata = {k: v for k, v in props.items()
                    if k not in {"content", "label", "memory_type", "node_category",
                                 "entity_type", "embedding", "category", "confidence"}}
        item = {
            "item_id": raw.get("item_id", ""),
            "content": props.get("content", props.get("label", "")),
            "memory_type": mt or props.get("memory_type", ""),
            "created_at": raw.get("created_at", props.get("created_at", "")),
        }
        if metadata:
            item["metadata"] = metadata
        items.append(item)
    return items


def search(
    query: str,
    top_k: int = 5,
    filters: dict[str, str] | None = None,
) -> list[dict]:
    """Search memories by semantic similarity with optional property filters.

    Args:
        query: Search query string. Use "*" to return all memories.
        top_k: Maximum results.
        filters: Optional property filters (e.g. {"project": "atlas"}).
                 Applied as post-filter on search results.
    """
    top_k = max(1, min(top_k, 200))  # clamp to [1, 200]
    mem = get_memory()
    from smartmemory_app.remote_backend import RemoteMemory
    if isinstance(mem, RemoteMemory):
        if filters:
            raise NotImplementedError(
                "Property filters are not supported in remote mode. "
                "Use local mode or remove --<property> flags."
            )
        return mem.search(query, top_k)
    # Wildcard: return ALL memory nodes (not entity/relation/pattern nodes).
    # top_k is intentionally not applied — "*" means "list everything".
    if query.strip() == "*":
        all_items = _list_all_memories(mem)
        if filters:
            all_items = [
                r for r in all_items
                if all(
                    r.get("metadata", {}).get(k) == v or r.get(k) == v
                    for k, v in filters.items()
                )
            ]
        return all_items
    if not filters:
        results = mem.search(query, top_k=top_k)
        return [r.to_dict() for r in results]
    # With filters: fetch a wider window, then post-filter. If still short,
    # widen progressively to avoid missing sparse matches.
    for multiplier in (5, 20, 50):
        fetch_k = top_k * multiplier
        results = mem.search(query, top_k=fetch_k)
        items = [r.to_dict() for r in results]
        matched = [
            r for r in items
            if all(
                r.get("metadata", {}).get(k) == v or r.get(k) == v
                for k, v in filters.items()
            )
        ]
        if len(matched) >= top_k or len(results) < fetch_k:
            break  # enough matches found, or exhausted all results
    return matched[:top_k]


def recall(cwd: str | None = None, top_k: int = 10) -> str:
    """Recall recent and relevant memories. Remote delegates to RemoteMemory.recall().

    Local uses sort_by="recency" which is not supported in the remote search API,
    so recall() cannot be unified — explicit branch required.
    """
    mem = get_memory()
    from smartmemory_app.remote_backend import RemoteMemory
    if isinstance(mem, RemoteMemory):
        return mem.recall(cwd, top_k)
    if cwd:
        recent_k, semantic_k = _split_recall_counts(top_k)
        recent = mem.search("", top_k=recent_k, sort_by="recency")
        semantic = mem.search(cwd, top_k=semantic_k) if semantic_k else []
    else:
        # No cwd — give full budget to recency
        recent = mem.search("", top_k=top_k, sort_by="recency")
        semantic = []
    seen: set[str] = set()
    items = []
    for r in recent + semantic:
        if r.item_id not in seen:
            seen.add(r.item_id)
            items.append(r)
    if not items:
        return ""
    lines = ["## SmartMemory Context"]
    for item in items[:top_k]:
        lines.append(f"- [{item.memory_type}] {item.content[:200]}")
    return "\n".join(lines)


def get(item_id: str) -> dict:
    """Get a single memory by item_id. Returns dict in both modes.

    Local returns MemoryItem (needs .to_dict()); remote returns dict | None directly.
    """
    mem = get_memory()
    from smartmemory_app.remote_backend import RemoteMemory
    if isinstance(mem, RemoteMemory):
        return mem.get(item_id) or {}  # already a dict
    item = mem.get(item_id)
    return item.to_dict() if item else {}
