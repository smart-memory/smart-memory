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

WRITE_LOCK_TIMEOUT = 2.0  # seconds — daemon ingest only, worker uses SmartMemory.add() (no lock)


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
        # Disable evolution — evolvers crash with missing typed configs
        # (EpisodicDecayEvolver, EpisodicToSemanticEvolver, etc.)
        # and ExponentialDecayEvolver hits datetime serialization errors.
        # These crash the daemon process silently.
        profile.evolve.run_evolution = False
        profile.evolve.run_clustering = False

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
        # Use SmartMemory.close() which shuts down evolution worker, ontology
        # store, and graph backend in the correct order.
        if hasattr(_memory, "close"):
            _memory.close()
        elif hasattr(_memory, "_graph") and hasattr(_memory._graph, "backend"):
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
                                 "entity_type", "embedding", "category",
                                 "confidence", "stale", "reference"}}
        item = {
            "item_id": raw.get("item_id", ""),
            "content": props.get("content", props.get("label", "")),
            "memory_type": mt or props.get("memory_type", ""),
            "created_at": raw.get("created_at", props.get("created_at", "")),
            # CORE-PROPS-1: Include confidence, stale, reference as top-level keys
            "confidence": props.get("confidence", 1.0),
            "stale": props.get("stale", False),
            "reference": props.get("reference", False),
        }
        if metadata:
            item["metadata"] = metadata
        items.append(item)
    return items


def search(
    query: str,
    top_k: int = 5,
    filters: dict[str, str] | None = None,
    include_reference: bool = False,
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
        # CORE-PROPS-1 Phase 6: exclude reference data from wildcard by default
        if not include_reference:
            all_items = [r for r in all_items if not r.get("reference", False)]
        return all_items
    if not filters:
        results = mem.search(query, top_k=top_k, include_reference=include_reference)
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


def recall(
    cwd: str | None = None,
    top_k: int = 10,
    *,
    query: str | None = None,
    include_snapshot: bool = True,
    workspace_id: str | None = None,
) -> str:
    """HOOK-RECALL-RELEVANCE-1: workspace-scoped, ranked, deduped recall.

    Bug fixes vs legacy:
      - empty-content items skipped
      - dedup by item_id (then content) — collapses `[concept] smartmemory` × 3
      - origin tier filter (default {1, 2}) — excludes tier-4 system noise
      - workspace metadata filter — Lite SQLiteBackend can't scope at storage layer
      - optional CORE-SUMMARY-1 snapshot frame (≤7d) prepended for SessionStart
      - optional `query` for UserPromptSubmit semantic search
      - JSONL trace at ~/.smartmemory/hook-recall.jsonl

    Remote mode delegates to RemoteMemory.recall (same param contract).
    """
    from smartmemory.origin_policy import filter_by_tiers, get_default_tiers
    from smartmemory_app.recall_format import (
        _item_to_recall_dict, _trace, derive_workspace_id,
        format_recall_lines, time_ms,
    )

    t0 = time_ms()
    mem = get_memory()
    from smartmemory_app.remote_backend import RemoteMemory
    if isinstance(mem, RemoteMemory):
        return mem.recall(
            cwd, top_k,
            query=query, include_snapshot=include_snapshot, workspace_id=workspace_id,
        )

    workspace_id = workspace_id or derive_workspace_id(cwd)

    # 1. Optional snapshot frame (graph-mirrored markdown from CORE-SUMMARY-1)
    frame = ""
    if include_snapshot:
        try:
            snaps = mem.search("", memory_type="snapshot", sort_by="recency", top_k=1)
        except Exception as exc:  # noqa: BLE001
            log.debug("snapshot lookup failed: %s", exc)
            snaps = []
        if snaps:
            snap = snaps[0]
            # Defensive: only honor results that are actually snapshots.
            # `mem.search(memory_type="snapshot")` should filter, but a mocked
            # backend may not — and we don't want to render a non-snapshot as a frame.
            if getattr(snap, "memory_type", None) == "snapshot":
                snap_meta = getattr(snap, "metadata", None) or {}
                ws_match = (
                    workspace_id is None
                    or snap_meta.get("workspace_id") in (None, workspace_id)
                )
                if ws_match and _snapshot_is_fresh(snap, max_days=7):
                    frame = (getattr(snap, "content", "") or "").strip()

    # 2. Recency + semantic candidates
    if query:
        results = mem.search(query, top_k=top_k * 2)
    elif cwd:
        recent_k, semantic_k = _split_recall_counts(top_k)
        recent = mem.search("", top_k=recent_k, sort_by="recency")
        semantic = mem.search(cwd, top_k=semantic_k) if semantic_k else []
        results = list(recent) + list(semantic)
    else:
        results = list(mem.search("", top_k=top_k, sort_by="recency"))

    # Drop snapshot rows from candidate set (the frame already covers them)
    results = [r for r in results if getattr(r, "memory_type", "") != "snapshot"]

    # 3. Origin tier filter (default {1, 2}; excludes tier-4 system noise)
    results = filter_by_tiers(results, get_default_tiers("recall"))

    # 4. Workspace metadata filter — items without workspace_id are visible
    #    (legacy backward compat; SQLiteBackend single-tenant can't scope at storage)
    if workspace_id:
        scoped = []
        for r in results:
            r_meta = getattr(r, "metadata", None) or {}
            r_ws = r_meta.get("workspace_id") if isinstance(r_meta, dict) else None
            if r_ws in (None, workspace_id):
                scoped.append(r)
        results = scoped

    # 5. Confidence floor + reference exclusion (preserved from legacy)
    recall_floor = float(os.environ.get("SMARTMEMORY_RECALL_FLOOR", "0.3"))
    results = [r for r in results if getattr(r, "confidence", 1.0) >= recall_floor]
    results = [r for r in results if not getattr(r, "reference", False)]

    # 6. Format (dedup + empty-suppress + top_k cap inside)
    item_dicts = [_item_to_recall_dict(r) for r in results]
    body = format_recall_lines(item_dicts, top_k=top_k)
    emitted = body.count("\n- ") if body else 0

    # 7. Compose: body + optional snapshot frame
    if body and frame:
        out = f"{body}\n\n{frame}"
    elif body:
        out = body
    elif frame:
        out = f"## SmartMemory Context\n\n{frame}"
    else:
        out = ""

    # 8. Trace (never raises)
    _trace(
        phase="user_prompt" if query else "session_start",
        workspace_id=workspace_id,
        cwd=cwd,
        query=query,
        candidate_count=len(results),
        emitted=emitted,
        snapshot_used=bool(frame),
        latency_ms=time_ms() - t0,
    )

    return out


def _snapshot_is_fresh(snap, max_days: int = 7) -> bool:
    """True if the snapshot's created_at is within max_days. Defaults open on parse failure."""
    from datetime import datetime, timezone
    meta = getattr(snap, "metadata", None) or {}
    created_str = meta.get("created_at") if isinstance(meta, dict) else None
    if not created_str:
        return True  # no timestamp recorded → don't suppress
    try:
        created = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        age = (datetime.now(timezone.utc) - created).days
        return age <= max_days
    except (ValueError, AttributeError):
        return True


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
