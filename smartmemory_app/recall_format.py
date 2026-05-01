"""HOOK-RECALL-RELEVANCE-1: Pure formatter, dedup, workspace derivation, JSONL trace.

This module is intentionally I/O-free except for `_trace()` (filesystem append).
All recall-path bug fixes (empty-bucket suppress, dedup, top_k cap) live in
`format_recall_lines()` so both local (`storage.recall`) and remote
(`remote_backend.RemoteMemory.recall`) paths converge on the same formatter.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

log = logging.getLogger(__name__)

DEFAULT_TRACE_PATH = Path(os.environ.get("SMARTMEMORY_HOOK_TRACE",
                                          str(Path.home() / ".smartmemory" / "hook-recall.jsonl")))
TRACE_MAX_BYTES = 10 * 1024 * 1024  # 10 MB rotate threshold


# --- Item adapter -----------------------------------------------------------


def _item_to_recall_dict(item: Any) -> dict:
    """Adapt a MemoryItem (or dict) to the uniform shape the formatter expects.

    Local path returns MemoryItem objects; remote path returns dicts. Normalize.
    """
    if isinstance(item, dict):
        return {
            "item_id": item.get("item_id") or item.get("id"),
            "memory_type": item.get("memory_type", "?"),
            "content": item.get("content", ""),
            "origin": item.get("origin", ""),
            "confidence": item.get("confidence", 1.0),
            "stale": item.get("stale", False),
            "metadata": item.get("metadata") or {},
        }
    metadata = getattr(item, "metadata", None) or {}
    return {
        "item_id": getattr(item, "item_id", None),
        "memory_type": getattr(item, "memory_type", "?"),
        "content": getattr(item, "content", "") or "",
        "origin": getattr(item, "origin", "") or "",
        "confidence": getattr(item, "confidence", 1.0),
        "stale": getattr(item, "stale", False),
        "metadata": metadata if isinstance(metadata, dict) else {},
    }


# --- Formatter --------------------------------------------------------------


def format_recall_lines(items: Iterable[dict], top_k: int) -> str:
    """Format items as the `## SmartMemory Context` block.

    Bug fixes vs legacy storage.recall():
      - empty/whitespace content suppressed
      - dedup by item_id (then case-folded content prefix as fallback)
      - top_k cap applied AFTER dedup, not before

    Returns "" if no items survive filtering.
    """
    seen_ids: set[str] = set()
    seen_bodies: set[tuple[str, str]] = set()
    lines: list[str] = []

    for it in items:
        body = (it.get("content") or "").strip()
        if not body:
            continue

        iid = it.get("item_id")
        if iid and iid in seen_ids:
            continue
        mtype = it.get("memory_type", "?") or "?"
        body_key = (mtype, body[:120].lower())
        if body_key in seen_bodies:
            continue
        if iid:
            seen_ids.add(iid)
        seen_bodies.add(body_key)

        if len(lines) >= top_k:
            break

        conf = it.get("confidence", 1.0)
        try:
            conf_marker = "~" if float(conf) < 0.5 else ""
        except (TypeError, ValueError):
            conf_marker = ""
        stale_marker = "⚠" if it.get("stale") else ""
        mtype = it.get("memory_type", "?") or "?"

        lines.append(f"- {stale_marker}{conf_marker}[{mtype}] {body[:200]}")

    if not lines:
        return ""
    return "## SmartMemory Context\n" + "\n".join(lines)


# --- Workspace derivation ---------------------------------------------------


def derive_workspace_id(cwd: str | None) -> str | None:
    """Derive a stable workspace_id from cwd.

    Strategy:
      1. SMARTMEMORY_WORKSPACE_ID env var wins.
      2. Otherwise: realpath(cwd) → optional `git rev-parse --show-toplevel` →
         realpath again → sha1[:12], prefixed `ws_`.
      3. None cwd returns None.

    Symlinked paths collapse to the same id as the real path.
    """
    env = os.environ.get("SMARTMEMORY_WORKSPACE_ID")
    if env:
        return env
    if not cwd:
        return None

    canonical = os.path.realpath(cwd)
    try:
        result = subprocess.run(
            ["git", "-C", canonical, "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=1,
            check=True,
        )
        toplevel = result.stdout.strip()
        if toplevel:
            canonical = os.path.realpath(toplevel)
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        pass

    digest = hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:12]
    return f"ws_{digest}"


# --- Trace -------------------------------------------------------------------


def _trace(
    *,
    phase: str,
    workspace_id: str | None,
    cwd: str | None,
    query: str | None,
    candidate_count: int,
    emitted: int,
    snapshot_used: bool,
    latency_ms: int,
    trace_path: Path | None = None,
) -> None:
    """Append one JSONL line per hook invocation. Never raises."""
    path = Path(trace_path) if trace_path is not None else DEFAULT_TRACE_PATH
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "workspace_id": workspace_id,
        "cwd": cwd,
        "query": query,
        "candidate_count": candidate_count,
        "emitted": emitted,
        "snapshot_used": snapshot_used,
        "latency_ms": latency_ms,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.stat().st_size > TRACE_MAX_BYTES:
            rotated = path.with_suffix(path.suffix + ".1")
            try:
                if rotated.exists():
                    rotated.unlink()
                path.rename(rotated)
            except OSError:
                pass
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as exc:  # noqa: BLE001 — trace must never break the hook
        log.debug("hook-recall trace failed: %s", exc)


def time_ms() -> int:
    """Helper: monotonic ms for latency measurement."""
    return int(time.monotonic() * 1000)
