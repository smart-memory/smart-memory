"""Tests for smartmemory_app.recall_format — pure formatter, dedup, workspace derivation.

These are unit tests (pure logic, no I/O) per testing-hierarchy rule:
"Unit tests — only when they replace multiple integration tests (pure logic, parsing)".
"""
from __future__ import annotations

import os
from unittest.mock import patch

import pytest


# --- format_recall_lines -----------------------------------------------------


def test_format_empty_list_returns_empty_string():
    from smartmemory_app.recall_format import format_recall_lines
    assert format_recall_lines([], top_k=10) == ""


def test_format_skips_empty_content_items():
    """Symptom #1 from filing: `- [semantic]` lines with no body."""
    from smartmemory_app.recall_format import format_recall_lines
    items = [
        {"item_id": "a", "memory_type": "semantic", "content": ""},
        {"item_id": "b", "memory_type": "semantic", "content": "   "},
        {"item_id": "c", "memory_type": "semantic", "content": "real content"},
    ]
    out = format_recall_lines(items, top_k=10)
    assert "real content" in out
    assert out.count("\n- ") == 1, f"only 1 line should be emitted, got: {out!r}"


def test_format_dedup_by_item_id():
    """Symptom #2: `- [concept] smartmemory` × 3 — duplicate item_ids."""
    from smartmemory_app.recall_format import format_recall_lines
    items = [
        {"item_id": "X", "memory_type": "concept", "content": "smartmemory"},
        {"item_id": "X", "memory_type": "concept", "content": "smartmemory"},
        {"item_id": "X", "memory_type": "concept", "content": "smartmemory"},
    ]
    out = format_recall_lines(items, top_k=10)
    assert out.count("\n- ") == 1


def test_format_dedup_by_content_when_no_id():
    from smartmemory_app.recall_format import format_recall_lines
    items = [
        {"memory_type": "concept", "content": "smartmemory"},
        {"memory_type": "concept", "content": "smartmemory"},
        {"memory_type": "concept", "content": "Smartmemory"},  # case-fold dupe
    ]
    out = format_recall_lines(items, top_k=10)
    assert out.count("\n- ") == 1


def test_format_dedup_distinct_ids_same_content():
    """Symptom from original observation: same content, different item_ids appears 3x."""
    from smartmemory_app.recall_format import format_recall_lines
    items = [
        {"item_id": "a", "memory_type": "concept", "content": "smartmemory"},
        {"item_id": "b", "memory_type": "concept", "content": "smartmemory"},
        {"item_id": "c", "memory_type": "concept", "content": "smartmemory"},
    ]
    out = format_recall_lines(items, top_k=10)
    assert out.count("\n- ") == 1, f"got: {out!r}"


def test_format_top_k_cap():
    from smartmemory_app.recall_format import format_recall_lines
    items = [
        {"item_id": str(i), "memory_type": "semantic", "content": f"item {i}"}
        for i in range(20)
    ]
    out = format_recall_lines(items, top_k=5)
    assert out.count("\n- ") == 5


def test_format_emits_header():
    from smartmemory_app.recall_format import format_recall_lines
    items = [{"item_id": "a", "memory_type": "semantic", "content": "x"}]
    out = format_recall_lines(items, top_k=10)
    assert out.startswith("## SmartMemory Context")


def test_format_truncates_long_content():
    from smartmemory_app.recall_format import format_recall_lines
    items = [{"item_id": "a", "memory_type": "semantic", "content": "x" * 500}]
    out = format_recall_lines(items, top_k=10)
    # Body line is "- [semantic] xxx..." — 200-char body cap
    body_line = [line for line in out.splitlines() if line.startswith("- ")][0]
    assert len(body_line) <= 200 + len("- [semantic] ") + 5  # marker headroom


def test_format_low_confidence_marker():
    from smartmemory_app.recall_format import format_recall_lines
    items = [
        {"item_id": "a", "memory_type": "semantic", "content": "x", "confidence": 0.4},
    ]
    out = format_recall_lines(items, top_k=10)
    assert "~[semantic]" in out


def test_format_stale_marker():
    from smartmemory_app.recall_format import format_recall_lines
    items = [
        {"item_id": "a", "memory_type": "semantic", "content": "x", "stale": True},
    ]
    out = format_recall_lines(items, top_k=10)
    assert "⚠" in out


def test_format_unknown_memory_type_falls_back():
    from smartmemory_app.recall_format import format_recall_lines
    items = [{"item_id": "a", "content": "x"}]  # no memory_type
    out = format_recall_lines(items, top_k=10)
    assert "[?]" in out


# --- derive_workspace_id -----------------------------------------------------


def test_derive_workspace_env_override():
    from smartmemory_app.recall_format import derive_workspace_id
    with patch.dict(os.environ, {"SMARTMEMORY_WORKSPACE_ID": "ws_explicit"}):
        assert derive_workspace_id("/any/path") == "ws_explicit"


def test_derive_workspace_none_cwd_returns_none():
    from smartmemory_app.recall_format import derive_workspace_id
    with patch.dict(os.environ, {}, clear=True):
        assert derive_workspace_id(None) is None


def test_derive_workspace_realpath_canonicalizes(tmp_path):
    """Symlinked path collapses to the same workspace_id as the real path."""
    from smartmemory_app.recall_format import derive_workspace_id
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)
    with patch.dict(os.environ, {}, clear=True):
        ws_real = derive_workspace_id(str(real))
        ws_link = derive_workspace_id(str(link))
    assert ws_real == ws_link
    assert ws_real.startswith("ws_")


def test_derive_workspace_deterministic(tmp_path):
    from smartmemory_app.recall_format import derive_workspace_id
    p = str(tmp_path)
    with patch.dict(os.environ, {}, clear=True):
        assert derive_workspace_id(p) == derive_workspace_id(p)


def test_derive_workspace_distinct_paths(tmp_path):
    from smartmemory_app.recall_format import derive_workspace_id
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    with patch.dict(os.environ, {}, clear=True):
        assert derive_workspace_id(str(a)) != derive_workspace_id(str(b))


# --- _item_to_recall_dict adapter --------------------------------------------


def test_item_to_recall_dict_handles_dict_passthrough():
    from smartmemory_app.recall_format import _item_to_recall_dict
    d = {"item_id": "a", "memory_type": "semantic", "content": "x", "origin": "cli:add"}
    out = _item_to_recall_dict(d)
    assert out["item_id"] == "a"
    assert out["origin"] == "cli:add"


def test_item_to_recall_dict_handles_object():
    from smartmemory_app.recall_format import _item_to_recall_dict

    class Item:
        item_id = "a"
        memory_type = "semantic"
        content = "hello"
        origin = "cli:add"
        confidence = 0.9
        stale = False
        metadata = {"workspace_id": "ws_test"}

    out = _item_to_recall_dict(Item())
    assert out["item_id"] == "a"
    assert out["content"] == "hello"
    assert out["origin"] == "cli:add"
    assert out["metadata"]["workspace_id"] == "ws_test"


# --- _trace ------------------------------------------------------------------


def test_trace_appends_jsonl(tmp_path):
    from smartmemory_app.recall_format import _trace
    trace_file = tmp_path / "trace.jsonl"
    _trace(
        trace_path=trace_file,
        phase="session_start",
        workspace_id="ws_x",
        cwd="/x",
        query=None,
        candidate_count=5,
        emitted=3,
        snapshot_used=True,
        latency_ms=42,
    )
    _trace(
        trace_path=trace_file,
        phase="user_prompt",
        workspace_id="ws_x",
        cwd="/x",
        query="hi",
        candidate_count=2,
        emitted=2,
        snapshot_used=False,
        latency_ms=18,
    )
    lines = trace_file.read_text().strip().splitlines()
    assert len(lines) == 2
    import json
    assert json.loads(lines[0])["phase"] == "session_start"
    assert json.loads(lines[1])["phase"] == "user_prompt"


def test_trace_silent_on_io_error(tmp_path):
    """Trace failures must never raise — hook output is what matters."""
    from smartmemory_app.recall_format import _trace
    bad_path = tmp_path / "nonexistent_dir" / "trace.jsonl"
    # Should not raise; just silently no-op or create the dir
    _trace(
        trace_path=bad_path,
        phase="session_start",
        workspace_id=None, cwd=None, query=None,
        candidate_count=0, emitted=0, snapshot_used=False, latency_ms=0,
    )
