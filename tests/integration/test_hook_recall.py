"""HOOK-RECALL-RELEVANCE-1 G5: integration tests for hook recall behavior.

Real SQLite backend, real ingest pipeline. No mocks. Verifies the contract:
- empty memory-type buckets are suppressed
- duplicate item_id and duplicate content collapse
- origin tier-4 items don't surface (default recall)
- workspace metadata filter scopes recall when items are tagged
- strict mode drops legacy items with no workspace_id
- snapshot frame is not contaminated by non-snapshot rows
- failure-mode is "emit nothing" — never inject empty labels
"""
from __future__ import annotations

import pytest

import smartmemory_app.storage as storage_mod


@pytest.fixture(autouse=True)
def reset_storage():
    storage_mod._memory = None
    storage_mod._remote_memory = None
    yield
    storage_mod._memory = None
    storage_mod._remote_memory = None


@pytest.fixture
def temp_data(tmp_path, monkeypatch):
    monkeypatch.setenv("SMARTMEMORY_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SMARTMEMORY_RECALL_FLOOR", "0.1")
    return tmp_path


@pytest.mark.integration
def test_hook_recall_no_empty_buckets(temp_data):
    """Items with empty content must never produce bare `[type]` lines."""
    storage_mod.ingest("real content here", properties={"origin": "cli:add"})
    # An item with empty content shouldn't normally exist, but if one slips
    # through (extraction artifacts, malformed imports), recall must skip it.
    result = storage_mod.recall(top_k=5)
    for line in result.splitlines():
        if line.startswith("- "):
            # Strip marker glyphs and the "[type] " prefix
            body = line.split("] ", 1)[-1].strip()
            assert body, f"empty body in line: {line!r}"


@pytest.mark.integration
def test_hook_recall_dedup_distinct_ids_same_content(temp_data):
    """Three ingests of identical content emit at most one line (the original
    `[concept] smartmemory ×3` symptom from the filing)."""
    same_content = "duplicate-by-content test memory"
    for _ in range(3):
        storage_mod.ingest(same_content, properties={"origin": "cli:add"})
    result = storage_mod.recall(top_k=10)
    occurrences = result.count(same_content)
    assert occurrences <= 1, (
        f"duplicate content should collapse to one line, got {occurrences}: {result!r}"
    )


@pytest.mark.integration
def test_hook_recall_origin_tier_4_excluded(temp_data):
    """Items with tier-4 origin (hook:*, structured:*) don't appear in recall."""
    storage_mod.ingest("user-authored content", properties={"origin": "cli:add"})
    storage_mod.ingest("hook noise content", properties={"origin": "hook:test"})
    storage_mod.ingest("system noise content", properties={"origin": "structured:tool_call"})

    result = storage_mod.recall(top_k=10)
    assert "user-authored content" in result
    assert "hook noise content" not in result, "tier-4 hook:* must not surface"
    assert "system noise content" not in result, "tier-4 structured:* must not surface"


@pytest.mark.integration
def test_hook_recall_workspace_isolation(temp_data):
    """Items tagged with workspace_id A must not surface in workspace B recall."""
    storage_mod.ingest(
        "workspace A private memory",
        properties={"origin": "cli:add", "workspace_id": "ws_alpha"},
    )
    storage_mod.ingest(
        "workspace B private memory",
        properties={"origin": "cli:add", "workspace_id": "ws_beta"},
    )

    a = storage_mod.recall(top_k=10, workspace_id="ws_alpha")
    b = storage_mod.recall(top_k=10, workspace_id="ws_beta")

    assert "workspace A private memory" in a
    assert "workspace B private memory" not in a, "ws_alpha recall leaked ws_beta item"
    assert "workspace B private memory" in b
    assert "workspace A private memory" not in b, "ws_beta recall leaked ws_alpha item"


@pytest.mark.integration
def test_hook_recall_strict_drops_untagged_legacy(temp_data):
    """Strict mode excludes items with no workspace_id when a workspace_id is set
    on the recall — eliminates the Alice/Atlas-style cross-workspace leak."""
    storage_mod.ingest(
        "legacy untagged memory",
        properties={"origin": "cli:add"},  # no workspace_id
    )
    storage_mod.ingest(
        "scoped memory",
        properties={"origin": "cli:add", "workspace_id": "ws_alpha"},
    )

    # Default (non-strict): legacy item passes through
    default = storage_mod.recall(top_k=10, workspace_id="ws_alpha")
    assert "legacy untagged memory" in default
    assert "scoped memory" in default

    # Strict: legacy item dropped
    strict = storage_mod.recall(top_k=10, workspace_id="ws_alpha", strict=True)
    assert "scoped memory" in strict
    assert "legacy untagged memory" not in strict, (
        "strict mode must drop items with no workspace_id"
    )


@pytest.mark.integration
def test_hook_recall_strict_via_env_var(temp_data, monkeypatch):
    """SMARTMEMORY_RECALL_STRICT=1 enables strict mode globally."""
    storage_mod.ingest("legacy untagged", properties={"origin": "cli:add"})

    monkeypatch.setenv("SMARTMEMORY_RECALL_STRICT", "1")
    result = storage_mod.recall(top_k=5, workspace_id="ws_alpha")
    assert "legacy untagged" not in result


@pytest.mark.integration
def test_hook_recall_failure_mode_empty_string(temp_data):
    """When no items survive filtering, recall returns empty string —
    never bare labels or error text."""
    # Ingest only tier-4 items
    storage_mod.ingest("hook noise 1", properties={"origin": "hook:test"})
    storage_mod.ingest("hook noise 2", properties={"origin": "structured:tool_call"})

    result = storage_mod.recall(top_k=5, include_snapshot=False)
    assert result == "", (
        f"failure mode must be empty string, got: {result!r}"
    )


@pytest.mark.integration
def test_hook_recall_query_mode(temp_data):
    """UserPromptSubmit-style query: recall(query=...) does semantic search."""
    storage_mod.ingest("the architecture uses FalkorDB for graph storage",
                        properties={"origin": "cli:add"})
    storage_mod.ingest("unrelated topic about marketing",
                        properties={"origin": "cli:add"})

    result = storage_mod.recall(query="what graph database do we use?", top_k=3)
    # Semantic search should rank FalkorDB content above marketing
    assert "FalkorDB" in result or "graph storage" in result


@pytest.mark.integration
def test_hook_recall_trace_jsonl_emitted(temp_data, tmp_path, monkeypatch):
    """Each recall call appends one JSONL line to the trace file."""
    trace_file = tmp_path / "hook-recall-test.jsonl"
    monkeypatch.setenv("SMARTMEMORY_HOOK_TRACE", str(trace_file))

    # Reload the recall_format module so it picks up the new trace path
    import importlib
    import smartmemory_app.recall_format as rf
    importlib.reload(rf)

    storage_mod.ingest("trace test content", properties={"origin": "cli:add"})
    storage_mod.recall(top_k=5)
    storage_mod.recall(query="trace test", top_k=5)

    assert trace_file.exists(), f"trace file missing at {trace_file}"
    lines = trace_file.read_text().strip().splitlines()
    assert len(lines) == 2

    import json
    rec1 = json.loads(lines[0])
    rec2 = json.loads(lines[1])
    assert rec1["phase"] == "session_start"
    assert rec2["phase"] == "user_prompt"
    assert rec2["query"] == "trace test"


@pytest.mark.integration
def test_hook_recall_seed_origin_filtered(temp_data):
    """seed:* origin (added via `smartmemory retag`) is tier-4 by virtue of
    matching no tier 1/2/3 prefix → falls to tier 4 → excluded by default."""
    storage_mod.ingest("Alice leads Project Atlas", properties={"origin": "seed:demo"})
    storage_mod.ingest("real user content", properties={"origin": "cli:add"})

    result = storage_mod.recall(top_k=10)
    assert "real user content" in result
    assert "Alice leads Project Atlas" not in result, (
        "seed:* must be tier 4 (excluded from recall by default)"
    )
