"""Additional coverage tests for SQLiteBackend — edge cases not covered by test_sqlite_backend.py.

Focuses on:
- add_edge FK violation on missing target_id (not just source_id)
- remove_edge without edge_type on node with single edge
- search_nodes with only content filter (no memory_type)
- serialize on empty graph returns empty lists
- deserialize followed by serialize is idempotent
- add_node with created_at tuple
- search_nodes filters arbitrary property keys (regression: DIST-LITE-1 P1)
- scope_provider warning when non-None (regression: DIST-LITE-1 P2)
- edge temporal fields stored and round-tripped (regression: DIST-LITE-1 P2)
"""
import sqlite3 as _sqlite3
import pytest

from smartmemory.graph.backends.sqlite import SQLiteBackend


@pytest.fixture
def b(tmp_path):
    return SQLiteBackend(db_path=str(tmp_path / "extra.db"))


# ── FK violation: missing TARGET (complement to existing source test) ─────────

def test_add_edge_fk_violation_missing_target(b):
    """add_edge with a non-existent target_id also raises IntegrityError."""
    b.add_node("src", {"content": "source", "memory_type": "working"})
    with pytest.raises(_sqlite3.IntegrityError):
        b.add_edge("src", "ghost_target", "linked", {})


# ── remove_edge without type: single-edge pair returns True ──────────────────

def test_remove_edge_no_type_single_edge_returns_true(b):
    """remove_edge(src, tgt) with no edge_type returns True when exactly one edge exists."""
    b.add_node("p", {"content": "P", "memory_type": "working"})
    b.add_node("q", {"content": "Q", "memory_type": "working"})
    b.add_edge("p", "q", "only_edge", {})
    assert b.remove_edge("p", "q") is True
    assert b.get_neighbors("p") == []


# ── search_nodes content-only filter (no memory_type) ────────────────────────

def test_search_nodes_content_only_no_memory_type_filter(b):
    """search_nodes({'content': 'X'}) works with no memory_type constraint."""
    b.add_node("n1", {"content": "needle in haystack", "memory_type": "working"})
    b.add_node("n2", {"content": "unrelated item", "memory_type": "semantic"})
    results = b.search_nodes({"content": "needle"})
    ids = [r["item_id"] for r in results]
    assert "n1" in ids
    assert "n2" not in ids


# ── serialize on empty graph ──────────────────────────────────────────────────

def test_serialize_empty_graph_returns_empty_lists(b):
    """serialize() on an empty backend returns {'nodes': [], 'edges': []}."""
    data = b.serialize()
    assert data["nodes"] == []
    assert data["edges"] == []


# ── deserialize + re-serialize is idempotent ─────────────────────────────────

def test_deserialize_then_serialize_is_idempotent(b):
    """deserialize(data) followed by serialize() produces the same node/edge counts."""
    b.add_node("a", {"content": "A", "memory_type": "working"})
    b.add_node("b", {"content": "B", "memory_type": "semantic"})
    b.add_edge("a", "b", "linked", {"weight": 1.0})
    original = b.serialize()

    # Clear and restore
    b.clear()
    b.deserialize(original)
    restored = b.serialize()

    assert len(restored["nodes"]) == len(original["nodes"])
    assert len(restored["edges"]) == len(original["edges"])
    # Node IDs match
    original_ids = {n["item_id"] for n in original["nodes"]}
    restored_ids = {n["item_id"] for n in restored["nodes"]}
    assert original_ids == restored_ids


# ── add_node with created_at tuple ───────────────────────────────────────────

def test_add_node_with_created_at_tuple(b):
    """created_at as a tuple — first element stored as created_at column."""
    b.add_node(
        "ct1",
        {"content": "timestamped", "memory_type": "episodic"},
        created_at=("2026-02-01T12:00:00",),
    )
    row = b._conn.execute("SELECT created_at FROM nodes WHERE item_id='ct1'").fetchone()
    assert row is not None
    assert row[0] == "2026-02-01T12:00:00"


# ── search_nodes: arbitrary property key filtering (regression: DIST-LITE-1 P1) ─

def test_search_nodes_filters_by_arbitrary_property_key(b):
    """search_nodes({'label': 'Note'}) must return only nodes whose 'label' property is 'Note'.

    Before the fix, unknown query keys were silently ignored, returning ALL nodes
    regardless of whether they matched.
    """
    b.add_node("note", {"content": "a note", "memory_type": "semantic", "label": "Note"})
    b.add_node("task", {"content": "a task", "memory_type": "semantic", "label": "Task"})

    results = b.search_nodes({"label": "Note"})
    ids = [r["item_id"] for r in results]
    assert "note" in ids, "Node with label=Note not returned"
    assert "task" not in ids, "Node with label=Task incorrectly returned for label=Note query"


def test_search_nodes_filters_by_name_property(b):
    """search_nodes({'name': 'Alice'}) returns only Alice's node."""
    b.add_node("alice", {"content": "Alice is an engineer", "memory_type": "semantic", "name": "Alice"})
    b.add_node("bob",   {"content": "Bob is a designer",   "memory_type": "semantic", "name": "Bob"})

    results = b.search_nodes({"name": "Alice"})
    ids = [r["item_id"] for r in results]
    assert "alice" in ids
    assert "bob" not in ids


def test_search_nodes_empty_query_returns_all(b):
    """search_nodes({}) returns all nodes (no filter applied)."""
    b.add_node("x", {"content": "X", "memory_type": "working"})
    b.add_node("y", {"content": "Y", "memory_type": "semantic"})
    results = b.search_nodes({})
    assert len(results) == 2


def test_search_nodes_combined_column_and_property_filter(b):
    """search_nodes can combine a top-level column filter with an arbitrary property filter."""
    b.add_node("s1", {"content": "S1", "memory_type": "semantic", "label": "Note"})
    b.add_node("s2", {"content": "S2", "memory_type": "semantic", "label": "Task"})
    b.add_node("w1", {"content": "W1", "memory_type": "working",  "label": "Note"})

    results = b.search_nodes({"memory_type": "semantic", "label": "Note"})
    ids = [r["item_id"] for r in results]
    assert ids == ["s1"]


# ── add_edge properties round-trip ───────────────────────────────────────────

def test_add_edge_properties_stored_in_json(b):
    """Edge properties with nested data are stored and retrievable via serialize."""
    b.add_node("x", {"content": "X", "memory_type": "working"})
    b.add_node("y", {"content": "Y", "memory_type": "working"})
    b.add_edge("x", "y", "weighted", {"weight": 0.75, "label": "strong"})
    data = b.serialize()
    edges = {(e["source_id"], e["target_id"], e["edge_type"]): e for e in data["edges"]}
    key = ("x", "y", "weighted")
    assert key in edges
    assert edges[key]["properties"]["weight"] == 0.75
    assert edges[key]["properties"]["label"] == "strong"


# ── clear removes both nodes and edges ───────────────────────────────────────

def test_clear_removes_nodes_and_edges(b):
    """After clear(), both nodes and edges tables are empty."""
    b.add_node("a", {"content": "A", "memory_type": "working"})
    b.add_node("b", {"content": "B", "memory_type": "working"})
    b.add_edge("a", "b", "linked", {})
    b.clear()
    assert b.search_nodes({}) == []
    data = b.serialize()
    assert data["edges"] == []


# ── get_neighbors bidirectional (regression: DIST-LITE-1 P2) ─────────────────

def test_get_neighbors_returns_incoming_neighbor(b):
    """get_neighbors(X) must include nodes that have an INCOMING edge to X.

    FalkorDB uses undirected MATCH (n)-[r]-(m) which traverses both directions.
    Before the fix, SQLite only returned outgoing neighbors (WHERE source_id=?),
    so querying from the target side returned nothing.
    """
    b.add_node("src", {"content": "source node", "memory_type": "working"})
    b.add_node("tgt", {"content": "target node", "memory_type": "working"})
    b.add_edge("src", "tgt", "points_to", {})

    # From target's perspective, src is an *incoming* neighbor — must be returned
    neighbors = b.get_neighbors("tgt")
    ids = [n["item_id"] for n in neighbors]
    assert "src" in ids, "Incoming neighbor 'src' not returned by get_neighbors('tgt')"


def test_get_neighbors_returns_both_directions(b):
    """get_neighbors returns nodes on both ends of all edges touching the query node."""
    b.add_node("center", {"content": "center", "memory_type": "working"})
    b.add_node("out1",   {"content": "outgoing1", "memory_type": "working"})
    b.add_node("in1",    {"content": "incoming1", "memory_type": "working"})

    b.add_edge("center", "out1", "outgoing", {})  # center → out1
    b.add_edge("in1",    "center", "incoming", {})  # in1 → center

    neighbors = b.get_neighbors("center")
    ids = [n["item_id"] for n in neighbors]
    assert "out1" in ids, "Outgoing neighbor 'out1' missing"
    assert "in1"  in ids, "Incoming neighbor 'in1' missing"


def test_get_neighbors_bidirectional_with_edge_type_filter(b):
    """get_neighbors(edge_type=X) returns both outgoing and incoming edges of that type."""
    b.add_node("a", {"content": "A", "memory_type": "working"})
    b.add_node("b", {"content": "B", "memory_type": "working"})
    b.add_node("c", {"content": "C", "memory_type": "working"})

    b.add_edge("a", "b", "relates", {})   # a → b
    b.add_edge("c", "a", "relates", {})   # c → a
    b.add_edge("a", "b", "other_type", {})  # a → b (different type, should not appear)

    neighbors = b.get_neighbors("a", edge_type="relates")
    ids = [n["item_id"] for n in neighbors]
    assert "b" in ids, "Outgoing 'relates' neighbor 'b' missing"
    assert "c" in ids, "Incoming 'relates' neighbor 'c' missing"
    # 'b' connected via 'other_type' should not appear for edge_type='relates' filter
    # (b IS already in via 'relates', so we check that the filter is type-correct, not count)


# ── _unpack_valid_to with single-element tuple ───────────────────────────────

def test_unpack_valid_to_single_element_tuple_returns_none():
    """_unpack_valid_to with a 1-element tuple returns None (no end time)."""
    result = SQLiteBackend._unpack_valid_to(("2026-01-01",))
    assert result is None


def test_unpack_valid_to_none_returns_none():
    """_unpack_valid_to(None) returns None."""
    assert SQLiteBackend._unpack_valid_to(None) is None


def test_unpack_time_non_tuple_returns_str():
    """_unpack_time with a plain string returns that string."""
    result = SQLiteBackend._unpack_time("2026-01-01")
    assert result == "2026-01-01"


def test_unpack_time_none_returns_none():
    """_unpack_time(None) returns None."""
    assert SQLiteBackend._unpack_time(None) is None


# ── scope_provider guard (regression: DIST-LITE-1 P2) ────────────────────────

def test_scope_provider_none_succeeds(tmp_path):
    """scope_provider=None (the normal Lite case) constructs without error."""
    b = SQLiteBackend(db_path=str(tmp_path / "noscope.db"), scope_provider=None)
    assert b._scope_provider is None


def test_scope_provider_non_none_raises(tmp_path):
    """A non-None scope_provider raises ValueError — SQLiteBackend is single-tenant only.

    Regression test for DIST-LITE-1 P2: previously only emitted a warning, allowing
    callers to silently proceed without any isolation being enforced.
    """
    class FakeScope:
        pass
    with pytest.raises(ValueError, match="scope_provider"):
        SQLiteBackend(db_path=str(tmp_path / "bad.db"), scope_provider=FakeScope())  # type: ignore[arg-type]


# ── edge temporal fields (regression: DIST-LITE-1 P2) ────────────────────────

def test_add_edge_stores_valid_time(b):
    """add_edge with valid_time tuple must persist valid_from and valid_to columns."""
    b.add_node("a", {"content": "A", "memory_type": "working"})
    b.add_node("b", {"content": "B", "memory_type": "working"})
    b.add_edge(
        "a", "b", "linked", {},
        valid_time=("2026-01-01T00:00:00", "2026-12-31T23:59:59"),
    )
    row = b._conn.execute(
        "SELECT valid_from, valid_to FROM edges WHERE source_id='a' AND target_id='b'"
    ).fetchone()
    assert row is not None
    assert row[0] == "2026-01-01T00:00:00"
    assert row[1] == "2026-12-31T23:59:59"


def test_add_edge_stores_created_at(b):
    """add_edge with created_at tuple must persist the created_at column."""
    b.add_node("x", {"content": "X", "memory_type": "working"})
    b.add_node("y", {"content": "Y", "memory_type": "working"})
    b.add_edge(
        "x", "y", "tagged", {},
        created_at=("2026-02-01T10:00:00",),
    )
    row = b._conn.execute(
        "SELECT created_at FROM edges WHERE source_id='x' AND target_id='y'"
    ).fetchone()
    assert row is not None
    assert row[0] == "2026-02-01T10:00:00"


def test_serialize_includes_edge_temporal_fields(b):
    """serialize() must include valid_from, valid_to, created_at for edges."""
    b.add_node("p", {"content": "P", "memory_type": "working"})
    b.add_node("q", {"content": "Q", "memory_type": "working"})
    b.add_edge(
        "p", "q", "temporal_edge", {},
        valid_time=("2026-03-01", "2026-03-31"),
        created_at=("2026-02-15",),
    )
    data = b.serialize()
    edge = data["edges"][0]
    assert edge["valid_from"] == "2026-03-01"
    assert edge["valid_to"] == "2026-03-31"
    assert edge["created_at"] == "2026-02-15"


def test_deserialize_restores_edge_temporal_fields(b):
    """deserialize(data) must restore valid_from, valid_to, created_at for edges."""
    b.add_node("m", {"content": "M", "memory_type": "working"})
    b.add_node("n", {"content": "N", "memory_type": "working"})
    b.add_edge(
        "m", "n", "temporal_edge", {},
        valid_time=("2026-04-01", "2026-04-30"),
        created_at=("2026-03-10",),
    )
    original = b.serialize()
    b.clear()
    b.deserialize(original)
    row = b._conn.execute(
        "SELECT valid_from, valid_to, created_at FROM edges WHERE source_id='m' AND target_id='n'"
    ).fetchone()
    assert row is not None
    assert row[0] == "2026-04-01"
    assert row[1] == "2026-04-30"
    assert row[2] == "2026-03-10"


def test_edge_migration_adds_temporal_columns_to_existing_db(tmp_path):
    """Opening an old-schema DB (no temporal edge columns) must migrate successfully."""
    db_path = str(tmp_path / "legacy.db")
    # Create a DB with the old edge schema (no temporal columns)
    import sqlite3
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS nodes (
            item_id TEXT PRIMARY KEY,
            memory_type TEXT,
            valid_from TEXT,
            valid_to TEXT,
            created_at TEXT,
            properties TEXT NOT NULL DEFAULT '{}'
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS edges (
            source_id TEXT NOT NULL REFERENCES nodes(item_id) ON DELETE CASCADE,
            target_id TEXT NOT NULL REFERENCES nodes(item_id) ON DELETE CASCADE,
            edge_type TEXT NOT NULL,
            memory_type TEXT,
            properties TEXT NOT NULL DEFAULT '{}',
            PRIMARY KEY (source_id, target_id, edge_type)
        )
    """)
    conn.execute("INSERT INTO nodes(item_id, memory_type, properties) VALUES ('a', 'working', '{}')")
    conn.execute("INSERT INTO nodes(item_id, memory_type, properties) VALUES ('b', 'working', '{}')")
    conn.execute("INSERT INTO edges(source_id, target_id, edge_type, properties) VALUES ('a', 'b', 'e', '{}')")
    conn.commit()
    conn.close()

    # Opening via SQLiteBackend must apply migrations without error
    b2 = SQLiteBackend(db_path=db_path)
    # After migration, add_edge with temporal fields must succeed
    b2.add_edge("a", "b", "e", {}, valid_time=("2026-05-01", "2026-05-31"))
    row = b2._conn.execute("SELECT valid_from FROM edges WHERE source_id='a'").fetchone()
    assert row is not None
    assert row[0] == "2026-05-01"
