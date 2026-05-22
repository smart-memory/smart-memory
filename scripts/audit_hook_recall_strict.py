#!/usr/bin/env python3
"""Audit HOOK-RECALL-RELEVANCE-1: should we flip SMARTMEMORY_RECALL_STRICT=1?

Usage:  python scripts/audit_hook_recall_strict.py
Reads ~/.smartmemory/memory.db via sqlite3 (no SmartMemory imports required).
Schema errors exit with a pointer to smartmemory_app/storage.py.
"""
import json
import sqlite3
import sys
from pathlib import Path

DB_PATH = Path(
    __import__("os").environ.get("SMARTMEMORY_DATA_DIR", str(Path.home() / ".smartmemory"))
) / "memory.db"
SCHEMA_REF = "smartmemory_app/storage.py + local_api.py (/reindex query)"
_EXCL = ("Version", "entity", "relation", "pattern", "snapshot")


def die(msg: str) -> None:
    sys.exit(f"ERROR: {msg}\n  Schema ref: {SCHEMA_REF}")


def ws_expr(con: sqlite3.Connection) -> tuple[str, str, str]:
    """Return (workspace_id_sql, origin_sql, content_sql) after schema validation."""
    tables = {r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    )}
    if "nodes" not in tables:
        die(f"Table 'nodes' not found. Tables present: {sorted(tables - {'sqlite_sequence'})}")

    cols = {r[1] for r in con.execute("PRAGMA table_info(nodes)")}
    if not {"item_id", "memory_type", "properties"} <= cols:
        die(f"'nodes' missing expected columns. Found: {sorted(cols)}")

    sample = con.execute("SELECT properties FROM nodes WHERE properties IS NOT NULL LIMIT 1").fetchone()
    if sample:
        try:
            props = json.loads(sample[0])
        except (json.JSONDecodeError, TypeError):
            die("nodes.properties is not valid JSON — schema mismatch.")
        ws = ("json_extract(properties,'$.metadata.workspace_id')"
              if isinstance(props.get("metadata"), dict) and "workspace_id" in props["metadata"]
              else "json_extract(properties,'$.workspace_id')")
    else:
        ws = "json_extract(properties,'$.workspace_id')"

    orig = "origin" if "origin" in cols else "json_extract(properties,'$.origin')"
    cont = "content" if "content" in cols else "json_extract(properties,'$.content')"
    return ws, orig, cont


def main() -> None:
    if not DB_PATH.exists():
        die(f"Database not found at {DB_PATH}")

    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        ws, orig, cont = ws_expr(con)
        excl = ",".join(f"'{t}'" for t in _EXCL)
        where = f"WHERE memory_type IS NULL OR memory_type NOT IN ({excl})"

        total = con.execute(f"SELECT COUNT(*) FROM nodes {where}").fetchone()[0]
        if total == 0:
            print("Database has 0 user memory items. Nothing to audit.")
            return

        tagged = con.execute(
            f"SELECT COUNT(*) FROM nodes {where} AND {ws} IS NOT NULL AND {ws} != ''"
        ).fetchone()[0]
        untagged = total - tagged
        pct = untagged / total * 100

        top_origins = con.execute(f"""
            SELECT COALESCE({orig},'unknown') AS o, COUNT(*) AS c
            FROM nodes {where} GROUP BY o ORDER BY c DESC LIMIT 10
        """).fetchall()

        top_unknown = con.execute(f"""
            SELECT SUBSTR(COALESCE({cont},''),1,80) AS prefix, COUNT(*) AS c
            FROM nodes {where} AND COALESCE({orig},'unknown')='unknown'
            GROUP BY prefix ORDER BY c DESC LIMIT 10
        """).fetchall()
    finally:
        con.close()

    W = 62
    print("=" * W)
    print("SmartMemory Strict-Mode Audit — HOOK-RECALL-RELEVANCE-1")
    print("=" * W)
    print(f"  DB              : {DB_PATH}")
    print(f"  Total user items: {total:,}")
    print(f"  Tagged (ws_id)  : {tagged:,}  ({100 - pct:.1f}%)")
    print(f"  Untagged        : {untagged:,}  ({pct:.1f}%)")

    print("\nTop 10 origins:")
    for o, c in top_origins:
        print(f"  {c:6,}  {o}")

    if top_unknown:
        print("\nTop content prefixes for origin='unknown' (retag candidates):")
        for prefix, c in top_unknown:
            print(f"  {c:6,}  {(prefix or '').replace(chr(10),' ').strip()[:68]!r}")

    print(f"\n{'─' * W}")
    if pct < 30:
        print("RECOMMENDATION: FLIP STRICT ON")
        print(f"\n  RECOMMEND flipping SMARTMEMORY_RECALL_STRICT=1 in shell rc;")
        print(f"  remaining {untagged:,} untagged items will be invisible in scoped recall.")
        print("\n  Add to ~/.zshrc / ~/.bashrc:")
        print("    export SMARTMEMORY_RECALL_STRICT=1")
    elif pct < 70:
        print("RECOMMENDATION: PARTIAL — keep strict opt-in for now")
        print(f"\n  {pct:.0f}% untagged. Run `smartmemory retag` for the patterns above,")
        print("  then re-run this audit.")
    else:
        print("RECOMMENDATION: NOT YET")
        print(f"\n  {pct:.0f}% untagged. Let workspace propagation accumulate or run:")
        print("    smartmemory retag")
    print("─" * W)


if __name__ == "__main__":
    main()
