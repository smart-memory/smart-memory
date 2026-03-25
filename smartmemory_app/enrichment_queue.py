"""SQLite-backed enrichment queue for two-tier ingest.

Tier 1 (ingest endpoint) writes jobs here. A separate worker process
reads and processes them. No threading, no shared state between processes.

Table: enrichment_queue in the main memory.db
"""

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS enrichment_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id TEXT NOT NULL,
    entity_ids TEXT DEFAULT '{}',
    workspace_id TEXT DEFAULT '',
    enqueued_at REAL NOT NULL,
    status TEXT DEFAULT 'pending',
    started_at REAL,
    completed_at REAL,
    error TEXT
)
"""


def _db_path() -> Path:
    from smartmemory_app.config import load_config
    data_dir = Path(load_config().data_dir).expanduser()
    return data_dir / "memory.db"


def _get_conn(db_path: Optional[Path] = None) -> sqlite3.Connection:
    path = db_path or _db_path()
    conn = sqlite3.connect(str(path), timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(_CREATE_TABLE)
    return conn


def enqueue(item_id: str, entity_ids: dict, workspace_id: str = "") -> None:
    """Add a job to the enrichment queue. Called by the ingest endpoint."""
    conn = _get_conn()
    try:
        conn.execute(
            "INSERT INTO enrichment_queue (item_id, entity_ids, workspace_id, enqueued_at) VALUES (?, ?, ?, ?)",
            (item_id, json.dumps(entity_ids), workspace_id, time.time()),
        )
        conn.commit()
    finally:
        conn.close()


def dequeue(batch_size: int = 1) -> list[dict]:
    """Claim pending jobs. Returns list of job dicts. Marks them 'processing'."""
    conn = _get_conn()
    try:
        cursor = conn.execute(
            "SELECT id, item_id, entity_ids, workspace_id FROM enrichment_queue "
            "WHERE status = 'pending' ORDER BY enqueued_at LIMIT ?",
            (batch_size,),
        )
        rows = cursor.fetchall()
        if not rows:
            return []

        jobs = []
        ids = []
        for row in rows:
            ids.append(row[0])
            jobs.append({
                "queue_id": row[0],
                "item_id": row[1],
                "entity_ids": json.loads(row[2] or "{}"),
                "workspace_id": row[3] or "",
            })

        conn.execute(
            f"UPDATE enrichment_queue SET status = 'processing', started_at = ? "
            f"WHERE id IN ({','.join('?' * len(ids))})",
            [time.time()] + ids,
        )
        conn.commit()
        return jobs
    finally:
        conn.close()


def mark_done(queue_id: int) -> None:
    conn = _get_conn()
    try:
        conn.execute(
            "UPDATE enrichment_queue SET status = 'done', completed_at = ? WHERE id = ?",
            (time.time(), queue_id),
        )
        conn.commit()
    finally:
        conn.close()


def mark_failed(queue_id: int, error: str) -> None:
    conn = _get_conn()
    try:
        conn.execute(
            "UPDATE enrichment_queue SET status = 'failed', completed_at = ?, error = ? WHERE id = ?",
            (time.time(), error, queue_id),
        )
        conn.commit()
    finally:
        conn.close()


def stats() -> dict:
    """Return queue stats for the health endpoint."""
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT status, COUNT(*) FROM enrichment_queue GROUP BY status"
        ).fetchall()
        counts = dict(rows)
        return {
            "pending": counts.get("pending", 0),
            "processing": counts.get("processing", 0),
            "done": counts.get("done", 0),
            "failed": counts.get("failed", 0),
        }
    finally:
        conn.close()


def cleanup(max_age_hours: int = 24) -> int:
    """Remove completed jobs older than max_age_hours."""
    conn = _get_conn()
    try:
        cutoff = time.time() - (max_age_hours * 3600)
        cursor = conn.execute(
            "DELETE FROM enrichment_queue WHERE status = 'done' AND completed_at < ?",
            (cutoff,),
        )
        conn.commit()
        return cursor.rowcount
    finally:
        conn.close()
