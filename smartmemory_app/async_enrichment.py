"""DIST-DAEMON-1: Async background enrichment queue + drain thread.

Two-tier ingest: Tier 1 (sync, ~4ms, spaCy + EntityRuler) stores immediately and
returns the item_id. If an LLM API key is available, the item_id is enqueued here
for Tier 2 background enrichment via process_extract_job().

The drain thread runs as a daemon thread inside viewer_server.py. It processes
one item at a time, acquiring _rw_lock to serialize with API endpoints.
"""
import logging
import threading
import time
from collections import deque
from typing import Any, Callable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_queue: "AsyncEnrichmentQueue | None" = None
_queue_lock = threading.Lock()

_drain_running: bool = False
_stop_event = threading.Event()


class AsyncEnrichmentQueue:
    """In-memory queue for deferred LLM enrichment of ingested items.

    Thread-safe. Bounded deque drops oldest items on overflow.
    """

    def __init__(self, maxlen: int = 10_000):
        self._queue: deque[dict] = deque(maxlen=maxlen)
        self._maxlen = maxlen
        self._lock = threading.Lock()
        self._event = threading.Event()

        # Stats (updated under _lock)
        self._total_enqueued: int = 0
        self._total_processed: int = 0
        self._total_failed: int = 0
        self._total_dropped: int = 0

    def enqueue(self, job: dict) -> None:
        """Add a job to the queue. Wakes the drain thread."""
        with self._lock:
            if len(self._queue) >= self._maxlen:
                self._total_dropped += 1
            self._queue.append(job)
            self._total_enqueued += 1
        self._event.set()

    def dequeue_all(self) -> list[dict]:
        """Drain ALL pending items. Returns empty list if queue is empty."""
        with self._lock:
            items = list(self._queue)
            self._queue.clear()
            if not items:
                self._event.clear()
            return items

    def wait(self, timeout: float = 5.0) -> bool:
        """Block until items available or timeout."""
        return self._event.wait(timeout=timeout)

    def clear(self) -> None:
        """Flush all pending jobs. Called by /clear endpoint."""
        with self._lock:
            self._queue.clear()
            self._event.clear()

    def record_processed(self) -> None:
        with self._lock:
            self._total_processed += 1

    def record_failed(self) -> None:
        with self._lock:
            self._total_failed += 1

    @property
    def size(self) -> int:
        return len(self._queue)

    @property
    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "pending": len(self._queue),
                "total_enqueued": self._total_enqueued,
                "total_processed": self._total_processed,
                "total_failed": self._total_failed,
                "total_dropped": self._total_dropped,
            }


def get_queue() -> AsyncEnrichmentQueue:
    """Lazy-init singleton queue."""
    global _queue
    if _queue is not None:
        return _queue
    with _queue_lock:
        if _queue is None:
            _queue = AsyncEnrichmentQueue()
    return _queue


def reset_queue() -> None:
    """Flush pending jobs in-place. Does NOT create a new queue object.

    The drain thread holds a reference to the singleton — replacing the object
    would leave the drain thread reading from the old queue.
    """
    q = _queue
    if q is not None:
        q.clear()


def stop_drain() -> None:
    """Signal the drain thread to stop gracefully."""
    _stop_event.set()
    # Wake the drain thread if it's blocked on queue.wait()
    q = _queue
    if q is not None:
        q._event.set()


# ---------------------------------------------------------------------------
# Drain thread
# ---------------------------------------------------------------------------


def enrichment_drain_loop(
    get_mem_fn: Callable[[], Any],
    queue: AsyncEnrichmentQueue,
    rw_lock: threading.RLock,
) -> None:
    """Background thread: drain queue and run Tier 2 LLM extraction.

    Args:
        get_mem_fn: Callable returning the SmartMemory singleton (re-called per
                    item to handle /clear invalidation).
        queue: The enrichment queue to drain.
        rw_lock: The _rw_lock from local_api.py — serializes with API endpoints.
    """
    global _drain_running
    _drain_running = True
    logger.info("Enrichment drain thread started")

    from smartmemory.background.extraction_worker import (
        _get_content_from_item,
        _run_llm_extraction,
        process_extract_job,
    )

    while not _stop_event.is_set():
        queue.wait(timeout=5.0)
        if _stop_event.is_set():
            break

        jobs = queue.dequeue_all()
        if not jobs:
            continue

        for job in jobs:
            if _stop_event.is_set():
                break

            item_id = job.get("item_id", "?")
            try:
                with rw_lock:
                    # Snapshot the current item while /clear and foreground writes
                    # are excluded. The expensive LLM call runs outside this lock.
                    mem = get_mem_fn()
                    item = mem.get(item_id)

                if item is None:
                    result = {
                        "status": "item_not_found",
                        "new_entities": 0,
                        "new_relations": 0,
                        "new_patterns": 0,
                    }
                else:
                    content = _get_content_from_item(item)
                    if not content.strip():
                        result = {
                            "status": "no_text",
                            "new_entities": 0,
                            "new_relations": 0,
                            "new_patterns": 0,
                        }
                    else:
                        llm_result = _run_llm_extraction(content)
                        if llm_result["status"] != "ok":
                            result = {
                                "status": "llm_failed",
                                "new_entities": 0,
                                "new_relations": 0,
                                "new_patterns": 0,
                            }
                        else:
                            with rw_lock:
                                # Re-fetch after the LLM call so /clear can replace the
                                # singleton safely while extraction is in flight.
                                mem = get_mem_fn()
                                fresh_item = mem.get(item_id)
                                if fresh_item is None:
                                    result = {
                                        "status": "item_not_found",
                                        "new_entities": 0,
                                        "new_relations": 0,
                                        "new_patterns": 0,
                                    }
                                else:
                                    # Wrap in _di_context so _crud.add() inside
                                    # process_extract_job sees the correct vector
                                    # backend, cache, and observability ContextVars.
                                    # Without this, each entity write creates a
                                    # temporary VectorStore that can clobber the
                                    # main backend's usearch state on _save().
                                    with mem._di_context():
                                        result = process_extract_job(
                                            mem,
                                            job,
                                            redis_client=None,
                                            item_override=fresh_item,
                                            extraction_override=llm_result.get("extraction"),
                                        )
                queue.record_processed()
                status = result.get("status", "?")
                new_e = result.get("new_entities", 0)
                new_r = result.get("new_relations", 0)
                if status == "ok" and (new_e or new_r):
                    logger.info("Enriched %s: +%d entities, +%d relations", item_id, new_e, new_r)
                else:
                    logger.debug("Enrichment %s: status=%s", item_id, status)
            except Exception:
                queue.record_failed()
                logger.warning("Enrichment failed for %s", item_id, exc_info=True)

            time.sleep(0.1)  # rate limit between items

    _drain_running = False
    logger.info("Enrichment drain thread stopped")
