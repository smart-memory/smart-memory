"""Unit tests for DIST-DAEMON-1 async enrichment queue + drain thread.

Tests the AsyncEnrichmentQueue, drain loop, and two-tier ingest integration
without requiring infrastructure or real LLM calls.
"""
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from smartmemory_app.async_enrichment import (
    AsyncEnrichmentQueue,
    enrichment_drain_loop,
    get_queue,
    reset_queue,
    stop_drain,
    _stop_event,
)


class TestAsyncEnrichmentQueue:
    """Queue behavior: enqueue, dequeue_all, wait, clear, overflow."""

    def test_enqueue_and_dequeue(self):
        q = AsyncEnrichmentQueue(maxlen=100)
        q.enqueue({"item_id": "a"})
        q.enqueue({"item_id": "b"})
        assert q.size == 2

        items = q.dequeue_all()
        assert len(items) == 2
        assert items[0]["item_id"] == "a"
        assert items[1]["item_id"] == "b"
        assert q.size == 0

    def test_dequeue_all_returns_empty_when_no_items(self):
        q = AsyncEnrichmentQueue(maxlen=100)
        assert q.dequeue_all() == []

    def test_wait_returns_true_when_items_available(self):
        q = AsyncEnrichmentQueue(maxlen=100)
        q.enqueue({"item_id": "x"})
        assert q.wait(timeout=0.1) is True

    def test_wait_returns_false_on_timeout(self):
        q = AsyncEnrichmentQueue(maxlen=100)
        assert q.wait(timeout=0.1) is False

    def test_overflow_drops_oldest_and_tracks_count(self):
        q = AsyncEnrichmentQueue(maxlen=3)
        q.enqueue({"item_id": "1"})
        q.enqueue({"item_id": "2"})
        q.enqueue({"item_id": "3"})
        q.enqueue({"item_id": "4"})  # drops "1"

        assert q.size == 3
        assert q.stats["total_dropped"] == 1
        assert q.stats["total_enqueued"] == 4

        items = q.dequeue_all()
        assert items[0]["item_id"] == "2"  # oldest surviving

    def test_clear_flushes_queue(self):
        q = AsyncEnrichmentQueue(maxlen=100)
        q.enqueue({"item_id": "a"})
        q.enqueue({"item_id": "b"})
        q.clear()
        assert q.size == 0
        assert q.dequeue_all() == []

    def test_stats_tracks_processed_and_failed(self):
        q = AsyncEnrichmentQueue(maxlen=100)
        q.enqueue({"item_id": "a"})
        q.record_processed()
        q.record_processed()
        q.record_failed()

        stats = q.stats
        assert stats["total_enqueued"] == 1
        assert stats["total_processed"] == 2
        assert stats["total_failed"] == 1


class TestResetQueue:
    """reset_queue() clears in-place, preserving the singleton object."""

    def test_reset_clears_items_in_place(self):
        q = get_queue()
        q.enqueue({"item_id": "x"})
        assert q.size == 1

        original_id = id(q)
        reset_queue()

        q2 = get_queue()
        assert id(q2) == original_id, "reset_queue must not create a new object"
        assert q2.size == 0


class TestDrainThread:
    """Drain thread: processes jobs, handles failures, stops gracefully."""

    def test_drain_processes_enqueued_jobs(self):
        """Drain thread should call process_extract_job for each enqueued job."""
        q = AsyncEnrichmentQueue(maxlen=100)
        lock = threading.RLock()
        mem = MagicMock()

        job = {"item_id": "test-1", "workspace_id": "", "entity_ids": {}, "enable_ontology": False}
        q.enqueue(job)

        mock_result = {"status": "ok", "new_entities": 2, "new_relations": 1, "new_patterns": 0}

        _stop_event.clear()

        with patch(
            "smartmemory.background.extraction_worker.process_extract_job",
            return_value=mock_result,
        ) as mock_pej:
            # Start drain in thread, stop after first batch
            def run_drain():
                enrichment_drain_loop(lambda: mem, q, lock)

            t = threading.Thread(target=run_drain, daemon=True)
            t.start()

            # Wait for job to be processed
            deadline = time.time() + 5
            while q.stats["total_processed"] == 0 and time.time() < deadline:
                time.sleep(0.05)

            stop_drain()
            t.join(timeout=3)

        mock_pej.assert_called_once_with(mem, job, redis_client=None)
        assert q.stats["total_processed"] == 1

    def test_drain_survives_job_failure(self):
        """Drain thread should continue after a job fails."""
        q = AsyncEnrichmentQueue(maxlen=100)
        lock = threading.RLock()
        mem = MagicMock()

        q.enqueue({"item_id": "fail-1", "workspace_id": "", "entity_ids": {}})
        q.enqueue({"item_id": "ok-1", "workspace_id": "", "entity_ids": {}})

        call_count = 0

        def side_effect(memory, payload, redis_client=None):
            nonlocal call_count
            call_count += 1
            if payload["item_id"] == "fail-1":
                raise RuntimeError("LLM exploded")
            return {"status": "ok", "new_entities": 0, "new_relations": 0, "new_patterns": 0}

        _stop_event.clear()

        with patch(
            "smartmemory.background.extraction_worker.process_extract_job",
            side_effect=side_effect,
        ):
            t = threading.Thread(
                target=enrichment_drain_loop,
                args=(lambda: mem, q, lock),
                daemon=True,
            )
            t.start()

            deadline = time.time() + 5
            while call_count < 2 and time.time() < deadline:
                time.sleep(0.05)

            stop_drain()
            t.join(timeout=3)

        assert q.stats["total_failed"] == 1
        assert q.stats["total_processed"] == 1

    def test_drain_stops_on_stop_event(self):
        """Drain thread should exit when _stop_event is set."""
        q = AsyncEnrichmentQueue(maxlen=100)
        lock = threading.RLock()

        _stop_event.clear()

        t = threading.Thread(
            target=enrichment_drain_loop,
            args=(MagicMock, q, lock),
            daemon=True,
        )
        t.start()

        time.sleep(0.3)  # Let it enter wait loop
        stop_drain()
        t.join(timeout=10)
        assert not t.is_alive(), "Drain thread should have stopped"

    def test_drain_refetches_memory_per_item(self):
        """Drain thread re-calls get_mem_fn() per item, not per batch."""
        q = AsyncEnrichmentQueue(maxlen=100)
        lock = threading.RLock()

        mem1 = MagicMock()
        mem2 = MagicMock()
        call_idx = 0

        def get_mem():
            nonlocal call_idx
            call_idx += 1
            return mem1 if call_idx <= 1 else mem2

        q.enqueue({"item_id": "a", "workspace_id": "", "entity_ids": {}})
        q.enqueue({"item_id": "b", "workspace_id": "", "entity_ids": {}})

        _stop_event.clear()

        with patch(
            "smartmemory.background.extraction_worker.process_extract_job",
            return_value={"status": "ok", "new_entities": 0, "new_relations": 0, "new_patterns": 0},
        ) as mock_pej:
            t = threading.Thread(
                target=enrichment_drain_loop,
                args=(get_mem, q, lock),
                daemon=True,
            )
            t.start()

            deadline = time.time() + 5
            while q.stats["total_processed"] < 2 and time.time() < deadline:
                time.sleep(0.05)

            stop_drain()
            t.join(timeout=3)

        # Should have been called with mem1 for first item, mem2 for second
        calls = mock_pej.call_args_list
        assert calls[0][0][0] is mem1
        assert calls[1][0][0] is mem2


class TestTwoTierIngest:
    """Integration: POST /ingest uses two-tier when LLM key present."""

    def test_ingest_enqueues_when_llm_available(self, monkeypatch):
        """With LLM key, ingest should use sync=False and enqueue for enrichment."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        from smartmemory_app.async_enrichment import _drain_running
        import smartmemory_app.async_enrichment as ae_mod
        # Simulate drain thread running
        ae_mod._drain_running = True

        try:
            from smartmemory_app.viewer_server import app
            from fastapi.testclient import TestClient

            mock_result = {"item_id": "tier1-id", "queued": False, "entity_ids": {"alice": "node-1"}}

            with patch("smartmemory_app.storage.ingest", return_value=mock_result) as mock_ingest:
                client = TestClient(app)
                r = client.post("/memory/ingest", json={"content": "Alice leads Atlas"})

            assert r.status_code == 200
            assert r.json()["item_id"] == "tier1-id"
            mock_ingest.assert_called_once_with("Alice leads Atlas", "episodic", sync=False)

            # Check job was enqueued
            q = get_queue()
            jobs = q.dequeue_all()
            assert len(jobs) == 1
            assert jobs[0]["item_id"] == "tier1-id"
            assert jobs[0]["entity_ids"] == {"alice": "node-1"}
        finally:
            ae_mod._drain_running = False

    def test_ingest_sync_when_no_llm(self, monkeypatch):
        """Without LLM key, ingest should use sync=True (no enqueue)."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GROQ_API_KEY", raising=False)

        from smartmemory_app.viewer_server import app
        from fastapi.testclient import TestClient

        with patch("smartmemory_app.storage.ingest", return_value="sync-id") as mock_ingest:
            client = TestClient(app)
            r = client.post("/memory/ingest", json={"content": "Bob tests", "memory_type": "semantic"})

        assert r.status_code == 200
        assert r.json()["item_id"] == "sync-id"
        mock_ingest.assert_called_once_with("Bob tests", "semantic")


class TestClearFlushesQueue:
    """POST /clear should flush the enrichment queue."""

    def test_clear_resets_queue(self):
        from smartmemory_app.viewer_server import app
        from fastapi.testclient import TestClient

        q = get_queue()
        q.enqueue({"item_id": "stale"})
        assert q.size == 1

        with patch("smartmemory_app.storage._shutdown"):
            with patch("smartmemory_app.storage._resolve_data_dir", return_value=MagicMock(exists=lambda: False)):
                with patch("smartmemory_app.setup._seed_data_dir"):
                    client = TestClient(app)
                    client.post("/memory/clear")

        assert q.size == 0, "Queue should be flushed after /clear"
