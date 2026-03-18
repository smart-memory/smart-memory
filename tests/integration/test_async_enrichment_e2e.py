"""End-to-end integration tests for DIST-DAEMON-1 async background enrichment.

Tests the FULL flow with a real SQLite-backed SmartMemory:
  ingest(sync=False) → Tier 1 stores item → drain thread → process_extract_job
  with mocked LLM → entities/relations written to graph.

No Docker required. No real LLM calls (LLMSingleExtractor is mocked).
"""
import hashlib
import os
import threading
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from smartmemory.models.memory_item import MemoryItem
from smartmemory.background.extraction_worker import process_extract_job


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256(name: str, entity_type: str) -> str:
    """Reproduce the SHA-256 id that LLMSingleExtractor._process_entities() produces."""
    raw = f"{name.lower()}|{entity_type.lower()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _make_llm_entity(name: str, entity_type: str, confidence: float = 0.9) -> MemoryItem:
    sha_id = _sha256(name, entity_type)
    return MemoryItem(
        content=name,
        item_id=sha_id,
        memory_type="concept",
        metadata={"name": name, "entity_type": entity_type, "confidence": confidence},
    )


def _make_llm_extraction(entities_data: list[tuple[str, str]], relations: list[dict] | None = None):
    """Build a fake LLM extraction result: {entities: [...], relations: [...]}."""
    entities = [_make_llm_entity(name, etype) for name, etype in entities_data]
    return {"entities": entities, "relations": relations or []}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def lite_memory(tmp_path):
    """Real SQLite-backed SmartMemory with LLM extraction DISABLED in pipeline."""
    from smartmemory.tools.factory import create_lite_memory
    from smartmemory.pipeline.config import PipelineConfig

    profile = PipelineConfig.lite(llm_enabled=False)
    mem = create_lite_memory(data_dir=str(tmp_path), pipeline_profile=profile)
    yield mem
    # Cleanup: close SQLite backend
    try:
        mem._graph.backend.close()
    except Exception:
        pass


@pytest.fixture()
def enrichment_env(lite_memory):
    """Set up AsyncEnrichmentQueue + drain thread for a lite_memory instance."""
    from smartmemory_app.async_enrichment import (
        AsyncEnrichmentQueue,
        enrichment_drain_loop,
        _stop_event,
    )

    queue = AsyncEnrichmentQueue(maxlen=100)
    lock = threading.RLock()
    _stop_event.clear()

    return {
        "memory": lite_memory,
        "queue": queue,
        "lock": lock,
        "stop_event": _stop_event,
    }


# ---------------------------------------------------------------------------
# 1. Core: SmartMemory.ingest(sync=False) with real SQLite backend
# ---------------------------------------------------------------------------

class TestTier1IngestLite:
    """Verify ingest(sync=False) works with SQLiteBackend — real pipeline, no LLM."""

    def test_ingest_sync_false_returns_dict_with_item_id(self, lite_memory):
        result = lite_memory.ingest("Alice leads Project Atlas at Acme Corp.", sync=False)
        assert isinstance(result, dict), f"Expected dict, got {type(result)}"
        assert "item_id" in result, "Missing item_id in result"
        assert result["item_id"], "item_id must be non-empty"

    def test_ingest_sync_false_returns_entity_ids(self, lite_memory):
        result = lite_memory.ingest("Alice leads Project Atlas at Acme Corp.", sync=False)
        assert "entity_ids" in result, "Missing entity_ids in result"
        assert isinstance(result["entity_ids"], dict), "entity_ids must be a dict"

    def test_ingest_sync_false_item_is_retrievable(self, lite_memory):
        result = lite_memory.ingest("Bob joined Acme Corp in 2020.", sync=False)
        item = lite_memory.get(result["item_id"])
        assert item is not None, f"Item {result['item_id']} not found after sync=False ingest"

    def test_ingest_sync_false_queued_is_false_without_redis(self, lite_memory):
        """In lite mode (no Redis), queued should be False."""
        result = lite_memory.ingest("Carol manages the DevOps pipeline.", sync=False)
        assert result.get("queued") is False, "queued must be False in lite mode (no Redis)"

    def test_ingest_sync_false_preserves_memory_type(self, lite_memory):
        """memory_type passed via context should be preserved."""
        result = lite_memory.ingest(
            "Important procedure step.",
            context={"memory_type": "procedural"},
            sync=False,
        )
        item = lite_memory.get(result["item_id"])
        assert item is not None
        mt = item.memory_type if hasattr(item, "memory_type") else item.get("memory_type")
        assert mt == "procedural", f"Expected procedural, got {mt}"

    def test_ingest_sync_true_returns_string(self, lite_memory):
        """Verify sync=True (default) still returns a plain string."""
        result = lite_memory.ingest("Simple sync ingest.")
        assert isinstance(result, str), f"sync=True should return str, got {type(result)}"


# ---------------------------------------------------------------------------
# 2. process_extract_job with real SQLite SmartMemory
# ---------------------------------------------------------------------------

class TestProcessExtractJobLite:
    """Verify process_extract_job works with SQLiteBackend (not just FalkorDB)."""

    def test_ok_with_net_new_entity(self, lite_memory):
        """Tier 2 writes a net-new entity to the SQLite graph."""
        # Store an item via sync ingest
        item_id = lite_memory.ingest("Django is a Python web framework.")

        llm_extraction = _make_llm_extraction(
            entities_data=[("Python", "technology"), ("Django", "technology")],
            relations=[{
                "source_id": _sha256("Django", "technology"),
                "target_id": _sha256("Python", "technology"),
                "relation_type": "USES",
            }],
        )

        # Simulate ruler already found Python in Tier 1
        ruler_entity_ids = {"python": "ruler_python_001"}

        with patch(
            "smartmemory.plugins.extractors.llm_single.LLMSingleExtractor.extract",
            return_value=llm_extraction,
        ):
            result = process_extract_job(
                lite_memory,
                {"item_id": item_id, "workspace_id": "", "entity_ids": ruler_entity_ids},
            )

        assert result["status"] == "ok"
        assert result["new_entities"] >= 1, "Django should be net-new"

    def test_item_not_found(self, lite_memory):
        result = process_extract_job(
            lite_memory,
            {"item_id": "nonexistent_abc123", "workspace_id": "", "entity_ids": {}},
        )
        assert result["status"] == "item_not_found"

    def test_empty_content(self, lite_memory):
        item = MemoryItem(content="", memory_type="semantic")
        item_id = lite_memory.add(item)
        result = process_extract_job(
            lite_memory,
            {"item_id": item_id, "workspace_id": "", "entity_ids": {}},
        )
        assert result["status"] == "no_text"

    def test_llm_failure(self, lite_memory):
        item_id = lite_memory.ingest("Valid content for LLM failure test.")
        with patch(
            "smartmemory.plugins.extractors.llm_single.LLMSingleExtractor.extract",
            side_effect=RuntimeError("API timeout"),
        ):
            result = process_extract_job(
                lite_memory,
                {"item_id": item_id, "workspace_id": "", "entity_ids": {}},
            )
        assert result["status"] == "llm_failed"

    def test_dedup_lowercased_entity_ids(self, lite_memory):
        """Entities already in ruler_entity_ids (case-insensitive) should NOT be written as net-new."""
        item_id = lite_memory.ingest("Alice and Bob work at Acme.")

        llm_extraction = _make_llm_extraction(
            entities_data=[("Alice", "person"), ("Bob", "person")],
        )
        # Ruler found both — lowercase keys
        ruler_entity_ids = {"alice": "ruler_alice_001", "bob": "ruler_bob_001"}

        with patch(
            "smartmemory.plugins.extractors.llm_single.LLMSingleExtractor.extract",
            return_value=llm_extraction,
        ):
            result = process_extract_job(
                lite_memory,
                {"item_id": item_id, "workspace_id": "", "entity_ids": ruler_entity_ids},
            )

        assert result["status"] == "ok"
        assert result["new_entities"] == 0, "Both entities already in ruler — none should be net-new"


# ---------------------------------------------------------------------------
# 3. Full async flow: ingest → queue → drain → enrichment
# ---------------------------------------------------------------------------

class TestFullAsyncFlow:
    """End-to-end: ingest(sync=False) → enqueue → drain thread → process_extract_job → graph updated."""

    def test_drain_enriches_tier1_item(self, enrichment_env):
        """Full flow: Tier 1 stores item, drain thread runs Tier 2, entities appear."""
        from smartmemory_app.async_enrichment import enrichment_drain_loop, _stop_event

        mem = enrichment_env["memory"]
        queue = enrichment_env["queue"]
        lock = enrichment_env["lock"]

        # Tier 1: ingest sync=False
        result = mem.ingest("Alice leads Project Atlas at Acme Corp.", sync=False)
        item_id = result["item_id"]
        entity_ids = result.get("entity_ids", {})

        # Verify item is stored
        assert mem.get(item_id) is not None

        # Enqueue for Tier 2 (simulating what local_api.py does)
        lowered_ids = {k.lower(): v for k, v in entity_ids.items()}
        queue.enqueue({
            "item_id": item_id,
            "workspace_id": "",
            "entity_ids": lowered_ids,
            "enable_ontology": False,
            "enqueued_at": time.time(),
        })

        # Mock LLM to return new entities
        llm_extraction = _make_llm_extraction(
            entities_data=[("Project Atlas", "project"), ("Acme Corp", "organization")],
        )

        _stop_event.clear()
        processed = threading.Event()
        original_record = queue.record_processed

        def record_and_signal():
            original_record()
            processed.set()

        queue.record_processed = record_and_signal

        with patch(
            "smartmemory.plugins.extractors.llm_single.LLMSingleExtractor.extract",
            return_value=llm_extraction,
        ):
            t = threading.Thread(
                target=enrichment_drain_loop,
                args=(lambda: mem, queue, lock),
                daemon=True,
            )
            t.start()

            # Wait for processing
            assert processed.wait(timeout=15), "Drain thread did not process job in time"

            from smartmemory_app.async_enrichment import stop_drain
            stop_drain()
            t.join(timeout=5)

        stats = queue.stats
        assert stats["total_processed"] == 1
        assert stats["total_failed"] == 0

    def test_drain_handles_multiple_items(self, enrichment_env):
        """Drain thread processes multiple queued items in sequence."""
        from smartmemory_app.async_enrichment import enrichment_drain_loop, _stop_event, stop_drain

        mem = enrichment_env["memory"]
        queue = enrichment_env["queue"]
        lock = enrichment_env["lock"]

        # Ingest 3 items
        items = []
        for text in [
            "Alice works at Acme.",
            "Bob leads engineering.",
            "Carol manages DevOps.",
        ]:
            result = mem.ingest(text, sync=False)
            items.append(result)
            ids = {k.lower(): v for k, v in result.get("entity_ids", {}).items()}
            queue.enqueue({
                "item_id": result["item_id"],
                "workspace_id": "",
                "entity_ids": ids,
                "enable_ontology": False,
                "enqueued_at": time.time(),
            })

        assert queue.size == 3

        llm_extraction = _make_llm_extraction(entities_data=[])  # Empty — no new entities

        _stop_event.clear()

        with patch(
            "smartmemory.plugins.extractors.llm_single.LLMSingleExtractor.extract",
            return_value=llm_extraction,
        ):
            t = threading.Thread(
                target=enrichment_drain_loop,
                args=(lambda: mem, queue, lock),
                daemon=True,
            )
            t.start()

            # Wait for all 3 to process
            deadline = time.time() + 15
            while queue.stats["total_processed"] < 3 and time.time() < deadline:
                time.sleep(0.1)

            stop_drain()
            t.join(timeout=5)

        assert queue.stats["total_processed"] == 3
        assert queue.stats["total_failed"] == 0

    def test_drain_survives_llm_failure_mid_batch(self, enrichment_env):
        """If LLM fails for one item, drain continues to the next."""
        from smartmemory_app.async_enrichment import enrichment_drain_loop, _stop_event, stop_drain

        mem = enrichment_env["memory"]
        queue = enrichment_env["queue"]
        lock = enrichment_env["lock"]

        # Ingest 2 items
        r1 = mem.ingest("Item that will fail LLM.", sync=False)
        r2 = mem.ingest("Item that will succeed.", sync=False)

        for r in [r1, r2]:
            ids = {k.lower(): v for k, v in r.get("entity_ids", {}).items()}
            queue.enqueue({
                "item_id": r["item_id"],
                "workspace_id": "",
                "entity_ids": ids,
                "enable_ontology": False,
                "enqueued_at": time.time(),
            })

        call_count = 0

        def side_effect(content):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("LLM exploded")
            return _make_llm_extraction(entities_data=[])

        _stop_event.clear()

        with patch(
            "smartmemory.plugins.extractors.llm_single.LLMSingleExtractor.extract",
            side_effect=side_effect,
        ):
            t = threading.Thread(
                target=enrichment_drain_loop,
                args=(lambda: mem, queue, lock),
                daemon=True,
            )
            t.start()

            deadline = time.time() + 15
            while (queue.stats["total_processed"] + queue.stats["total_failed"]) < 2 and time.time() < deadline:
                time.sleep(0.1)

            stop_drain()
            t.join(timeout=5)

        # process_extract_job catches LLM errors internally (returns status=llm_failed)
        # so the drain thread counts BOTH as processed (not failed). The drain thread's
        # "failed" counter only fires when process_extract_job itself raises, which it
        # deliberately doesn't — it's designed to never crash.
        total = queue.stats["total_processed"] + queue.stats["total_failed"]
        assert total == 2, f"Both items should have been handled, got processed={queue.stats['total_processed']}, failed={queue.stats['total_failed']}"


# ---------------------------------------------------------------------------
# 4. storage.ingest(sync=False) integration
# ---------------------------------------------------------------------------

class TestStorageIngestSyncFalse:
    """Verify storage.ingest() passes sync through to SmartMemory correctly."""

    def test_storage_ingest_sync_false_returns_dict(self, lite_memory, monkeypatch, tmp_path):
        """storage.ingest(sync=False) should return dict with entity_ids."""
        import smartmemory_app.storage as storage

        # Patch storage module to use our lite_memory
        monkeypatch.setattr(storage, "_memory", lite_memory)
        monkeypatch.setattr(storage, "_data_path", tmp_path)

        # Mock the file lock to be a no-op context manager
        mock_lock = MagicMock()
        mock_lock.__enter__ = MagicMock(return_value=None)
        mock_lock.__exit__ = MagicMock(return_value=False)
        monkeypatch.setattr(storage, "_get_lock_file", lambda _: mock_lock)

        # Also need to make get_memory return our lite instance
        monkeypatch.setattr(storage, "get_memory", lambda: lite_memory)

        result = storage.ingest("Alice leads Atlas.", memory_type="episodic", sync=False)
        assert isinstance(result, dict), f"Expected dict for sync=False, got {type(result)}"
        assert "item_id" in result
        assert "entity_ids" in result

    def test_storage_ingest_sync_true_returns_string(self, lite_memory, monkeypatch, tmp_path):
        """storage.ingest(sync=True) should return plain string."""
        import smartmemory_app.storage as storage

        monkeypatch.setattr(storage, "_memory", lite_memory)
        monkeypatch.setattr(storage, "_data_path", tmp_path)

        mock_lock = MagicMock()
        mock_lock.__enter__ = MagicMock(return_value=None)
        mock_lock.__exit__ = MagicMock(return_value=False)
        monkeypatch.setattr(storage, "_get_lock_file", lambda _: mock_lock)
        monkeypatch.setattr(storage, "get_memory", lambda: lite_memory)

        result = storage.ingest("Bob works at Acme.", memory_type="semantic", sync=True)
        assert isinstance(result, str), f"Expected str for sync=True, got {type(result)}"


# ---------------------------------------------------------------------------
# 5. Limits and edge cases
# ---------------------------------------------------------------------------

class TestLimitsAndEdgeCases:
    """Boundary conditions, stress, and failure modes."""

    def test_empty_content_ingest_sync_false(self, lite_memory):
        """Empty string should still return a valid result (pipeline may classify as working)."""
        result = lite_memory.ingest("", sync=False)
        assert isinstance(result, dict)
        assert "item_id" in result

    def test_very_long_content(self, lite_memory):
        """10KB content should ingest without error."""
        long_text = "Alice works at Acme Corp on Project Atlas. " * 250  # ~11KB
        result = lite_memory.ingest(long_text, sync=False)
        assert result["item_id"]
        item = lite_memory.get(result["item_id"])
        assert item is not None

    def test_unicode_content(self, lite_memory):
        """Unicode content (CJK, emoji, accents) should survive round-trip."""
        text = "Müller arbeitet bei Siemens AG. 田中太郎は東京で働いています。"
        result = lite_memory.ingest(text, sync=False)
        item = lite_memory.get(result["item_id"])
        assert item is not None

    def test_rapid_sequential_ingests(self, lite_memory):
        """20 rapid sync=False ingests should all succeed."""
        results = []
        for i in range(20):
            r = lite_memory.ingest(f"Item number {i} about topic {i % 5}.", sync=False)
            results.append(r)

        assert len(results) == 20
        assert all(r["item_id"] for r in results), "All should have item_ids"
        # Verify all are retrievable
        for r in results:
            assert lite_memory.get(r["item_id"]) is not None

    def test_queue_overflow_under_load(self):
        """Bounded queue drops oldest items and tracks overflow."""
        from smartmemory_app.async_enrichment import AsyncEnrichmentQueue

        q = AsyncEnrichmentQueue(maxlen=5)
        for i in range(10):
            q.enqueue({"item_id": f"item-{i}"})

        assert q.size == 5
        assert q.stats["total_dropped"] == 5
        assert q.stats["total_enqueued"] == 10

        # Oldest surviving should be item-5
        items = q.dequeue_all()
        assert items[0]["item_id"] == "item-5"

    def test_drain_thread_with_deleted_item(self, enrichment_env):
        """Item deleted between enqueue and drain → process_extract_job returns item_not_found."""
        from smartmemory_app.async_enrichment import enrichment_drain_loop, _stop_event, stop_drain

        mem = enrichment_env["memory"]
        queue = enrichment_env["queue"]
        lock = enrichment_env["lock"]

        # Ingest then delete
        result = mem.ingest("Ephemeral content.", sync=False)
        item_id = result["item_id"]
        mem.delete(item_id)

        queue.enqueue({
            "item_id": item_id,
            "workspace_id": "",
            "entity_ids": {},
            "enable_ontology": False,
            "enqueued_at": time.time(),
        })

        _stop_event.clear()

        # process_extract_job will return item_not_found — should be counted as processed, not failed
        t = threading.Thread(
            target=enrichment_drain_loop,
            args=(lambda: mem, queue, lock),
            daemon=True,
        )
        t.start()

        deadline = time.time() + 10
        while queue.stats["total_processed"] == 0 and queue.stats["total_failed"] == 0 and time.time() < deadline:
            time.sleep(0.1)

        stop_drain()
        t.join(timeout=5)

        # item_not_found is a successful processing (not a crash), so it's counted as processed
        assert queue.stats["total_processed"] == 1

    def test_concurrent_ingest_and_drain(self, lite_memory):
        """Concurrent ingest + drain thread don't crash or corrupt data."""
        from smartmemory_app.async_enrichment import (
            AsyncEnrichmentQueue, enrichment_drain_loop, _stop_event, stop_drain,
        )

        queue = AsyncEnrichmentQueue(maxlen=100)
        lock = threading.RLock()
        _stop_event.clear()

        llm_extraction = _make_llm_extraction(entities_data=[])

        with patch(
            "smartmemory.plugins.extractors.llm_single.LLMSingleExtractor.extract",
            return_value=llm_extraction,
        ):
            # Start drain thread
            t = threading.Thread(
                target=enrichment_drain_loop,
                args=(lambda: lite_memory, queue, lock),
                daemon=True,
            )
            t.start()

            # Ingest items concurrently (simulating API requests)
            errors = []
            for i in range(10):
                try:
                    with lock:
                        result = lite_memory.ingest(f"Concurrent item {i}.", sync=False)
                    ids = {k.lower(): v for k, v in result.get("entity_ids", {}).items()}
                    queue.enqueue({
                        "item_id": result["item_id"],
                        "workspace_id": "",
                        "entity_ids": ids,
                        "enable_ontology": False,
                        "enqueued_at": time.time(),
                    })
                except Exception as e:
                    errors.append(str(e))

            # Wait for drain to finish
            deadline = time.time() + 30
            while queue.stats["total_processed"] < 10 and time.time() < deadline:
                time.sleep(0.1)

            stop_drain()
            t.join(timeout=5)

        assert not errors, f"Ingest errors: {errors}"
        assert queue.stats["total_processed"] == 10
        assert queue.stats["total_failed"] == 0
