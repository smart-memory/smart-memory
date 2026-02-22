"""Integration tests for AsyncFalkorDBBackend against a real FalkorDB instance.

Requires FalkorDB on localhost:9010 (Docker: ``docker compose up -d``).

Auto-marked as ``integration`` by the conftest hook — no decorator needed.

Run:
    PYTHONPATH=. pytest tests/integration/test_async_falkordb_backend.py -v
    PYTHONPATH=. pytest tests/integration/test_async_falkordb_backend.py -v -k "perf" -s
"""

import asyncio
import time
from uuid import uuid4

import pytest

from smartmemory.graph.backends.async_falkordb import AsyncFalkorDBBackend
from smartmemory.graph.backends.falkordb import FalkorDBBackend
from smartmemory.scope_provider import DefaultScopeProvider


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
async def async_backend():
    """Isolated async backend per-test using a unique graph name."""
    graph_name = f"test_async_{uuid4().hex[:10]}"
    backend = AsyncFalkorDBBackend(host="localhost", port=9010, graph_name=graph_name)
    try:
        await backend.connect()
    except Exception as exc:
        pytest.skip(f"FalkorDB not available on localhost:9010: {exc}")
    yield backend
    try:
        await backend.clear()
        await backend.close()
    except Exception:
        pass


@pytest.fixture
def sync_backend():
    """Sync FalkorDB backend for performance comparison."""
    graph_name = f"test_sync_{uuid4().hex[:10]}"
    try:
        backend = FalkorDBBackend(host="localhost", port=9010, graph_name=graph_name)
    except Exception as exc:
        pytest.skip(f"FalkorDB not available: {exc}")
    yield backend
    try:
        backend.clear()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# CRUD Parity (13 tests)
# ---------------------------------------------------------------------------

class TestCrudParity:
    async def test_add_and_get_node(self, async_backend):
        """Round-trip: add_node → get_node returns correct properties."""
        item_id = f"node-{uuid4().hex[:8]}"
        await async_backend.add_node(item_id, {"content": "hello world", "score": 42}, memory_type="semantic")

        node = await async_backend.get_node(item_id)

        assert node is not None
        assert node["item_id"] == item_id
        assert node["content"] == "hello world"
        assert node["score"] == 42

    async def test_add_node_upsert_semantics(self, async_backend):
        """Second add_node updates the node without creating a duplicate."""
        item_id = f"node-{uuid4().hex[:8]}"
        await async_backend.add_node(item_id, {"content": "v1"}, memory_type="semantic")
        await async_backend.add_node(item_id, {"content": "v2"}, memory_type="semantic")

        node = await async_backend.get_node(item_id)
        count_res = await async_backend._ro_query(
            "MATCH (n {item_id: $id}) RETURN count(n)", {"id": item_id}
        )
        count = count_res[0][0] if count_res and count_res[0] else 0

        assert count == 1
        assert node["content"] == "v2"

    async def test_get_node_nonexistent_returns_none(self, async_backend):
        result = await async_backend.get_node("does-not-exist-xyz")
        assert result is None

    async def test_add_edge_and_get_neighbors(self, async_backend):
        """add_edge → get_neighbors returns target node."""
        src_id = f"src-{uuid4().hex[:8]}"
        tgt_id = f"tgt-{uuid4().hex[:8]}"
        await async_backend.add_node(src_id, {"content": "source"}, memory_type="semantic")
        await async_backend.add_node(tgt_id, {"content": "target"}, memory_type="semantic")

        created = await async_backend.add_edge(src_id, tgt_id, "LINKS", {"weight": 1})
        assert created is True

        neighbors = await async_backend.get_neighbors(src_id)
        neighbor_ids = [n[0].get("item_id") for n in neighbors]
        assert tgt_id in neighbor_ids

    async def test_get_neighbors_filtered_by_edge_type(self, async_backend):
        """Edge type filter returns only matching relationships."""
        src_id = f"src-{uuid4().hex[:8]}"
        n1 = f"n1-{uuid4().hex[:8]}"
        n2 = f"n2-{uuid4().hex[:8]}"
        for nid, content in [(src_id, "src"), (n1, "n1"), (n2, "n2")]:
            await async_backend.add_node(nid, {"content": content}, memory_type="semantic")

        await async_backend.add_edge(src_id, n1, "LINKS", {})
        await async_backend.add_edge(src_id, n2, "MENTIONS", {})

        neighbors = await async_backend.get_neighbors(src_id, edge_type="LINKS")
        neighbor_ids = [n[0].get("item_id") for n in neighbors]
        assert n1 in neighbor_ids
        assert n2 not in neighbor_ids

    async def test_get_neighbors_empty(self, async_backend):
        """Node with no edges returns empty list."""
        node_id = f"isolated-{uuid4().hex[:8]}"
        await async_backend.add_node(node_id, {"content": "alone"}, memory_type="semantic")

        neighbors = await async_backend.get_neighbors(node_id)
        assert neighbors == []

    async def test_remove_node(self, async_backend):
        """Removed node returns None from get_node."""
        node_id = f"rm-{uuid4().hex[:8]}"
        await async_backend.add_node(node_id, {"content": "to remove"}, memory_type="semantic")
        assert await async_backend.get_node(node_id) is not None

        await async_backend.remove_node(node_id)

        assert await async_backend.get_node(node_id) is None

    async def test_remove_node_cascades_edges(self, async_backend):
        """DETACH DELETE removes connected edges too."""
        src_id = f"src-{uuid4().hex[:8]}"
        tgt_id = f"tgt-{uuid4().hex[:8]}"
        await async_backend.add_node(src_id, {"content": "s"}, memory_type="semantic")
        await async_backend.add_node(tgt_id, {"content": "t"}, memory_type="semantic")
        await async_backend.add_edge(src_id, tgt_id, "LINKS", {})

        await async_backend.remove_node(src_id)

        neighbors = await async_backend.get_neighbors(tgt_id)
        neighbor_ids = [n[0].get("item_id") for n in neighbors]
        assert src_id not in neighbor_ids

    async def test_remove_edge(self, async_backend):
        """Specific edge removed, others preserved."""
        src_id = f"src-{uuid4().hex[:8]}"
        n1 = f"n1-{uuid4().hex[:8]}"
        n2 = f"n2-{uuid4().hex[:8]}"
        for nid, content in [(src_id, "src"), (n1, "n1"), (n2, "n2")]:
            await async_backend.add_node(nid, {"content": content}, memory_type="semantic")
        await async_backend.add_edge(src_id, n1, "LINKS", {})
        await async_backend.add_edge(src_id, n2, "LINKS", {})

        await async_backend.remove_edge(src_id, n1, "LINKS")

        neighbors = await async_backend.get_neighbors(src_id)
        neighbor_ids = [n[0].get("item_id") for n in neighbors]
        assert n1 not in neighbor_ids
        assert n2 in neighbor_ids

    async def test_search_nodes_by_property(self, async_backend):
        """Property filter matches exactly the right nodes."""
        unique_tag = uuid4().hex
        n1 = f"match-{uuid4().hex[:8]}"
        n2 = f"nomatch-{uuid4().hex[:8]}"
        await async_backend.add_node(n1, {"tag": unique_tag}, memory_type="semantic")
        await async_backend.add_node(n2, {"tag": "other"}, memory_type="semantic")

        results = await async_backend.search_nodes({"tag": unique_tag})
        result_ids = [r.get("item_id") for r in results]

        assert n1 in result_ids
        assert n2 not in result_ids

    async def test_serialize_deserialize_round_trip(self, async_backend):
        """serialize → clear → deserialize preserves graph structure."""
        node_id = f"serde-{uuid4().hex[:8]}"
        await async_backend.add_node(node_id, {"content": "persisted"}, memory_type="semantic")

        snapshot = await async_backend.serialize()
        await async_backend.clear()

        node_after_clear = await async_backend.get_node(node_id)
        assert node_after_clear is None

        await async_backend.deserialize(snapshot)
        restored = await async_backend.get_node(node_id)
        assert restored is not None
        assert restored.get("content") == "persisted"

    async def test_clear_idempotent(self, async_backend):
        """clear() called twice does not raise."""
        node_id = f"clr-{uuid4().hex[:8]}"
        await async_backend.add_node(node_id, {"content": "x"}, memory_type="semantic")
        await async_backend.clear()
        await async_backend.clear()  # second clear on empty graph — must not raise

    async def test_bulk_nodes_and_edges(self, async_backend):
        """UNWIND-based bulk ops insert correct counts."""
        nodes = [
            {"item_id": f"bulk-{i}-{uuid4().hex[:6]}", "memory_type": "semantic", "content": f"item {i}"}
            for i in range(5)
        ]
        total_nodes = await async_backend.add_nodes_bulk(nodes)
        assert total_nodes == 5

        edges = [
            (nodes[i]["item_id"], nodes[i + 1]["item_id"], "NEXT", {})
            for i in range(4)
        ]
        total_edges = await async_backend.add_edges_bulk(edges)
        assert total_edges == 4


# ---------------------------------------------------------------------------
# Concurrency & Backpressure (5 tests)
# ---------------------------------------------------------------------------

class TestConcurrency:
    async def test_concurrent_reads_no_serialization(self, async_backend):
        """20 concurrent ro_query reads complete without serializing."""
        node_id = f"rd-{uuid4().hex[:8]}"
        await async_backend.add_node(node_id, {"content": "read me"}, memory_type="semantic")

        start = time.perf_counter()
        results = await asyncio.gather(*[async_backend.get_node(node_id) for _ in range(20)])
        elapsed = time.perf_counter() - start

        assert all(r is not None for r in results)
        # 20 concurrent reads must be faster than 20 sequential (heuristic: <5s)
        assert elapsed < 5.0, f"Concurrent reads took {elapsed:.2f}s — likely serialized"

    async def test_concurrent_writes_no_corruption(self, async_backend):
        """50 concurrent add_node calls produce the correct total node count."""
        ids = [f"cw-{uuid4().hex[:8]}" for _ in range(50)]

        await asyncio.gather(*[
            async_backend.add_node(nid, {"content": "x"}, memory_type="semantic")
            for nid in ids
        ])

        count = await async_backend.get_node_count()
        assert count == 50

    async def test_concurrent_read_write_mix(self, async_backend):
        """Mixed reads/writes under gather: no deadlock or corruption."""
        # Pre-seed some nodes
        seed_ids = [f"seed-{uuid4().hex[:8]}" for _ in range(10)]
        for nid in seed_ids:
            await async_backend.add_node(nid, {"content": "seed"}, memory_type="semantic")

        new_ids = [f"new-{uuid4().hex[:8]}" for _ in range(10)]

        async def write(nid):
            await async_backend.add_node(nid, {"content": "new"}, memory_type="semantic")

        async def read(nid):
            return await async_backend.get_node(nid)

        tasks = [write(nid) for nid in new_ids] + [read(nid) for nid in seed_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        errors = [r for r in results if isinstance(r, Exception)]
        assert not errors, f"Errors during mixed r/w: {errors}"

    async def test_bounded_pool_queues_under_load(self):
        """max_connections=2, 10 callers sharing a Semaphore(2) all complete.

        Spike finding: redis.asyncio raises ConnectionError immediately when
        pool is exhausted — it does NOT queue.  Callers must use a Semaphore
        to cap concurrency at the pool size for graceful backpressure.
        """
        graph_name = f"test_pool_{uuid4().hex[:10]}"
        pool_size = 2
        backend = AsyncFalkorDBBackend(
            host="localhost", port=9010, graph_name=graph_name, max_connections=pool_size
        )
        try:
            await backend.connect()
        except Exception as exc:
            pytest.skip(f"FalkorDB not available: {exc}")

        node_id = f"pool-{uuid4().hex[:8]}"
        await backend.add_node(node_id, {"content": "x"}, memory_type="semantic")

        sem = asyncio.Semaphore(pool_size)

        async def guarded_read():
            async with sem:
                return await backend.get_node(node_id)

        try:
            results = await asyncio.gather(*[guarded_read() for _ in range(10)])
            assert all(r is not None for r in results)
        finally:
            await backend.clear()
            await backend.close()

    async def test_bounded_pool_no_deadlock(self):
        """max_connections=1, sequential ops via Semaphore(1) complete without deadlock.

        Spike finding: pool size 1 serializes all I/O correctly when callers
        use a semaphore — no deadlock occurs because queries complete and
        release the connection before the next acquires the semaphore.
        """
        graph_name = f"test_pool1_{uuid4().hex[:10]}"
        backend = AsyncFalkorDBBackend(
            host="localhost", port=9010, graph_name=graph_name, max_connections=1
        )
        try:
            await backend.connect()
        except Exception as exc:
            pytest.skip(f"FalkorDB not available: {exc}")

        sem = asyncio.Semaphore(1)

        async def guarded_write(nid):
            async with sem:
                await backend.add_node(nid, {"content": "x"}, memory_type="semantic")

        try:
            ids = [f"pd-{uuid4().hex[:8]}" for _ in range(5)]
            await asyncio.gather(*[guarded_write(nid) for nid in ids])
            async with sem:
                count = await backend.get_node_count()
            assert count == 5
        finally:
            await backend.clear()
            await backend.close()


# ---------------------------------------------------------------------------
# Connection Lifecycle (3 tests)
# ---------------------------------------------------------------------------

class TestConnectionLifecycleIntegration:
    async def test_operations_after_close_raise(self, async_backend):
        """connect → close → operation raises RuntimeError."""
        await async_backend.close()
        with pytest.raises(RuntimeError, match="connect()"):
            await async_backend.get_node("n1")

    async def test_close_idempotent(self, async_backend):
        """close() twice does not raise."""
        await async_backend.close()
        await async_backend.close()

    async def test_connect_to_unavailable_host(self):
        """connect() to a bad host raises within a reasonable timeout."""
        backend = AsyncFalkorDBBackend(
            host="127.0.0.1",
            port=19999,  # unused port
            graph_name="noop",
            max_connections=1,
        )
        await backend.connect()  # FalkorDB is lazy — connection happens on first query
        with pytest.raises(Exception):
            # This should fail because port 19999 is not listening
            await backend.get_node("n1")
        await backend.close()


# ---------------------------------------------------------------------------
# Multi-Tenant Isolation (4 tests)
# ---------------------------------------------------------------------------

class TestMultiTenantIsolation:
    async def test_scope_provider_stamps_write_context(self, async_backend):
        """Nodes receive workspace_id/user_id from scope provider."""
        scope = DefaultScopeProvider(workspace_id="ws-A", user_id="user-1")
        graph_name = f"test_scope_{uuid4().hex[:10]}"
        backend = AsyncFalkorDBBackend(
            host="localhost", port=9010, graph_name=graph_name, scope_provider=scope
        )
        try:
            await backend.connect()
        except Exception as exc:
            pytest.skip(f"FalkorDB not available: {exc}")

        node_id = f"scoped-{uuid4().hex[:8]}"
        try:
            await backend.add_node(node_id, {"content": "scoped content"}, memory_type="semantic")

            node = await backend.get_node(node_id)
            assert node is not None
            # workspace_id and user_id are stripped by get_node (via is_global), but
            # we can verify they were stored via a raw query
            raw = await backend._ro_query(
                "MATCH (n {item_id: $id}) RETURN n.workspace_id, n.user_id",
                {"id": node_id},
            )
            ws, uid = raw[0][0], raw[0][1]
            assert ws == "ws-A"
            assert uid == "user-1"
        finally:
            await backend.clear()
            await backend.close()

    async def test_scope_provider_filters_reads(self, async_backend):
        """search_nodes only returns nodes from the correct workspace."""
        graph_name = f"test_filter_{uuid4().hex[:10]}"
        scope_a = DefaultScopeProvider(workspace_id="ws-filter-A")
        scope_b = DefaultScopeProvider(workspace_id="ws-filter-B")

        b_a = AsyncFalkorDBBackend(host="localhost", port=9010, graph_name=graph_name, scope_provider=scope_a)
        b_b = AsyncFalkorDBBackend(host="localhost", port=9010, graph_name=graph_name, scope_provider=scope_b)
        try:
            await b_a.connect()
            await b_b.connect()
        except Exception as exc:
            pytest.skip(f"FalkorDB not available: {exc}")

        unique_tag = uuid4().hex
        id_a = f"a-{uuid4().hex[:8]}"
        id_b = f"b-{uuid4().hex[:8]}"
        try:
            await b_a.add_node(id_a, {"tag": unique_tag}, memory_type="semantic")
            await b_b.add_node(id_b, {"tag": unique_tag}, memory_type="semantic")

            results_a = await b_a.search_nodes({"tag": unique_tag})
            results_b = await b_b.search_nodes({"tag": unique_tag})

            ids_a = {r.get("item_id") for r in results_a}
            ids_b = {r.get("item_id") for r in results_b}

            assert id_a in ids_a
            assert id_b not in ids_a
            assert id_b in ids_b
            assert id_a not in ids_b
        finally:
            await b_a.clear()
            await b_a.close()
            await b_b.close()

    async def test_cross_workspace_isolation(self, async_backend):
        """Two backends with different workspace_ids see different data."""
        graph_name = f"test_cross_{uuid4().hex[:10]}"
        scope_x = DefaultScopeProvider(workspace_id="ws-X")
        scope_y = DefaultScopeProvider(workspace_id="ws-Y")
        bx = AsyncFalkorDBBackend(host="localhost", port=9010, graph_name=graph_name, scope_provider=scope_x)
        by = AsyncFalkorDBBackend(host="localhost", port=9010, graph_name=graph_name, scope_provider=scope_y)
        try:
            await bx.connect()
            await by.connect()
        except Exception as exc:
            pytest.skip(f"FalkorDB not available: {exc}")

        tag = uuid4().hex
        xid = f"x-{uuid4().hex[:8]}"
        yid = f"y-{uuid4().hex[:8]}"
        try:
            await bx.add_node(xid, {"tag": tag}, memory_type="semantic")
            await by.add_node(yid, {"tag": tag}, memory_type="semantic")

            rx = await bx.search_nodes({"tag": tag})
            ry = await by.search_nodes({"tag": tag})

            assert {r["item_id"] for r in rx} == {xid}
            assert {r["item_id"] for r in ry} == {yid}
        finally:
            await bx.clear()
            await bx.close()
            await by.close()

    async def test_global_nodes_skip_scoping(self, async_backend):
        """is_global=True nodes are stored without workspace_id."""
        graph_name = f"test_global_{uuid4().hex[:10]}"
        scope = DefaultScopeProvider(workspace_id="ws-G", user_id="u-G")
        backend = AsyncFalkorDBBackend(
            host="localhost", port=9010, graph_name=graph_name, scope_provider=scope
        )
        try:
            await backend.connect()
        except Exception as exc:
            pytest.skip(f"FalkorDB not available: {exc}")

        global_id = f"global-{uuid4().hex[:8]}"
        try:
            await backend.add_node(global_id, {"content": "shared"}, memory_type="semantic", is_global=True)

            raw = await backend._ro_query(
                "MATCH (n {item_id: $id}) RETURN n.workspace_id, n.user_id",
                {"id": global_id},
            )
            ws, uid = raw[0][0], raw[0][1]
            assert ws is None
            assert uid is None
        finally:
            await backend.clear()
            await backend.close()


# ---------------------------------------------------------------------------
# Performance Baseline (3 tests, informational — no hard thresholds)
# ---------------------------------------------------------------------------

class TestPerformanceBaseline:
    """Informational only — results are printed, no assertions fail on latency."""

    async def test_single_query_latency_async(self, async_backend):
        """Log async single-query latency for baseline."""
        node_id = f"perf-{uuid4().hex[:8]}"
        await async_backend.add_node(node_id, {"content": "perf"}, memory_type="semantic")

        iterations = 20
        start = time.perf_counter()
        for _ in range(iterations):
            await async_backend.get_node(node_id)
        elapsed = time.perf_counter() - start
        avg_ms = (elapsed / iterations) * 1000
        print(f"\n[perf] Async single-query avg: {avg_ms:.2f}ms over {iterations} iterations")

    async def test_concurrent_read_throughput(self, async_backend):
        """Compare 100 concurrent async reads vs baseline."""
        node_id = f"perf-rd-{uuid4().hex[:8]}"
        await async_backend.add_node(node_id, {"content": "throughput"}, memory_type="semantic")

        n = 100
        start = time.perf_counter()
        await asyncio.gather(*[async_backend.get_node(node_id) for _ in range(n)])
        concurrent_elapsed = time.perf_counter() - start

        # Sequential baseline
        start = time.perf_counter()
        for _ in range(n):
            await async_backend.get_node(node_id)
        sequential_elapsed = time.perf_counter() - start

        print(
            f"\n[perf] {n} reads — concurrent: {concurrent_elapsed:.3f}s | "
            f"sequential: {sequential_elapsed:.3f}s | "
            f"speedup: {sequential_elapsed / concurrent_elapsed:.1f}x"
        )

    async def test_concurrent_write_throughput(self, async_backend):
        """Compare 100 concurrent writes vs sequential baseline."""
        n = 50  # writes are heavier; 50 is sufficient for spike validation

        ids_concurrent = [f"cwr-{uuid4().hex[:8]}" for _ in range(n)]
        start = time.perf_counter()
        await asyncio.gather(*[
            async_backend.add_node(nid, {"content": "c"}, memory_type="semantic")
            for nid in ids_concurrent
        ])
        concurrent_elapsed = time.perf_counter() - start

        await async_backend.clear()
        # need to reconnect after clear (graph is deleted)
        try:
            await async_backend.connect()
        except Exception:
            pass  # already connected

        ids_seq = [f"swr-{uuid4().hex[:8]}" for _ in range(n)]
        start = time.perf_counter()
        for nid in ids_seq:
            await async_backend.add_node(nid, {"content": "s"}, memory_type="semantic")
        sequential_elapsed = time.perf_counter() - start

        print(
            f"\n[perf] {n} writes — concurrent: {concurrent_elapsed:.3f}s | "
            f"sequential: {sequential_elapsed:.3f}s | "
            f"speedup: {sequential_elapsed / concurrent_elapsed:.1f}x"
        )
