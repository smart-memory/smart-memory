"""Unit tests for DIST-LITE-3: events_server.py and event_sink.py.

Tests cover:
  - get_event_sink() singleton semantics (sequential + concurrent)
  - start_background() idempotency (sequential + concurrent)
  - _serve() OSError handling + attach_loop(None) in finally
  - Broadcaster: delivers item to connected mock client
  - Broadcaster: broken client does not kill loop (return_exceptions=True)
  - stop_event causes broadcaster to exit
"""
import asyncio
import threading
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# get_event_sink() singleton
# ---------------------------------------------------------------------------


class TestGetEventSink:
    def setup_method(self):
        """Reset singleton before each test."""
        import smartmemory_pkg.event_sink as _mod
        _mod._sink = None

    def teardown_method(self):
        import smartmemory_pkg.event_sink as _mod
        _mod._sink = None

    def test_sequential_calls_return_same_instance(self):
        from smartmemory_pkg.event_sink import get_event_sink
        a = get_event_sink()
        b = get_event_sink()
        assert a is b

    def test_concurrent_calls_create_exactly_one_instance(self):
        """Two threads racing through get_event_sink() must create exactly one sink."""
        from smartmemory_pkg.event_sink import get_event_sink

        results = []
        barrier = threading.Barrier(2)

        def _worker():
            barrier.wait()
            results.append(get_event_sink())

        t1 = threading.Thread(target=_worker)
        t2 = threading.Thread(target=_worker)
        t1.start(); t2.start()
        t1.join(timeout=5); t2.join(timeout=5)

        assert len(results) == 2
        # Both must be the exact same object
        assert results[0] is results[1]


# ---------------------------------------------------------------------------
# start_background() idempotency
# ---------------------------------------------------------------------------


class TestStartBackground:
    def setup_method(self):
        import smartmemory_pkg.events_server as _mod
        _mod._server_thread = None
        _mod._stop_event.clear()

    def teardown_method(self):
        import smartmemory_pkg.events_server as _mod
        _mod._stop_event.set()
        if _mod._server_thread is not None:
            _mod._server_thread.join(timeout=2)
        _mod._server_thread = None
        _mod._stop_event.clear()

    def test_sequential_idempotent(self):
        """Second call while thread is alive spawns no new thread."""
        import smartmemory_pkg.events_server as _mod

        # Patch _serve so the thread doesn't actually try to bind
        async def _fake_serve(port=9004):
            await asyncio.sleep(10)  # block until stop_event or timeout

        with patch("smartmemory_pkg.events_server._serve", side_effect=_fake_serve):
            _mod.start_background(port=9999)
            first_thread = _mod._server_thread

            _mod.start_background(port=9999)
            second_thread = _mod._server_thread

        assert first_thread is second_thread

    def test_concurrent_safety(self):
        """Two concurrent callers each acquiring the lock still spawn exactly one thread."""
        import smartmemory_pkg.events_server as _mod

        async def _fake_serve(port=9004):
            await asyncio.sleep(10)

        with patch("smartmemory_pkg.events_server._serve", side_effect=_fake_serve):
            threads_spawned = []
            barrier = threading.Barrier(2)

            def _caller():
                barrier.wait()
                _mod.start_background(port=9999)
                threads_spawned.append(_mod._server_thread)

            t1 = threading.Thread(target=_caller)
            t2 = threading.Thread(target=_caller)
            t1.start(); t2.start()
            t1.join(timeout=5); t2.join(timeout=5)

        # Both callers should observe the same thread object
        assert len(threads_spawned) == 2
        assert threads_spawned[0] is threads_spawned[1]


# ---------------------------------------------------------------------------
# _serve() error handling
# ---------------------------------------------------------------------------


class TestServe:
    def test_oserror_logs_warning_does_not_raise(self):
        """OSError binding failure logs a warning and exits cleanly."""
        import smartmemory_pkg.events_server as _mod
        from smartmemory.observability.events import InProcessQueueSink

        mock_sink = MagicMock(spec=InProcessQueueSink)
        mock_sink._q = asyncio.Queue()

        with (
            patch("smartmemory_pkg.events_server.log") as mock_log,
            patch("smartmemory_pkg.event_sink.get_event_sink", return_value=mock_sink),
            patch("websockets.serve", side_effect=OSError("address in use")),
        ):
            asyncio.run(_mod._serve(port=19999))

        mock_log.warning.assert_called_once()
        # The warning format string contains %d for port — verify port is in the args
        warn_args = mock_log.warning.call_args[0]
        assert any(str(19999) in str(a) for a in warn_args)

    def test_attach_loop_none_called_in_finally(self):
        """attach_loop(None) is always called even when OSError is raised."""
        import smartmemory_pkg.events_server as _mod
        from smartmemory.observability.events import InProcessQueueSink

        mock_sink = MagicMock(spec=InProcessQueueSink)
        mock_sink._q = asyncio.Queue()
        attach_calls = []
        mock_sink.attach_loop.side_effect = lambda loop: attach_calls.append(loop)

        with (
            patch("smartmemory_pkg.event_sink.get_event_sink", return_value=mock_sink),
            patch("websockets.serve", side_effect=OSError("fail")),
            patch("smartmemory_pkg.events_server.log"),
        ):
            asyncio.run(_mod._serve(port=19999))

        # First call: real loop. Second (finally): None.
        assert attach_calls[-1] is None

    def test_stop_event_exits_broadcast_loop(self):
        """Setting stop_event causes _serve to exit the while loop."""
        import smartmemory_pkg.events_server as _mod
        from smartmemory.observability.events import InProcessQueueSink

        mock_sink = MagicMock(spec=InProcessQueueSink)
        mock_sink._q = asyncio.Queue()
        mock_sink.attach_loop = MagicMock()

        _mod._stop_event.clear()

        async def _run():
            # Set stop_event after a brief delay so _serve exits quickly
            async def _set_stop():
                await asyncio.sleep(0.05)
                _mod._stop_event.set()

            mock_ws_server = MagicMock()
            mock_ws_server.__aenter__ = AsyncMock(return_value=mock_ws_server)
            mock_ws_server.__aexit__ = AsyncMock(return_value=False)

            with (
                patch("smartmemory_pkg.event_sink.get_event_sink", return_value=mock_sink),
                patch("websockets.serve", return_value=mock_ws_server),
                patch("smartmemory_pkg.events_server.log"),
            ):
                await asyncio.gather(
                    _mod._serve(port=19999),
                    _set_stop(),
                )

        asyncio.run(_run())
        # If we get here without hanging, stop_event correctly terminated the loop.


# ---------------------------------------------------------------------------
# Broadcaster
# ---------------------------------------------------------------------------


class TestBroadcast:
    def test_item_sent_to_connected_client(self):
        """An item in the queue is broadcast as a new_event envelope to connected clients."""
        import smartmemory_pkg.events_server as _mod
        from smartmemory.observability.events import InProcessQueueSink

        import json

        async def _run():
            sink = InProcessQueueSink()
            # Simulate a graph.add_node span event as produced by _emit_span
            item = {
                "event_type": "span_event",
                "component": "graph",
                "operation": "add_node",
                "name": "graph.add_node",
                "trace_id": "",
                "span_id": "abc123",
                "parent_span_id": None,
                "memory_id": "item-1",
                "memory_type": "semantic",
                "label": "hello",
            }
            await sink._q.put(item)

            mock_ws = AsyncMock()
            clients = {mock_ws}

            loop = asyncio.get_running_loop()
            sink.attach_loop(loop)

            await _mod._broadcast(sink, clients)

            # Must be wrapped in new_event envelope; data contains node-specific fields
            sent_msg = json.loads(mock_ws.send.call_args[0][0])
            assert sent_msg["type"] == "new_event"
            assert sent_msg["component"] == "graph"
            assert sent_msg["operation"] == "add_node"
            assert sent_msg["data"]["memory_id"] == "item-1"
            assert sent_msg["trace_id"] is None  # empty string → None

        asyncio.run(_run())

    def test_broken_client_does_not_kill_loop(self):
        """return_exceptions=True: an exception from one client doesn't crash the broadcast."""
        import smartmemory_pkg.events_server as _mod
        from smartmemory.observability.events import InProcessQueueSink

        async def _run():
            sink = InProcessQueueSink()
            await sink._q.put({"event_type": "test"})

            broken = AsyncMock()
            broken.send.side_effect = Exception("connection reset")
            healthy = AsyncMock()
            clients = {broken, healthy}

            loop = asyncio.get_running_loop()
            sink.attach_loop(loop)

            # Must not raise
            await _mod._broadcast(sink, clients)

            # healthy client still received the message
            healthy.send.assert_called_once()

        asyncio.run(_run())

    def test_idle_queue_returns_on_timeout(self):
        """asyncio.wait_for timeout on empty queue causes _broadcast to return without error."""
        import smartmemory_pkg.events_server as _mod
        from smartmemory.observability.events import InProcessQueueSink

        async def _run():
            sink = InProcessQueueSink()
            loop = asyncio.get_running_loop()
            sink.attach_loop(loop)
            clients: set = set()
            # Empty queue — _broadcast uses wait_for(timeout=1.0). Wrap the call so
            # the test completes in ≤2s regardless of the internal timeout.
            await asyncio.wait_for(_mod._broadcast(sink, clients), timeout=2.0)

        asyncio.run(_run())
