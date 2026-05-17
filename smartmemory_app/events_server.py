"""DIST-LITE-3: Lite WebSocket events server.

Runs as a daemon thread inside the MCP process. Bridges the InProcessQueueSink
(fed by pipeline emit_event() calls) to connected WebSocket clients (graph viewer).

Correctness constraints (all present):
- _server_thread_lock guards start_background() check+spawn as an atomic unit
- loop captured inside _serve() (after asyncio.run() starts it)
- sink.attach_loop(loop) called immediately after capture
- sink.attach_loop(None) called in _serve() finally regardless of exit path
- asyncio.wait_for(sink._q.get(), timeout=1.0) allows stop_event check on idle queue
- return_exceptions=True in asyncio.gather — broken clients never kill broadcast loop
- Thread is daemon=True — exits automatically with the process
- OSError caught in _serve() → log warning, do not re-raise
"""
import asyncio
import logging
import threading
import time

log = logging.getLogger(__name__)

# Span-envelope fields that belong at the message top level.
# Everything else in a queue item becomes the nested ``data`` payload
# that classifyEvent.js reads from ``raw.data``.
_SPAN_STRUCTURAL_FIELDS = frozenset({
    "event_type", "component", "operation", "name",
    "trace_id", "span_id", "parent_span_id", "duration_ms",
    "error", "status",
})


def _to_viewer_message(item: dict) -> dict:
    """Reshape a raw sink queue item to the viewer's ``new_event`` message schema.

    classifyEvent.js expects::

        {
            "type": "new_event",
            "component": "graph",
            "operation": "add_node",
            "name": "graph.add_node",
            "event_type": "span_event",
            "trace_id": "...",        # may be null in lite mode
            "span_id": "...",
            "parent_span_id": null,
            "duration_ms": null,
            "data": {                 # node/edge-specific payload
                "memory_id": "...",
                ...
            }
        }
    """
    data = {k: v for k, v in item.items() if k not in _SPAN_STRUCTURAL_FIELDS}
    return {
        "type": "new_event",
        "event_type": item.get("event_type", "span"),
        "component": item.get("component", "unknown"),
        "operation": item.get("operation", ""),
        "name": item.get("name", ""),
        "trace_id": item.get("trace_id") or None,
        "span_id": item.get("span_id") or None,
        "parent_span_id": item.get("parent_span_id"),
        "duration_ms": item.get("duration_ms"),
        "data": data,
    }


_server_thread: threading.Thread | None = None
_server_thread_lock = threading.Lock()
_stop_event = threading.Event()

# ---------------------------------------------------------------------------
# SSE fan-out (DIST-LITE graph viewer migrated WS -> SSE).
#
# The graph package (smart-memory-graph) dropped its WebSocket transport;
# useGraphStream now only consumes SSE via GET /memory/progress/stream. The
# lite daemon still exposes this ws://:9015 server (the recorder's settled()
# depends on it), so SSE is added as an ADDITIONAL broadcast target at the
# single existing sink._q drain point — NOT a second queue consumer (a second
# sink._q.get() would steal items from the WS broadcast and break settled()).
#
# Subscribers register an asyncio.Queue + the loop it lives on (uvicorn's
# loop, a different thread/loop from this events-server loop). Cross-loop
# delivery uses loop.call_soon_threadsafe — the same bridge pattern as
# InProcessQueueSink.emit.
# ---------------------------------------------------------------------------
_sse_lock = threading.Lock()
_sse_subscribers: list = []  # list of (loop, asyncio.Queue)
_sse_seq: int = 0  # monotonic; mutated only on the single _broadcast loop


def register_sse_subscriber(loop, queue) -> tuple:
    """Register an SSE subscriber queue. Returns an opaque entry for unregister."""
    entry = (loop, queue)
    with _sse_lock:
        _sse_subscribers.append(entry)
    return entry


def unregister_sse_subscriber(entry) -> None:
    """Remove a previously registered SSE subscriber. Idempotent."""
    with _sse_lock:
        try:
            _sse_subscribers.remove(entry)
        except ValueError:
            pass


def _to_progress_event(item: dict, seq: int) -> dict | None:
    """Reshape a raw sink item into a ProgressEvent contract frame.

    Contract: progress-event-contract.json v1.5.0 — the shape
    classifyProgressEvent() (useGraphStream.js) expects. Only graph
    node/edge operations are projected (the viewer paints from these);
    other span events are dropped to keep the lite stream lean.
    """
    viewer = _to_viewer_message(item)
    op = viewer.get("operation")
    if op == "add_node":
        kind = "graph.node"
    elif op == "add_edge":
        kind = "graph.edge"
    else:
        return None
    now = time.time()
    return {
        "run_id": "lite",
        "scope": "workspace:local",
        "seq": seq,
        "ts": now,
        "kind": kind,
        "status": "ok",
        # payload.data is the same dict the WS path renders from — parity by
        # construction. original_ts drives the viewer's replay pacing.
        "payload": {"data": viewer["data"], "original_ts": now},
    }


def _fanout_sse(item: dict) -> None:
    """Push a reshaped progress frame to every registered SSE subscriber.

    Called from the single _broadcast loop. Cross-loop hand-off via
    call_soon_threadsafe; QueueFull is dropped (slow consumer, same policy
    as InProcessQueueSink).
    """
    global _sse_seq
    with _sse_lock:
        subs = list(_sse_subscribers)
    if not subs:
        return
    _sse_seq += 1
    frame = _to_progress_event(item, _sse_seq)
    if frame is None:
        return
    for sub_loop, q in subs:
        def _push(q=q, frame=frame) -> None:
            try:
                q.put_nowait(frame)
            except asyncio.QueueFull:
                pass

        try:
            sub_loop.call_soon_threadsafe(_push)
        except RuntimeError:
            # Subscriber loop closed mid-flight — unregister handles cleanup.
            pass


async def _broadcast(sink, clients: set) -> None:
    """Send one queued event to all connected clients.

    Uses asyncio.wait_for to bound the get() wait so the caller can check
    stop_event without blocking forever on an idle queue.
    """
    try:
        item = await asyncio.wait_for(sink._q.get(), timeout=1.0)
    except asyncio.TimeoutError:
        return

    # SSE fan-out (graph viewer). Done at the single drain point so the WS
    # path below still receives every event — recorder settled() unaffected.
    _fanout_sse(item)

    import json
    message = json.dumps(_to_viewer_message(item))
    if clients:
        await asyncio.gather(
            *[client.send(message) for client in clients],
            return_exceptions=True,
        )


async def _serve(port: int = 9015) -> None:
    """Run the WebSocket server and broadcast loop."""
    from smartmemory_app.event_sink import get_event_sink

    sink = get_event_sink()
    loop = asyncio.get_running_loop()
    sink.attach_loop(loop)

    clients: set = set()

    try:
        import websockets

        async def _handler(ws) -> None:
            clients.add(ws)
            log.info("events-server: client connected (total: %d)", len(clients))
            try:
                # Listen for incoming messages and rebroadcast to all other clients.
                # This allows CLI commands (e.g. `smartmemory clear`) to push events
                # to the viewer by connecting as a WebSocket client.
                async for message in ws:
                    others = {c for c in clients if c is not ws}
                    if others:
                        await asyncio.gather(
                            *[c.send(message) for c in others],
                            return_exceptions=True,
                        )
            except Exception:
                pass
            finally:
                clients.discard(ws)
                log.info("events-server: client disconnected (total: %d)", len(clients))

        async with websockets.serve(_handler, "localhost", port, subprotocols=["sm.v1"]):
            log.info("events-server: listening on ws://localhost:%d", port)
            while not _stop_event.is_set():
                await _broadcast(sink, clients)
    except OSError as exc:
        log.warning(
            "events-server: could not bind to port %d (%s). "
            "Graph viewer animations will not be available.",
            port,
            exc,
        )
    finally:
        sink.attach_loop(None)
        log.info("events-server: stopped")


def start_background(port: int = 9015) -> None:
    """Start the events server as a background daemon thread. Idempotent.

    The _server_thread_lock makes the is_alive() check + Thread() spawn atomic —
    two concurrent callers cannot both pass the check and spawn two threads.
    """
    global _server_thread

    with _server_thread_lock:
        if _server_thread is not None and _server_thread.is_alive():
            return

        _stop_event.clear()

        def _run() -> None:
            asyncio.run(_serve(port=port))

        _server_thread = threading.Thread(target=_run, daemon=True, name="smartmemory-events-server")
        _server_thread.start()
        log.info("events-server: background thread started (port %d)", port)


def stop_background() -> None:
    """Signal the background server to stop. Used for clean shutdown in tests."""
    _stop_event.set()


def main(port: int = 9015) -> None:
    """Standalone entry point. Run via: smartmemory events-server."""
    asyncio.run(_serve(port=port))
