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


async def _broadcast(sink, clients: set) -> None:
    """Send one queued event to all connected clients.

    Uses asyncio.wait_for to bound the get() wait so the caller can check
    stop_event without blocking forever on an idle queue.
    """
    try:
        item = await asyncio.wait_for(sink._q.get(), timeout=1.0)
    except asyncio.TimeoutError:
        return

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
                await ws.wait_closed()
            finally:
                clients.discard(ws)
                log.info("events-server: client disconnected (total: %d)", len(clients))

        async with websockets.serve(_handler, "localhost", port):
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
