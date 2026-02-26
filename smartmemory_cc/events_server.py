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
    message = json.dumps(item)
    if clients:
        await asyncio.gather(
            *[client.send(message) for client in clients],
            return_exceptions=True,
        )


async def _serve(port: int = 9004) -> None:
    """Run the WebSocket server and broadcast loop."""
    from smartmemory_cc.event_sink import get_event_sink

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


def start_background(port: int = 9004) -> None:
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


def main(port: int = 9004) -> None:
    """Standalone entry point. Run via: smartmemory-cc events-server."""
    asyncio.run(_serve(port=port))
