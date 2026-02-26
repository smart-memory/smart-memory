"""DIST-LITE-3: Process-singleton InProcessQueueSink for the MCP plugin.

get_event_sink() is called from two threads:
  - main thread (via storage.get_memory())
  - daemon thread (via events_server._serve())

Double-checked locking ensures exactly one InProcessQueueSink is created.
Without the lock, both threads can race past the ``_sink is None`` check and
produce separate sink instances — the memory singleton and the events server
would then hold different queues, producing no animations.
"""
import threading

from smartmemory.observability.events import InProcessQueueSink

_sink: InProcessQueueSink | None = None
_sink_lock = threading.Lock()


def get_event_sink() -> InProcessQueueSink:
    """Return the process-singleton InProcessQueueSink, creating it on first call."""
    global _sink
    if _sink is None:
        with _sink_lock:
            if _sink is None:
                _sink = InProcessQueueSink()
    return _sink
