"""DIST-DAEMON-1: SmartMemory daemon — HTTP API + static viewer + events WebSocket.

Single uvicorn process serving:
  GET  /health     →  daemon health check (root app, not /memory sub-app)
  GET  /           →  static/index.html (LocalApp.jsx build)
  /memory/*        →  local_api.py (graph + ingest + search + recall + clear)
  ws://:9015       →  events_server.start_background() (DIST-LITE-3, daemon thread)

The module-level ``app = _build_app()`` is side-effect-free — it does not start uvicorn
or the events server. This makes the module safely importable by tests.
"""
import atexit
import os
import threading
import time
import webbrowser
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from smartmemory_app.local_api import api as _local_api

STATIC_DIR = Path(__file__).parent / "static"
DEFAULT_PORT = 9014


def _build_app() -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    def health():
        """Daemon health check. Used by daemon.is_running() to verify ownership."""
        from smartmemory_app.config import load_config
        cfg = load_config()
        backend_ok = False
        node_count = -1
        try:
            from smartmemory_app.storage import get_memory
            mem = get_memory()
            backend_ok = mem is not None
            from smartmemory_app.remote_backend import RemoteMemory
            if not isinstance(mem, RemoteMemory):
                try:
                    from smartmemory_app.local_api import _rw_lock
                    with _rw_lock:
                        snapshot = mem._graph.backend.serialize()
                    nodes = snapshot.get("nodes", [])
                    node_count = len([n for n in nodes if n.get("memory_type") != "Version"])
                except Exception:
                    node_count = 0  # backend exists but empty/new — still healthy
        except Exception:
            pass
        # Async enrichment status
        async_info = {"enabled": False}
        try:
            from smartmemory_app.async_enrichment import _drain_running
            if _drain_running:
                from smartmemory_app.async_enrichment import get_queue
                async_info = {"enabled": True, **get_queue().stats}
        except Exception:
            pass

        return {
            "service": "smartmemory",
            "status": "ok" if backend_ok else "degraded",
            "memories": node_count,
            "llm_provider": cfg.llm_provider,
            "embedding_provider": cfg.embedding_provider,
            "pid": os.getpid(),
            "async_enrichment": async_info,
        }

    # Mount local_api at /memory — sub-app routes (e.g. /graph/full) become /memory/graph/full,
    # matching createFetchAdapter's expected paths (fetchAdapter.js:34-49).
    app.mount("/memory", _local_api)
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
    return app


# Module-level app — importable by tests without starting uvicorn or events server.
app = _build_app()


def main(port: int = DEFAULT_PORT, open_browser: bool = True) -> None:
    """Start the SmartMemory daemon.

    Eagerly warms the memory backend (spaCy + embedding model load) before
    starting uvicorn so the first API request doesn't time out.
    Writes PID file after successful warmup for daemon lifecycle management.
    """
    from smartmemory_app.storage import get_memory, _shutdown, _resolve_data_dir

    data_path = _resolve_data_dir()
    data_path.mkdir(parents=True, exist_ok=True)
    pid_file = data_path / "daemon.pid"

    print("Loading SmartMemory backend...", flush=True)
    t0 = time.time()
    try:
        get_memory()
        print(f"Backend ready ({time.time() - t0:.1f}s)", flush=True)
    except Exception as e:
        print(f"Warning: backend init failed ({e}) — daemon running in degraded mode", flush=True)

    # Write PID file after warmup
    pid_file.write_text(str(os.getpid()))

    def _cleanup():
        _shutdown()
        pid_file.unlink(missing_ok=True)

    # atexit handles cleanup on normal exit AND uvicorn's graceful SIGTERM shutdown.
    # Do NOT install a custom SIGTERM handler — uvicorn needs SIGTERM to trigger
    # its graceful shutdown (drain active requests, then exit → atexit fires).
    atexit.register(_cleanup)

    # Start events WebSocket server as background daemon thread
    from smartmemory_app.events_server import start_background
    start_background()

    # Start background LLM enrichment drain thread if:
    # 1. LLM API key available, AND
    # 2. Backend is local (not RemoteMemory — drain thread uses SmartMemory internals)
    _has_llm = bool(os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY"))
    _is_local = True
    try:
        from smartmemory_app.remote_backend import RemoteMemory
        _is_local = not isinstance(get_memory(), RemoteMemory)
    except Exception:
        pass
    if _has_llm and _is_local:
        from smartmemory_app.async_enrichment import (
            get_queue, enrichment_drain_loop, stop_drain, _stop_event,
        )
        from smartmemory_app.local_api import _rw_lock
        _stop_event.clear()
        drain_thread = threading.Thread(
            target=enrichment_drain_loop,
            args=(get_memory, get_queue(), _rw_lock),
            daemon=True,
            name="enrichment-drain",
        )
        drain_thread.start()
        atexit.register(stop_drain)
        print("Background LLM enrichment enabled", flush=True)
    else:
        print("Background enrichment disabled (no LLM API key)", flush=True)

    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(f"http://localhost:{port}")).start()

    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
