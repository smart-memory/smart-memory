"""DIST-LITE-4: Viewer HTTP server — static SPA + local graph API + events server.

Single uvicorn process serving:
  GET /          →  static/index.html (LocalApp.jsx build)
  GET /memory/*  →  local_api.py (SQLite-backed, read-only)
  ws://:9004     →  events_server.start_background() (DIST-LITE-3, daemon thread)

The module-level ``app = _build_app()`` is side-effect-free — it does not start uvicorn
or the events server. This makes the module safely importable by tests.
"""
import threading
import webbrowser
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from smartmemory_app.local_api import api as _local_api

STATIC_DIR = Path(__file__).parent / "static"
DEFAULT_PORT = 9005


def _build_app() -> FastAPI:
    app = FastAPI()
    # Mount local_api at /memory — sub-app routes (e.g. /graph/full) become /memory/graph/full,
    # matching createFetchAdapter's expected paths (fetchAdapter.js:34-49).
    app.mount("/memory", _local_api)
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
    return app


# Module-level app — importable by tests without starting uvicorn or events server.
app = _build_app()


def main(port: int = DEFAULT_PORT, open_browser: bool = True) -> None:
    """Start the viewer server.

    Starts the DIST-LITE-3 events server as a background daemon thread (idempotent),
    optionally opens the browser, then runs uvicorn on the given port.
    """
    # Start DIST-LITE-3 events server — idempotent, lock-protected (events_server.py:131-150)
    from smartmemory_app.events_server import start_background
    start_background()

    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(f"http://localhost:{port}")).start()

    uvicorn.run(app, host="localhost", port=port, log_level="warning")
