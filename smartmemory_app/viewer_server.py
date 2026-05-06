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

    # Capture versions at startup — used to detect pip upgrades.
    # On every request, compare against installed versions. If different,
    # exit cleanly — launchd KeepAlive restarts with new code.
    _startup_versions: dict[str, str] = {}
    try:
        from importlib.metadata import version as _pkg_version
        _startup_versions = {
            "smartmemory": _pkg_version("smartmemory"),
            "smartmemory-core": _pkg_version("smartmemory-core"),
        }
    except Exception:
        pass

    _version_check_counter = 0  # Only check every 10th request to avoid I/O overhead

    @app.middleware("http")
    async def _version_guard(request, call_next):
        """Auto-restart daemon when pip upgrade is detected.

        Checks installed package version every 10th request. If a version
        mismatch is found, the daemon exits cleanly and launchd restarts it.
        """
        nonlocal _version_check_counter
        _version_check_counter += 1
        if _startup_versions and _version_check_counter % 10 == 0:
            try:
                from importlib.metadata import version as _pkg_version
                for pkg, startup_ver in _startup_versions.items():
                    current = _pkg_version(pkg)
                    if current != startup_ver:
                        print(f"{pkg} version changed ({startup_ver} → {current}), restarting...", flush=True)
                        os._exit(0)
            except Exception:
                pass
        return await call_next(request)

    @app.get("/health")
    def health():
        """Daemon health check. Used by daemon.is_running() to verify ownership."""

        from smartmemory_app.config import load_config
        cfg = load_config()
        backend_ok = False
        node_count = -1
        mem = None
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
        # Enrichment queue status (SQLite-backed, separate worker process)
        async_info: dict = {"enabled": False}
        try:
            from smartmemory_app.enrichment_queue import stats as queue_stats
            qs = queue_stats()
            async_info = {"enabled": True, **qs}
        except Exception:
            pass

        # DIST-OBSIDIAN-LITE-1: capability block lets the Obsidian plugin (and
        # any other client) detect lite vs remote-proxy mode + which write
        # operations are available. `mode` is the source of truth — clients
        # should branch on this rather than infer from llm_provider, etc.
        from smartmemory_app.remote_backend import RemoteMemory as _RM
        if isinstance(mem, _RM):
            mode = "remote"
            capabilities = {
                "delete": False,
                "patch": False,
                "neighbors_direction": True,
                "quota": False,
                "auth": False,
            }
        else:
            mode = "lite"
            capabilities = {
                "delete": True,
                "patch": True,
                "neighbors_direction": True,
                "quota": False,
                "auth": False,
            }

        return {
            "service": "smartmemory",
            "status": "ok" if backend_ok else "degraded",
            "memories": node_count,
            "llm_provider": cfg.llm_provider,
            "embedding_provider": cfg.embedding_provider,
            "pid": os.getpid(),
            "async_enrichment": async_info,
            "mode": mode,
            "capabilities": capabilities,
        }

    # DIST-AGENT-HOOKS-1: Mount lifecycle API at /lifecycle on root app
    from smartmemory_app.lifecycle_api import lifecycle_router
    app.include_router(lifecycle_router, prefix="/lifecycle")

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

    # Load LLM API keys: env → keychain → shell profile.
    # setup stores keys in all three locations. Keychain and profile
    # are available immediately without sourcing .zshrc in a new shell.
    _LLM_KEYS = ["GROQ_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY"]
    for key_name in _LLM_KEYS:
        if os.environ.get(key_name):
            continue
        # Try keychain
        try:
            import keyring
            stored = keyring.get_password("smartmemory", key_name)
            if stored:
                os.environ[key_name] = stored
                print(f"  Loaded {key_name} from keychain", flush=True)
                continue
        except Exception:
            pass
        # Try shell profile
        try:
            from smartmemory_app.setup import _read_env_from_profile
            stored = _read_env_from_profile(key_name)
            if stored:
                os.environ[key_name] = stored
                print(f"  Loaded {key_name} from shell profile", flush=True)
        except Exception:
            pass

    print("Loading SmartMemory backend...", flush=True)
    t0 = time.time()
    try:
        get_memory()
        print(f"Backend ready ({time.time() - t0:.1f}s)", flush=True)
    except Exception as e:
        print(f"Warning: backend init failed ({e}) — daemon running in degraded mode", flush=True)

    # Warm embedding model — first embed() triggers lazy model load (~3-5s).
    # Do it here so the first ingest/search request doesn't time out.
    try:
        from smartmemory.plugins.embedding import EmbeddingService
        t1 = time.time()
        svc = EmbeddingService()
        svc.embed("warmup")
        print(f"Embedding model ready ({time.time() - t1:.1f}s, provider={svc.provider})", flush=True)
    except Exception as e:
        print(f"Warning: embedding warmup failed ({e})", flush=True)

    # Sync hook scripts from package → ~/.claude/hooks/ on every daemon start.
    # Ensures pip upgrades that change hook content (e.g. persist→add rename)
    # take effect without requiring users to re-run `smartmemory setup`.
    try:
        from smartmemory_app.setup import _copy_hooks
        _copy_hooks()
    except Exception as e:
        print(f"Warning: hook sync failed ({e})", flush=True)

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

    # Enrichment is handled by a separate worker process (smartmemory worker --loop).
    # The ingest endpoint enqueues to a SQLite table; the worker drains it.
    # No in-process threading — avoids the _drain_running import bug and
    # keeps the daemon process stable.
    print("Enrichment queue: SQLite-backed (run `smartmemory worker --loop` for Tier 2)", flush=True)

    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(f"http://localhost:{port}")).start()

    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
