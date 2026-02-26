# Changelog — smartmemory-cc

## [Unreleased]

### Added

#### DIST-LITE-3 — Lite Mode Graph Viewer Animations (Phase 2 — plugin)

- `event_sink.py` — process-singleton `get_event_sink()` with double-checked locking; safe for concurrent calls from main thread (`get_memory()`) and daemon thread (`_serve()`).
- `events_server.py` — asyncio WebSocket broadcaster on port 9004. `start_background()` spawns a daemon thread; `_server_thread_lock` makes it idempotent under concurrent callers. Loop captured inside `_serve()` after `asyncio.run()` starts it; `sink.attach_loop(None)` called in `finally` regardless of exit path. `asyncio.wait_for(timeout=1.0)` allows `stop_event` check on idle queue. `return_exceptions=True` in `asyncio.gather` so a broken client never kills the broadcast loop. `OSError` on bind logged as warning, not raised.
- `_to_viewer_message()` — reshapes raw sink queue items to the `new_event` schema expected by `classifyEvent.js`: `type`, `component`, `operation`, `name` at top level; node/edge payload nested in `data`.
- `storage.py` — `get_memory()` now passes `event_sink=get_event_sink()` to `create_lite_memory()`.
- `server.py` — `main()` calls `start_background()` before `mcp.run()`; wrapped in `try/except` so MCP always starts even if the port is unavailable.
- `cli.py` — `smartmemory-cc events-server [--port N]` command for standalone debug use.
- `pyproject.toml` — `websockets>=12.0` added to dependencies.

## [0.1.0] — 2026-02-23

### Added

- **LitePatternManager** (`smartmemory_cc/patterns.py`) — JSONL-backed entity pattern store
  duck-typing `PatternManager.get_patterns()` for `EntityRulerStage`. Bundled seed patterns,
  frequency gate (≥2), atomic rewrite via `.tmp`. Graceful degradation when seed file absent.

- **Storage singleton** (`smartmemory_cc/storage.py`) — double-checked-locked `get_memory()`,
  filelock-protected `ingest()`, `_data_path` module var so lock and singleton always share the
  same directory. `_shutdown()` logs warnings on save failure (never silent data loss).

- **FastMCP server** (`smartmemory_cc/server.py`) — `memory_ingest`, `memory_search`,
  `memory_recall`, `memory_get` tools; all error-safe (return strings, never raise).

- **CLI** (`smartmemory_cc/cli.py`) — `persist`, `ingest`, `recall`, `setup`, `uninstall`
  commands. Entry point: `smartmemory-cc`.

- **Setup/teardown** (`smartmemory_cc/setup.py`) — idempotent hook registration via
  `_get_hook_registrations()` (lazy, patchable in tests), skill copy with no-overwrite,
  `SMARTMEMORY_DATA_DIR` env var honoured throughout.

- **Session hooks** — `session-start.sh` (recall on session open), `session-end.sh`
  (persist last assistant message), `post-tool-failure.sh` (ingest tool errors, skips interrupts).
  All hooks exit 0 and redirect stderr to `~/.smartmemory/hooks.log`.

- **Skills** — `/remember`, `/search`, `/ingest`, `/orient`.

- **Sync stub** (`smartmemory_cc/sync.py`) — raises `NotImplementedError` if
  `SMARTMEMORY_SYNC_TOKEN` set, silent no-op otherwise.

- **Test suite** — 31 unit tests + 12 integration tests (real SQLite + usearch, no mocks).
