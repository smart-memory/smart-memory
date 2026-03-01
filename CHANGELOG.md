# Changelog — smartmemory

## [1.0.0] - 2026-03-01

### Breaking Changes

- **Python 3.11+ required.** Dropped Python 3.10 support (intentional product decision — `tomllib` stdlib, `asyncio.TaskGroup` alignment).
- **Package renamed** from `smartmemory-cc` to `smartmemory`. CLI entry point renamed from `smartmemory-cc` to `smartmemory`.
- **Core library renamed** from `smartmemory` (PyPI) to `smartmemory-core`. Python import name (`import smartmemory`) is unchanged.

### Added

#### DIST-LITE-5 — Setup, Config & Dual-Mode Backend

- `config.py` — XDG-aware config at `~/.config/smartmemory/config.toml`. `SmartMemoryConfig` dataclass, `load_config()` / `save_config()` (UTF-8 explicit), env var overlay, `UnconfiguredError`, `get_api_key()` / `set_api_key()` with OS keychain via `keyring`, `_detect_and_migrate()` for upgrade path. Mode validation: invalid `mode=` in config file warns and returns unconfigured; invalid `SMARTMEMORY_MODE` env var raises `ValueError` immediately.
- `remote_backend.py` — `RemoteMemory` class: httpx client wrapping the hosted API. Same MCP tool interface as local storage (`ingest`, `search`, `get`, `recall`), plus graph methods for the viewer (`get_graph_full`, `get_edges_bulk`, `get_neighbors`). `_request()` never raises. `get_neighbors()` normalises response to always include `edges` key (service omits it). `recall()` deduplicates on both `item_id` and `id` field names. `login()` persists `team_id` to config after successful auth.
- `storage.py` — dual-mode dispatch: `get_memory()` returns local `SmartMemory` or `RemoteMemory` based on config. Unconfigured state raises `UnconfiguredError` (upgrade auto-migration attempted first). All four operations (`ingest`, `search`, `get`, `recall`) have explicit remote branches — no duck-typing across asymmetric return types (`MemoryItem` vs `dict`, `sort_by` support). `filelock` imported lazily inside `ingest()` only (not available in base install).
- `local_api.py` — `_get_mem()` wrapper: `UnconfiguredError` → HTTP 503 with setup instructions; `ValueError` (invalid mode) → HTTP 400. `_get_backend()` routes through `_get_mem()` so all local paths share the 503 guard. All viewer endpoints dispatch to `RemoteMemory` graph methods in remote mode.
- `setup.py` — `smartmemory setup` questionnaire: choose local or remote mode, install `[local]` deps on demand (uv or pip), ask pipeline questions (coreference, LLM provider, data dir), write config, wire Claude Code hooks (local mode) or validate API key + store in keychain (remote mode).
- `server.py` — `login`, `whoami`, `switch_team` MCP tools (delegate to `RemoteMemory` in remote mode; no-op in local).
- `pyproject.toml` — `[local]` extra: `smartmemory-core[lite]>=0.3.1` + `filelock>=3.12`. Base install adds `httpx`, `keyring`, `tomli-w`. `requires-python` raised to `>=3.11` (intentional — `tomllib` stdlib).
- Tests — 102 unit tests (was 78): `test_config.py` (21), `test_remote_backend.py` (13 new), plus coverage for 503/400 error paths, remote dispatch branches, singleton isolation, and auth tool delegation.

### Added

#### DIST-LITE-4 — Loginless Pip-Bundled Graph Viewer

- `local_api.py` — FastAPI sub-app mounted at `/memory`; `GET /graph/full`, `POST /graph/edges`, `GET /list`, `GET /{id}/neighbors`, `GET /{id}`. `_flatten_node()` promotes `node_category`, `entity_type`, and other viewer fields from SQLite's nested `properties` blob to top-level. Route ordering ensures `/{id}/neighbors` resolves before `/{id}`.
- `viewer_server.py` — module-level `app = _build_app()` (side-effect-free); mounts `local_api` at `/memory` and serves built static assets via `StaticFiles(html=True)`. `main()` calls `start_background()` (idempotent) then `uvicorn.run()`.
- `static/index.html` — built viewer bundle (replaced placeholder by `make build-viewer`); hatchling picks it up via `smartmemory_cc/static/**/*` in `[tool.hatch.build.targets.wheel].include`.
- `cli.py` — `viewer [--port N] [--no-browser]` command with lazy import matching the `events-server` pattern.
- `pyproject.toml` — added `fastapi>=0.110`, `uvicorn>=0.29` dependencies; added static asset glob to wheel include list.
- Option B delete endpoints (`DELETE /{id}` and `DELETE /graph/nodes/{id}` → 405) added because `GraphExplorer` has no `readOnly` prop.

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
