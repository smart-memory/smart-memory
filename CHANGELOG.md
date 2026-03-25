# Changelog — smartmemory

## [1.1.5] — 2026-03-24

### Fixed

- **Real-time graph viewer updates work.** Lite WebSocket events server now negotiates the `sm.v1` subprotocol. Browser WebSocket API (per RFC 6455) closes connections when the server doesn't echo back a requested subprotocol — the viewer showed "disconnected" and real-time node animations never fired.

## [Unreleased]

### Added

- **CORE-PROPS-1 Phase 1b: Confidence surface wiring.** Recall excludes items below configurable confidence floor (default 0.3, `SMARTMEMORY_RECALL_FLOOR` env var). Low-confidence items (< 0.5) show `~` prefix in both CLI search and recall output. Remote recall path has parity with local path.

### Changed

- **DIST-QA-2: CLI cleanup.** `persist` renamed to `add`. Admin commands (`export`, `import`, `mine`, `convert-rebel`, `list-packs`, `install-pack`, `reindex`) moved under `smartmemory admin` subgroup.
- **Input validation.** `smartmemory add` rejects invalid `--type` values and empty content with clear error messages.
- **Wildcard search.** `smartmemory search "*"` returns all memory nodes (excludes entity/relation/pattern nodes from enrichment).
- **Admin reindex guard.** `smartmemory admin reindex` rejects with clear error in remote mode (local-only operation).
- **Auto-sync hooks on daemon start.** Hook scripts are recopied from the package to `~/.claude/hooks/` on every daemon start, so pip upgrades that change hook content take effect without re-running `smartmemory setup`.
- **`ingest` command removed.** `add` is now the single entry point — `ingest` was identical (both called `/memory/ingest`).

### Fixed

- **Daemon no longer crashes after ingest.** Disabled evolution worker in daemon pipeline profile — evolvers crash with missing typed configs (EpisodicDecayEvolver, ExponentialDecayEvolver datetime serialization). Evolution re-enabled when configs are wired properly.
- **Sync ingest for reliable LLM extraction.** Ingest endpoint uses sync pipeline instead of two-tier async. The `_drain_running` bool was captured by value on import (Python immutable import semantics), so async enrichment never enqueued jobs. Sync path works correctly with the direct Groq SDK.
- **Launchd plist has GROQ_API_KEY.** Daemon managed by launchd didn't inherit shell env vars — LLM enrichment was silently disabled.
- **Async enrichment no longer drops nodes on SQLite.** Tier 2 LLM entities used deterministic SHA256[:16] item_ids that could collide with existing nodes via SQLite's `ON CONFLICT DO UPDATE`. Entity IDs are now stripped before persist on backends without dual-node support.
- **Recall works with few memories.** Recency sort key returned `""` for `None` created_at, pushing items with missing timestamps to the end. Fixed to use `"0000-00-00"` fallback.

## [1.0.10] — 2026-03-21

### Fixed

- **Search returns only matching results.** Lite mode search no longer returns unrelated memories. Text-first fallback chain (substring + keyword) replaces unreliable vector search on small corpora.
- **Add no longer drops nodes with LLM key.** Batch evolution disabled in Tier 1 config — destructive evolvers were deleting nodes during rapid sequential ingests.
- **Recall endpoint works.** `/recall` route moved before `/{memory_id}` wildcard to prevent 404 capture.
- **Recall returns results.** Recency sort and empty-query handling fixed in search fallback.

### Added

- **`smartmemory get <item_id>`** — retrieve a memory by ID via CLI.
- **Auto-restart on pip upgrade.** Daemon middleware checks installed package version every 10th request. Version mismatch triggers clean exit; launchd restarts with new code.
- **CLI retry on daemon restart.** `_daemon_request` retries once with 2s wait on connection drop for seamless upgrades.
- **Arbitrary properties.** `smartmemory add "text" --project atlas --domain legal` passes extra properties.

## [1.0.5] — 2026-03-19

### Added

#### DIST-SETUP-TUI-1 — Interactive Setup TUI (COMPLETE)

- **Textual TUI for `smartmemory setup`**: Arrow-key selection for mode, LLM provider, embedding provider. 6 screens: Welcome, LLM, Model Discovery, Embedding, Summary, Progress.
- **Live model discovery**: `@work` async worker fetches available models from ollama/lmstudio with loading indicator and graceful failure.
- **Summary screen**: Edit data directory, toggle coreference, review all choices before confirming.
- **Progress screen**: Per-step checklist with indeterminate spinner for daemon startup.
- **Graceful fallback**: Non-interactive environments (pipes, CI, `TERM=dumb`, missing textual) fall back to existing click prompts automatically.
- **Optional dependency**: `pip install smartmemory[tui]` adds Textual. Base install falls back to click.
- **`SetupResult` dataclass**: Clean contract between TUI and business logic.
- **`_can_run_tui()`**: Detects interactive terminal availability.
- **`_apply_setup_result()`**: Shared post-config logic with `on_step` callback for per-step progress updates.

### Fixed

- **`_seed_data_dir()`**: Now accepts explicit `data_dir` parameter with `expanduser()`. Previously ignored user-selected directory from setup.

## [1.0.4] — 2026-03-19

### Added

#### DIST-DAEMON-1 — Async Background Enrichment (COMPLETE)

- **Two-tier ingest pipeline**: When LLM API key is available, `POST /memory/ingest` runs Tier 1 (spaCy + EntityRuler, ~4ms) synchronously and returns immediately. Tier 2 (LLM extraction via `process_extract_job()`) runs in a background drain thread, progressively improving extraction quality from 96.9% to 100% E-F1.
- **`AsyncEnrichmentQueue`**: Thread-safe bounded deque (`maxlen=10_000`) with enqueue/dequeue_all/wait/clear/stats. Overflow drops oldest items and tracks `total_dropped`.
- **Background drain thread**: Daemon thread in `viewer_server.py` processes queued items one at a time, acquiring `_rw_lock` to serialize with API endpoints. Graceful shutdown via `_stop_event`. Re-fetches SmartMemory singleton per item to handle `/clear` invalidation.
- **Health endpoint enrichment stats**: `/health` now includes `async_enrichment` object with `enabled`, `pending`, `total_enqueued`, `total_processed`, `total_failed`, `total_dropped`.
- **`smartmemory status` enrichment display**: Shows enrichment queue stats when drain thread is active.
- **Thread safety hardened**: All read endpoints (`/graph/full`, `/graph/edges`, `/list`, `/{id}/neighbors`, `/{id}`) and `/health` now acquire `_rw_lock` for backend access.

### Changed

- **`storage.ingest()`**: Now accepts `sync` parameter. Passes `memory_type` via `context={"memory_type": ...}` instead of loose kwarg (fixes memory_type not being preserved for raw string inputs).
- **`SmartMemory.ingest(sync=False)`**: Now returns `entity_ids` in result dict alongside `item_id` and `queued`.

## [1.0.3] — 2026-03-16

### Fixed

- **Hook safety**: Namespaced hook files (`smartmemory-session-start.sh` etc.) so `smartmemory setup` never clobbers other apps' hooks. Migrates legacy registrations in `settings.json`.
- **Hook registration format**: Updated to current Claude Code hooks API format (`{matcher, hooks: [{type, command}]}`).
- **Wheel packaging**: Added `hooks/*.sh` and `skills/*.md` to `pyproject.toml` includes — were missing from built wheels.

### Added

- **`smartmemory server`** command — starts MCP server (with events-server in background).
- **`smartmemory clear`** command — deletes all local memories and resets vector index.
- **Pinned embedding provider**: `embedding_provider` config field (`local`/`openai`/`ollama`) prevents env var changes from silently switching providers and causing dimension mismatches.

### Changed

- **Default to local**: Setup question 1 defaults to local mode.
- **`events-server`** command hidden from help (still accessible for debugging).

## [1.0.2] — 2026-03-05

### Changed

- Bumped `smartmemory-core[lite]` minimum to `>=0.5.4` (DIST-QA-1: fixes Version node duplication in search results, adds `get()` to usearch backend for embedding retrieval).

#### CORE-EXT-2 — Unified PatternManager Backend (COMPLETE)

- **`LitePatternManager` deleted** from `smartmemory_app/patterns.py`. Replaced by `JSONLPatternStore`, a `PatternStore`-protocol-conformant class with atomic `FileLock`-protected writes and source provenance preservation on update.
- **`storage.py`**: `get_memory()` now constructs `PatternManager(store=JSONLPatternStore(data_path))` instead of `LitePatternManager(data_path)`.
- **`setup.py`**: `_seed_data_dir()` instantiates `JSONLPatternStore(data_dir)` (seeds `entity_patterns.jsonl` on first run via `__init__` side-effect).
- **`JSONLPatternStore.save()`**: source field is written on CREATE only — not overwritten on frequency increment updates. Matches FalkorDB `ON MATCH SET` provenance semantics.
- 108 unit tests + 5 integration tests green; zero `LitePatternManager` references remain in source or test files.

---

## [1.0.1] - 2026-03-01

### Fixed

- Bumped `smartmemory-core[lite]` minimum to `>=0.5.3` (fixes pymongo lazy import, VectorStore lazy init, and no-op VersionTracker for SQLiteBackend).

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
