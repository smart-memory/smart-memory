# smartmemory

AI memory for Claude Code and other MCP-compatible tools. Stores memories locally (SQLite, no Docker) or connects to the SmartMemory hosted service — one package, one install, configured at first run.

## Requirements

- Python 3.11+
- macOS, Linux, or Windows

## Install

```bash
pip install smartmemory
```

Includes everything for local mode: core library, spaCy NLP, USearch vectors, MCP server, viewer, CLI, persistent daemon.

For the interactive setup TUI (arrow-key selection, model discovery):

```bash
pip install smartmemory[tui]
```

## First run

```bash
smartmemory setup
```

With `smartmemory[tui]` installed, this launches a Textual TUI with arrow-key selection for LLM provider, live model discovery from ollama/lmstudio, embedding provider, and a summary screen. Without `[tui]`, falls back to text prompts.

**Local mode** wires Claude Code hooks, downloads the spaCy language model (~15MB), and starts a persistent daemon.

**Remote mode** validates your API key and stores it in the OS keychain.

## Daemon

SmartMemory runs a persistent background daemon so CLI commands respond in <200ms instead of cold-starting Python every time (~22s).

```bash
smartmemory start       # Start daemon (auto-started by setup)
smartmemory stop        # Stop daemon
smartmemory restart     # Restart daemon
smartmemory status      # Show status, memory count, enrichment queue
```

On macOS, `smartmemory setup` installs a launchd plist — the daemon auto-starts on login and restarts on crash.

### Two-tier ingest

When an LLM API key is available, the daemon runs **two-tier ingestion**:

- **Tier 1 (sync, ~4ms):** spaCy + EntityRuler extracts entities immediately, returns item_id
- **Tier 2 (async, ~740ms):** Background drain thread runs LLM extraction, adds net-new entities and relations

This means `smartmemory add` returns instantly while quality improves in the background.

## Commands

```bash
smartmemory setup              # First-run questionnaire (TUI or text)
smartmemory start              # Start daemon
smartmemory stop               # Stop daemon
smartmemory status             # Daemon health + enrichment stats
smartmemory add "text"         # Add text as a memory
smartmemory search "query"     # Semantic search (use "*" for all)
smartmemory recall             # Session context for Claude Code
smartmemory get <item_id>      # Fetch a single memory by ID
smartmemory viewer             # Open knowledge graph viewer
smartmemory models             # List available LLM models
smartmemory config             # View/edit settings
smartmemory clear              # Delete all memories
smartmemory server             # Start MCP server (used by Claude Code)
smartmemory uninstall          # Remove hooks, plist, and optionally data

# Admin commands
smartmemory admin export out.jsonl   # Export memories
smartmemory admin import data.jsonl  # Import memories
smartmemory admin reindex            # Re-embed with current model
smartmemory admin list-packs         # List seed packs
smartmemory admin install-pack NAME  # Install a seed pack
smartmemory admin mine               # Mine Wikidata entities
smartmemory admin convert-rebel      # Convert REBEL dataset
```

## Non-interactive / CI

```bash
# Local
SMARTMEMORY_MODE=local smartmemory server

# Remote
SMARTMEMORY_MODE=remote SMARTMEMORY_API_KEY=sk_... smartmemory server
```

Env vars always override config file — the correct path for Docker and CI. The TUI is automatically disabled in non-interactive environments.

## Configuration

Config file: `~/.config/smartmemory/config.toml` (XDG on Linux/macOS, `%APPDATA%\smartmemory\config.toml` on Windows).

API keys are stored in the OS keychain, never in the config file. Set `SMARTMEMORY_API_KEY` as an env var on headless systems where the keychain is unavailable.

## Env vars

| Variable | Description |
|----------|-------------|
| `SMARTMEMORY_MODE` | `local` or `remote` — overrides config file |
| `SMARTMEMORY_API_KEY` | API key for remote mode — bypasses keychain |
| `SMARTMEMORY_API_URL` | Remote API URL (default: `https://api.smartmemory.ai`) |
| `SMARTMEMORY_TEAM_ID` | Team/workspace ID for remote mode |
| `SMARTMEMORY_DATA_DIR` | Local data directory (default: `~/.smartmemory`) |
| `SMARTMEMORY_LLM_PROVIDER` | LLM provider for local enrichment |
| `SMARTMEMORY_EMBEDDING_PROVIDER` | Embedding provider (`local`, `openai`, `ollama`) |
| `SMARTMEMORY_DAEMON_PORT` | Daemon port (default: `9014`) |
| `SMARTMEMORY_ASYNC_ENRICHMENT` | Enable/disable background enrichment |
