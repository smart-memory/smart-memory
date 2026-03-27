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

### Core

```bash
smartmemory add "text"                # Add a memory (default type: episodic)
smartmemory add --type semantic "..."  # Add with specific memory type
smartmemory add - < notes.txt          # Add from stdin, one memory per line
smartmemory add --all - < doc.txt      # Add entire stdin as one memory
smartmemory add --project atlas "..."  # Add with arbitrary property flags

smartmemory search "query"             # Semantic search
smartmemory search "*"                 # List all memories
smartmemory search --top-k 20 "query"  # Control result count (default: 5)
smartmemory search --project atlas "q" # Filter by property

smartmemory get <item_id>              # Fetch a single memory by ID
smartmemory recall                     # Session context for Claude Code hooks
smartmemory recall --cwd /path         # Recall with working directory context
smartmemory viewer                     # Open knowledge graph viewer in browser
smartmemory viewer --port 8080         # Custom port for viewer
smartmemory models                     # List available LLM models
smartmemory config                     # View all settings
smartmemory config llm_provider        # View one setting
smartmemory config llm_provider groq   # Change a setting
smartmemory clear                      # Delete all memories and reset vectors
```

### Daemon

```bash
smartmemory start                      # Start daemon + enrichment workers
smartmemory stop                       # Stop daemon
smartmemory restart                    # Restart daemon
smartmemory status                     # Daemon health + enrichment stats
smartmemory worker                     # Run enrichment worker (drain and exit)
smartmemory worker --loop              # Run enrichment worker continuously
```

### Setup & Lifecycle

```bash
smartmemory setup                      # Interactive first-run questionnaire
smartmemory setup --mode local         # Skip questionnaire, set local mode
smartmemory setup --mode remote --api-key sk_...  # Non-interactive remote setup
smartmemory setup --for cursor         # Configure for Cursor instead of Claude Code
smartmemory server                     # Start MCP server (called by MCP clients)
smartmemory uninstall                  # Remove hooks, skills, plist, and data
smartmemory uninstall --keep-data      # Remove hooks/plist but keep memories
```

### Admin

```bash
smartmemory admin export out.jsonl     # Export memories to corpus JSONL
smartmemory admin import data.jsonl    # Import corpus JSONL into SmartMemory
smartmemory admin reindex              # Re-embed all memories with current model
smartmemory admin reextract            # Re-run entity extraction on all memories
smartmemory admin list-packs           # List available seed packs
smartmemory admin install-pack NAME    # Install a seed pack
smartmemory admin mine                 # Mine Wikidata entities via SPARQL
smartmemory admin convert-rebel        # Convert REBEL dataset to corpus JSONL
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
