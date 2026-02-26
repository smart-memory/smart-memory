# smartmemory

AI memory for Claude Code and other MCP-compatible tools. Stores memories locally (SQLite, no Docker) or connects to the SmartMemory hosted service — one package, one install, configured at first run.

## Requirements

- Python 3.11+
- macOS, Linux, or Windows

## Install

```bash
pip install smartmemory
```

Base install (~50MB): MCP server, viewer, events server, CLI. Works immediately for remote mode.

## First run

```bash
smartmemory setup
```

Asks whether to store memories locally or use the hosted service, then configures accordingly.

**Local mode** installs additional deps (~250MB: spaCy, USearch) and wires Claude Code hooks.

**Remote mode** validates your API key and stores it in the OS keychain.

## Non-interactive / CI

```bash
# Local
pip install smartmemory[local]
SMARTMEMORY_MODE=local smartmemory server

# Remote
SMARTMEMORY_MODE=remote SMARTMEMORY_API_KEY=sk_... smartmemory server
```

Env vars always override config file — the correct path for Docker and CI.

## Commands

```bash
smartmemory setup            # First-run questionnaire
smartmemory server           # Start MCP server
smartmemory viewer           # Open knowledge graph viewer
smartmemory events-server    # Run WebSocket events server standalone
smartmemory uninstall        # Remove hooks, skills, and optionally data
```

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
