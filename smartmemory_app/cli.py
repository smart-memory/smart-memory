"""SmartMemory CLI — daemon management + memory operations.

DIST-DAEMON-1: All memory commands (persist, ingest, search, recall) try the
daemon HTTP API first (<200ms). Falls back to direct storage calls if daemon
is not running (~22s cold start).
"""
import logging

import click

log = logging.getLogger(__name__)


def _daemon_url() -> str:
    from smartmemory_app.config import load_config
    return f"http://127.0.0.1:{load_config().daemon_port}"


def _daemon_request(method: str, path: str, **kwargs):
    """Try daemon HTTP API. Returns parsed JSON or None if daemon unreachable."""
    import httpx
    try:
        r = httpx.request(method, f"{_daemon_url()}{path}", timeout=120, **kwargs)
        r.raise_for_status()
        return r.json() if r.status_code != 204 else {}
    except (httpx.ConnectError, httpx.ConnectTimeout):
        return None  # daemon not running — fall back to direct
    except httpx.ReadTimeout:
        return None  # daemon busy (model loading) — fall back to direct


@click.group()
def cli() -> None:
    """SmartMemory — persistent AI memory system."""


# Register setup/uninstall commands from setup module
from smartmemory_app.setup import setup as _setup_cmd, uninstall as _uninstall_cmd  # noqa: E402

cli.add_command(_setup_cmd, name="setup")
cli.add_command(_uninstall_cmd, name="uninstall")


# ── Daemon lifecycle ────────────────────────────────────────────────────────


@cli.command("start")
def start_cmd() -> None:
    """Start the SmartMemory daemon."""
    from smartmemory_app.daemon import start_daemon, is_running
    if is_running():
        click.echo("SmartMemory daemon is already running.")
        return
    click.echo("Starting SmartMemory daemon (loading models)...")
    try:
        start_daemon()
        click.echo("Daemon ready.")
    except Exception as e:
        click.echo(f"Failed to start daemon: {e}", err=True)
        raise SystemExit(1)


@cli.command("stop")
def stop_cmd() -> None:
    """Stop the SmartMemory daemon."""
    from smartmemory_app.daemon import stop_daemon, is_running
    if not is_running(require_healthy=False):
        click.echo("Daemon is not running.")
        return
    stop_daemon()
    click.echo("Daemon stopped.")


@cli.command("restart")
def restart_cmd() -> None:
    """Restart the SmartMemory daemon."""
    from smartmemory_app.daemon import stop_daemon, start_daemon, is_running
    if is_running(require_healthy=False):
        click.echo("Stopping daemon...")
        stop_daemon()
    click.echo("Starting daemon...")
    start_daemon()
    click.echo("Daemon ready.")


@cli.command("status")
def status_cmd() -> None:
    """Show SmartMemory daemon status."""
    from smartmemory_app.daemon import get_status
    info = get_status()
    if info is None:
        click.echo("SmartMemory daemon is not running.")
        click.echo("Start with: smartmemory start")
        return
    click.echo(f"SmartMemory daemon: {info.get('status', '?')}")
    click.echo(f"  Memories:   {info.get('memories', '?')}")
    click.echo(f"  LLM:        {info.get('llm_provider', '?')}")
    click.echo(f"  Embeddings: {info.get('embedding_provider', '?')}")
    click.echo(f"  PID:        {info.get('pid', '?')}")


@cli.command("viewer")
@click.option("--port", default=None, type=int, help="Port override.")
def viewer_cmd(port: int | None) -> None:
    """Open the knowledge graph viewer in the browser."""
    import webbrowser
    from smartmemory_app.daemon import start_daemon, is_running, _port
    if not is_running():
        click.echo("Starting daemon...")
        start_daemon()
    p = port or _port()
    webbrowser.open(f"http://localhost:{p}")


# ── Memory operations ───────────────────────────────────────────────────────


@cli.command("persist")
@click.argument("text")
@click.option("--type", "memory_type", default="episodic", show_default=True)
def persist_cmd(text: str, memory_type: str) -> None:
    """Persist text as a memory (Stop hook)."""
    result = _daemon_request("POST", "/memory/ingest", json={"content": text, "memory_type": memory_type})
    if result:
        click.echo(result.get("item_id", "?"))
    else:
        from smartmemory_app.storage import ingest
        click.echo(ingest(text, memory_type))


@cli.command("ingest")
@click.argument("text")
@click.option("--type", "memory_type", default="episodic", show_default=True)
def ingest_cmd(text: str, memory_type: str) -> None:
    """Ingest text through the full pipeline."""
    result = _daemon_request("POST", "/memory/ingest", json={"content": text, "memory_type": memory_type})
    if result:
        click.echo(result.get("item_id", "?"))
    else:
        from smartmemory_app.storage import ingest
        click.echo(ingest(text, memory_type))


@cli.command("recall")
@click.option("--cwd", default=None, help="Current working directory for context.")
@click.option("--top-k", default=10, show_default=True)
def recall_cmd(cwd: str, top_k: int) -> None:
    """Recall memories (SessionStart hook)."""
    result = _daemon_request("GET", "/memory/recall", params={"cwd": cwd or "", "top_k": top_k})
    if result:
        click.echo(result.get("context", ""))
    else:
        from smartmemory_app.storage import recall
        click.echo(recall(cwd, top_k))


@cli.command("search")
@click.argument("query")
@click.option("--top-k", default=5, show_default=True)
def search_cmd(query: str, top_k: int) -> None:
    """Search memories by semantic similarity."""
    results = _daemon_request("POST", "/memory/search", json={"query": query, "top_k": top_k})
    if results is None:
        from smartmemory_app.storage import search
        results = search(query, top_k)
    if not results:
        click.echo("No results.")
        return
    for r in results:
        content = r.get("content", "")[:200]
        mem_type = r.get("memory_type", "?")
        item_id = r.get("item_id", "?")
        click.echo(f"[{mem_type}] {item_id[:8]}  {content}")


# ── Discovery + config ──────────────────────────────────────────────────────


@cli.command("models")
@click.option("--provider", default=None, help="Filter by provider (ollama, lmstudio, groq, openai)")
def models_cmd(provider: str | None) -> None:
    """List available models from local and cloud providers."""
    import httpx

    providers_to_check = [provider] if provider else ["ollama", "lmstudio", "groq", "openai"]

    for p in providers_to_check:
        if p == "ollama":
            try:
                r = httpx.get("http://localhost:11434/api/tags", timeout=3)
                r.raise_for_status()
                models = r.json().get("models", [])
                if models:
                    click.echo(f"\nOllama ({len(models)} models):")
                    for m in models:
                        name = m.get("name", "?")
                        size = m.get("size", 0)
                        size_gb = f"{size / 1e9:.1f}GB" if size else "?"
                        click.echo(f"  ollama/{name}  ({size_gb})")
                else:
                    click.echo("\nOllama: running but no models pulled")
            except Exception:
                click.echo("\nOllama: not running (start with: ollama serve)")

        elif p == "lmstudio":
            try:
                r = httpx.get("http://localhost:1234/v1/models", timeout=3)
                r.raise_for_status()
                models = r.json().get("data", [])
                if models:
                    click.echo(f"\nLM Studio ({len(models)} models):")
                    for m in models:
                        click.echo(f"  lmstudio/{m.get('id', '?')}")
                else:
                    click.echo("\nLM Studio: running but no models loaded")
            except Exception:
                click.echo("\nLM Studio: not running (start LM Studio and load a model)")

        elif p == "groq":
            import os
            key = os.environ.get("GROQ_API_KEY")
            if not key:
                click.echo("\nGroq: GROQ_API_KEY not set (get one at console.groq.com)")
                continue
            try:
                r = httpx.get(
                    "https://api.groq.com/openai/v1/models",
                    headers={"Authorization": f"Bearer {key}"},
                    timeout=5,
                )
                r.raise_for_status()
                models = sorted(r.json().get("data", []), key=lambda m: m.get("id", ""))
                if models:
                    click.echo(f"\nGroq ({len(models)} models):")
                    for m in models:
                        click.echo(f"  groq/{m.get('id', '?')}")
            except Exception as e:
                click.echo(f"\nGroq: API error ({e})")

        elif p == "openai":
            import os
            key = os.environ.get("OPENAI_API_KEY")
            if not key:
                click.echo("\nOpenAI: OPENAI_API_KEY not set")
                continue
            try:
                r = httpx.get(
                    "https://api.openai.com/v1/models",
                    headers={"Authorization": f"Bearer {key}"},
                    timeout=5,
                )
                r.raise_for_status()
                models = sorted(r.json().get("data", []), key=lambda m: m.get("id", ""))
                chat_models = [m for m in models if any(
                    m.get("id", "").startswith(prefix)
                    for prefix in ("gpt-4", "gpt-3.5", "o1", "o3", "o4")
                )]
                if chat_models:
                    click.echo(f"\nOpenAI ({len(chat_models)} chat models):")
                    for m in chat_models:
                        click.echo(f"  openai/{m.get('id', '?')}")
            except Exception as e:
                click.echo(f"\nOpenAI: API error ({e})")

        elif p == "claude-agent":
            click.echo("\nClaude Agent SDK: uses Claude Code OAuth (no model selection needed)")

        elif p == "anthropic":
            click.echo("\nAnthropic: claude-3-5-haiku-latest, claude-sonnet-4-5-20250514, claude-opus-4-6-20250603")


@cli.command("config")
@click.argument("key", required=False)
@click.argument("value", required=False)
def config_cmd(key: str | None, value: str | None) -> None:
    """View or change SmartMemory configuration.

    \b
    smartmemory config                  Show all settings
    smartmemory config llm_provider     Show one setting
    smartmemory config llm_provider groq  Change a setting
    """
    from smartmemory_app.config import load_config, save_config, config_path

    cfg = load_config()

    if key is None:
        click.echo(f"Config: {config_path()}\n")
        click.echo(f"  mode              = {cfg.mode}")
        click.echo(f"  llm_provider      = {cfg.llm_provider}")
        click.echo(f"  llm_model         = {cfg.llm_model or '(auto)'}")
        click.echo(f"  embedding_provider = {cfg.embedding_provider}")
        click.echo(f"  daemon_port       = {cfg.daemon_port}")
        click.echo(f"  coreference       = {cfg.coreference}")
        click.echo(f"  data_dir          = {cfg.data_dir}")
        click.echo(f"  api_url           = {cfg.api_url}")
        click.echo(f"  api_key_set       = {cfg.api_key_set}")
        click.echo(f"  team_id           = {cfg.team_id or '(none)'}")
        return

    settable = {
        "llm_provider": {"values": ["groq", "claude-agent", "anthropic", "openai", "ollama", "lmstudio", "none"]},
        "llm_model": {"values": None},
        "embedding_provider": {"values": ["local", "openai", "ollama"]},
        "daemon_port": {"values": None},
        "coreference": {"values": ["true", "false"]},
        "data_dir": {"values": None},
        "mode": {"values": ["local", "remote"]},
    }

    if key not in settable:
        current = getattr(cfg, key, None)
        if current is not None:
            click.echo(f"{key} = {current}")
        else:
            click.echo(f"Unknown key: {key}")
            click.echo(f"Settable keys: {', '.join(settable)}")
        return

    if value is None:
        click.echo(f"{key} = {getattr(cfg, key)}")
        allowed = settable[key]["values"]
        if allowed:
            click.echo(f"Allowed: {', '.join(allowed)}")
        return

    allowed = settable[key]["values"]
    if allowed and value not in allowed:
        click.echo(f"Invalid value '{value}'. Allowed: {', '.join(allowed)}")
        return

    coerced: object = value
    if key == "coreference":
        coerced = value.lower() == "true"
    elif key == "daemon_port":
        coerced = int(value)

    setattr(cfg, key, coerced)
    save_config(cfg)
    click.echo(f"{key} = {value}")


# ── Data management ─────────────────────────────────────────────────────────


@cli.command("clear")
@click.confirmation_option(prompt="This will delete all local memories. Are you sure?")
def clear_cmd() -> None:
    """Delete all local memories and reset the vector index."""
    result = _daemon_request("POST", "/memory/clear")
    if result is not None:
        click.echo(f"Cleared {result.get('cleared', '?')} files via daemon.")
        return

    # Daemon not running — clear files directly
    from smartmemory_app.storage import _resolve_data_dir, _shutdown
    _shutdown()

    data_path = _resolve_data_dir()
    if not data_path.exists():
        click.echo("No data directory found. Nothing to clear.")
        return

    removed = 0
    for pattern in [
        "*.db", "*.db-shm", "*.db-wal", "*.db-journal",
        "*.usearch", "*.json", "*.jsonl", "*.log", ".write.lock",
    ]:
        for f in data_path.glob(pattern):
            try:
                f.unlink()
                removed += 1
            except OSError as e:
                click.echo(f"Warning: could not remove {f.name}: {e}")

    click.echo(f"Cleared {removed} files from {data_path}")

    from smartmemory_app.setup import _seed_data_dir
    _seed_data_dir()
    click.echo("Re-seeded entity patterns.")


# ── Hidden/internal ─────────────────────────────────────────────────────────


@cli.command("server", hidden=True)
def server_cmd() -> None:
    """Start the SmartMemory MCP server (called by MCP clients, not users)."""
    from smartmemory_app.server import main
    main()


@cli.command("events-server", hidden=True)
@click.option("--port", default=9015, show_default=True, help="WebSocket port")
def events_server_cmd(port: int) -> None:
    """Run the lite WebSocket events server standalone (debugging only)."""
    from smartmemory_app.events_server import main
    main(port=port)


if __name__ == "__main__":
    cli()
