import logging

import click
from smartmemory_app.storage import ingest, recall, search

log = logging.getLogger(__name__)


@click.group()
def cli() -> None:
    """SmartMemory CLI — used by Claude Code hooks."""


# Register setup/uninstall commands from setup module
from smartmemory_app.setup import setup as _setup_cmd, uninstall as _uninstall_cmd  # noqa: E402

cli.add_command(_setup_cmd, name="setup")
cli.add_command(_uninstall_cmd, name="uninstall")


@cli.command("persist")
@click.argument("text")
@click.option("--type", "memory_type", default="episodic", show_default=True)
def persist_cmd(text: str, memory_type: str) -> None:
    """Persist text as a memory (Stop hook)."""
    item_id = ingest(text, memory_type)
    click.echo(item_id)


@cli.command("ingest")
@click.argument("text")
@click.option("--type", "memory_type", default="episodic", show_default=True)
def ingest_cmd(text: str, memory_type: str) -> None:
    """Full pipeline ingest (PostToolUseFailure hook)."""
    item_id = ingest(text, memory_type)
    click.echo(item_id)


@cli.command("recall")
@click.option("--cwd", default=None, help="Current working directory for context.")
@click.option("--top-k", default=10, show_default=True)
def recall_cmd(cwd: str, top_k: int) -> None:
    """Recall memories (SessionStart hook)."""
    result = recall(cwd, top_k)
    click.echo(result)


@cli.command("search")
@click.argument("query")
@click.option("--top-k", default=5, show_default=True)
def search_cmd(query: str, top_k: int) -> None:
    """Search memories by semantic similarity."""
    results = search(query, top_k)
    if not results:
        click.echo("No results.")
        return
    for r in results:
        content = r.get("content", "")[:200]
        mem_type = r.get("memory_type", "?")
        item_id = r.get("item_id", "?")
        click.echo(f"[{mem_type}] {item_id[:8]}  {content}")


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
        # Show all
        click.echo(f"Config: {config_path()}\n")
        click.echo(f"  mode              = {cfg.mode}")
        click.echo(f"  llm_provider      = {cfg.llm_provider}")
        click.echo(f"  llm_model         = {cfg.llm_model or '(auto)'}")
        click.echo(f"  embedding_provider = {cfg.embedding_provider}")
        click.echo(f"  coreference       = {cfg.coreference}")
        click.echo(f"  data_dir          = {cfg.data_dir}")
        click.echo(f"  api_url           = {cfg.api_url}")
        click.echo(f"  api_key_set       = {cfg.api_key_set}")
        click.echo(f"  team_id           = {cfg.team_id or '(none)'}")
        return

    # Settable fields and their validation
    settable = {
        "llm_provider": {"values": ["groq", "claude-agent", "anthropic", "openai", "ollama", "lmstudio", "none"]},
        "llm_model": {"values": None},  # freeform
        "embedding_provider": {"values": ["local", "openai", "ollama"]},
        "coreference": {"values": ["true", "false"]},
        "data_dir": {"values": None},  # freeform
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

    # Validate
    allowed = settable[key]["values"]
    if allowed and value not in allowed:
        click.echo(f"Invalid value '{value}'. Allowed: {', '.join(allowed)}")
        return

    # Special handling for bool
    if key == "coreference":
        value = value.lower() == "true"

    setattr(cfg, key, value)
    save_config(cfg)
    click.echo(f"{key} = {value}")


@cli.command("server", hidden=True)
def server_cmd() -> None:
    """Start the SmartMemory MCP server (called by MCP clients, not users)."""
    from smartmemory_app.server import main
    main()


@cli.command("clear")
@click.confirmation_option(prompt="This will delete all local memories. Are you sure?")
def clear_cmd() -> None:
    """Delete all local memories and reset the vector index."""
    from smartmemory_app.storage import _resolve_data_dir, _shutdown

    _shutdown()  # flush and release any open handles

    data_path = _resolve_data_dir()
    if not data_path.exists():
        click.echo("No data directory found. Nothing to clear.")
        return

    removed = []
    for pattern in [
        "*.db", "*.db-shm", "*.db-wal", "*.db-journal",  # SQLite + WAL
        "*.usearch",                                        # vector index
        "*.json",                                           # usearch metadata
        "*.jsonl",                                          # patterns
        "*.log",                                            # plugin.log, hooks.log
        ".write.lock",                                      # filelock
    ]:
        for f in data_path.glob(pattern):
            try:
                f.unlink()
                removed.append(f.name)
            except OSError as e:
                click.echo(f"Warning: could not remove {f.name}: {e}")

    if removed:
        click.echo(f"Cleared {len(removed)} files from {data_path}")
    else:
        click.echo(f"No data files found in {data_path}")

    # Re-seed patterns file
    from smartmemory_app.setup import _seed_data_dir
    _seed_data_dir()
    click.echo("Re-seeded entity patterns.")

    # Notify viewer to refresh via events WebSocket
    _notify_viewer_cleared()


def _notify_viewer_cleared() -> None:
    """Send graph_cleared event to the viewer via WebSocket."""
    try:
        import json
        import websockets.sync.client as ws_sync

        with ws_sync.connect("ws://localhost:9015", close_timeout=2) as ws:
            ws.send(json.dumps({
                "type": "new_event",
                "event_type": "span",
                "component": "graph",
                "operation": "clear_all",
                "name": "graph.clear_all",
                "data": {"nuclear": True},
            }))
        click.echo("Notified viewer to refresh.")
    except Exception:
        # Viewer not running — that's fine
        pass


@cli.command("viewer")
@click.option("--port", default=9014, show_default=True, help="Port for the viewer server.")
@click.option("--no-browser", is_flag=True, default=False, help="Don't auto-open browser.")
def viewer_cmd(port: int, no_browser: bool) -> None:
    """Open the knowledge graph viewer (loginless, no Docker required)."""
    from smartmemory_app.viewer_server import main  # lazy — avoids fastapi import at CLI startup
    main(port=port, open_browser=not no_browser)


@cli.command("events-server", hidden=True)
@click.option("--port", default=9015, show_default=True, help="WebSocket port")
def events_server_cmd(port: int) -> None:
    """Run the lite WebSocket events server standalone (debugging only)."""
    from smartmemory_app.events_server import main
    main(port=port)


if __name__ == "__main__":
    cli()
