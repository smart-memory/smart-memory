import logging
import click
from smartmemory_cc.storage import ingest, recall

log = logging.getLogger(__name__)


@click.group()
def cli() -> None:
    """SmartMemory CLI — used by Claude Code hooks."""


# Register setup/uninstall commands from setup module
from smartmemory_cc.setup import setup as _setup_cmd, uninstall as _uninstall_cmd  # noqa: E402

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


@cli.command("events-server")
@click.option("--port", default=9004, show_default=True, help="WebSocket port")
def events_server_cmd(port: int) -> None:
    """Run the lite WebSocket events server (graph animations without Redis).

    Runs automatically inside the MCP server process. Use this command
    only for debugging or to run the events server standalone.
    """
    from smartmemory_cc.events_server import main
    main(port=port)


if __name__ == "__main__":
    cli()
