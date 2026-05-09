"""`sm mcp install <client>` — write MCP config for the named client.

Pointer-only wrapper around the already-shipped ``smartmemory-mcp`` server.
We do not reimplement the server — we just write config that points at the
installed console script.

Supported clients
-----------------
* **claude-code** — writes to ``~/.claude.json`` under ``mcpServers.smartmemory``.
* **cursor**      — writes to ``~/.cursor/mcp.json`` under ``mcpServers.smartmemory``
                    (Cursor reads MCP config from this path; project-local
                    ``.cursor/mcp.json`` is also a supported override but we
                    target the user-global file by default).
* **codex**       — writes to ``~/.codex/config.toml`` under
                    ``[mcp_servers.smartmemory]``. Codex CLI uses TOML, not
                    JSON, per its current docs. We write only the smartmemory
                    section, leaving any other servers / settings intact.

Failure mode is loud: if a config file cannot be parsed or written we raise
``ClickException`` rather than silently downgrading.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import click

CLIENTS = ("claude-code", "cursor", "codex")

_SERVER_KEY = "smartmemory"


def _resolve_server_command() -> str:
    """Locate the ``smartmemory-mcp`` console script.

    Falls back to the bare command name when ``which`` cannot find it — that
    way the config still works once the user finishes activating a venv.
    """
    found = shutil.which("smartmemory-mcp")
    return found or "smartmemory-mcp"


def _server_block() -> dict:
    """The portable MCP server descriptor (used by JSON-based clients)."""
    return {
        "command": _resolve_server_command(),
        "args": [],
        "env": {},
    }


def _write_json_config(path: Path, *, dry_run: bool) -> dict:
    """Merge the smartmemory server block into a JSON-style MCP config file.

    Returns the new config dict (always — even in dry-run mode), so the caller
    can echo / assert against the exact shape that would be persisted.
    """
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as f:
                existing = json.load(f)
        except json.JSONDecodeError as exc:
            raise click.ClickException(
                f"Refusing to overwrite malformed JSON at {path}: {exc}. "
                "Fix or remove the file and rerun."
            )
        if not isinstance(existing, dict):
            raise click.ClickException(
                f"Expected object at top level of {path}, got {type(existing).__name__}."
            )
    else:
        existing = {}

    servers = existing.setdefault("mcpServers", {})
    if not isinstance(servers, dict):
        raise click.ClickException(f"`mcpServers` in {path} is not an object.")
    servers[_SERVER_KEY] = _server_block()

    if not dry_run:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, sort_keys=True)
            f.write("\n")
        tmp.replace(path)

    return existing


def _write_codex_toml(path: Path, *, dry_run: bool) -> str:
    """Codex stores MCP servers in ``~/.codex/config.toml`` under
    ``[mcp_servers.<name>]`` tables.

    We use a minimal hand-rolled TOML emitter for the smartmemory block so we
    do not pull in tomli-w as a hard dep. Any existing ``[mcp_servers.smartmemory]``
    block is replaced; everything else in the file is preserved verbatim.

    Returns the full file contents that would be written.
    """
    cmd = _resolve_server_command()
    new_block = f'[mcp_servers.{_SERVER_KEY}]\ncommand = "{cmd}"\nargs = []\n'

    if path.exists():
        original = path.read_text(encoding="utf-8")
        # Strip any existing [mcp_servers.smartmemory] block (until next [tag] or EOF).
        import re

        pattern = re.compile(
            rf"^\[mcp_servers\.{_SERVER_KEY}\][^\[]*",
            flags=re.MULTILINE,
        )
        cleaned = pattern.sub("", original).rstrip() + (
            "\n\n" if original.strip() else ""
        )
    else:
        cleaned = ""

    new_contents = (cleaned + new_block).lstrip("\n")

    if not dry_run:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(new_contents, encoding="utf-8")
        tmp.replace(path)

    return new_contents


def _config_path_for(client: str) -> Path:
    home = Path.home()
    if client == "claude-code":
        return home / ".claude.json"
    if client == "cursor":
        return home / ".cursor" / "mcp.json"
    if client == "codex":
        return home / ".codex" / "config.toml"
    raise click.ClickException(f"Unknown MCP client: {client}")


@click.group("mcp")
def mcp_group() -> None:
    """MCP client configuration commands."""


@mcp_group.command("install")
@click.argument("client", type=click.Choice(CLIENTS))
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Print the config that would be written without modifying any file.",
)
@click.option(
    "--path",
    "override_path",
    default=None,
    type=click.Path(),
    help="Override the default config file location.",
)
def mcp_install_cmd(client: str, dry_run: bool, override_path: str | None) -> None:
    """Install MCP config pointing this client at the smartmemory-mcp server.

    \b
    Examples:
        sm mcp install claude-code
        sm mcp install cursor --dry-run
        sm mcp install codex --path /tmp/codex.toml
    """
    target = (
        Path(override_path).expanduser() if override_path else _config_path_for(client)
    )

    click.echo(f"[mcp:install] client={client} target={target} dry_run={dry_run}")

    if client in {"claude-code", "cursor"}:
        result = _write_json_config(target, dry_run=dry_run)
        block = result["mcpServers"][_SERVER_KEY]
        click.echo(
            json.dumps({"mcpServers": {_SERVER_KEY: block}}, indent=2, sort_keys=True)
        )
    else:  # codex
        text = _write_codex_toml(target, dry_run=dry_run)
        click.echo(text.rstrip())

    if dry_run:
        click.echo("[mcp:install] dry-run — no file written")
    else:
        click.echo(f"[mcp:install] wrote {target}")
