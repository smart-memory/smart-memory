"""Integration tests for `sm init`, `sm code index`, `sm mcp install`.

Real backends only — per project rule (no mocks). The code-index test runs
the lite local backend (SQLite + usearch) against a tiny fixture repo built
inline; no daemon and no FalkorDB required for this layer.

Skipped automatically if the wrapper is not installed (importing
``smartmemory_app.storage.get_memory`` requires the full lite extras).
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest
from click.testing import CliRunner


# ── `sm init` is an alias for `sm setup` ──────────────────────────────────


def test_init_alias_registered():
    """`sm init` resolves to the same command object as `sm setup`."""
    from smartmemory_app.cli import cli

    init_cmd = cli.get_command(ctx=None, cmd_name="init")
    setup_cmd = cli.get_command(ctx=None, cmd_name="setup")
    assert init_cmd is not None, "`sm init` not registered"
    assert setup_cmd is not None, "`sm setup` not registered (sanity)"
    assert init_cmd is setup_cmd, (
        "`sm init` must alias the setup command, not duplicate it"
    )


def test_init_help_renders():
    """`sm init --help` produces the same help text as `sm setup --help`."""
    from smartmemory_app.cli import cli

    runner = CliRunner()
    init_help = runner.invoke(cli, ["init", "--help"])
    setup_help = runner.invoke(cli, ["setup", "--help"])
    assert init_help.exit_code == 0
    assert setup_help.exit_code == 0
    # Same command, only the program-name in usage differs ("init" vs "setup").
    assert init_help.output.replace("init", "X") == setup_help.output.replace(
        "setup", "X"
    )


# ── `sm mcp install` ──────────────────────────────────────────────────────


@pytest.mark.parametrize("client", ["claude-code", "cursor"])
def test_mcp_install_dry_run_json_clients(client, tmp_path):
    """Dry-run prints valid mcpServers JSON and writes nothing."""
    from smartmemory_app.cli import cli

    target = tmp_path / "config.json"
    runner = CliRunner()
    result = runner.invoke(
        cli, ["mcp", "install", client, "--dry-run", "--path", str(target)]
    )
    assert result.exit_code == 0, result.output
    assert not target.exists(), "dry-run must not write the config file"
    # The command echoes the new mcpServers block as JSON; parse it.
    json_lines = [
        line for line in result.output.splitlines() if line.startswith(("{", " ", "}"))
    ]
    parsed = json.loads("\n".join(json_lines))
    block = parsed["mcpServers"]["smartmemory"]
    assert "command" in block
    assert "args" in block
    assert (
        block["command"].endswith("smartmemory-mcp")
        or block["command"] == "smartmemory-mcp"
    )


def test_mcp_install_claude_code_writes_and_merges(tmp_path):
    """Writing into a pre-existing JSON config preserves other keys."""
    from smartmemory_app.cli import cli

    target = tmp_path / ".claude.json"
    target.write_text(
        json.dumps(
            {"existingKey": "preserved", "mcpServers": {"other": {"command": "x"}}}
        )
    )

    runner = CliRunner()
    result = runner.invoke(
        cli, ["mcp", "install", "claude-code", "--path", str(target)]
    )
    assert result.exit_code == 0, result.output

    written = json.loads(target.read_text())
    assert written["existingKey"] == "preserved"
    assert "smartmemory" in written["mcpServers"]
    assert written["mcpServers"]["other"] == {"command": "x"}, (
        "other servers must survive"
    )


def test_mcp_install_codex_toml(tmp_path):
    """Codex client writes [mcp_servers.smartmemory] TOML block."""
    from smartmemory_app.cli import cli

    target = tmp_path / "config.toml"
    target.write_text("# pre-existing\n[other]\nkey = 'val'\n")

    runner = CliRunner()
    result = runner.invoke(cli, ["mcp", "install", "codex", "--path", str(target)])
    assert result.exit_code == 0, result.output

    contents = target.read_text()
    assert "[mcp_servers.smartmemory]" in contents
    assert "command = " in contents
    assert "[other]" in contents, "other TOML sections must be preserved"


def test_mcp_install_rejects_malformed_json(tmp_path):
    """Refuse to overwrite a malformed JSON config; loud failure."""
    from smartmemory_app.cli import cli

    target = tmp_path / ".claude.json"
    target.write_text("{ this is not valid json")

    runner = CliRunner()
    result = runner.invoke(
        cli, ["mcp", "install", "claude-code", "--path", str(target)]
    )
    assert result.exit_code != 0
    assert "malformed" in result.output.lower()


# ── `sm code index` against a tiny real fixture repo ─────────────────────


@pytest.fixture
def tiny_repo(tmp_path):
    """A minimal Python "repo" for the code indexer to chew on."""
    root = tmp_path / "tiny"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "__init__.py").write_text("")
    (root / "pkg" / "a.py").write_text(
        textwrap.dedent(
            '''
            """Module a."""

            def hello(name: str) -> str:
                """Say hi."""
                return f"hi {name}"


            class Greeter:
                def greet(self, name: str) -> str:
                    return hello(name)
            '''
        ).strip()
    )
    (root / "pkg" / "b.py").write_text(
        textwrap.dedent(
            """
            from pkg.a import hello


            def call_hello() -> str:
                return hello("world")
            """
        ).strip()
    )
    return root


def _try_get_memory(tmp_data_dir: Path):
    """Return (memory, skip_reason). Skip if lite deps unavailable."""
    import os

    os.environ.setdefault("SMARTMEMORY_DATA_DIR", str(tmp_data_dir))
    try:
        from smartmemory_app.storage import get_memory
    except Exception as exc:  # pragma: no cover
        return None, f"storage layer not importable: {exc}"
    try:
        mem = get_memory(data_dir=str(tmp_data_dir))
    except Exception as exc:
        return None, f"get_memory unavailable: {exc}"
    if not hasattr(mem, "ingest_code"):
        return None, "active backend lacks ingest_code"
    return mem, None


def test_code_index_writes_origin_code_index(tiny_repo, tmp_path, monkeypatch):
    """`sm code index <fixture>` creates code:index-tagged entities in the graph."""
    from smartmemory_app.cli import cli

    data_dir = tmp_path / "smdata"
    data_dir.mkdir()
    monkeypatch.setenv("SMARTMEMORY_DATA_DIR", str(data_dir))

    mem, skip = _try_get_memory(data_dir)
    if mem is None:
        pytest.skip(skip)

    runner = CliRunner()
    result = runner.invoke(cli, ["code", "index", str(tiny_repo), "--repo", "tiny"])
    assert result.exit_code == 0, result.output
    assert "[code:index] phase=start" in result.output
    assert "[code:index] phase=done" in result.output
    assert "entities=" in result.output

    # Verify entities landed with origin=code:index.
    backend = getattr(getattr(mem, "_graph", None), "backend", None)
    if backend is None or not hasattr(backend, "serialize"):
        pytest.skip("graph backend does not expose serialize() for assertion")

    snapshot = backend.serialize()
    code_nodes = [
        n
        for n in snapshot.get("nodes", [])
        if (n.get("properties") or {}).get("memory_type") == "code"
        or n.get("memory_type") == "code"
    ]
    assert code_nodes, "expected at least one code:* node after `sm code index`"
    origins = {
        (n.get("properties") or {}).get("origin") or n.get("origin") for n in code_nodes
    }
    assert "code:index" in origins, f"expected origin=code:index, got {origins!r}"


def test_code_index_help_lists_languages():
    from smartmemory_app.cli import cli

    runner = CliRunner()
    result = runner.invoke(cli, ["code", "index", "--help"])
    assert result.exit_code == 0
    assert "python" in result.output
    assert "typescript" in result.output
