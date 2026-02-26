"""Tests for the Click CLI commands.

Patches must target the imported names in smartmemory_pkg.cli (not the source module),
since cli.py uses `from smartmemory_pkg.storage import ingest, recall` — the names
are bound at import time in the cli module's namespace.
"""

import pytest
from click.testing import CliRunner
from unittest.mock import patch


@pytest.fixture
def runner():
    return CliRunner()


def test_persist_cmd(runner):
    """persist command calls ingest and echoes item_id."""
    with patch("smartmemory_pkg.cli.ingest", return_value="item-abc") as mock_ingest:
        from smartmemory_pkg.cli import cli

        result = runner.invoke(cli, ["persist", "test memory text"])
    assert result.exit_code == 0
    assert "item-abc" in result.output
    mock_ingest.assert_called_once_with("test memory text", "episodic")


def test_ingest_cmd(runner):
    """ingest command calls ingest and echoes item_id."""
    with patch("smartmemory_pkg.cli.ingest", return_value="item-xyz") as mock_ingest:
        from smartmemory_pkg.cli import cli

        result = runner.invoke(cli, ["ingest", "some content"])
    assert result.exit_code == 0
    assert "item-xyz" in result.output
    mock_ingest.assert_called_once_with("some content", "episodic")


def test_recall_cmd(runner):
    """recall command calls recall and echoes output."""
    with patch(
        "smartmemory_pkg.cli.recall", return_value="## SmartMemory Context\n- hello"
    ) as mock_recall:
        from smartmemory_pkg.cli import cli

        result = runner.invoke(cli, ["recall", "--cwd", "/my/project"])
    assert result.exit_code == 0
    assert "SmartMemory" in result.output
    mock_recall.assert_called_once_with("/my/project", 10)


def test_recall_cmd_no_cwd(runner):
    """recall command passes None for cwd when --cwd not provided."""
    with patch("smartmemory_pkg.cli.recall", return_value="") as mock_recall:
        from smartmemory_pkg.cli import cli

        result = runner.invoke(cli, ["recall"])
    assert result.exit_code == 0
    mock_recall.assert_called_once_with(None, 10)
