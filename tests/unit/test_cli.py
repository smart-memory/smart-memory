"""Tests for the Click CLI commands.

DIST-DAEMON-1: CLI commands try daemon HTTP API first (_daemon_request), then
fall back to storage.ingest/search/recall via lazy import inside function bodies.
Patches target storage module functions (the fallback path) and _daemon_request
(to simulate daemon-down).
"""

import pytest
from click.testing import CliRunner
from unittest.mock import patch


@pytest.fixture
def runner():
    return CliRunner()


def test_persist_cmd_fallback(runner):
    """persist command falls back to storage.ingest when daemon is down."""
    with (
        patch("smartmemory_app.cli._daemon_request", return_value=None),
        patch("smartmemory_app.storage.ingest", return_value="item-abc") as mock_ingest,
    ):
        from smartmemory_app.cli import cli
        result = runner.invoke(cli, ["persist", "test memory text"])

    assert result.exit_code == 0
    assert "item-abc" in result.output
    mock_ingest.assert_called_once_with("test memory text", "episodic")


def test_persist_cmd_daemon_path(runner):
    """persist command uses daemon response when available."""
    with patch("smartmemory_app.cli._daemon_request", return_value={"item_id": "daemon-id"}):
        from smartmemory_app.cli import cli
        result = runner.invoke(cli, ["persist", "test memory text"])

    assert result.exit_code == 0
    assert "daemon-id" in result.output


def test_ingest_cmd_fallback(runner):
    """ingest command falls back to storage.ingest when daemon is down."""
    with (
        patch("smartmemory_app.cli._daemon_request", return_value=None),
        patch("smartmemory_app.storage.ingest", return_value="item-xyz") as mock_ingest,
    ):
        from smartmemory_app.cli import cli
        result = runner.invoke(cli, ["ingest", "some content"])

    assert result.exit_code == 0
    assert "item-xyz" in result.output
    mock_ingest.assert_called_once_with("some content", "episodic")


def test_recall_cmd_fallback(runner):
    """recall command falls back to storage.recall when daemon is down."""
    with (
        patch("smartmemory_app.cli._daemon_request", return_value=None),
        patch("smartmemory_app.storage.recall", return_value="## SmartMemory Context\n- hello") as mock_recall,
    ):
        from smartmemory_app.cli import cli
        result = runner.invoke(cli, ["recall", "--cwd", "/my/project"])

    assert result.exit_code == 0
    assert "SmartMemory" in result.output
    mock_recall.assert_called_once_with("/my/project", 10)


def test_recall_cmd_no_cwd_fallback(runner):
    """recall command passes None for cwd when --cwd not provided (fallback path)."""
    with (
        patch("smartmemory_app.cli._daemon_request", return_value=None),
        patch("smartmemory_app.storage.recall", return_value="") as mock_recall,
    ):
        from smartmemory_app.cli import cli
        result = runner.invoke(cli, ["recall"])

    assert result.exit_code == 0
    mock_recall.assert_called_once_with(None, 10)


def test_recall_cmd_daemon_path(runner):
    """recall command uses daemon response when available."""
    with patch("smartmemory_app.cli._daemon_request", return_value={"context": "daemon context"}):
        from smartmemory_app.cli import cli
        result = runner.invoke(cli, ["recall"])

    assert result.exit_code == 0
    assert "daemon context" in result.output


def test_search_cmd_fallback(runner):
    """search command falls back to storage.search when daemon is down."""
    mock_results = [{"item_id": "abc12345", "content": "test result", "memory_type": "semantic"}]
    with (
        patch("smartmemory_app.cli._daemon_request", return_value=None),
        patch("smartmemory_app.storage.search", return_value=mock_results) as mock_search,
    ):
        from smartmemory_app.cli import cli
        result = runner.invoke(cli, ["search", "test query"])

    assert result.exit_code == 0
    assert "abc12345" in result.output
    mock_search.assert_called_once_with("test query", 5)


def test_search_cmd_no_results(runner):
    """search command prints 'No results.' when empty."""
    with patch("smartmemory_app.cli._daemon_request", return_value=[]):
        from smartmemory_app.cli import cli
        result = runner.invoke(cli, ["search", "nonexistent"])

    assert result.exit_code == 0
    assert "No results" in result.output
