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
    mock_ingest.assert_called_once_with("test memory text", "episodic", properties={})


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
    mock_ingest.assert_called_once_with("some content", "episodic", properties={})


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
    mock_search.assert_called_once_with("test query", 5, filters={})


def test_search_cmd_no_results(runner):
    """search command prints 'No results.' when empty."""
    with patch("smartmemory_app.cli._daemon_request", return_value=[]):
        from smartmemory_app.cli import cli
        result = runner.invoke(cli, ["search", "nonexistent"])

    assert result.exit_code == 0
    assert "No results" in result.output


def test_get_cmd_fallback(runner):
    """get command falls back to storage.get when daemon is unavailable."""
    with (
        patch("smartmemory_app.cli._daemon_request", return_value=None),
        patch("smartmemory_app.storage.get", return_value={"item_id": "abc", "content": "hello", "memory_type": "episodic"}) as mock_get,
    ):
        from smartmemory_app.cli import cli
        result = runner.invoke(cli, ["get", "abc"])

    assert result.exit_code == 0
    assert '"item_id": "abc"' in result.output
    mock_get.assert_called_once_with("abc")


def test_get_cmd_daemon_path(runner):
    """get command prints daemon response when available."""
    with patch("smartmemory_app.cli._daemon_request", return_value={"item_id": "daemon-id", "content": "from daemon"}):
        from smartmemory_app.cli import cli
        result = runner.invoke(cli, ["get", "daemon-id"])

    assert result.exit_code == 0
    assert '"item_id": "daemon-id"' in result.output


def test_get_cmd_not_found_exits_nonzero(runner):
    """get command exits with an error when the memory is missing."""
    with (
        patch("smartmemory_app.cli._daemon_request", return_value=None),
        patch("smartmemory_app.storage.get", return_value={}),
    ):
        from smartmemory_app.cli import cli
        result = runner.invoke(cli, ["get", "missing-id"])

    assert result.exit_code == 1
    assert "Memory not found" in result.output


def test_search_cmd_filters_unsupported_exits_nonzero(runner):
    """search with filters raises ClickException when storage raises NotImplementedError."""
    with (
        patch("smartmemory_app.cli._daemon_request", return_value=None),
        patch("smartmemory_app.storage.search", side_effect=NotImplementedError("filters not supported")),
    ):
        from smartmemory_app.cli import cli
        result = runner.invoke(cli, ["search", "test", "--project", "atlas"])

    assert result.exit_code != 0
    assert "filters not supported" in result.output


def test_get_cmd_daemon_http_error_surfaces(runner):
    """get command surfaces ClickException from daemon HTTP errors instead of swallowing it."""
    import click

    with patch("smartmemory_app.cli._daemon_request", side_effect=click.ClickException("501: Not Implemented")):
        from smartmemory_app.cli import cli
        result = runner.invoke(cli, ["get", "some-id"])

    assert result.exit_code != 0
    assert "Not Implemented" in result.output


def test_persist_cmd_with_properties(runner):
    """persist command passes extra --key value flags as properties."""
    with (
        patch("smartmemory_app.cli._daemon_request", return_value=None),
        patch("smartmemory_app.storage.ingest", return_value="prop-id") as mock_ingest,
    ):
        from smartmemory_app.cli import cli
        result = runner.invoke(cli, ["persist", "test text", "--project", "atlas", "--domain", "legal"])

    assert result.exit_code == 0
    assert "prop-id" in result.output
    mock_ingest.assert_called_once_with("test text", "episodic", properties={"project": "atlas", "domain": "legal"})


def test_search_cmd_with_filters(runner):
    """search command passes extra --key value flags as filters."""
    mock_results = [{"item_id": "abc12345", "content": "filtered result", "memory_type": "semantic"}]
    with (
        patch("smartmemory_app.cli._daemon_request", return_value=None),
        patch("smartmemory_app.storage.search", return_value=mock_results) as mock_search,
    ):
        from smartmemory_app.cli import cli
        result = runner.invoke(cli, ["search", "test", "--project", "atlas"])

    assert result.exit_code == 0
    assert "abc12345" in result.output
    mock_search.assert_called_once_with("test", 5, filters={"project": "atlas"})
