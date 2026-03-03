"""Tests for the FastMCP server tool functions.

FastMCP 2.x wraps @mcp.tool() functions as FunctionTool objects.
Call the underlying function via tool.fn(args) for synchronous testing.
"""

from unittest.mock import patch


def test_memory_ingest_returns_item_id():
    """memory_ingest.fn() returns the item_id string on success."""
    with patch("smartmemory_app.server.ingest", return_value="item-abc-123"):
        from smartmemory_app.server import memory_ingest

        result = memory_ingest.fn("some text")
    assert result == "item-abc-123"


def test_memory_ingest_error_returns_string():
    """memory_ingest.fn() returns 'Error: ...' string when storage raises."""
    with patch("smartmemory_app.server.ingest", side_effect=RuntimeError("disk full")):
        from smartmemory_app.server import memory_ingest

        result = memory_ingest.fn("some text")
    assert result.startswith("Error:")
    assert "disk full" in result


def test_memory_search_returns_list():
    """memory_search.fn() returns a list of dicts on success."""
    mock_results = [{"item_id": "1", "content": "hello"}]
    with patch("smartmemory_app.server.search", return_value=mock_results):
        from smartmemory_app.server import memory_search

        result = memory_search.fn("query")
    assert isinstance(result, list)
    assert result == mock_results


def test_memory_recall_returns_string():
    """memory_recall.fn() returns a string context block."""
    with patch(
        "smartmemory_app.server.recall",
        return_value="## SmartMemory Context\n- [episodic] hello",
    ):
        from smartmemory_app.server import memory_recall

        result = memory_recall.fn(cwd="/project")
    assert isinstance(result, str)
    assert "SmartMemory" in result


def test_memory_get_returns_dict():
    """memory_get.fn() returns a dict for the given item_id."""
    mock_item = {"item_id": "abc", "content": "some memory"}
    with patch("smartmemory_app.server.get", return_value=mock_item):
        from smartmemory_app.server import memory_get

        result = memory_get.fn("abc")
    assert isinstance(result, dict)
    assert result["item_id"] == "abc"


# ---------------------------------------------------------------------------
# Auth tools — local mode fast-path (DIST-LITE-5, Phase 3b)
# ---------------------------------------------------------------------------


def test_login_local_mode_returns_no_auth_message():
    """login.fn() returns a 'no authentication' message in local mode."""
    from unittest.mock import MagicMock
    from smartmemory_app.remote_backend import RemoteMemory

    local_mem = MagicMock()  # not a RemoteMemory instance
    with patch("smartmemory_app.server.get_memory", return_value=local_mem):
        from smartmemory_app.server import login

        result = login.fn("sk_test")
    assert "Local mode" in result
    assert "authentication" in result.lower()


def test_whoami_local_mode_returns_no_auth_message():
    """whoami.fn() returns a 'no authentication' message in local mode."""
    from unittest.mock import MagicMock

    local_mem = MagicMock()
    with patch("smartmemory_app.server.get_memory", return_value=local_mem):
        from smartmemory_app.server import whoami

        result = whoami.fn()
    assert "Local mode" in result


def test_switch_team_local_mode_returns_not_applicable():
    """switch_team.fn() returns a 'not applicable' message in local mode."""
    from unittest.mock import MagicMock

    local_mem = MagicMock()
    with patch("smartmemory_app.server.get_memory", return_value=local_mem):
        from smartmemory_app.server import switch_team

        result = switch_team.fn("team-xyz")
    assert "Local mode" in result


def test_login_remote_mode_delegates_to_remote_memory():
    """login.fn() calls RemoteMemory.login() in remote mode."""
    from unittest.mock import MagicMock
    from smartmemory_app.remote_backend import RemoteMemory

    remote_mem = MagicMock(spec=RemoteMemory)
    remote_mem.login.return_value = "Logged in. Team: t1"
    with patch("smartmemory_app.server.get_memory", return_value=remote_mem):
        from smartmemory_app.server import login

        result = login.fn("sk_live_key")
    remote_mem.login.assert_called_once_with("sk_live_key")
    assert result == "Logged in. Team: t1"
