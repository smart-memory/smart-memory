"""Tests for the FastMCP server tool functions.

FastMCP 2.x wraps @mcp.tool() functions as FunctionTool objects.
Call the underlying function via tool.fn(args) for synchronous testing.
"""

from unittest.mock import patch


def test_memory_ingest_returns_item_id():
    """memory_ingest.fn() returns the item_id string on success."""
    with patch("smartmemory_pkg.server.ingest", return_value="item-abc-123"):
        from smartmemory_pkg.server import memory_ingest

        result = memory_ingest.fn("some text")
    assert result == "item-abc-123"


def test_memory_ingest_error_returns_string():
    """memory_ingest.fn() returns 'Error: ...' string when storage raises."""
    with patch("smartmemory_pkg.server.ingest", side_effect=RuntimeError("disk full")):
        from smartmemory_pkg.server import memory_ingest

        result = memory_ingest.fn("some text")
    assert result.startswith("Error:")
    assert "disk full" in result


def test_memory_search_returns_list():
    """memory_search.fn() returns a list of dicts on success."""
    mock_results = [{"item_id": "1", "content": "hello"}]
    with patch("smartmemory_pkg.server.search", return_value=mock_results):
        from smartmemory_pkg.server import memory_search

        result = memory_search.fn("query")
    assert isinstance(result, list)
    assert result == mock_results


def test_memory_recall_returns_string():
    """memory_recall.fn() returns a string context block."""
    with patch(
        "smartmemory_pkg.server.recall",
        return_value="## SmartMemory Context\n- [episodic] hello",
    ):
        from smartmemory_pkg.server import memory_recall

        result = memory_recall.fn(cwd="/project")
    assert isinstance(result, str)
    assert "SmartMemory" in result


def test_memory_get_returns_dict():
    """memory_get.fn() returns a dict for the given item_id."""
    mock_item = {"item_id": "abc", "content": "some memory"}
    with patch("smartmemory_pkg.server.get", return_value=mock_item):
        from smartmemory_pkg.server import memory_get

        result = memory_get.fn("abc")
    assert isinstance(result, dict)
    assert result["item_id"] == "abc"
