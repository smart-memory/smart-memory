from fastmcp import FastMCP
from smartmemory_cc.storage import ingest, search, recall, get

mcp = FastMCP("smartmemory")


@mcp.tool()
def memory_ingest(content: str, memory_type: str = "episodic") -> str:
    """Ingest content into SmartMemory. Returns the memory item_id."""
    try:
        return ingest(content, memory_type)
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def memory_search(query: str, top_k: int = 5) -> list[dict]:
    """Search memories by semantic similarity."""
    try:
        return search(query, top_k)
    except Exception as e:
        return [{"error": str(e)}]


@mcp.tool()
def memory_recall(cwd: str = None, top_k: int = 10) -> str:
    """Recall recent and relevant memories for the current directory."""
    try:
        return recall(cwd, top_k)
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def memory_get(item_id: str) -> dict:
    """Get a single memory by item_id."""
    try:
        return get(item_id)
    except Exception as e:
        return {"error": str(e)}


def main() -> None:
    mcp.run(show_banner=False)


if __name__ == "__main__":
    main()
