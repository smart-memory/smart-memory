from typing import Dict, Optional

try:
    from smartmemory.models.memory_item import MemoryItem
    from smartmemory.stores.external.mcp_handler import MCPHandler
except ImportError:
    MemoryItem = None
    MCPHandler = None

# Global MCP handler instance
_mcp_handler = None


def get_mcp_handler():
    """Get or create MCP handler instance."""
    global _mcp_handler
    if _mcp_handler is None and MCPHandler is not None:
        _mcp_handler = MCPHandler()
    return _mcp_handler


def mcp_memory_add(content: str, memory_type: str = "semantic", metadata: Optional[Dict] = None) -> str:
    """Add a memory item via MCP interface."""
    handler = get_mcp_handler()
    if handler is None or MemoryItem is None:
        return "MCP handler or MemoryItem not available"

    try:
        item = MemoryItem(
            content=content,
            memory_type=memory_type,
            metadata=metadata or {}
        )
        result = handler.add(item)
        return f"Added memory item: {result.item_id}"
    except Exception as e:
        return f"Error adding memory: {str(e)}"


def mcp_memory_get(item_id: str) -> str:
    """Get a memory item by ID via MCP interface."""
    handler = get_mcp_handler()
    if handler is None:
        return "MCP handler not available"

    try:
        item = handler.get(item_id)
        if item is None:
            return f"Memory item not found: {item_id}"
        return f"Memory item {item_id}: {item.content}"
    except Exception as e:
        return f"Error getting memory item: {e}"


def smart_memory_search(query: str, top_k: int = 5, user_id: str = None) -> str:
    """Search for memories using enhanced contextual matching."""
    from smartmemory.smart_memory import SmartMemory

    smart_memory = SmartMemory()
    results = smart_memory.search(query, top_k=top_k, user_id=user_id)

    if not results:
        return f"No results found for query: {query}"

    result_strs = []
    for item in results:
        content_preview = item.content[:150] + "..." if len(item.content) > 150 else item.content
        metadata_info = ""
        if hasattr(item, 'metadata') and item.metadata:
            item_type = item.metadata.get('type', '')
            timestamp = item.metadata.get('timestamp', '')
            user_context = item.metadata.get('user_id', '')
            if item_type:
                metadata_info += f" [{item_type}]"
            if timestamp:
                metadata_info += f" ({timestamp[:10]})"
            if user_context and user_context == user_id:
                metadata_info += f" [user: {user_context}]"

        result_strs.append(f"- {item.item_id}: {content_preview}{metadata_info}")

    return f"Found {len(results)} results for query '{query}':\n" + "\n".join(result_strs)


def mcp_memory_update(item_id: str, content: str = None, metadata: Dict = None) -> str:
    """Update a memory item via MCP interface."""
    handler = get_mcp_handler()
    if handler is None:
        return "MCP handler not available"

    try:
        # Get existing item
        existing_item = handler.get(item_id)
        if existing_item is None:
            return f"Memory item not found: {item_id}"

        # Update fields
        if content is not None:
            existing_item.content = content
        if metadata is not None:
            existing_item.metadata.update(metadata)

        # Save updated item
        handler.update(existing_item)
        return f"Updated memory item: {item_id}"
    except Exception as e:
        return f"Error updating memory item: {e}"


def mcp_memory_delete(item_id: str) -> str:
    """Delete a memory item via MCP interface."""
    handler = get_mcp_handler()
    if handler is None:
        return "MCP handler not available"

    try:
        success = handler.delete(item_id)
        if success:
            return f"Deleted memory item: {item_id}"
        else:
            return f"Failed to delete memory item: {item_id}"
    except Exception as e:
        return f"Error deleting memory item: {e}"


def mcp_memory_ingest(content: str, extractor_name: str = "spacy") -> str:
    """Ingest content with full extraction pipeline via MCP interface."""
    handler = get_mcp_handler()
    if handler is None:
        return "MCP handler not available"

    try:
        # Now MCPHandler.ingest accepts string content directly
        result = handler.ingest(content, extractor_name=extractor_name)

        # Extract meaningful info from result
        if isinstance(result, dict):
            item = result.get('item')
            entities = result.get('entities', [])
            return f"Ingested content successfully. Item ID: {item.item_id if item else 'unknown'}, Entities: {len(entities)}"
        else:
            return f"Ingested content successfully. Result: {result}"
    except Exception as e:
        return f"Error ingesting content: {e}"


# Simple tool class for MCP tools
class MCPTool:
    def __init__(self, name, description, func, args_schema=None, tags=None):
        self.name = name
        self.description = description
        self.func = func
        self.args_schema = args_schema or {}
        self.tags = tags or []
        self.service = "mcp"


def find_tools(metadata_only=False, names=None, tags=None, types=None, custom_filter=None):
    """Find MCP memory tools."""
    if MCPHandler is None or MemoryItem is None:
        return []

    # Create simple tool objects
    tools = [
        MCPTool(
            name="smart_memory_add",
            description="Add a memory item to the agentic memory system.",
            func=mcp_memory_add,
            args_schema={"content": str, "memory_type": str, "metadata": dict},
            tags=["memory", "smart_memory", "add"]
        ),
        MCPTool(
            name="smart_memory_get",
            description="Get a memory item by ID from the agentic memory system.",
            func=mcp_memory_get,
            args_schema={"item_id": str},
            tags=["memory", "smart_memory", "get"]
        ),
        MCPTool(
            name="smart_memory_search",
            description="Search memory items in the agentic memory system.",
            func=smart_memory_search,
            args_schema={"query": str, "top_k": int, "user_id": str},
            tags=["memory", "smart_memory", "search"]
        ),
        MCPTool(
            name="smart_memory_update",
            description="Update a memory item in the agentic memory system.",
            func=mcp_memory_update,
            args_schema={"item_id": str, "content": str, "metadata": dict},
            tags=["memory", "smart_memory", "update"]
        ),
        MCPTool(
            name="smart_memory_delete",
            description="Delete a memory item from the agentic memory system.",
            func=mcp_memory_delete,
            args_schema={"item_id": str},
            tags=["memory", "smart_memory", "delete"]
        ),
        MCPTool(
            name="smart_memory_ingest",
            description="Ingest content with full entity/relation extraction pipeline.",
            func=mcp_memory_ingest,
            args_schema={"content": str, "extractor_name": str},
            tags=["memory", "smart_memory", "ingest", "extraction"]
        )
    ]

    if metadata_only:
        return [{
            "name": tool.name,
            "description": tool.description,
            "args_schema": tool.args_schema,
            "tags": tool.tags,
            "type": "MCPTool",
            "service": tool.service
        } for tool in tools]

    return tools
