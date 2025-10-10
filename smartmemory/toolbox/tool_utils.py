import inspect
from typing import Any, Callable, Dict, List


def auto_args_schema(fn: Callable) -> Dict[str, Any]:
    """
    Generate an argument schema dictionary from function type annotations.
    Ignores 'self' and 'cls'.
    """
    sig = inspect.signature(fn)
    schema = {}
    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue
        if param.annotation is inspect.Parameter.empty:
            schema[name] = Any
        else:
            schema[name] = param.annotation
    return schema


def ensure_tags(fn: Callable, tags: List[str]) -> None:
    """
    Ensure that the function has a .tags attribute and set it if missing.
    """
    if not hasattr(fn, "tags"):
        fn.tags = tags


def ensure_all_tools_have_tags_and_schema(module) -> None:
    """
    For all callables in a module, auto-generate .args_schema and ensure .tags exists.
    """
    for name in dir(module):
        obj = getattr(module, name)
        if callable(obj) and not name.startswith("__"):
            # Only set args_schema if not present
            if not hasattr(obj, "args_schema"):
                obj.args_schema = auto_args_schema(obj)
            if not hasattr(obj, "tags"):
                obj.tags = []


# --- Tool factory utilities ---

def tool_to_metadata(tool):
    """
    Convert a Tool object to a metadata dict with all required fields.
    Assumes tool is a Tool object with .name, .description, .args_schema, .tags, .service.
    """
    name = getattr(tool, "name", getattr(tool, "__name__", str(tool)))
    description = getattr(tool, "description", getattr(tool, "__doc__", ""))
    args_schema = getattr(tool, "args_schema", {})
    tags = list(getattr(tool, "tags", []))
    tool_type = type(tool).__name__
    service = getattr(tool, "service", None)
    if not service and hasattr(tool, "__module__"):
        mod = tool.__module__
        if "smolagents" in mod:
            service = "smolagents"
        elif "mcp_tools" in mod:
            service = "mcp"
        elif "smartmemory.tools" in mod:
            service = mod.split('.')[-1]
        else:
            service = mod.split('.')[-1] or "unknown"
    elif not service:
        service = "unknown"
    return {
        "name": name,
        "description": description,
        "args_schema": args_schema,
        "tags": tags,
        "type": tool_type,
        "service": service,
    }


def smolagents_tool_from_function(fn: Callable) -> Any:
    """
    Wrap a function as a SmolAgents Tool using the official @tool decorator pattern.
    This ensures compatibility and extracts all required metadata automatically.
    Usage: decorate your function with @tool or call this utility to wrap it dynamically.
    """
    try:
        from smolagents import tool as smol_tool_decorator
    except ImportError:
        raise ImportError("SmolAgents is not installed.")
    return smol_tool_decorator(fn)


def mcp_tool_from_function(fn: Callable) -> Any:
    """
    Create an MCP Tool object from a pure function with metadata.
    Requires MCP SDK to be installed.
    """
    try:
        from mcp.server.fastmcp.tools.base import Tool as MCPTool
    except ImportError:
        raise ImportError("MCP SDK is not installed.")
    return MCPTool.from_function(
        fn,
        name=fn.__name__,
        description=fn.__doc__.strip() if fn.__doc__ else fn.__name__,
    )
