# Agentic toolbox auto-discovery and filtering (see docs/api_reference.md for details)

from typing import Callable, Set, List, Any, Optional


def discover_tools(
        metadata_only: bool = False,
        names: Optional[Set[str]] = None,
        tags: Optional[Set[str]] = None,
        types: Optional[Set[str]] = None,
        services: Optional[Set[str]] = None,
        custom_filter: Optional[Callable[[Any], bool]] = None
) -> List[Any]:
    """
    Auto-discover and filter agentic tools from all registered finders.
    Aggregates SmolAgents, MCP, and custom tools. Deduplicates by tool name.
    Args:
        metadata_only: If True, return metadata dicts instead of Tool objects.
        names: Restrict to these tool names (set of str).
        tags: Restrict to tools with these tags (set of str).
        types: Restrict to tools of these types (set of str/class names).
        services: Restrict to tools with these service names (set of str).
        custom_filter: Optional callable to filter tools.
    Returns:
        List of Tool objects or metadata dicts.
    """
    from smartmemory.toolbox.finders import mcp_tools
    # Use MCP tools only as requested
    finders = [mcp_tools]
    seen = set()
    tools = []
    for finder in finders:
        found_tools = finder.find_tools(metadata_only, names, tags, types, custom_filter)
        # Group tool names by service for debug
        if not metadata_only:
            from collections import defaultdict
            service_tools = defaultdict(list)
            for tool in found_tools:
                service = getattr(tool, 'service', None)
                name = getattr(tool, 'name', None)
                service_tools[service].append(name)
            for service, names in service_tools.items():
                print(f"[discover_tools] Service: {service}, Tools: {names}")
        for tool in found_tools:
            tool_name = tool['name'] if metadata_only else getattr(tool, 'name', None)
            tool_service = tool['service'] if metadata_only else getattr(tool, 'service', None)
            key = (tool_service, tool_name)
            if tool_name and key not in seen:
                seen.add(key)
                # Assign .service to the module name if not already set (for Tool objects, not dicts)
                if not metadata_only and not hasattr(tool, 'service'):
                    import sys
                    module = sys.modules.get(getattr(tool, '__module__', ''))
                    if module and hasattr(module, '__file__'):
                        import os
                        filename = os.path.splitext(os.path.basename(module.__file__))[0]
                        # Any tool from smartmemory.tools uses its filename as service
                        if "smartmemory.tools" in getattr(tool, "__module__", ""):
                            tool.service = filename
                        # SmolAgents tools: module name contains 'smolagents'
                        elif "smolagents" in getattr(tool, "__module__", ""):
                            tool.service = "smolagents"
                        # MCP tools: module name contains 'mcp_tools'
                        elif "mcp_tools" in getattr(tool, "__module__", ""):
                            tool.service = "mcp"
                        else:
                            tool.service = filename
                    else:
                        mod = getattr(tool, "__module__", "")
                        if "smolagents" in mod:
                            tool.service = "smolagents"
                        elif "mcp_tools" in mod:
                            tool.service = "mcp"
                        else:
                            tool.service = mod.split(".")[-1] or "default"
                # Service filter (only after .service is set)
                if services and (getattr(tool, 'service', None) not in services):
                    continue
                tools.append(tool)
    return tools


if __name__ == "__main__":
    print("Actionable tools:")
    for tool in discover_tools():
        print(f"{tool.name}: {tool.description} (args: {getattr(tool, 'args_schema', {})})")
    print("\nMetadata only:")
    for meta in discover_tools(metadata_only=True):
        print(f"{meta['name']}: {meta['description']} (args: {meta['args_schema']})")


def select_tools(
        tools: List[Any],
        names: Optional[Set[str]] = None,
        tags: Optional[Set[str]] = None,
        types: Optional[Set[str]] = None,
        services: Optional[Set[str]] = None,
) -> List[Any]:
    """
    Filter a list of tools by name, tags, type, or service.
    """
    filtered = []
    for tool in tools:
        if names and getattr(tool, "name", None) not in names:
            continue
        if tags and not (tags & set(getattr(tool, "tags", []))):
            continue
        if types and type(tool).__name__ not in types:
            continue
        if services and getattr(tool, "service", None) not in services:
            continue
        filtered.append(tool)
    return filtered
