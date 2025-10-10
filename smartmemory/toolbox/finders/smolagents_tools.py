import smolagents
from smolagents import Tool


def is_leaf_tool(cls):
    return issubclass(cls, Tool) and cls is not Tool and not any(
        issubclass(sub, cls) and sub is not cls for sub in Tool.__subclasses__()
    )


import pkgutil
import importlib


def _import_all_submodules(package):
    # Recursively import all submodules of a package
    for loader, module_name, is_pkg in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        try:
            importlib.import_module(module_name)
        except Exception:
            pass  # Ignore import errors for now


def get_smolagents_tools():
    # Recursively import all submodules so all Tool subclasses are registered
    _import_all_submodules(smolagents)

    # Gather all leaf subclasses of Tool (even those in submodules)
    def all_leaf_tool_subclasses(cls):
        # Recursively find all leaf subclasses
        subclasses = set(cls.__subclasses__())
        leafs = set()
        for sub in subclasses:
            sub_leafs = all_leaf_tool_subclasses(sub)
            if sub_leafs:
                leafs.update(sub_leafs)
            else:
                leafs.add(sub)
        return leafs

    tool_classes = [c for c in all_leaf_tool_subclasses(Tool) if c is not Tool]
    # Instantiate each tool (if possible)
    tools = []
    for clazz in tool_classes:
        try:
            if clazz.__name__ == "SimpleTool":
                continue
            tool_instance = clazz()
            if hasattr(tool_instance, 'tags') and isinstance(tool_instance.tags, list):
                tool_instance.tags.append('external')
            else:
                tool_instance.tags = ['external']
            tools.append(tool_instance)
        except Exception as e:
            pass  # Skip if can't instantiate without args
    return tools


def tool_to_metadata(tool):
    # Dummy implementation, should be replaced with actual metadata extraction
    return {"name": getattr(tool, "name", None), "tags": getattr(tool, "tags", [])}


def find_tools(metadata_only=False, names=None, tags=None, types=None, custom_filter=None):
    """
    Discover SmolAgents tools via get_smolagents_tools().
    Efficiently filter by tool name (tool.name), class name (type(tool).__name__), tags, type, or custom_filter.
    If 'names' is provided, matches if either tool.name or class name is in the set.
    Returns a list of Tool objects (default) or metadata dicts (if metadata_only=True).
    """
    try:
        all_tools = list(get_smolagents_tools())
    except ImportError:
        all_tools = []
    tools = []
    metadata = []
    for tool in all_tools:
        tool_name = getattr(tool, "name", str(tool))
        tool_class_name = type(tool).__name__
        if names and tool_name not in names and tool_class_name not in names:
            continue
        if metadata_only:
            # Determine service for metadata dict
            if hasattr(tool, 'service') and tool.service:
                service = tool.service
            elif hasattr(tool, '__module__'):
                mod = tool.__module__
                if "smolagents" in mod:
                    service = "smolagents"
                else:
                    service = mod.split('.')[-1] or 'unknown'
            else:
                service = 'unknown'
            meta = {
                "name": getattr(tool, "name", str(tool)),
                "description": getattr(tool, "description", ""),
                "args_schema": getattr(tool, "args_schema", {}),
                "tags": getattr(tool, "tags", []),
                "type": type(tool).__name__,
                "service": service,
            }
            metadata.append(meta)
    # Further filter by tags/types/custom_filter
    filtered = []
    for tool in tools:
        if isinstance(tool, Tool):
            name = getattr(tool, "name", None)
            tool_tags = set(getattr(tool, "tags", []))
            tool_type = type(tool).__name__
        if tags and not (tags & tool_tags):
            continue
        if types and tool_type not in types:
            continue
        if custom_filter and not custom_filter(tool):
            continue
        filtered.append(tool)
    if metadata_only:
        return [tool_to_metadata(tool) for tool in filtered]
    return filtered if any([tags, types, custom_filter]) else tools
