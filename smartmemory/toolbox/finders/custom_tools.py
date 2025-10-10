# ALWAYS avoid import side effects in tool modules.
import importlib
import pkgutil
import sys
import traceback

import smartmemory.tools
from smartmemory.toolbox.tool_utils import ensure_tags, auto_args_schema, smolagents_tool_from_function


def find_tools(metadata_only=False, names=None, tags=None, types=None, custom_filter=None):
    """
    Discover custom tools in smartmemory.tools, efficiently filtering by names (module filenames) if provided.
    Returns a list of SmolAgents Tool objects (default) or metadata dicts (if metadata_only=True).
    """
    tools = []
    name_set = set(names) if names else None
    for loader, module_name, is_pkg in pkgutil.iter_modules(smartmemory.tools.__path__):
        try:
            module = importlib.import_module(f"smartmemory.tools.{module_name}")
        except Exception as e:
            print(f"[custom_tools] Skipping module {module_name} due to import error: {e}", file=sys.stderr)
            traceback.print_exc()
            continue
        # Discover all public functions in the module
        for attr_name in dir(module):
            if attr_name.startswith("_"):
                continue
            tool_func = getattr(module, attr_name)
            if callable(tool_func):
                # Only apply to real functions, not typing objects
                import types
                if isinstance(tool_func, (types.FunctionType, types.BuiltinFunctionType)):
                    # Skip imported functions - only process functions defined in this module
                    if hasattr(tool_func, '__module__') and tool_func.__module__ != module.__name__:
                        continue
                    # If name_set is provided, only include functions whose name is in name_set
                    if name_set and tool_func.__name__ not in name_set:
                        continue
                    ensure_tags(tool_func, [module_name, attr_name])
                    tool_func.args_schema = auto_args_schema(tool_func)
                    wrapped_tool = smolagents_tool_from_function(tool_func)
                    # Set .service to the filename for custom tools
                    if hasattr(module, "__file__"):
                        import os
                        wrapped_tool.service = os.path.splitext(os.path.basename(module.__file__))[0]
                    tools.append(wrapped_tool)
    # Filtering logic (by name, tags, types, custom_filter)
    filtered = []
    for tool in tools:
        name = getattr(tool, "name", None)
        tool_tags = set(getattr(tool, "tags", []))
        tool_type = type(tool).__name__
        if names and name not in names:
            continue
        if tags and not (tags & tool_tags):
            continue
        # Only perform type filtering if 'types' is an iterable (not a module)
        if types and isinstance(types, (set, list, tuple)):
            if tool_type not in types:
                continue
        if custom_filter and not custom_filter(tool):
            continue
        filtered.append(tool)
    if metadata_only:
        from smartmemory.toolbox.tool_utils import tool_to_metadata
        return [tool_to_metadata(tool) for tool in filtered]
    return filtered
