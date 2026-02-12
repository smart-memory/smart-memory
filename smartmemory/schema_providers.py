"""
Schema Providers for CFS-4: Self-Healing Procedures

SchemaProvider is the protocol that all schema sources implement. It defines
how tool schemas are fetched — whether from a static dict, an MCP server,
or another source.

StaticSchemaProvider is the simplest implementation: it wraps a pre-built
dict of tool_name -> JSON Schema mappings. Useful for testing, manual schema
registration, and offline operation.
"""

import copy
from typing import Protocol, runtime_checkable


@runtime_checkable
class SchemaProvider(Protocol):
    """Protocol for fetching tool schemas from any source.

    Implementations may pull schemas from MCP servers, static files,
    databases, or other registries. The protocol is kept minimal so
    that new sources can be added without changing consumers.
    """

    @property
    def source_type(self) -> str:
        """The source type identifier (e.g., "mcp", "static", "evolution", "manual")."""
        ...

    def get_schemas(self, tool_names: list[str]) -> dict[str, dict]:
        """Fetch schemas for the specified tool names.

        Args:
            tool_names: List of tool names to retrieve schemas for.

        Returns:
            Mapping of tool_name to JSON Schema dict. Only includes tools
            that exist in the provider; unknown names are silently skipped.
        """
        ...

    def get_all_schemas(self) -> dict[str, dict]:
        """Fetch all available schemas from this provider.

        Returns:
            Mapping of every known tool_name to its JSON Schema dict.
        """
        ...


class StaticSchemaProvider:
    """Schema provider backed by an in-memory dict.

    Wraps a pre-built mapping of tool_name -> JSON Schema. Useful for
    testing, manual registration, and scenarios where schemas are known
    ahead of time.

    Args:
        schemas: Mapping of tool_name to JSON Schema dict. A defensive
            copy is stored so mutations to the original dict don't
            affect the provider.
        source_type: Override the default source type identifier.
    """

    def __init__(self, schemas: dict[str, dict], source_type: str = "static") -> None:
        self._schemas: dict[str, dict] = copy.deepcopy(schemas)
        self._source_type: str = source_type

    @property
    def source_type(self) -> str:
        """The source type identifier for this provider."""
        return self._source_type

    def get_schemas(self, tool_names: list[str]) -> dict[str, dict]:
        """Fetch schemas for the specified tool names.

        Unknown tool names are silently skipped — only tools present
        in the provider's internal dict are returned.

        Args:
            tool_names: List of tool names to retrieve schemas for.

        Returns:
            Mapping of matching tool_name to JSON Schema dict.
        """
        return {name: copy.deepcopy(self._schemas[name]) for name in tool_names if name in self._schemas}

    def get_all_schemas(self) -> dict[str, dict]:
        """Fetch all schemas held by this provider.

        Returns:
            Full copy of the internal tool_name -> JSON Schema mapping.
        """
        return copy.deepcopy(self._schemas)
