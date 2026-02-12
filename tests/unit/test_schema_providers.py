"""Unit tests for CFS-4: SchemaProvider protocol and StaticSchemaProvider."""

from smartmemory.schema_providers import SchemaProvider, StaticSchemaProvider


def _sample_schemas() -> dict[str, dict]:
    """Build a small set of tool schemas for testing."""
    return {
        "search_web": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_results": {"type": "integer", "default": 10},
            },
            "required": ["query"],
        },
        "read_file": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
            },
            "required": ["path"],
        },
        "write_file": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    }


class TestStaticSchemaProvider:
    """Tests for StaticSchemaProvider."""

    def test_get_schemas_returns_matching_tools(self):
        """Requesting 2 of 3 tools returns exactly those 2."""
        provider = StaticSchemaProvider(_sample_schemas())
        result = provider.get_schemas(["search_web", "write_file"])

        assert set(result.keys()) == {"search_web", "write_file"}
        assert result["search_web"]["properties"]["query"]["type"] == "string"
        assert result["write_file"]["required"] == ["path", "content"]

    def test_get_schemas_unknown_tools_skipped(self):
        """Unknown tool names are silently excluded from the result."""
        provider = StaticSchemaProvider(_sample_schemas())
        result = provider.get_schemas(["search_web", "nonexistent_tool"])

        assert set(result.keys()) == {"search_web"}
        assert "nonexistent_tool" not in result

    def test_get_all_schemas_returns_full_dict(self):
        """get_all_schemas returns every tool in the provider."""
        schemas = _sample_schemas()
        provider = StaticSchemaProvider(schemas)
        result = provider.get_all_schemas()

        assert set(result.keys()) == {"search_web", "read_file", "write_file"}
        for name in schemas:
            assert result[name] == schemas[name]

    def test_source_type_default(self):
        """Default source_type is 'static'."""
        provider = StaticSchemaProvider({})
        assert provider.source_type == "static"

    def test_source_type_custom(self):
        """source_type is configurable via constructor."""
        provider = StaticSchemaProvider({}, source_type="manual")
        assert provider.source_type == "manual"

    def test_isinstance_schema_provider(self):
        """StaticSchemaProvider satisfies the SchemaProvider protocol."""
        provider = StaticSchemaProvider({})
        assert isinstance(provider, SchemaProvider)

    def test_empty_schemas(self):
        """Provider with empty dict works without errors."""
        provider = StaticSchemaProvider({})

        assert provider.get_all_schemas() == {}
        assert provider.get_schemas(["any_tool"]) == {}

    def test_get_schemas_empty_list(self):
        """Requesting no tools returns an empty dict."""
        provider = StaticSchemaProvider(_sample_schemas())
        result = provider.get_schemas([])

        assert result == {}

    def test_get_schemas_returns_copies(self):
        """Returned schemas are copies, not references to internal state."""
        provider = StaticSchemaProvider(_sample_schemas())
        result = provider.get_schemas(["search_web"])

        # Mutate the returned schema
        result["search_web"]["properties"]["query"]["type"] = "integer"

        # Internal state should be unaffected
        fresh = provider.get_schemas(["search_web"])
        assert fresh["search_web"]["properties"]["query"]["type"] == "string"

    def test_get_all_schemas_returns_copies(self):
        """get_all_schemas returns copies, not references to internal state."""
        provider = StaticSchemaProvider(_sample_schemas())
        result = provider.get_all_schemas()

        # Mutate the returned schema
        result["read_file"]["required"] = []

        # Internal state should be unaffected
        fresh = provider.get_all_schemas()
        assert fresh["read_file"]["required"] == ["path"]

    def test_constructor_takes_defensive_copy(self):
        """Mutating the original dict after construction does not affect the provider."""
        schemas = _sample_schemas()
        provider = StaticSchemaProvider(schemas)

        # Mutate the original dict
        schemas["search_web"]["properties"]["query"]["type"] = "number"
        schemas["new_tool"] = {"type": "object"}

        # Provider should be unaffected at the top level
        result = provider.get_all_schemas()
        assert "new_tool" not in result
