"""Unit tests for CFS-4: SchemaDiffEngine.

This is a logic kernel — every diff rule must be tested exhaustively.
"""

from smartmemory.schema_diff import (
    SchemaChange,
    SchemaDiffResult,
    diff_schemas,
    diff_tool_schemas,
)

# ---------------------------------------------------------------------------
# Shared sample schemas
# ---------------------------------------------------------------------------

SCHEMA_V1 = {
    "type": "object",
    "required": ["name", "content"],
    "properties": {
        "name": {"type": "string", "description": "Tool name"},
        "content": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}},
        "status": {"type": "string", "enum": ["active", "archived"]},
    },
}


class TestDiffSchemas:
    """Tests for diff_schemas() — property-level schema comparison."""

    def test_identical_schemas_no_changes(self):
        """Same schema compared to itself yields no changes."""
        result = diff_schemas(SCHEMA_V1, SCHEMA_V1)
        assert result.has_changes is False
        assert result.has_breaking_changes is False
        assert result.changes == []
        assert result.breaking_count == 0
        assert result.non_breaking_count == 0
        assert result.summary == ""

    def test_empty_schemas_no_changes(self):
        """Two empty dicts produce no changes."""
        result = diff_schemas({}, {})
        assert result.has_changes is False
        assert result.has_breaking_changes is False
        assert result.changes == []

    def test_empty_old_all_additions(self):
        """Empty old schema with non-empty new: every property is an addition."""
        new_schema = {
            "type": "object",
            "properties": {
                "alpha": {"type": "string"},
                "beta": {"type": "integer"},
            },
        }
        result = diff_schemas({}, new_schema)
        assert result.has_changes is True
        assert result.has_breaking_changes is False
        assert result.non_breaking_count == 2
        assert result.breaking_count == 0
        for change in result.changes:
            assert change.change_type == "added"
            assert change.breaking is False

    def test_empty_new_all_removals(self):
        """Non-empty old, empty new: required properties removed are breaking."""
        result = diff_schemas(SCHEMA_V1, {})
        assert result.has_changes is True
        assert result.has_breaking_changes is True
        # "name" and "content" are required -> breaking
        # "tags" and "status" are optional -> non-breaking
        breaking = [c for c in result.changes if c.breaking]
        non_breaking = [c for c in result.changes if not c.breaking]
        assert len(breaking) == 2
        assert len(non_breaking) == 2
        breaking_paths = {c.path for c in breaking}
        assert "/properties/name" in breaking_paths
        assert "/properties/content" in breaking_paths

    def test_required_property_removed_is_breaking(self):
        """Removing a property listed in 'required' is a breaking change."""
        new_schema = {
            "type": "object",
            "required": ["content"],
            "properties": {
                "content": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "status": {"type": "string", "enum": ["active", "archived"]},
            },
        }
        result = diff_schemas(SCHEMA_V1, new_schema)
        assert result.has_breaking_changes is True
        removed = [c for c in result.changes if c.path == "/properties/name"]
        assert len(removed) == 1
        assert removed[0].breaking is True
        assert removed[0].change_type == "removed"

    def test_optional_property_removed_is_not_breaking(self):
        """Removing a property NOT in 'required' is non-breaking."""
        new_schema = {
            "type": "object",
            "required": ["name", "content"],
            "properties": {
                "name": {"type": "string", "description": "Tool name"},
                "content": {"type": "string"},
                "status": {"type": "string", "enum": ["active", "archived"]},
                # "tags" removed — it was optional
            },
        }
        result = diff_schemas(SCHEMA_V1, new_schema)
        assert result.has_changes is True
        removed = [c for c in result.changes if c.path == "/properties/tags"]
        assert len(removed) == 1
        assert removed[0].breaking is False
        assert removed[0].change_type == "removed"

    def test_property_type_changed_is_breaking(self):
        """Changing a property's type is always breaking."""
        new_schema = {
            "type": "object",
            "required": ["name", "content"],
            "properties": {
                "name": {"type": "integer", "description": "Tool name"},  # was string
                "content": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "status": {"type": "string", "enum": ["active", "archived"]},
            },
        }
        result = diff_schemas(SCHEMA_V1, new_schema)
        assert result.has_breaking_changes is True
        type_changes = [c for c in result.changes if c.change_type == "type_changed"]
        assert len(type_changes) == 1
        assert type_changes[0].path == "/properties/name/type"
        assert type_changes[0].old_value == "string"
        assert type_changes[0].new_value == "integer"
        assert type_changes[0].breaking is True

    def test_new_required_property_is_breaking(self):
        """Adding a new property that is also required is breaking."""
        new_schema = {
            "type": "object",
            "required": ["name", "content", "priority"],
            "properties": {
                "name": {"type": "string", "description": "Tool name"},
                "content": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "status": {"type": "string", "enum": ["active", "archived"]},
                "priority": {"type": "integer"},  # new and required
            },
        }
        result = diff_schemas(SCHEMA_V1, new_schema)
        assert result.has_breaking_changes is True
        added = [c for c in result.changes if "/properties/priority" in c.path and c.change_type == "added"]
        assert len(added) == 1
        assert added[0].breaking is True

    def test_new_optional_property_is_not_breaking(self):
        """Adding a new property that is NOT required is non-breaking."""
        new_schema = {
            "type": "object",
            "required": ["name", "content"],
            "properties": {
                "name": {"type": "string", "description": "Tool name"},
                "content": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "status": {"type": "string", "enum": ["active", "archived"]},
                "metadata": {"type": "object"},  # new and optional
            },
        }
        result = diff_schemas(SCHEMA_V1, new_schema)
        assert result.has_changes is True
        assert result.has_breaking_changes is False
        added = [c for c in result.changes if c.path == "/properties/metadata"]
        assert len(added) == 1
        assert added[0].breaking is False

    def test_enum_value_removed_is_breaking(self):
        """Removing a value from an enum is a breaking change."""
        new_schema = {
            "type": "object",
            "required": ["name", "content"],
            "properties": {
                "name": {"type": "string", "description": "Tool name"},
                "content": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "status": {"type": "string", "enum": ["active"]},  # "archived" removed
            },
        }
        result = diff_schemas(SCHEMA_V1, new_schema)
        assert result.has_breaking_changes is True
        enum_removed = [c for c in result.changes if c.path == "/properties/status/enum" and c.change_type == "removed"]
        assert len(enum_removed) == 1
        assert enum_removed[0].old_value == "archived"
        assert enum_removed[0].breaking is True

    def test_enum_value_added_is_not_breaking(self):
        """Adding a value to an enum is non-breaking."""
        new_schema = {
            "type": "object",
            "required": ["name", "content"],
            "properties": {
                "name": {"type": "string", "description": "Tool name"},
                "content": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "status": {"type": "string", "enum": ["active", "archived", "draft"]},  # "draft" added
            },
        }
        result = diff_schemas(SCHEMA_V1, new_schema)
        assert result.has_changes is True
        assert result.has_breaking_changes is False
        enum_added = [c for c in result.changes if c.path == "/properties/status/enum" and c.change_type == "added"]
        assert len(enum_added) == 1
        assert enum_added[0].new_value == "draft"
        assert enum_added[0].breaking is False

    def test_description_change_is_not_breaking(self):
        """Changing only a description is a non-breaking modification."""
        new_schema = {
            "type": "object",
            "required": ["name", "content"],
            "properties": {
                "name": {"type": "string", "description": "The tool's name"},  # description changed
                "content": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "status": {"type": "string", "enum": ["active", "archived"]},
            },
        }
        result = diff_schemas(SCHEMA_V1, new_schema)
        assert result.has_changes is True
        assert result.has_breaking_changes is False
        desc_changes = [c for c in result.changes if c.change_type == "modified"]
        assert len(desc_changes) == 1
        assert desc_changes[0].path == "/properties/name/description"
        assert desc_changes[0].old_value == "Tool name"
        assert desc_changes[0].new_value == "The tool's name"

    def test_nested_property_changes(self):
        """Changes inside a nested properties object are detected."""
        old_schema = {
            "type": "object",
            "properties": {
                "address": {
                    "type": "object",
                    "required": ["street"],
                    "properties": {
                        "street": {"type": "string"},
                        "city": {"type": "string"},
                    },
                },
            },
        }
        new_schema = {
            "type": "object",
            "properties": {
                "address": {
                    "type": "object",
                    "required": ["street"],
                    "properties": {
                        "street": {"type": "integer"},  # type changed
                        "city": {"type": "string"},
                        "zip": {"type": "string"},  # added
                    },
                },
            },
        }
        result = diff_schemas(old_schema, new_schema)
        assert result.has_changes is True
        assert result.has_breaking_changes is True
        type_changes = [c for c in result.changes if c.change_type == "type_changed"]
        assert len(type_changes) == 1
        assert type_changes[0].path == "/properties/address/properties/street/type"
        added = [c for c in result.changes if c.change_type == "added"]
        assert len(added) == 1
        assert added[0].path == "/properties/address/properties/zip"

    def test_mixed_breaking_and_non_breaking(self):
        """A single diff can contain both breaking and non-breaking changes."""
        new_schema = {
            "type": "object",
            "required": ["name", "content"],
            "properties": {
                "name": {"type": "integer", "description": "Tool name"},  # type changed (breaking)
                "content": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "status": {"type": "string", "enum": ["active", "archived", "draft"]},  # enum added (non-breaking)
                "metadata": {"type": "object"},  # new optional (non-breaking)
            },
        }
        result = diff_schemas(SCHEMA_V1, new_schema)
        assert result.has_changes is True
        assert result.has_breaking_changes is True
        assert result.breaking_count >= 1
        assert result.non_breaking_count >= 1
        assert result.breaking_count + result.non_breaking_count == len(result.changes)

    def test_summary_string_generated(self):
        """Summary is a non-empty string when changes exist."""
        new_schema = {
            "type": "object",
            "required": ["name", "content"],
            "properties": {
                "name": {"type": "string", "description": "Tool name"},
                "content": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "status": {"type": "string", "enum": ["active", "archived"]},
                "extra": {"type": "string"},  # new optional property
            },
        }
        result = diff_schemas(SCHEMA_V1, new_schema)
        assert result.has_changes is True
        assert len(result.summary) > 0
        assert "non-breaking" in result.summary.lower() or "1" in result.summary

    def test_summary_empty_when_no_changes(self):
        """Summary is empty string when schemas are identical."""
        result = diff_schemas(SCHEMA_V1, SCHEMA_V1)
        assert result.summary == ""

    def test_breaking_count_and_non_breaking_count(self):
        """Counts accurately reflect the number of breaking/non-breaking changes."""
        new_schema = {
            "type": "object",
            "required": ["content"],  # "name" removed from required AND properties
            "properties": {
                # "name" removed (was required -> breaking)
                "content": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "status": {"type": "string", "enum": ["active", "archived", "pending"]},  # enum added (non-breaking)
                "notes": {"type": "string"},  # added optional (non-breaking)
            },
        }
        result = diff_schemas(SCHEMA_V1, new_schema)
        assert result.has_changes is True
        breaking = [c for c in result.changes if c.breaking]
        non_breaking = [c for c in result.changes if not c.breaking]
        assert result.breaking_count == len(breaking)
        assert result.non_breaking_count == len(non_breaking)
        assert result.breaking_count + result.non_breaking_count == len(result.changes)
        # "name" removal is breaking (1), "pending" enum added (1), "notes" added (1)
        assert result.breaking_count == 1
        assert result.non_breaking_count == 2

    def test_existing_property_made_required_is_breaking(self):
        """Making an existing optional property required is a breaking change."""
        new_schema = {
            "type": "object",
            "required": ["name", "content", "tags"],  # "tags" now required
            "properties": {
                "name": {"type": "string", "description": "Tool name"},
                "content": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "status": {"type": "string", "enum": ["active", "archived"]},
            },
        }
        result = diff_schemas(SCHEMA_V1, new_schema)
        assert result.has_breaking_changes is True
        req_changes = [c for c in result.changes if "/required/tags" in c.path]
        assert len(req_changes) == 1
        assert req_changes[0].breaking is True
        assert req_changes[0].change_type == "added"


class TestSchemaChangeToDict:
    """Tests for SchemaChange.to_dict() serialization."""

    def test_to_dict_roundtrip(self):
        """to_dict() returns a dict with exactly the expected keys and values."""
        change = SchemaChange(
            path="/properties/name",
            change_type="removed",
            old_value={"type": "string"},
            new_value=None,
            breaking=True,
        )
        d = change.to_dict()
        assert isinstance(d, dict)
        assert set(d.keys()) == {"path", "change_type", "old_value", "new_value", "breaking"}
        assert d["path"] == "/properties/name"
        assert d["change_type"] == "removed"
        assert d["old_value"] == {"type": "string"}
        assert d["new_value"] is None
        assert d["breaking"] is True

    def test_schema_diff_result_to_dict(self):
        """SchemaDiffResult.to_dict() produces a valid dict with serialized changes."""
        change = SchemaChange(path="/properties/x", change_type="added", new_value={"type": "string"})
        result = SchemaDiffResult(
            has_changes=True,
            has_breaking_changes=False,
            changes=[change],
            summary="1 non-breaking change",
            breaking_count=0,
            non_breaking_count=1,
        )
        d = result.to_dict()
        assert isinstance(d, dict)
        assert set(d.keys()) == {
            "has_changes",
            "has_breaking_changes",
            "changes",
            "summary",
            "breaking_count",
            "non_breaking_count",
        }
        assert d["has_changes"] is True
        assert d["has_breaking_changes"] is False
        assert len(d["changes"]) == 1
        assert isinstance(d["changes"][0], dict)
        assert d["changes"][0]["path"] == "/properties/x"
        assert d["breaking_count"] == 0
        assert d["non_breaking_count"] == 1

    def test_empty_result_to_dict(self):
        """Default SchemaDiffResult serializes correctly."""
        result = SchemaDiffResult()
        d = result.to_dict()
        assert d["has_changes"] is False
        assert d["has_breaking_changes"] is False
        assert d["changes"] == []
        assert d["summary"] == ""
        assert d["breaking_count"] == 0
        assert d["non_breaking_count"] == 0


class TestDiffToolSchemas:
    """Tests for diff_tool_schemas() — multi-tool comparison."""

    def test_tool_removed_is_breaking(self):
        """A tool present in old but not new is a breaking removal."""
        old_tools = {
            "search": SCHEMA_V1,
            "create": {"type": "object", "properties": {"title": {"type": "string"}}},
        }
        new_tools = {
            "search": SCHEMA_V1,
        }
        results = diff_tool_schemas(old_tools, new_tools)
        assert "create" in results
        assert results["create"].has_changes is True
        assert results["create"].has_breaking_changes is True
        assert results["create"].breaking_count == 1
        assert results["create"].changes[0].change_type == "removed"
        # "search" is identical
        assert "search" in results
        assert results["search"].has_changes is False

    def test_tool_added_is_not_breaking(self):
        """A tool present in new but not old is a non-breaking addition."""
        old_tools = {
            "search": SCHEMA_V1,
        }
        new_tools = {
            "search": SCHEMA_V1,
            "delete": {"type": "object", "properties": {"id": {"type": "string"}}},
        }
        results = diff_tool_schemas(old_tools, new_tools)
        assert "delete" in results
        assert results["delete"].has_changes is True
        assert results["delete"].has_breaking_changes is False
        assert results["delete"].non_breaking_count == 1
        assert results["delete"].changes[0].change_type == "added"

    def test_tool_schema_changed(self):
        """A tool present in both with schema differences delegates to diff_schemas."""
        new_schema = {
            "type": "object",
            "required": ["name", "content"],
            "properties": {
                "name": {"type": "integer", "description": "Tool name"},  # type change
                "content": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "status": {"type": "string", "enum": ["active", "archived"]},
            },
        }
        old_tools = {"search": SCHEMA_V1}
        new_tools = {"search": new_schema}
        results = diff_tool_schemas(old_tools, new_tools)
        assert "search" in results
        assert results["search"].has_changes is True
        assert results["search"].has_breaking_changes is True

    def test_empty_both(self):
        """Two empty tool sets produce no results."""
        results = diff_tool_schemas({}, {})
        assert results == {}

    def test_multiple_tools_mixed(self):
        """Mix of added, removed, and unchanged tools."""
        old_tools = {
            "tool_a": SCHEMA_V1,
            "tool_b": {"type": "object", "properties": {"x": {"type": "string"}}},
            "tool_c": {"type": "object", "properties": {"y": {"type": "integer"}}},
        }
        new_tools = {
            "tool_a": SCHEMA_V1,  # unchanged
            # tool_b removed
            "tool_c": {"type": "object", "properties": {"y": {"type": "string"}}},  # type changed
            "tool_d": {"type": "object", "properties": {"z": {"type": "boolean"}}},  # added
        }
        results = diff_tool_schemas(old_tools, new_tools)
        # tool_a: no changes
        assert results["tool_a"].has_changes is False
        # tool_b: removed (breaking)
        assert results["tool_b"].has_changes is True
        assert results["tool_b"].has_breaking_changes is True
        # tool_c: type changed (breaking)
        assert results["tool_c"].has_changes is True
        assert results["tool_c"].has_breaking_changes is True
        # tool_d: added (non-breaking)
        assert results["tool_d"].has_changes is True
        assert results["tool_d"].has_breaking_changes is False
