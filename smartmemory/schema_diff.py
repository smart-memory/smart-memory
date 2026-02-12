"""CFS-4: Schema Diff Engine.

Compares two JSON Schema dicts and produces a structured diff result
identifying breaking vs non-breaking changes. Used by the self-healing
procedures system to detect when MCP tool schemas have changed and
determine if stored procedures need adaptation.

Breaking changes:
- Required property removed
- Property type changed
- New required property added
- Enum value removed
- Tool removed entirely

Non-breaking changes:
- Optional property added or removed
- Enum value added
- Description changed
- Tool added
"""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class SchemaChange:
    """A single change detected between two schemas.

    Attributes:
        path: JSON pointer to the changed location (e.g. "/properties/name/type").
        change_type: One of "added", "removed", "modified", "type_changed".
        old_value: Previous value (None for additions).
        new_value: New value (None for removals).
        breaking: Whether this change could break existing consumers.
    """

    path: str
    change_type: str  # "added" | "removed" | "modified" | "type_changed"
    old_value: Any = None
    new_value: Any = None
    breaking: bool = False

    def to_dict(self) -> dict:
        """Serialize to a plain dict."""
        return {
            "path": self.path,
            "change_type": self.change_type,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "breaking": self.breaking,
        }


@dataclass
class SchemaDiffResult:
    """Aggregated result of comparing two schemas.

    Attributes:
        has_changes: True if any differences were detected.
        has_breaking_changes: True if at least one change is breaking.
        changes: List of individual SchemaChange objects.
        summary: Human-readable summary of the diff.
        breaking_count: Number of breaking changes.
        non_breaking_count: Number of non-breaking changes.
    """

    has_changes: bool = False
    has_breaking_changes: bool = False
    changes: list["SchemaChange"] = field(default_factory=list)
    summary: str = ""
    breaking_count: int = 0
    non_breaking_count: int = 0

    def to_dict(self) -> dict:
        """Serialize to a plain dict."""
        return {
            "has_changes": self.has_changes,
            "has_breaking_changes": self.has_breaking_changes,
            "changes": [c.to_dict() if isinstance(c, SchemaChange) else c for c in self.changes],
            "summary": self.summary,
            "breaking_count": self.breaking_count,
            "non_breaking_count": self.non_breaking_count,
        }


def diff_schemas(old_schema: dict, new_schema: dict) -> SchemaDiffResult:
    """Compare two JSON Schema dicts and return a structured diff.

    Detects added/removed properties, type changes, required-list changes,
    enum changes, and description changes. Recurses one level into nested
    properties that themselves have a ``properties`` key.

    Args:
        old_schema: The previous schema dict.
        new_schema: The updated schema dict.

    Returns:
        SchemaDiffResult with all detected changes and a human-readable summary.
    """
    changes: list[SchemaChange] = []
    _diff_properties(old_schema, new_schema, "", changes)
    return _build_result(changes)


def _diff_properties(old_schema: dict, new_schema: dict, prefix: str, changes: list[SchemaChange]) -> None:
    """Compare properties between two schemas, optionally with a path prefix.

    Args:
        old_schema: Old schema (or sub-schema).
        new_schema: New schema (or sub-schema).
        prefix: Path prefix for nested comparisons (e.g. "/properties/address").
        changes: Accumulator list for detected changes.
    """
    old_props = old_schema.get("properties", {})
    new_props = new_schema.get("properties", {})
    old_required: list[str] = old_schema.get("required", [])
    new_required: list[str] = new_schema.get("required", [])

    old_keys = set(old_props.keys())
    new_keys = set(new_props.keys())

    # --- Removed properties ---
    for key in sorted(old_keys - new_keys):
        is_breaking = key in old_required
        changes.append(
            SchemaChange(
                path=f"{prefix}/properties/{key}",
                change_type="removed",
                old_value=old_props[key],
                new_value=None,
                breaking=is_breaking,
            )
        )

    # --- Added properties ---
    for key in sorted(new_keys - old_keys):
        is_breaking = key in new_required
        changes.append(
            SchemaChange(
                path=f"{prefix}/properties/{key}",
                change_type="added",
                old_value=None,
                new_value=new_props[key],
                breaking=is_breaking,
            )
        )

    # --- Properties present in both ---
    for key in sorted(old_keys & new_keys):
        old_def = old_props[key]
        new_def = new_props[key]
        prop_path = f"{prefix}/properties/{key}"

        # Type change
        old_type = old_def.get("type")
        new_type = new_def.get("type")
        if old_type is not None and new_type is not None and old_type != new_type:
            changes.append(
                SchemaChange(
                    path=f"{prop_path}/type",
                    change_type="type_changed",
                    old_value=old_type,
                    new_value=new_type,
                    breaking=True,
                )
            )

        # Enum changes
        old_enum = old_def.get("enum")
        new_enum = new_def.get("enum")
        if old_enum is not None or new_enum is not None:
            _diff_enum(old_enum, new_enum, prop_path, changes)

        # Description change
        old_desc = old_def.get("description")
        new_desc = new_def.get("description")
        if old_desc != new_desc and old_desc is not None and new_desc is not None:
            changes.append(
                SchemaChange(
                    path=f"{prop_path}/description",
                    change_type="modified",
                    old_value=old_desc,
                    new_value=new_desc,
                    breaking=False,
                )
            )

        # Recurse one level into nested properties
        if "properties" in old_def or "properties" in new_def:
            _diff_properties(old_def, new_def, prop_path, changes)

    # --- Required list changes (additions only; removals are covered above) ---
    # A property that existed before but is now newly required
    for key in sorted(set(new_required) - set(old_required)):
        # Only flag as breaking if the property already existed (new props handled above)
        if key in old_keys and key in new_keys:
            changes.append(
                SchemaChange(
                    path=f"{prefix}/required/{key}",
                    change_type="added",
                    old_value=None,
                    new_value=key,
                    breaking=True,
                )
            )


def _diff_enum(
    old_enum: Optional[list],
    new_enum: Optional[list],
    prop_path: str,
    changes: list[SchemaChange],
) -> None:
    """Compare enum lists for a single property.

    Args:
        old_enum: Previous enum values (may be None).
        new_enum: Updated enum values (may be None).
        prop_path: JSON pointer to the property.
        changes: Accumulator for detected changes.
    """
    old_set = set(old_enum) if old_enum else set()
    new_set = set(new_enum) if new_enum else set()

    removed = old_set - new_set
    added = new_set - old_set

    for val in sorted(removed, key=str):
        changes.append(
            SchemaChange(
                path=f"{prop_path}/enum",
                change_type="removed",
                old_value=val,
                new_value=None,
                breaking=True,
            )
        )

    for val in sorted(added, key=str):
        changes.append(
            SchemaChange(
                path=f"{prop_path}/enum",
                change_type="added",
                old_value=None,
                new_value=val,
                breaking=False,
            )
        )


def _build_result(changes: list[SchemaChange]) -> SchemaDiffResult:
    """Construct a SchemaDiffResult from a list of changes.

    Args:
        changes: All detected SchemaChange objects.

    Returns:
        Fully populated SchemaDiffResult with counts and summary.
    """
    if not changes:
        return SchemaDiffResult()

    breaking_count = sum(1 for c in changes if c.breaking)
    non_breaking_count = len(changes) - breaking_count

    summary_parts: list[str] = []
    if breaking_count:
        summary_parts.append(f"{breaking_count} breaking change{'s' if breaking_count != 1 else ''}")
    if non_breaking_count:
        summary_parts.append(f"{non_breaking_count} non-breaking change{'s' if non_breaking_count != 1 else ''}")

    # Add detail lines for each change
    for change in changes:
        label = "BREAKING" if change.breaking else "minor"
        if change.change_type == "type_changed":
            summary_parts.append(f"  [{label}] {change.path}: type {change.old_value} -> {change.new_value}")
        elif change.change_type == "removed":
            summary_parts.append(f"  [{label}] {change.path}: removed")
        elif change.change_type == "added":
            summary_parts.append(f"  [{label}] {change.path}: added")
        elif change.change_type == "modified":
            summary_parts.append(f"  [{label}] {change.path}: modified")

    return SchemaDiffResult(
        has_changes=True,
        has_breaking_changes=breaking_count > 0,
        changes=changes,
        summary="\n".join(summary_parts),
        breaking_count=breaking_count,
        non_breaking_count=non_breaking_count,
    )


def diff_tool_schemas(old_schemas: dict[str, dict], new_schemas: dict[str, dict]) -> dict[str, SchemaDiffResult]:
    """Compare tool schemas across two versions, tool by tool.

    Each key is a tool name, each value is that tool's JSON Schema dict.
    Tools present in old but not new are treated as removed (breaking).
    Tools present in new but not old are treated as added (non-breaking).
    Tools present in both are compared with ``diff_schemas``.

    Args:
        old_schemas: Mapping of tool name to schema for the old version.
        new_schemas: Mapping of tool name to schema for the new version.

    Returns:
        Dict mapping tool name to its SchemaDiffResult.
    """
    results: dict[str, SchemaDiffResult] = {}
    old_tools = set(old_schemas.keys())
    new_tools = set(new_schemas.keys())

    # Tools removed
    for tool in sorted(old_tools - new_tools):
        change = SchemaChange(
            path=f"/{tool}",
            change_type="removed",
            old_value=tool,
            new_value=None,
            breaking=True,
        )
        results[tool] = SchemaDiffResult(
            has_changes=True,
            has_breaking_changes=True,
            changes=[change],
            summary=f"Tool '{tool}' removed",
            breaking_count=1,
            non_breaking_count=0,
        )

    # Tools added
    for tool in sorted(new_tools - old_tools):
        change = SchemaChange(
            path=f"/{tool}",
            change_type="added",
            old_value=None,
            new_value=tool,
            breaking=False,
        )
        results[tool] = SchemaDiffResult(
            has_changes=True,
            has_breaking_changes=False,
            changes=[change],
            summary=f"Tool '{tool}' added",
            breaking_count=0,
            non_breaking_count=1,
        )

    # Tools in both
    for tool in sorted(old_tools & new_tools):
        result = diff_schemas(old_schemas[tool], new_schemas[tool])
        results[tool] = result

    return results
