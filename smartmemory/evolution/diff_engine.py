"""
Procedure diff engine for computing differences between versions.

Provides utilities to compare two procedure versions and generate
human-readable diffs for the evolution timeline.
"""

import difflib
from typing import Any, Dict, List, Set

from smartmemory.evolution.models import ContentSnapshot, EventDiff


class ProcedureDiffEngine:
    """Engine for computing diffs between procedure versions.

    Compares content, skills, tools, and steps to generate a comprehensive
    diff with a human-readable summary.
    """

    def compute_diff(
        self,
        old_snapshot: ContentSnapshot,
        new_snapshot: ContentSnapshot,
    ) -> EventDiff:
        """Compute the diff between two content snapshots.

        Args:
            old_snapshot: The previous version's content snapshot
            new_snapshot: The new version's content snapshot

        Returns:
            EventDiff with has_changes, summary, and detailed diff
        """
        diff: Dict[str, Any] = {}
        changes: List[str] = []

        # Compare content
        if old_snapshot.content != new_snapshot.content:
            content_diff = self._compute_text_diff(old_snapshot.content, new_snapshot.content)
            if content_diff:
                diff["content"] = {
                    "old": old_snapshot.content[:500] + ("..." if len(old_snapshot.content) > 500 else ""),
                    "new": new_snapshot.content[:500] + ("..." if len(new_snapshot.content) > 500 else ""),
                    "unified_diff": content_diff,
                }
                changes.append("content modified")

        # Compare name
        if old_snapshot.name != new_snapshot.name:
            diff["name"] = {
                "old": old_snapshot.name,
                "new": new_snapshot.name,
            }
            changes.append(f"renamed from '{old_snapshot.name}' to '{new_snapshot.name}'")

        # Compare description
        if old_snapshot.description != new_snapshot.description:
            diff["description"] = {
                "old": old_snapshot.description,
                "new": new_snapshot.description,
            }
            changes.append("description updated")

        # Compare skills
        skills_diff = self._compute_list_diff(old_snapshot.skills, new_snapshot.skills)
        if skills_diff["added"] or skills_diff["removed"]:
            diff["skills"] = skills_diff
            if skills_diff["added"]:
                changes.append(f"+{len(skills_diff['added'])} skills")
            if skills_diff["removed"]:
                changes.append(f"-{len(skills_diff['removed'])} skills")

        # Compare tools
        tools_diff = self._compute_list_diff(old_snapshot.tools, new_snapshot.tools)
        if tools_diff["added"] or tools_diff["removed"]:
            diff["tools"] = tools_diff
            if tools_diff["added"]:
                changes.append(f"+{len(tools_diff['added'])} tools")
            if tools_diff["removed"]:
                changes.append(f"-{len(tools_diff['removed'])} tools")

        # Compare steps
        steps_diff = self._compute_ordered_list_diff(old_snapshot.steps, new_snapshot.steps)
        if steps_diff["has_changes"]:
            diff["steps"] = steps_diff
            step_changes = []
            if steps_diff.get("added"):
                step_changes.append(f"+{len(steps_diff['added'])} steps")
            if steps_diff.get("removed"):
                step_changes.append(f"-{len(steps_diff['removed'])} steps")
            if steps_diff.get("modified"):
                step_changes.append(f"~{len(steps_diff['modified'])} steps modified")
            if step_changes:
                changes.append(", ".join(step_changes))

        has_changes = bool(diff)
        summary = "; ".join(changes) if changes else "No changes"

        return EventDiff(
            has_changes=has_changes,
            summary=summary,
            diff=diff,
        )

    def _compute_text_diff(self, old_text: str, new_text: str) -> List[str]:
        """Compute unified diff between two text strings.

        Args:
            old_text: The previous version of the text
            new_text: The new version of the text

        Returns:
            List of diff lines in unified format
        """
        old_lines = old_text.splitlines(keepends=True)
        new_lines = new_text.splitlines(keepends=True)

        diff = list(
            difflib.unified_diff(
                old_lines,
                new_lines,
                fromfile="previous",
                tofile="current",
                lineterm="",
            )
        )

        return diff

    def _compute_list_diff(
        self,
        old_list: List[str],
        new_list: List[str],
    ) -> Dict[str, List[str]]:
        """Compute set diff between two lists (order-independent).

        Args:
            old_list: The previous version of the list
            new_list: The new version of the list

        Returns:
            Dict with 'added' and 'removed' lists
        """
        old_set: Set[str] = set(old_list)
        new_set: Set[str] = set(new_list)

        return {
            "added": sorted(new_set - old_set),
            "removed": sorted(old_set - new_set),
        }

    def _compute_ordered_list_diff(
        self,
        old_list: List[str],
        new_list: List[str],
    ) -> Dict[str, Any]:
        """Compute diff between two ordered lists (e.g., steps).

        Uses sequence matching to identify added, removed, and modified items.

        Args:
            old_list: The previous version of the list
            new_list: The new version of the list

        Returns:
            Dict with 'has_changes', 'added', 'removed', 'modified', and 'reordered' keys
        """
        if old_list == new_list:
            return {"has_changes": False}

        matcher = difflib.SequenceMatcher(None, old_list, new_list)
        opcodes = matcher.get_opcodes()

        added: List[Dict[str, Any]] = []
        removed: List[Dict[str, Any]] = []
        modified: List[Dict[str, Any]] = []

        for tag, i1, i2, j1, j2 in opcodes:
            if tag == "insert":
                for j in range(j1, j2):
                    added.append({"index": j, "value": new_list[j]})
            elif tag == "delete":
                for i in range(i1, i2):
                    removed.append({"index": i, "value": old_list[i]})
            elif tag == "replace":
                # Items at these positions were modified
                for k, (i, j) in enumerate(zip(range(i1, i2), range(j1, j2), strict=False)):
                    modified.append(
                        {
                            "old_index": i,
                            "new_index": j,
                            "old_value": old_list[i],
                            "new_value": new_list[j],
                        }
                    )
                # Handle length differences in replace
                if i2 - i1 > j2 - j1:
                    # More items removed than added
                    for i in range(i1 + (j2 - j1), i2):
                        removed.append({"index": i, "value": old_list[i]})
                elif j2 - j1 > i2 - i1:
                    # More items added than removed
                    for j in range(j1 + (i2 - i1), j2):
                        added.append({"index": j, "value": new_list[j]})

        return {
            "has_changes": True,
            "added": added if added else None,
            "removed": removed if removed else None,
            "modified": modified if modified else None,
        }
