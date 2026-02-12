"""Unit tests for ProcedureDiffEngine."""

import pytest

from smartmemory.evolution.diff_engine import ProcedureDiffEngine
from smartmemory.evolution.models import ContentSnapshot


class TestProcedureDiffEngine:
    """Tests for ProcedureDiffEngine."""

    @pytest.fixture
    def engine(self):
        """Create a ProcedureDiffEngine instance."""
        return ProcedureDiffEngine()

    def test_identical_snapshots_no_changes(self, engine):
        """Test that identical snapshots produce no changes."""
        snapshot = ContentSnapshot(
            content="Test content",
            name="Test",
            description="Description",
            skills=["skill1"],
            tools=["tool1"],
            steps=["step1"],
        )
        diff = engine.compute_diff(snapshot, snapshot)
        assert diff.has_changes is False
        assert diff.summary == "No changes"
        assert diff.diff == {}

    def test_content_change_detected(self, engine):
        """Test that content changes are detected."""
        old = ContentSnapshot(content="Old content")
        new = ContentSnapshot(content="New content")
        diff = engine.compute_diff(old, new)
        assert diff.has_changes is True
        assert "content modified" in diff.summary
        assert "content" in diff.diff
        assert diff.diff["content"]["old"] == "Old content"
        assert diff.diff["content"]["new"] == "New content"

    def test_name_change_detected(self, engine):
        """Test that name changes are detected."""
        old = ContentSnapshot(name="Old Name")
        new = ContentSnapshot(name="New Name")
        diff = engine.compute_diff(old, new)
        assert diff.has_changes is True
        assert "renamed" in diff.summary
        assert "name" in diff.diff
        assert diff.diff["name"]["old"] == "Old Name"
        assert diff.diff["name"]["new"] == "New Name"

    def test_description_change_detected(self, engine):
        """Test that description changes are detected."""
        old = ContentSnapshot(description="Old description")
        new = ContentSnapshot(description="New description")
        diff = engine.compute_diff(old, new)
        assert diff.has_changes is True
        assert "description updated" in diff.summary
        assert "description" in diff.diff

    def test_skills_added(self, engine):
        """Test that added skills are detected."""
        old = ContentSnapshot(skills=["skill1"])
        new = ContentSnapshot(skills=["skill1", "skill2", "skill3"])
        diff = engine.compute_diff(old, new)
        assert diff.has_changes is True
        assert "+2 skills" in diff.summary
        assert "skills" in diff.diff
        assert sorted(diff.diff["skills"]["added"]) == ["skill2", "skill3"]
        assert diff.diff["skills"]["removed"] == []

    def test_skills_removed(self, engine):
        """Test that removed skills are detected."""
        old = ContentSnapshot(skills=["skill1", "skill2", "skill3"])
        new = ContentSnapshot(skills=["skill1"])
        diff = engine.compute_diff(old, new)
        assert diff.has_changes is True
        assert "-2 skills" in diff.summary
        assert "skills" in diff.diff
        assert diff.diff["skills"]["added"] == []
        assert sorted(diff.diff["skills"]["removed"]) == ["skill2", "skill3"]

    def test_skills_mixed_changes(self, engine):
        """Test that mixed skill changes are detected."""
        old = ContentSnapshot(skills=["skill1", "skill2"])
        new = ContentSnapshot(skills=["skill2", "skill3"])
        diff = engine.compute_diff(old, new)
        assert diff.has_changes is True
        assert "+1 skills" in diff.summary
        assert "-1 skills" in diff.summary
        assert diff.diff["skills"]["added"] == ["skill3"]
        assert diff.diff["skills"]["removed"] == ["skill1"]

    def test_tools_added(self, engine):
        """Test that added tools are detected."""
        old = ContentSnapshot(tools=[])
        new = ContentSnapshot(tools=["tool1", "tool2"])
        diff = engine.compute_diff(old, new)
        assert diff.has_changes is True
        assert "+2 tools" in diff.summary
        assert "tools" in diff.diff
        assert sorted(diff.diff["tools"]["added"]) == ["tool1", "tool2"]

    def test_tools_removed(self, engine):
        """Test that removed tools are detected."""
        old = ContentSnapshot(tools=["tool1"])
        new = ContentSnapshot(tools=[])
        diff = engine.compute_diff(old, new)
        assert diff.has_changes is True
        assert "-1 tools" in diff.summary
        assert diff.diff["tools"]["removed"] == ["tool1"]

    def test_steps_added(self, engine):
        """Test that added steps are detected."""
        old = ContentSnapshot(steps=["step1"])
        new = ContentSnapshot(steps=["step1", "step2"])
        diff = engine.compute_diff(old, new)
        assert diff.has_changes is True
        assert "+1 steps" in diff.summary
        assert "steps" in diff.diff
        assert diff.diff["steps"]["has_changes"] is True

    def test_steps_removed(self, engine):
        """Test that removed steps are detected."""
        old = ContentSnapshot(steps=["step1", "step2", "step3"])
        new = ContentSnapshot(steps=["step1"])
        diff = engine.compute_diff(old, new)
        assert diff.has_changes is True
        assert "-2 steps" in diff.summary
        assert "steps" in diff.diff

    def test_steps_modified(self, engine):
        """Test that modified steps are detected."""
        old = ContentSnapshot(steps=["step1", "step2"])
        new = ContentSnapshot(steps=["step1", "step2 modified"])
        diff = engine.compute_diff(old, new)
        assert diff.has_changes is True
        assert "steps" in diff.diff
        # Should show step2 was modified to "step2 modified"

    def test_multiple_changes(self, engine):
        """Test that multiple changes are all detected."""
        old = ContentSnapshot(
            content="Old content",
            name="Old Name",
            skills=["skill1"],
            tools=["tool1"],
        )
        new = ContentSnapshot(
            content="New content",
            name="New Name",
            skills=["skill1", "skill2"],
            tools=[],
        )
        diff = engine.compute_diff(old, new)
        assert diff.has_changes is True
        # Should have multiple changes in summary
        summary_parts = diff.summary.split("; ")
        assert len(summary_parts) >= 3  # At least 3 different changes

    def test_long_content_truncated_in_diff(self, engine):
        """Test that long content is truncated in diff output."""
        old_content = "a" * 1000
        new_content = "b" * 1000
        old = ContentSnapshot(content=old_content)
        new = ContentSnapshot(content=new_content)
        diff = engine.compute_diff(old, new)
        assert diff.has_changes is True
        # Content should be truncated with "..."
        assert len(diff.diff["content"]["old"]) <= 503  # 500 + "..."
        assert diff.diff["content"]["old"].endswith("...")

    def test_empty_to_populated_snapshot(self, engine):
        """Test diff from empty to populated snapshot."""
        old = ContentSnapshot()
        new = ContentSnapshot(
            content="New content",
            name="New Procedure",
            skills=["skill1", "skill2"],
            tools=["tool1"],
            steps=["step1", "step2"],
        )
        diff = engine.compute_diff(old, new)
        assert diff.has_changes is True
        assert "content" in diff.diff
        assert "skills" in diff.diff
        assert "tools" in diff.diff
        assert "steps" in diff.diff


class TestTextDiff:
    """Tests for internal text diff computation."""

    @pytest.fixture
    def engine(self):
        return ProcedureDiffEngine()

    def test_empty_to_text(self, engine):
        """Test diff from empty string to text."""
        diff_lines = engine._compute_text_diff("", "New text")
        # Should produce unified diff output
        assert len(diff_lines) > 0

    def test_text_to_empty(self, engine):
        """Test diff from text to empty string."""
        diff_lines = engine._compute_text_diff("Old text", "")
        assert len(diff_lines) > 0

    def test_multiline_diff(self, engine):
        """Test diff with multiline content."""
        old = "Line 1\nLine 2\nLine 3"
        new = "Line 1\nModified Line 2\nLine 3"
        diff_lines = engine._compute_text_diff(old, new)
        assert len(diff_lines) > 0


class TestListDiff:
    """Tests for internal list diff computation."""

    @pytest.fixture
    def engine(self):
        return ProcedureDiffEngine()

    def test_empty_lists(self, engine):
        """Test diff between empty lists."""
        result = engine._compute_list_diff([], [])
        assert result["added"] == []
        assert result["removed"] == []

    def test_add_to_empty(self, engine):
        """Test adding items to empty list."""
        result = engine._compute_list_diff([], ["a", "b"])
        assert sorted(result["added"]) == ["a", "b"]
        assert result["removed"] == []

    def test_remove_all(self, engine):
        """Test removing all items."""
        result = engine._compute_list_diff(["a", "b"], [])
        assert result["added"] == []
        assert sorted(result["removed"]) == ["a", "b"]

    def test_no_changes(self, engine):
        """Test identical lists."""
        result = engine._compute_list_diff(["a", "b"], ["a", "b"])
        assert result["added"] == []
        assert result["removed"] == []

    def test_order_independent(self, engine):
        """Test that order doesn't matter for set diff."""
        result = engine._compute_list_diff(["a", "b"], ["b", "a"])
        assert result["added"] == []
        assert result["removed"] == []


class TestOrderedListDiff:
    """Tests for ordered list diff (steps)."""

    @pytest.fixture
    def engine(self):
        return ProcedureDiffEngine()

    def test_identical_lists(self, engine):
        """Test identical ordered lists."""
        result = engine._compute_ordered_list_diff(["a", "b", "c"], ["a", "b", "c"])
        assert result["has_changes"] is False

    def test_item_added_at_end(self, engine):
        """Test item added at end."""
        result = engine._compute_ordered_list_diff(["a", "b"], ["a", "b", "c"])
        assert result["has_changes"] is True
        assert result["added"] is not None
        assert any(item["value"] == "c" for item in result["added"])

    def test_item_removed(self, engine):
        """Test item removed."""
        result = engine._compute_ordered_list_diff(["a", "b", "c"], ["a", "c"])
        assert result["has_changes"] is True
        assert result["removed"] is not None
        assert any(item["value"] == "b" for item in result["removed"])

    def test_item_modified(self, engine):
        """Test item modified."""
        result = engine._compute_ordered_list_diff(["a", "b", "c"], ["a", "B", "c"])
        assert result["has_changes"] is True
        assert result["modified"] is not None
        assert any(item["old_value"] == "b" and item["new_value"] == "B" for item in result["modified"])
