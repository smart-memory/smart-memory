"""Unit tests for PipelineState dataclass."""

import dataclasses
from datetime import datetime, timezone

from smartmemory.pipeline.state import PipelineState


class TestPipelineStateConstruction:
    """Tests for default construction and field defaults."""

    def test_default_construction(self):
        """All fields have defaults, so no-arg construction works."""
        state = PipelineState()
        assert state.mode == "sync"
        assert state.text == ""
        assert state.workspace_id is None
        assert state.user_id is None
        assert state.team_id is None
        assert state.memory_type is None
        assert state.resolved_text is None
        assert state.simplified_text is None
        assert state.classified_types == []
        assert state.ruler_entities == []
        assert state.llm_entities == []
        assert state.llm_relations == []
        assert state.entities == []
        assert state.relations == []
        assert state.rejected == []
        assert state.promotion_candidates == []
        assert state.item_id is None
        assert state.entity_ids == {}
        assert state.links == {}
        assert state.enrichments == {}
        assert state.evolutions == {}
        assert state.stage_history == []
        assert state.stage_timings == {}
        assert state.started_at is None
        assert state.completed_at is None
        assert state._context == {}

    def test_construction_with_text(self):
        """Construction with text argument sets the field."""
        state = PipelineState(text="hello world")
        assert state.text == "hello world"

    def test_construction_with_workspace_and_mode(self):
        """Construction with workspace_id and mode sets both."""
        state = PipelineState(workspace_id="ws-123", mode="preview")
        assert state.workspace_id == "ws-123"
        assert state.mode == "preview"

    def test_list_defaults_are_independent(self):
        """Each instance gets its own list instances (no shared mutable default)."""
        state_a = PipelineState()
        state_b = PipelineState()
        state_a.stage_history.append("classify")
        assert state_b.stage_history == []

    def test_dict_defaults_are_independent(self):
        """Each instance gets its own dict instances (no shared mutable default)."""
        state_a = PipelineState()
        state_b = PipelineState()
        state_a.stage_timings["classify"] = 42.0
        assert state_b.stage_timings == {}


class TestPipelineStateReplace:
    """Tests for dataclasses.replace() immutability pattern."""

    def test_replace_creates_new_instance(self):
        """dataclasses.replace() returns a different object."""
        state = PipelineState(text="original")
        new_state = dataclasses.replace(state, text="modified")
        assert state is not new_state
        assert id(state) != id(new_state)

    def test_replace_preserves_unchanged_fields(self):
        """Fields not specified in replace() keep their original values."""
        state = PipelineState(text="hello", workspace_id="ws-1")
        new_state = dataclasses.replace(state, text="world")
        assert new_state.text == "world"
        assert new_state.workspace_id == "ws-1"

    def test_replace_does_not_mutate_original(self):
        """The original state is unchanged after replace()."""
        state = PipelineState(text="original")
        dataclasses.replace(state, text="modified")
        assert state.text == "original"


class TestPipelineStateToDict:
    """Tests for to_dict() serialization."""

    def test_to_dict_produces_plain_dict(self):
        """to_dict() returns a dict, not a PipelineState."""
        state = PipelineState(text="hello")
        d = state.to_dict()
        assert isinstance(d, dict)
        assert d["text"] == "hello"
        assert d["mode"] == "sync"

    def test_to_dict_has_no_datetime_objects(self):
        """Datetime fields are serialized to ISO strings, not datetime objects."""
        now = datetime.now(timezone.utc)
        state = PipelineState(started_at=now, completed_at=now)
        d = state.to_dict()
        assert isinstance(d["started_at"], str)
        assert isinstance(d["completed_at"], str)

    def test_to_dict_datetime_none_remains_none(self):
        """None datetime fields stay None in the dict."""
        state = PipelineState()
        d = state.to_dict()
        assert d["started_at"] is None
        assert d["completed_at"] is None

    def test_to_dict_datetime_iso_format(self):
        """Datetime values are serialized in ISO 8601 format."""
        dt = datetime(2025, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        state = PipelineState(started_at=dt)
        d = state.to_dict()
        assert "2025-01-15" in d["started_at"]
        assert "10:30:00" in d["started_at"]

    def test_to_dict_excludes_underscore_fields(self):
        """Fields starting with _ (like _context) are excluded from to_dict()."""
        state = PipelineState()
        state._context = {"internal": "data"}
        d = state.to_dict()
        assert "_context" not in d

    def test_to_dict_includes_all_public_fields(self):
        """All non-underscore fields appear in the dict."""
        state = PipelineState()
        d = state.to_dict()
        expected_fields = {
            "mode",
            "workspace_id",
            "user_id",
            "team_id",
            "text",
            "raw_metadata",
            "memory_type",
            "resolved_text",
            "simplified_text",
            "classified_types",
            "ruler_entities",
            "llm_entities",
            "llm_relations",
            "entities",
            "relations",
            "rejected",
            "promotion_candidates",
            "item_id",
            "entity_ids",
            "links",
            "enrichments",
            "evolutions",
            "stage_history",
            "stage_timings",
            "started_at",
            "completed_at",
        }
        for field_name in expected_fields:
            assert field_name in d, f"Missing field: {field_name}"


class TestPipelineStateFromDict:
    """Tests for from_dict() deserialization and round-trips."""

    def test_from_dict_basic(self):
        """from_dict() constructs a PipelineState from a plain dict."""
        d = {"text": "hello", "mode": "preview", "workspace_id": "ws-1"}
        state = PipelineState.from_dict(d)
        assert state.text == "hello"
        assert state.mode == "preview"
        assert state.workspace_id == "ws-1"

    def test_round_trip_to_dict_from_dict(self):
        """state -> to_dict() -> from_dict() preserves all public values."""
        original = PipelineState(
            text="round trip test",
            mode="async",
            workspace_id="ws-42",
            classified_types=["semantic", "episodic"],
            stage_history=["classify", "extract"],
            stage_timings={"classify": 12.5, "extract": 88.3},
        )
        d = original.to_dict()
        restored = PipelineState.from_dict(d)
        assert restored.text == original.text
        assert restored.mode == original.mode
        assert restored.workspace_id == original.workspace_id
        assert restored.classified_types == original.classified_types
        assert restored.stage_history == original.stage_history
        assert restored.stage_timings == original.stage_timings

    def test_round_trip_with_datetime(self):
        """Datetime fields survive a to_dict() -> from_dict() round-trip."""
        now = datetime.now(timezone.utc)
        original = PipelineState(started_at=now, completed_at=now)
        d = original.to_dict()
        restored = PipelineState.from_dict(d)
        # After round-trip through ISO string, microseconds may be preserved
        assert isinstance(restored.started_at, datetime)
        assert isinstance(restored.completed_at, datetime)
        # Values should be equal (within ISO string precision)
        assert restored.started_at.replace(microsecond=0) == now.replace(microsecond=0)

    def test_from_dict_ignores_unknown_keys(self):
        """Unknown keys in the dict are silently ignored."""
        d = {"text": "hello", "unknown_field": "should be ignored"}
        state = PipelineState.from_dict(d)
        assert state.text == "hello"
        assert not hasattr(state, "unknown_field") or state.text == "hello"

    def test_from_dict_with_empty_dict(self):
        """An empty dict produces a state with all defaults."""
        state = PipelineState.from_dict({})
        assert state.text == ""
        assert state.mode == "sync"
        assert state.stage_history == []
