"""Unit tests for CFS-4 models: SchemaSnapshot and DriftEvent."""

from datetime import datetime, timezone

from smartmemory.models.schema_snapshot import SchemaSnapshot
from smartmemory.models.drift_event import DriftEvent


class TestSchemaSnapshot:
    """Tests for SchemaSnapshot serialization and hashing."""

    def test_to_dict_keys(self):
        snap = SchemaSnapshot(snapshot_id="s1", procedure_id="p1", workspace_id="ws1")
        d = snap.to_dict()
        expected_keys = {
            "snapshot_id",
            "procedure_id",
            "workspace_id",
            "captured_at",
            "source_type",
            "schemas",
            "schema_hash",
            "version_tag",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_datetime_serialization(self):
        ts = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        snap = SchemaSnapshot(snapshot_id="s1", captured_at=ts)
        d = snap.to_dict()
        assert d["captured_at"] == "2026-01-15T12:00:00+00:00"

    def test_to_dict_none_datetime(self):
        snap = SchemaSnapshot(snapshot_id="s1", captured_at=None)
        d = snap.to_dict()
        assert d["captured_at"] is None

    def test_from_dict_round_trip(self):
        snap = SchemaSnapshot(
            snapshot_id="s1",
            procedure_id="p1",
            workspace_id="ws1",
            source_type="mcp",
            schemas={"tool_a": {"type": "object"}},
            schema_hash="abc",
            version_tag="v1",
        )
        d = snap.to_dict()
        restored = SchemaSnapshot.from_dict(d)
        assert restored.snapshot_id == "s1"
        assert restored.procedure_id == "p1"
        assert restored.workspace_id == "ws1"
        assert restored.source_type == "mcp"
        assert restored.schemas == {"tool_a": {"type": "object"}}
        assert restored.schema_hash == "abc"
        assert restored.version_tag == "v1"
        assert isinstance(restored.captured_at, datetime)

    def test_from_dict_iso_string_parsing(self):
        d = {"snapshot_id": "s1", "captured_at": "2026-01-15T12:00:00+00:00"}
        snap = SchemaSnapshot.from_dict(d)
        assert snap.captured_at == datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)

    def test_from_dict_defaults(self):
        snap = SchemaSnapshot.from_dict({})
        assert snap.snapshot_id == ""
        assert snap.source_type == "static"
        assert snap.schemas == {}

    def test_compute_hash_deterministic_regardless_of_key_order(self):
        snap_a = SchemaSnapshot(schemas={"b": {"type": "string"}, "a": {"type": "integer"}})
        snap_b = SchemaSnapshot(schemas={"a": {"type": "integer"}, "b": {"type": "string"}})
        assert snap_a.compute_hash() == snap_b.compute_hash()
        assert len(snap_a.compute_hash()) == 64  # SHA-256 hex digest

    def test_compute_hash_differs_for_different_schemas(self):
        snap_a = SchemaSnapshot(schemas={"a": {"type": "string"}})
        snap_b = SchemaSnapshot(schemas={"a": {"type": "integer"}})
        assert snap_a.compute_hash() != snap_b.compute_hash()


class TestDriftEvent:
    """Tests for DriftEvent serialization."""

    def test_to_dict_keys(self):
        evt = DriftEvent(event_id="e1")
        d = evt.to_dict()
        expected_keys = {
            "event_id",
            "procedure_id",
            "workspace_id",
            "snapshot_id",
            "detected_at",
            "diff_summary",
            "breaking_count",
            "non_breaking_count",
            "changes",
            "action_taken",
            "resolved",
            "resolved_at",
            "resolved_by",
            "resolution_note",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_datetime_serialization(self):
        ts = datetime(2026, 2, 10, 8, 30, 0, tzinfo=timezone.utc)
        evt = DriftEvent(event_id="e1", detected_at=ts)
        d = evt.to_dict()
        assert d["detected_at"] == "2026-02-10T08:30:00+00:00"

    def test_to_dict_none_datetimes(self):
        evt = DriftEvent(event_id="e1", detected_at=None, resolved_at=None)
        d = evt.to_dict()
        assert d["detected_at"] is None
        assert d["resolved_at"] is None

    def test_from_dict_round_trip(self):
        evt = DriftEvent(
            event_id="e1",
            procedure_id="p1",
            workspace_id="ws1",
            snapshot_id="s1",
            diff_summary="1 breaking",
            breaking_count=1,
            non_breaking_count=0,
            changes=[{"path": "/x", "breaking": True}],
            action_taken="confidence_reduced",
            resolved=True,
            resolved_by="user-1",
            resolution_note="Fixed",
        )
        d = evt.to_dict()
        restored = DriftEvent.from_dict(d)
        assert restored.event_id == "e1"
        assert restored.procedure_id == "p1"
        assert restored.breaking_count == 1
        assert restored.action_taken == "confidence_reduced"
        assert restored.resolved is True
        assert restored.resolved_by == "user-1"
        assert isinstance(restored.detected_at, datetime)

    def test_from_dict_iso_string_parsing(self):
        d = {
            "event_id": "e1",
            "detected_at": "2026-02-10T08:30:00+00:00",
            "resolved_at": "2026-02-11T10:00:00+00:00",
        }
        evt = DriftEvent.from_dict(d)
        assert evt.detected_at == datetime(2026, 2, 10, 8, 30, 0, tzinfo=timezone.utc)
        assert evt.resolved_at == datetime(2026, 2, 11, 10, 0, 0, tzinfo=timezone.utc)

    def test_from_dict_defaults(self):
        evt = DriftEvent.from_dict({})
        assert evt.event_id == ""
        assert evt.action_taken == "none"
        assert evt.resolved is False
        assert evt.changes == []
