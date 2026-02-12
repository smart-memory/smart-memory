"""Unit tests for evolution models."""

from datetime import datetime

from smartmemory.evolution.models import (
    ContentSnapshot,
    EventDiff,
    EventSource,
    EvolutionEvent,
    MatchStatsSnapshot,
)


class TestContentSnapshot:
    """Tests for ContentSnapshot model."""

    def test_default_values(self):
        """Test ContentSnapshot has sensible defaults."""
        snapshot = ContentSnapshot()
        assert snapshot.content == ""
        assert snapshot.name == ""
        assert snapshot.description == ""
        assert snapshot.skills == []
        assert snapshot.tools == []
        assert snapshot.steps == []

    def test_with_values(self):
        """Test ContentSnapshot with provided values."""
        snapshot = ContentSnapshot(
            content="Test content",
            name="Test Procedure",
            description="A test procedure",
            skills=["skill1", "skill2"],
            tools=["tool1"],
            steps=["step1", "step2", "step3"],
        )
        assert snapshot.content == "Test content"
        assert snapshot.name == "Test Procedure"
        assert snapshot.skills == ["skill1", "skill2"]
        assert len(snapshot.steps) == 3

    def test_to_dict(self):
        """Test ContentSnapshot serialization."""
        snapshot = ContentSnapshot(
            content="Test",
            name="Name",
            skills=["s1"],
        )
        d = snapshot.to_dict()
        assert d["content"] == "Test"
        assert d["name"] == "Name"
        assert d["skills"] == ["s1"]
        assert d["tools"] == []

    def test_from_dict(self):
        """Test ContentSnapshot deserialization."""
        data = {
            "content": "Test",
            "name": "Name",
            "description": "Desc",
            "skills": ["s1", "s2"],
            "tools": [],
            "steps": ["step1"],
        }
        snapshot = ContentSnapshot.from_dict(data)
        assert snapshot.content == "Test"
        assert snapshot.name == "Name"
        assert snapshot.skills == ["s1", "s2"]
        assert snapshot.steps == ["step1"]

    def test_from_dict_with_missing_fields(self):
        """Test ContentSnapshot handles missing fields gracefully."""
        snapshot = ContentSnapshot.from_dict({})
        assert snapshot.content == ""
        assert snapshot.skills == []


class TestEventDiff:
    """Tests for EventDiff model."""

    def test_default_values(self):
        """Test EventDiff has sensible defaults."""
        diff = EventDiff()
        assert diff.has_changes is False
        assert diff.summary == ""
        assert diff.diff == {}

    def test_with_changes(self):
        """Test EventDiff with changes."""
        diff = EventDiff(
            has_changes=True,
            summary="content modified; +2 skills",
            diff={
                "content": {"old": "old text", "new": "new text"},
                "skills": {"added": ["skill1", "skill2"], "removed": []},
            },
        )
        assert diff.has_changes is True
        assert "content modified" in diff.summary
        assert "skills" in diff.diff

    def test_round_trip(self):
        """Test EventDiff serialization round trip."""
        original = EventDiff(
            has_changes=True,
            summary="test",
            diff={"key": "value"},
        )
        d = original.to_dict()
        restored = EventDiff.from_dict(d)
        assert restored.has_changes == original.has_changes
        assert restored.summary == original.summary
        assert restored.diff == original.diff


class TestEventSource:
    """Tests for EventSource model."""

    def test_default_values(self):
        """Test EventSource defaults to working_memory."""
        source = EventSource()
        assert source.type == "working_memory"
        assert source.source_items == []
        assert source.pattern_count == 0

    def test_with_values(self):
        """Test EventSource with values."""
        source = EventSource(
            type="manual",
            source_items=["item1", "item2"],
            pattern_count=5,
        )
        assert source.type == "manual"
        assert len(source.source_items) == 2
        assert source.pattern_count == 5

    def test_round_trip(self):
        """Test EventSource serialization round trip."""
        original = EventSource(
            type="merge",
            source_items=["a", "b"],
            pattern_count=10,
        )
        d = original.to_dict()
        restored = EventSource.from_dict(d)
        assert restored.type == original.type
        assert restored.source_items == original.source_items
        assert restored.pattern_count == original.pattern_count


class TestMatchStatsSnapshot:
    """Tests for MatchStatsSnapshot model."""

    def test_default_values(self):
        """Test MatchStatsSnapshot defaults."""
        stats = MatchStatsSnapshot()
        assert stats.total_matches == 0
        assert stats.success_rate == 0.0

    def test_with_values(self):
        """Test MatchStatsSnapshot with values."""
        stats = MatchStatsSnapshot(
            total_matches=100,
            success_rate=0.85,
        )
        assert stats.total_matches == 100
        assert stats.success_rate == 0.85

    def test_round_trip(self):
        """Test MatchStatsSnapshot serialization round trip."""
        original = MatchStatsSnapshot(total_matches=50, success_rate=0.9)
        d = original.to_dict()
        restored = MatchStatsSnapshot.from_dict(d)
        assert restored.total_matches == original.total_matches
        assert restored.success_rate == original.success_rate


class TestEvolutionEvent:
    """Tests for EvolutionEvent model."""

    def test_default_values(self):
        """Test EvolutionEvent has sensible defaults."""
        event = EvolutionEvent()
        assert event.event_id  # Should have auto-generated UUID
        assert event.procedure_id == ""
        assert event.workspace_id == ""
        assert event.event_type == "created"
        assert event.version == 1
        assert isinstance(event.timestamp, datetime)
        assert isinstance(event.content_snapshot, ContentSnapshot)
        assert isinstance(event.source, EventSource)
        assert isinstance(event.changes_from_previous, EventDiff)

    def test_with_all_fields(self):
        """Test EvolutionEvent with all fields populated."""
        now = datetime.utcnow()
        event = EvolutionEvent(
            event_id="evt-123",
            procedure_id="proc-456",
            workspace_id="ws-789",
            user_id="user-abc",
            event_type="refined",
            timestamp=now,
            version=3,
            content_snapshot=ContentSnapshot(content="Test"),
            source=EventSource(type="manual", pattern_count=5),
            confidence_at_event=0.85,
            match_stats_at_event=MatchStatsSnapshot(total_matches=10, success_rate=0.9),
            changes_from_previous=EventDiff(has_changes=True, summary="content modified"),
        )
        assert event.event_id == "evt-123"
        assert event.procedure_id == "proc-456"
        assert event.event_type == "refined"
        assert event.version == 3
        assert event.confidence_at_event == 0.85

    def test_to_dict(self):
        """Test EvolutionEvent serialization."""
        event = EvolutionEvent(
            event_id="evt-123",
            procedure_id="proc-456",
            workspace_id="ws-789",
            event_type="created",
            version=1,
        )
        d = event.to_dict()
        assert d["event_id"] == "evt-123"
        assert d["procedure_id"] == "proc-456"
        assert d["event_type"] == "created"
        assert "content_snapshot" in d
        assert "source" in d

    def test_from_dict(self):
        """Test EvolutionEvent deserialization."""
        data = {
            "event_id": "evt-123",
            "procedure_id": "proc-456",
            "workspace_id": "ws-789",
            "user_id": "user-abc",
            "event_type": "updated",
            "timestamp": "2026-01-15T10:00:00",
            "version": 2,
            "content_snapshot": {"content": "Test content", "name": "Test"},
            "source": {"type": "working_memory", "pattern_count": 5},
            "confidence_at_event": 0.78,
            "match_stats_at_event": {"total_matches": 5, "success_rate": 0.8},
            "changes_from_previous": {"has_changes": True, "summary": "updated"},
        }
        event = EvolutionEvent.from_dict(data)
        assert event.event_id == "evt-123"
        assert event.event_type == "updated"
        assert event.version == 2
        assert event.content_snapshot.content == "Test content"
        assert event.source.pattern_count == 5
        assert event.confidence_at_event == 0.78

    def test_to_api_response(self):
        """Test EvolutionEvent API response format."""
        event = EvolutionEvent(
            event_id="evt-123",
            event_type="created",
            version=1,
            source=EventSource(type="working_memory", pattern_count=5),
            confidence_at_event=0.78,
            changes_from_previous=EventDiff(has_changes=True, summary="Promoted from working memory"),
        )
        resp = event.to_api_response()
        assert resp["event_id"] == "evt-123"
        assert resp["event_type"] == "created"
        assert resp["version"] == 1
        assert resp["source_type"] == "working_memory"
        assert resp["pattern_count"] == 5
        assert resp["confidence_at_event"] == 0.78
        assert "summary" in resp

    def test_to_api_detail_response(self):
        """Test EvolutionEvent detailed API response format."""
        event = EvolutionEvent(
            event_id="evt-123",
            content_snapshot=ContentSnapshot(content="Test", name="Test Proc"),
        )
        resp = event.to_api_detail_response()
        assert resp["event_id"] == "evt-123"
        assert "content_snapshot" in resp
        assert resp["content_snapshot"]["content"] == "Test"
        assert "source" in resp
        assert "match_stats_at_event" in resp
        assert "changes_from_previous" in resp

    def test_generate_summary_created(self):
        """Test summary generation for created event."""
        event = EvolutionEvent(
            event_type="created",
            source=EventSource(type="working_memory", pattern_count=5),
        )
        summary = event._generate_summary()
        assert "Promoted from working memory" in summary
        assert "5 pattern observations" in summary

    def test_generate_summary_refined(self):
        """Test summary generation for refined event."""
        event = EvolutionEvent(event_type="refined")
        summary = event._generate_summary()
        assert "refined" in summary.lower()

    def test_generate_summary_merged(self):
        """Test summary generation for merged event."""
        event = EvolutionEvent(event_type="merged")
        summary = event._generate_summary()
        assert "Merged" in summary
