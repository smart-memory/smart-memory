"""Unit tests for EvolutionTracker."""

import pytest
from unittest.mock import MagicMock

from smartmemory.evolution.tracker import EvolutionTracker
from smartmemory.evolution.store import EvolutionEventStore
from smartmemory.evolution.models import (
    ContentSnapshot,
    EvolutionEvent,
)


class TestEvolutionTracker:
    """Tests for EvolutionTracker."""

    @pytest.fixture
    def mock_store(self):
        """Create a mock EvolutionEventStore."""
        store = MagicMock(spec=EvolutionEventStore)
        return store

    @pytest.fixture
    def tracker(self, mock_store):
        """Create an EvolutionTracker with mock store."""
        return EvolutionTracker(store=mock_store)

    def test_track_creation(self, tracker, mock_store):
        """Test tracking a procedure creation event."""
        mock_store.save.return_value = "evt-123"

        event = tracker.track_creation(
            procedure_id="proc-456",
            content="Test procedure content",
            metadata={
                "name": "Test Procedure",
                "description": "A test procedure",
                "skills": ["skill1", "skill2"],
                "tools": ["tool1"],
                "steps": ["step1", "step2"],
            },
            source={
                "type": "working_memory",
                "source_items": ["item1", "item2"],
                "pattern_count": 5,
            },
            workspace_id="ws-789",
            user_id="user-abc",
            confidence=0.85,
            match_stats={"total_matches": 5, "success_rate": 0.8},
        )

        # Verify store was called
        mock_store.save.assert_called_once()
        saved_event = mock_store.save.call_args[0][0]

        # Verify event properties
        assert saved_event.procedure_id == "proc-456"
        assert saved_event.workspace_id == "ws-789"
        assert saved_event.user_id == "user-abc"
        assert saved_event.event_type == "created"
        assert saved_event.version == 1
        assert saved_event.confidence_at_event == 0.85

        # Verify content snapshot
        assert saved_event.content_snapshot.content == "Test procedure content"
        assert saved_event.content_snapshot.name == "Test Procedure"
        assert saved_event.content_snapshot.skills == ["skill1", "skill2"]

        # Verify source
        assert saved_event.source.type == "working_memory"
        assert saved_event.source.pattern_count == 5

        # Verify match stats
        assert saved_event.match_stats_at_event.total_matches == 5
        assert saved_event.match_stats_at_event.success_rate == 0.8

    def test_track_creation_minimal(self, tracker, mock_store):
        """Test tracking creation with minimal metadata."""
        mock_store.save.return_value = "evt-123"

        event = tracker.track_creation(
            procedure_id="proc-456",
            content="Simple content",
            metadata={},
            source={},
            workspace_id="ws-789",
        )

        mock_store.save.assert_called_once()
        saved_event = mock_store.save.call_args[0][0]

        assert saved_event.procedure_id == "proc-456"
        assert saved_event.event_type == "created"
        assert saved_event.version == 1
        assert saved_event.content_snapshot.content == "Simple content"
        assert saved_event.source.type == "working_memory"  # Default

    def test_track_update_with_previous_event(self, tracker, mock_store):
        """Test tracking an update when there's a previous event."""
        # Set up previous event
        previous_event = EvolutionEvent(
            event_id="evt-100",
            procedure_id="proc-456",
            workspace_id="ws-789",
            version=2,
            content_snapshot=ContentSnapshot(
                content="Old content",
                name="Old Name",
                skills=["skill1"],
            ),
        )
        mock_store.get_previous_event.return_value = previous_event
        mock_store.save.return_value = "evt-200"

        event = tracker.track_update(
            procedure_id="proc-456",
            new_content="New content",
            new_metadata={
                "name": "New Name",
                "skills": ["skill1", "skill2"],
            },
            source={"type": "manual"},
            workspace_id="ws-789",
            confidence=0.9,
        )

        mock_store.save.assert_called_once()
        saved_event = mock_store.save.call_args[0][0]

        # Should be version 3 (previous was 2)
        assert saved_event.version == 3
        assert saved_event.event_type == "updated"
        assert saved_event.content_snapshot.content == "New content"
        assert saved_event.content_snapshot.name == "New Name"

        # Should have computed diff
        assert saved_event.changes_from_previous.has_changes is True

    def test_track_update_no_previous_event(self, tracker, mock_store):
        """Test tracking update when no previous event exists."""
        mock_store.get_previous_event.return_value = None
        mock_store.save.return_value = "evt-123"

        event = tracker.track_update(
            procedure_id="proc-456",
            new_content="Content",
            new_metadata={},
            source={},
            workspace_id="ws-789",
        )

        mock_store.save.assert_called_once()
        saved_event = mock_store.save.call_args[0][0]

        # Should be version 1 since no previous
        assert saved_event.version == 1

    def test_track_refinement(self, tracker, mock_store):
        """Test tracking a refinement event."""
        mock_store.get_previous_event.return_value = EvolutionEvent(version=1)
        mock_store.save.return_value = "evt-123"

        event = tracker.track_refinement(
            procedure_id="proc-456",
            new_content="Refined content",
            new_metadata={"name": "Refined Procedure"},
            source={"type": "working_memory", "pattern_count": 10},
            workspace_id="ws-789",
        )

        mock_store.save.assert_called_once()
        saved_event = mock_store.save.call_args[0][0]

        # Refinement should use "refined" event type
        assert saved_event.event_type == "refined"
        assert saved_event.version == 2

    def test_get_history(self, tracker, mock_store):
        """Test getting evolution history."""
        expected_events = [
            EvolutionEvent(event_id="evt-3", version=3),
            EvolutionEvent(event_id="evt-2", version=2),
            EvolutionEvent(event_id="evt-1", version=1),
        ]
        mock_store.get_history.return_value = expected_events

        events = tracker.get_history(
            procedure_id="proc-456",
            workspace_id="ws-789",
            limit=20,
            offset=0,
        )

        mock_store.get_history.assert_called_once_with(
            procedure_id="proc-456",
            workspace_id="ws-789",
            limit=20,
            offset=0,
        )
        assert len(events) == 3
        assert events[0].event_id == "evt-3"

    def test_get_event(self, tracker, mock_store):
        """Test getting a specific event."""
        expected_event = EvolutionEvent(
            event_id="evt-123",
            procedure_id="proc-456",
        )
        mock_store.get_by_procedure_and_event.return_value = expected_event

        event = tracker.get_event(
            procedure_id="proc-456",
            event_id="evt-123",
            workspace_id="ws-789",
        )

        mock_store.get_by_procedure_and_event.assert_called_once_with(
            procedure_id="proc-456",
            event_id="evt-123",
            workspace_id="ws-789",
        )
        assert event.event_id == "evt-123"

    def test_get_event_not_found(self, tracker, mock_store):
        """Test getting a non-existent event."""
        mock_store.get_by_procedure_and_event.return_value = None

        event = tracker.get_event(
            procedure_id="proc-456",
            event_id="evt-999",
            workspace_id="ws-789",
        )

        assert event is None

    def test_get_confidence_trajectory(self, tracker, mock_store):
        """Test getting confidence trajectory data."""
        expected_trajectory = [
            {"timestamp": "2026-01-15T10:00:00", "confidence": 0.7, "matches": 5, "success_rate": 0.8},
            {"timestamp": "2026-01-16T10:00:00", "confidence": 0.8, "matches": 10, "success_rate": 0.85},
            {"timestamp": "2026-01-17T10:00:00", "confidence": 0.9, "matches": 20, "success_rate": 0.9},
        ]
        mock_store.get_confidence_trajectory.return_value = expected_trajectory

        trajectory = tracker.get_confidence_trajectory(
            procedure_id="proc-456",
            workspace_id="ws-789",
        )

        mock_store.get_confidence_trajectory.assert_called_once_with(
            procedure_id="proc-456",
            workspace_id="ws-789",
        )
        assert len(trajectory) == 3
        assert trajectory[0]["confidence"] == 0.7
        assert trajectory[2]["confidence"] == 0.9

    def test_get_current_version(self, tracker, mock_store):
        """Test getting current version number."""
        mock_store.get_latest_version.return_value = 5

        version = tracker.get_current_version(
            procedure_id="proc-456",
            workspace_id="ws-789",
        )

        assert version == 5

    def test_get_current_version_no_events(self, tracker, mock_store):
        """Test getting current version when no events exist."""
        mock_store.get_latest_version.return_value = 0

        version = tracker.get_current_version(
            procedure_id="proc-456",
            workspace_id="ws-789",
        )

        assert version == 0

    def test_get_event_count(self, tracker, mock_store):
        """Test getting total event count."""
        mock_store.count_events.return_value = 10

        count = tracker.get_event_count(
            procedure_id="proc-456",
            workspace_id="ws-789",
        )

        assert count == 10


class TestEvolutionTrackerDiffComputation:
    """Tests for diff computation during updates."""

    @pytest.fixture
    def mock_store(self):
        store = MagicMock(spec=EvolutionEventStore)
        return store

    @pytest.fixture
    def tracker(self, mock_store):
        return EvolutionTracker(store=mock_store)

    def test_diff_computed_on_update(self, tracker, mock_store):
        """Test that diff is properly computed when updating."""
        previous_event = EvolutionEvent(
            version=1,
            content_snapshot=ContentSnapshot(
                content="Original procedure",
                skills=["skill1"],
            ),
        )
        mock_store.get_previous_event.return_value = previous_event

        tracker.track_update(
            procedure_id="proc-456",
            new_content="Modified procedure",
            new_metadata={"skills": ["skill1", "skill2"]},
            source={},
            workspace_id="ws-789",
        )

        saved_event = mock_store.save.call_args[0][0]

        # Diff should show content changed and skill added
        assert saved_event.changes_from_previous.has_changes is True
        assert "content" in saved_event.changes_from_previous.summary.lower()
        # Should have skill change in summary
        diff = saved_event.changes_from_previous
        assert "+1 skills" in diff.summary or "skill" in diff.summary.lower()
