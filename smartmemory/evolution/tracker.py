"""
Evolution tracker for monitoring procedure changes.

Provides a high-level API for tracking procedure evolution events,
including creation, updates, and refinements.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from smartmemory.evolution.diff_engine import ProcedureDiffEngine
from smartmemory.evolution.models import (
    ContentSnapshot,
    EventDiff,
    EventSource,
    EvolutionEvent,
    MatchStatsSnapshot,
)
from smartmemory.evolution.store import EvolutionEventStore


def _utc_now() -> datetime:
    """Return current UTC time (timezone-aware)."""
    return datetime.now(timezone.utc)


logger = logging.getLogger(__name__)


class EvolutionTracker:
    """Tracks procedure evolution events.

    Provides methods for tracking creation, update, and refinement events,
    computing diffs, and retrieving evolution history.
    """

    def __init__(self, store: EvolutionEventStore):
        """Initialize the tracker.

        Args:
            store: The evolution event store for persistence
        """
        self.store = store
        self.diff_engine = ProcedureDiffEngine()

    def track_creation(
        self,
        procedure_id: str,
        content: str,
        metadata: Dict[str, Any],
        source: Dict[str, Any],
        workspace_id: str,
        user_id: str = "",
        confidence: float = 0.0,
        match_stats: Optional[Dict[str, Any]] = None,
    ) -> EvolutionEvent:
        """Track a procedure creation event.

        Args:
            procedure_id: The ID of the newly created procedure
            content: The procedure content
            metadata: Procedure metadata (name, description, skills, tools, steps)
            source: Source information (type, source_items, pattern_count)
            workspace_id: The workspace ID
            user_id: The user ID (optional)
            confidence: Initial confidence score
            match_stats: Initial match statistics

        Returns:
            The created EvolutionEvent
        """
        snapshot = ContentSnapshot(
            content=content,
            name=metadata.get("name", ""),
            description=metadata.get("description", ""),
            skills=metadata.get("skills", []),
            tools=metadata.get("tools", []),
            steps=metadata.get("steps", []),
        )

        event_source = EventSource(
            type=source.get("type", "working_memory"),
            source_items=source.get("source_items", []),
            pattern_count=source.get("pattern_count", 0),
        )

        stats = MatchStatsSnapshot(
            total_matches=match_stats.get("total_matches", 0) if match_stats else 0,
            success_rate=match_stats.get("success_rate", 0.0) if match_stats else 0.0,
        )

        event = EvolutionEvent(
            procedure_id=procedure_id,
            workspace_id=workspace_id,
            user_id=user_id,
            event_type="created",
            timestamp=_utc_now(),
            version=1,
            content_snapshot=snapshot,
            source=event_source,
            confidence_at_event=confidence,
            match_stats_at_event=stats,
            changes_from_previous=EventDiff(
                has_changes=True,
                summary=f"Procedure created from {event_source.type}",
                diff={},
            ),
        )

        self.store.save(event)
        logger.info(
            "Tracked creation event for procedure %s (version 1)",
            procedure_id,
        )

        return event

    def track_update(
        self,
        procedure_id: str,
        new_content: str,
        new_metadata: Dict[str, Any],
        source: Dict[str, Any],
        workspace_id: str,
        user_id: str = "",
        confidence: float = 0.0,
        match_stats: Optional[Dict[str, Any]] = None,
        event_type: str = "updated",
    ) -> EvolutionEvent:
        """Track a procedure update event.

        Args:
            procedure_id: The procedure ID
            new_content: The new procedure content
            new_metadata: New metadata
            source: Source information
            workspace_id: The workspace ID
            user_id: The user ID (optional)
            confidence: Confidence score at this update
            match_stats: Match statistics at this update
            event_type: Type of update (updated, refined, merged)

        Returns:
            The created EvolutionEvent
        """
        # Get the previous event to compute diff
        previous_event = self.store.get_previous_event(procedure_id, workspace_id)

        new_snapshot = ContentSnapshot(
            content=new_content,
            name=new_metadata.get("name", ""),
            description=new_metadata.get("description", ""),
            skills=new_metadata.get("skills", []),
            tools=new_metadata.get("tools", []),
            steps=new_metadata.get("steps", []),
        )

        # Compute diff from previous version
        if previous_event:
            diff = self.diff_engine.compute_diff(
                previous_event.content_snapshot,
                new_snapshot,
            )
            new_version = previous_event.version + 1
        else:
            # First event - should not happen but handle gracefully
            diff = EventDiff(
                has_changes=True,
                summary="Initial version",
                diff={},
            )
            new_version = 1

        event_source = EventSource(
            type=source.get("type", "working_memory"),
            source_items=source.get("source_items", []),
            pattern_count=source.get("pattern_count", 0),
        )

        stats = MatchStatsSnapshot(
            total_matches=match_stats.get("total_matches", 0) if match_stats else 0,
            success_rate=match_stats.get("success_rate", 0.0) if match_stats else 0.0,
        )

        event = EvolutionEvent(
            procedure_id=procedure_id,
            workspace_id=workspace_id,
            user_id=user_id,
            event_type=event_type,
            timestamp=_utc_now(),
            version=new_version,
            content_snapshot=new_snapshot,
            source=event_source,
            confidence_at_event=confidence,
            match_stats_at_event=stats,
            changes_from_previous=diff,
        )

        self.store.save(event)
        logger.info(
            "Tracked %s event for procedure %s (version %d)",
            event_type,
            procedure_id,
            new_version,
        )

        return event

    def track_refinement(
        self,
        procedure_id: str,
        new_content: str,
        new_metadata: Dict[str, Any],
        source: Dict[str, Any],
        workspace_id: str,
        user_id: str = "",
        confidence: float = 0.0,
        match_stats: Optional[Dict[str, Any]] = None,
    ) -> EvolutionEvent:
        """Track a procedure refinement event.

        Refinement is a special type of update that occurs when the evolver
        improves an existing procedure based on new pattern observations.

        Args:
            procedure_id: The procedure ID
            new_content: The refined procedure content
            new_metadata: New metadata
            source: Source information
            workspace_id: The workspace ID
            user_id: The user ID (optional)
            confidence: Confidence score
            match_stats: Match statistics

        Returns:
            The created EvolutionEvent
        """
        return self.track_update(
            procedure_id=procedure_id,
            new_content=new_content,
            new_metadata=new_metadata,
            source=source,
            workspace_id=workspace_id,
            user_id=user_id,
            confidence=confidence,
            match_stats=match_stats,
            event_type="refined",
        )

    def get_history(
        self,
        procedure_id: str,
        workspace_id: str,
        limit: int = 20,
        offset: int = 0,
    ) -> List[EvolutionEvent]:
        """Get evolution history for a procedure.

        Args:
            procedure_id: The procedure ID
            workspace_id: The workspace ID
            limit: Maximum events to return
            offset: Number of events to skip

        Returns:
            List of EvolutionEvents, newest first
        """
        return self.store.get_history(
            procedure_id=procedure_id,
            workspace_id=workspace_id,
            limit=limit,
            offset=offset,
        )

    def get_event(
        self,
        procedure_id: str,
        event_id: str,
        workspace_id: str,
    ) -> Optional[EvolutionEvent]:
        """Get a specific evolution event.

        Args:
            procedure_id: The procedure ID
            event_id: The event ID
            workspace_id: The workspace ID

        Returns:
            The EvolutionEvent or None if not found
        """
        return self.store.get_by_procedure_and_event(
            procedure_id=procedure_id,
            event_id=event_id,
            workspace_id=workspace_id,
        )

    def get_confidence_trajectory(
        self,
        procedure_id: str,
        workspace_id: str,
    ) -> List[Dict[str, Any]]:
        """Get confidence trajectory for charting.

        Args:
            procedure_id: The procedure ID
            workspace_id: The workspace ID

        Returns:
            List of data points for charting
        """
        return self.store.get_confidence_trajectory(
            procedure_id=procedure_id,
            workspace_id=workspace_id,
        )

    def get_current_version(
        self,
        procedure_id: str,
        workspace_id: str,
    ) -> int:
        """Get the current version number for a procedure.

        Args:
            procedure_id: The procedure ID
            workspace_id: The workspace ID

        Returns:
            The current version number, or 0 if no events exist
        """
        return self.store.get_latest_version(
            procedure_id=procedure_id,
            workspace_id=workspace_id,
        )

    def get_event_count(
        self,
        procedure_id: str,
        workspace_id: str,
    ) -> int:
        """Get total count of evolution events.

        Args:
            procedure_id: The procedure ID
            workspace_id: The workspace ID

        Returns:
            Total event count
        """
        return self.store.count_events(
            procedure_id=procedure_id,
            workspace_id=workspace_id,
        )
