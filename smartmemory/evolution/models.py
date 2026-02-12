"""
Evolution timeline data models for procedure tracking.

Provides dataclasses for tracking how procedures evolve over time:
- EvolutionEvent: A single evolution event (creation, update, refinement)
- ContentSnapshot: Snapshot of procedure content at a point in time
- EventDiff: Diff between two versions of a procedure
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List
import uuid

from smartmemory.models.base import MemoryBaseModel


@dataclass
class ContentSnapshot(MemoryBaseModel):
    """Snapshot of procedure content at a specific version.

    Captures the full state of a procedure at a point in time,
    enabling diff computation between versions.
    """

    content: str = ""
    name: str = ""
    description: str = ""
    skills: List[str] = field(default_factory=list)
    tools: List[str] = field(default_factory=list)
    steps: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for MongoDB storage."""
        return {
            "content": self.content,
            "name": self.name,
            "description": self.description,
            "skills": self.skills,
            "tools": self.tools,
            "steps": self.steps,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContentSnapshot":
        """Create from dictionary."""
        return cls(
            content=data.get("content", ""),
            name=data.get("name", ""),
            description=data.get("description", ""),
            skills=data.get("skills", []),
            tools=data.get("tools", []),
            steps=data.get("steps", []),
        )


@dataclass
class EventDiff(MemoryBaseModel):
    """Diff between two procedure versions.

    Captures what changed between versions for display in the timeline.
    """

    has_changes: bool = False
    summary: str = ""
    diff: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for MongoDB storage."""
        return {
            "has_changes": self.has_changes,
            "summary": self.summary,
            "diff": self.diff,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EventDiff":
        """Create from dictionary."""
        return cls(
            has_changes=data.get("has_changes", False),
            summary=data.get("summary", ""),
            diff=data.get("diff", {}),
        )


@dataclass
class EventSource(MemoryBaseModel):
    """Source information for an evolution event.

    Tracks where the procedure or update came from.
    """

    type: str = "working_memory"  # working_memory, manual, import, merge
    source_items: List[str] = field(default_factory=list)
    pattern_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for MongoDB storage."""
        return {
            "type": self.type,
            "source_items": self.source_items,
            "pattern_count": self.pattern_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EventSource":
        """Create from dictionary."""
        return cls(
            type=data.get("type", "working_memory"),
            source_items=data.get("source_items", []),
            pattern_count=data.get("pattern_count", 0),
        )


@dataclass
class MatchStatsSnapshot(MemoryBaseModel):
    """Snapshot of match statistics at a point in time."""

    total_matches: int = 0
    success_rate: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for MongoDB storage."""
        return {
            "total_matches": self.total_matches,
            "success_rate": self.success_rate,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MatchStatsSnapshot":
        """Create from dictionary."""
        return cls(
            total_matches=data.get("total_matches", 0),
            success_rate=data.get("success_rate", 0.0),
        )


@dataclass
class EvolutionEvent(MemoryBaseModel):
    """A single evolution event in a procedure's history.

    Tracks one change to a procedure, including the full content snapshot,
    source information, and diff from the previous version.
    """

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    procedure_id: str = ""
    workspace_id: str = ""
    user_id: str = ""
    event_type: str = "created"  # created, updated, refined, merged, superseded
    timestamp: datetime = field(default_factory=datetime.utcnow)
    version: int = 1

    content_snapshot: ContentSnapshot = field(default_factory=ContentSnapshot)
    source: EventSource = field(default_factory=EventSource)
    confidence_at_event: float = 0.0
    match_stats_at_event: MatchStatsSnapshot = field(default_factory=MatchStatsSnapshot)
    changes_from_previous: EventDiff = field(default_factory=EventDiff)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for MongoDB storage."""
        return {
            "event_id": self.event_id,
            "procedure_id": self.procedure_id,
            "workspace_id": self.workspace_id,
            "user_id": self.user_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            "version": self.version,
            "content_snapshot": self.content_snapshot.to_dict(),
            "source": self.source.to_dict(),
            "confidence_at_event": self.confidence_at_event,
            "match_stats_at_event": self.match_stats_at_event.to_dict(),
            "changes_from_previous": self.changes_from_previous.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvolutionEvent":
        """Create from dictionary (MongoDB document)."""
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        elif timestamp is None:
            timestamp = datetime.utcnow()

        return cls(
            event_id=data.get("event_id", str(uuid.uuid4())),
            procedure_id=data.get("procedure_id", ""),
            workspace_id=data.get("workspace_id", ""),
            user_id=data.get("user_id", ""),
            event_type=data.get("event_type", "created"),
            timestamp=timestamp,
            version=data.get("version", 1),
            content_snapshot=ContentSnapshot.from_dict(data.get("content_snapshot", {})),
            source=EventSource.from_dict(data.get("source", {})),
            confidence_at_event=data.get("confidence_at_event", 0.0),
            match_stats_at_event=MatchStatsSnapshot.from_dict(data.get("match_stats_at_event", {})),
            changes_from_previous=EventDiff.from_dict(data.get("changes_from_previous", {})),
        )

    def to_api_response(self) -> Dict[str, Any]:
        """Convert to API response format (for list endpoints)."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "version": self.version,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "source_type": self.source.type,
            "pattern_count": self.source.pattern_count,
            "confidence_at_event": self.confidence_at_event,
            "summary": self.changes_from_previous.summary or self._generate_summary(),
        }

    def to_api_detail_response(self) -> Dict[str, Any]:
        """Convert to API detail response format (for single event endpoints)."""
        return {
            **self.to_api_response(),
            "content_snapshot": self.content_snapshot.to_dict(),
            "source": self.source.to_dict(),
            "match_stats_at_event": self.match_stats_at_event.to_dict(),
            "changes_from_previous": self.changes_from_previous.to_dict(),
        }

    def _generate_summary(self) -> str:
        """Generate a human-readable summary for the event."""
        if self.event_type == "created":
            if self.source.type == "working_memory":
                return f"Promoted from working memory after {self.source.pattern_count} pattern observations"
            return "Procedure created"
        elif self.event_type == "refined":
            return "Procedure refined based on new pattern observations"
        elif self.event_type == "updated":
            return "Procedure content updated"
        elif self.event_type == "merged":
            return "Merged with similar procedure"
        elif self.event_type == "superseded":
            return "Superseded by newer procedure"
        return "Unknown event"
