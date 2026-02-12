"""
DriftEvent Model for CFS-4: Self-Healing Procedures

A DriftEvent records a detected schema drift between two SchemaSnapshots.
It captures the diff summary, individual changes, whether the drift is breaking
or non-breaking, and what action was taken in response.

DriftEvents support resolution tracking so operators can acknowledge and
close out detected drift.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from smartmemory.models.base import MemoryBaseModel


@dataclass
class DriftEvent(MemoryBaseModel):
    """A record of detected schema drift for a procedure.

    Attributes:
        event_id: Unique identifier for this drift event.
        procedure_id: The procedure whose schemas drifted.
        workspace_id: Tenant isolation scope.
        snapshot_id: The new snapshot that triggered drift detection.
        detected_at: When the drift was detected.
        diff_summary: Human-readable summary of what changed.
        breaking_count: Number of breaking changes detected.
        non_breaking_count: Number of non-breaking changes detected.
        changes: List of individual change dicts (from SchemaChange.to_dict()).
        action_taken: Response action ("confidence_reduced", "flagged", "none").
        resolved: Whether this drift event has been resolved.
        resolved_at: When resolution occurred.
        resolved_by: Who or what resolved it (user ID, "auto", etc.).
        resolution_note: Explanation of the resolution.
    """

    event_id: str = ""
    procedure_id: str = ""
    workspace_id: str = ""
    snapshot_id: str = ""
    detected_at: Optional[datetime] = field(default_factory=lambda: datetime.now(timezone.utc))
    diff_summary: str = ""
    breaking_count: int = 0
    non_breaking_count: int = 0
    changes: list[dict[str, Any]] = field(default_factory=list)
    action_taken: str = "none"
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None
    resolution_note: Optional[str] = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DriftEvent":
        """Deserialize from a plain dict (e.g. MongoDB document).

        Handles ISO 8601 string parsing for datetime fields.
        """
        detected_at = d.get("detected_at")
        if isinstance(detected_at, str):
            detected_at = datetime.fromisoformat(detected_at)
        elif detected_at is None:
            detected_at = datetime.now(timezone.utc)

        resolved_at = d.get("resolved_at")
        if isinstance(resolved_at, str):
            resolved_at = datetime.fromisoformat(resolved_at)

        return cls(
            event_id=d.get("event_id", ""),
            procedure_id=d.get("procedure_id", ""),
            workspace_id=d.get("workspace_id", ""),
            snapshot_id=d.get("snapshot_id", ""),
            detected_at=detected_at,
            diff_summary=d.get("diff_summary", ""),
            breaking_count=d.get("breaking_count", 0),
            non_breaking_count=d.get("non_breaking_count", 0),
            changes=d.get("changes", []),
            action_taken=d.get("action_taken", "none"),
            resolved=d.get("resolved", False),
            resolved_at=resolved_at,
            resolved_by=d.get("resolved_by"),
            resolution_note=d.get("resolution_note"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize all fields to a plain dict suitable for storage.

        Datetime fields are converted to ISO 8601 strings.

        Returns:
            Dictionary representation of the drift event.
        """
        return {
            "event_id": self.event_id,
            "procedure_id": self.procedure_id,
            "workspace_id": self.workspace_id,
            "snapshot_id": self.snapshot_id,
            "detected_at": self.detected_at.isoformat() if self.detected_at else None,
            "diff_summary": self.diff_summary,
            "breaking_count": self.breaking_count,
            "non_breaking_count": self.non_breaking_count,
            "changes": self.changes,
            "action_taken": self.action_taken,
            "resolved": self.resolved,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "resolved_by": self.resolved_by,
            "resolution_note": self.resolution_note,
        }
