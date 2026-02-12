"""
SchemaSnapshot Model for CFS-4: Self-Healing Procedures

A SchemaSnapshot captures the JSON schemas of tools referenced by a procedure
at a specific point in time. Snapshots are compared to detect schema drift —
changes in tool signatures that may break procedure reliability.

Each snapshot is linked to a procedure and workspace for tenant isolation,
and includes a SHA-256 hash for fast equality comparison.
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from smartmemory.models.base import MemoryBaseModel


@dataclass
class SchemaSnapshot(MemoryBaseModel):
    """A point-in-time capture of tool schemas referenced by a procedure.

    Attributes:
        snapshot_id: Unique identifier for this snapshot.
        procedure_id: Links to the FalkorDB procedure node.
        workspace_id: Tenant isolation scope.
        captured_at: When this snapshot was taken.
        source_type: How schemas were obtained ("mcp", "static", "evolution", "manual").
        schemas: Mapping of tool_name to its JSON Schema definition.
        schema_hash: SHA-256 digest for fast equality checking.
        version_tag: Optional human-readable version label.
    """

    snapshot_id: str = ""
    procedure_id: str = ""
    workspace_id: str = ""
    captured_at: Optional[datetime] = field(default_factory=lambda: datetime.now(timezone.utc))
    source_type: str = "static"
    schemas: dict[str, dict] = field(default_factory=dict)
    schema_hash: str = ""
    version_tag: Optional[str] = None

    def compute_hash(self) -> str:
        """Compute SHA-256 hash of the schemas dict for fast equality comparison.

        Uses deterministic serialization (sorted keys, compact separators) so
        identical schemas always produce the same hash regardless of insertion order.

        Returns:
            Hex-encoded SHA-256 digest string.
        """
        canonical = json.dumps(self.schemas, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SchemaSnapshot":
        """Deserialize from a plain dict (e.g. MongoDB document).

        Handles ISO 8601 string parsing for datetime fields.
        """
        captured_at = d.get("captured_at")
        if isinstance(captured_at, str):
            captured_at = datetime.fromisoformat(captured_at)
        elif captured_at is None:
            captured_at = datetime.now(timezone.utc)

        return cls(
            snapshot_id=d.get("snapshot_id", ""),
            procedure_id=d.get("procedure_id", ""),
            workspace_id=d.get("workspace_id", ""),
            captured_at=captured_at,
            source_type=d.get("source_type", "static"),
            schemas=d.get("schemas", {}),
            schema_hash=d.get("schema_hash", ""),
            version_tag=d.get("version_tag"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize all fields to a plain dict suitable for storage.

        Datetime fields are converted to ISO 8601 strings.

        Returns:
            Dictionary representation of the snapshot.
        """
        return {
            "snapshot_id": self.snapshot_id,
            "procedure_id": self.procedure_id,
            "workspace_id": self.workspace_id,
            "captured_at": self.captured_at.isoformat() if self.captured_at else None,
            "source_type": self.source_type,
            "schemas": self.schemas,
            "schema_hash": self.schema_hash,
            "version_tag": self.version_tag,
        }
