"""CFS-4: Drift Detector.

Compares stored SchemaSnapshots against current tool schemas to detect
schema drift. When drift is detected, creates DriftEvent records and
applies confidence degradation to procedure match scores.

The DriftDetector is stateless -- it receives snapshots and schemas as
arguments and returns results. Persistence and provider management
happen at the service layer.
"""

import logging
from dataclasses import dataclass
from typing import Optional
from uuid import uuid4

from smartmemory.models.drift_event import DriftEvent
from smartmemory.models.schema_snapshot import SchemaSnapshot
from smartmemory.schema_diff import diff_tool_schemas
from smartmemory.schema_providers import SchemaProvider

logger = logging.getLogger(__name__)


@dataclass
class DriftDetectorConfig:
    """Configuration for the drift detector.

    Attributes:
        enabled: Master switch for drift detection (default False, opt-in).
        confidence_multiplier: Factor applied to confidence when drift detected (0.0-1.0).
        breaking_only: If True, only report breaking changes as drift.
    """

    enabled: bool = False
    confidence_multiplier: float = 0.5
    breaking_only: bool = False


class DriftDetector:
    """Detects schema drift between stored snapshots and current tool schemas.

    Stateless engine -- receives data as arguments, returns results.
    Service layer handles persistence and provider management.

    Args:
        providers: List of SchemaProvider instances for resolving current schemas.
        config: DriftDetectorConfig controlling behavior.
    """

    def __init__(self, providers: list[SchemaProvider] | None = None, config: DriftDetectorConfig | None = None):
        # v2: providers will resolve current schemas internally (MCPSchemaProvider).
        # v1: current_schemas are passed directly to check_procedure().
        self._providers: list[SchemaProvider] = providers or []
        self._config = config or DriftDetectorConfig()

    @property
    def config(self) -> DriftDetectorConfig:
        """Access the detector configuration."""
        return self._config

    def check_procedure(self, snapshot: SchemaSnapshot, current_schemas: dict[str, dict]) -> Optional[DriftEvent]:
        """Check a single procedure's snapshot against current schemas for drift.

        Args:
            snapshot: The stored SchemaSnapshot for the procedure.
            current_schemas: Current tool_name -> JSON Schema mapping.

        Returns:
            DriftEvent if drift was detected, None otherwise.
        """
        if not self._config.enabled:
            return None

        try:
            tool_diffs = diff_tool_schemas(snapshot.schemas, current_schemas)
        except Exception as e:
            logger.debug("CFS-4: Schema diff failed: %s", e)
            return None

        # Aggregate changes across all tools
        all_changes = []
        total_breaking = 0
        total_non_breaking = 0
        summary_parts = []

        for tool_name, diff_result in tool_diffs.items():
            if not diff_result.has_changes:
                continue
            all_changes.extend([c.to_dict() for c in diff_result.changes])
            total_breaking += diff_result.breaking_count
            total_non_breaking += diff_result.non_breaking_count
            if diff_result.summary:
                summary_parts.append(f"{tool_name}: {diff_result.summary}")

        # No changes at all
        if not all_changes:
            return None

        # If breaking_only mode and no breaking changes, skip
        if self._config.breaking_only and total_breaking == 0:
            return None

        # Determine action
        if total_breaking > 0:
            action_taken = "confidence_reduced"
        else:
            action_taken = "flagged"

        event = DriftEvent(
            event_id=str(uuid4()),
            procedure_id=snapshot.procedure_id,
            workspace_id=snapshot.workspace_id,
            snapshot_id=snapshot.snapshot_id,
            diff_summary="\n".join(summary_parts),
            breaking_count=total_breaking,
            non_breaking_count=total_non_breaking,
            changes=all_changes,
            action_taken=action_taken,
        )

        return event

    def apply_confidence(self, original_confidence: float) -> float:
        """Apply confidence degradation for detected drift.

        Multiplies the original confidence by the configured multiplier.

        Args:
            original_confidence: The original procedure match confidence (0.0-1.0).

        Returns:
            Degraded confidence value.
        """
        return original_confidence * self._config.confidence_multiplier
