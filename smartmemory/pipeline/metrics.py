"""Pipeline metrics emission via Redis Streams.

Fire-and-forget metrics for pipeline stage execution. Reuses EventSpooler
infrastructure. Emits to 'smartmemory:metrics:pipeline'.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, Optional

from smartmemory.pipeline.state import PipelineState

logger = logging.getLogger(__name__)

METRICS_STREAM = "smartmemory:metrics:pipeline"


class PipelineMetricsEmitter:
    """Emit pipeline stage metrics to Redis Streams.

    Creates an EventSpooler lazily on first emission. All errors are
    swallowed — metrics must never break the pipeline.
    """

    def __init__(self, workspace_id: Optional[str] = None):
        self._spooler = None  # Lazy-init
        self._workspace_id = workspace_id
        self._pipeline_id = str(uuid.uuid4())
        self._stage_timings: Dict[str, float] = {}

    def _get_spooler(self):
        """Lazy-init the EventSpooler. Returns None if Redis is unavailable."""
        if self._spooler is not None:
            return self._spooler
        try:
            from smartmemory.observability.events import EventSpooler

            self._spooler = EventSpooler(stream_name=METRICS_STREAM)
            return self._spooler
        except Exception:
            return None

    def on_stage_complete(
        self,
        stage_name: str,
        elapsed_ms: float,
        state: PipelineState,
        error: Optional[Exception] = None,
    ) -> None:
        """Called after each stage completes. Emits stage_complete event."""
        self._stage_timings[stage_name] = elapsed_ms
        spooler = self._get_spooler()
        if spooler is None:
            return

        entity_count = len(state.entities) if state.entities else 0
        relation_count = len(state.relations) if state.relations else 0

        data: Dict[str, Any] = {
            "event_type": "stage_complete",
            "stage_name": stage_name,
            "elapsed_ms": round(elapsed_ms, 2),
            "status": "error" if error else "success",
            "workspace_id": self._workspace_id or "",
            "pipeline_id": self._pipeline_id,
            "entity_count": entity_count,
            "relation_count": relation_count,
        }
        if error:
            data["error"] = str(error)[:200]

        try:
            spooler.emit_event(
                event_type="stage_complete",
                component="pipeline",
                operation=stage_name,
                data=data,
            )
        except Exception:
            pass

    def on_pipeline_complete(self, state: PipelineState) -> None:
        """Called after all stages. Emits pipeline_complete summary."""
        spooler = self._get_spooler()
        if spooler is None:
            return

        total_ms = sum(self._stage_timings.values())
        entity_count = len(state.entities) if state.entities else 0
        relation_count = len(state.relations) if state.relations else 0

        data: Dict[str, Any] = {
            "event_type": "pipeline_complete",
            "total_ms": round(total_ms, 2),
            "stage_timings": {k: round(v, 2) for k, v in self._stage_timings.items()},
            "entity_count": entity_count,
            "relation_count": relation_count,
            "stages_completed": len(self._stage_timings),
            "workspace_id": self._workspace_id or "",
            "pipeline_id": self._pipeline_id,
        }

        try:
            spooler.emit_event(
                event_type="pipeline_complete",
                component="pipeline",
                operation="complete",
                data=data,
            )
        except Exception:
            pass
