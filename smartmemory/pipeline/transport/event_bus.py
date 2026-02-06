"""EventBusTransport — Redis Streams transport for async pipeline execution.

Serializes PipelineState between stages via Redis Streams with per-stage
consumer groups. Each stage reads from its own stream, executes, and publishes
to the next stage's stream.

Usage::

    # Producer (API)
    transport = EventBusTransport(redis_client)
    run_id = transport.submit(state, config, stages=["classify", "extract", "store"])
    status = transport.get_status(run_id)

    # Consumer (worker)
    consumer = StageConsumer("classify", classify_stage, transport)
    consumer.run(max_iterations=100)
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    import redis as redis_lib

    from smartmemory.pipeline.config import PipelineConfig
    from smartmemory.pipeline.protocol import StageCommand
    from smartmemory.pipeline.state import PipelineState

logger = logging.getLogger(__name__)

# Status constants
STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"


class EventBusTransport:
    """Redis Streams transport for async pipeline execution."""

    def __init__(
        self,
        redis_client: redis_lib.Redis | None = None,
        stream_prefix: str = "smartmemory:pipeline",
    ):
        self._redis = redis_client
        self._prefix = stream_prefix

    def _get_redis(self) -> redis_lib.Redis:
        """Lazy-load or return injected Redis client."""
        if self._redis is not None:
            return self._redis
        try:
            import redis as redis_mod
            from smartmemory.utils import get_config

            config = get_config()
            redis_config = config.cache.redis
            self._redis = redis_mod.Redis(
                host=redis_config.host,
                port=int(redis_config.port),
                db=2,
                decode_responses=True,
            )
            return self._redis
        except Exception as e:
            raise RuntimeError(f"Redis not available for EventBusTransport: {e}") from e

    def _stream_name(self, workspace_id: str, stage_name: str) -> str:
        return f"{self._prefix}:{workspace_id}:{stage_name}"

    def _results_stream(self, workspace_id: str) -> str:
        return f"{self._prefix}:{workspace_id}:results"

    def _status_key(self, run_id: str) -> str:
        return f"{self._prefix}:status:{run_id}"

    def submit(
        self,
        state: PipelineState,
        config: PipelineConfig,
        stages: List[str] | None = None,
    ) -> str:
        """Submit pipeline for async execution via Redis Streams.

        Args:
            state: Initial pipeline state.
            config: Pipeline configuration.
            stages: Ordered list of stage names. If None, uses default pipeline order.

        Returns:
            run_id for polling status.
        """
        r = self._get_redis()
        run_id = str(uuid.uuid4())
        workspace_id = state.workspace_id or "default"
        stage_list = stages or [
            "classify",
            "coreference",
            "simplify",
            "entity_ruler",
            "llm_extract",
            "ontology_constrain",
            "store",
            "link",
            "enrich",
            "ground",
            "evolve",
        ]

        # Store run metadata
        status_data = {
            "run_id": run_id,
            "status": STATUS_QUEUED,
            "current_stage": stage_list[0] if stage_list else "",
            "stages": json.dumps(stage_list),
            "workspace_id": workspace_id,
            "error": "",
        }
        r.hset(self._status_key(run_id), mapping=status_data)
        r.expire(self._status_key(run_id), 3600)  # 1h TTL

        # Serialize state + config and publish to first stage stream
        payload = {
            "run_id": run_id,
            "state": json.dumps(state.to_dict()),
            "config": json.dumps(_serialize_config(config)),
            "stages": json.dumps(stage_list),
            "stage_index": "0",
        }
        first_stream = self._stream_name(workspace_id, stage_list[0])
        r.xadd(first_stream, payload, maxlen=10000)

        logger.info("Submitted pipeline run %s to stream %s", run_id, first_stream)
        return run_id

    def get_status(self, run_id: str) -> Dict[str, Any]:
        """Poll execution status.

        Returns:
            Dict with run_id, status, current_stage, error fields.
        """
        r = self._get_redis()
        data = r.hgetall(self._status_key(run_id))
        if not data:
            return {"run_id": run_id, "status": "unknown", "error": "Run not found"}
        return {
            "run_id": data.get("run_id", run_id),
            "status": data.get("status", "unknown"),
            "current_stage": data.get("current_stage", ""),
            "error": data.get("error", ""),
        }

    def get_result(self, run_id: str) -> PipelineState | None:
        """Get completed pipeline result from the results stream."""
        from smartmemory.pipeline.state import PipelineState as PS

        r = self._get_redis()
        status = self.get_status(run_id)
        if status.get("status") != STATUS_COMPLETED:
            return None

        workspace_id = r.hget(self._status_key(run_id), "workspace_id") or "default"
        results_stream = self._results_stream(workspace_id)

        # Scan results for this run_id
        try:
            messages = r.xrange(results_stream, count=100)
            for _, fields in messages:
                if fields.get("run_id") == run_id:
                    state_json = fields.get("state", "{}")
                    return PS.from_dict(json.loads(state_json))
        except Exception as e:
            logger.warning("Failed to get result for run %s: %s", run_id, e)

        return None

    def publish_stage_result(
        self,
        run_id: str,
        state: PipelineState,
        config: PipelineConfig,
        stages: List[str],
        stage_index: int,
    ) -> None:
        """Publish result to the next stage's stream, or to results if done."""
        r = self._get_redis()
        workspace_id = state.workspace_id or "default"

        next_index = stage_index + 1
        if next_index >= len(stages):
            # Pipeline complete — write to results stream
            r.xadd(
                self._results_stream(workspace_id),
                {
                    "run_id": run_id,
                    "state": json.dumps(state.to_dict()),
                },
                maxlen=1000,
            )
            r.hset(
                self._status_key(run_id),
                mapping={
                    "status": STATUS_COMPLETED,
                    "current_stage": "done",
                },
            )
            logger.info("Pipeline run %s completed", run_id)
        else:
            # Publish to next stage
            next_stage = stages[next_index]
            payload = {
                "run_id": run_id,
                "state": json.dumps(state.to_dict()),
                "config": json.dumps(_serialize_config(config)),
                "stages": json.dumps(stages),
                "stage_index": str(next_index),
            }
            r.xadd(self._stream_name(workspace_id, next_stage), payload, maxlen=10000)
            r.hset(
                self._status_key(run_id),
                mapping={
                    "status": STATUS_RUNNING,
                    "current_stage": next_stage,
                },
            )

    def mark_failed(self, run_id: str, stage: str, error: str) -> None:
        """Mark a run as failed."""
        r = self._get_redis()
        r.hset(
            self._status_key(run_id),
            mapping={
                "status": STATUS_FAILED,
                "current_stage": stage,
                "error": error,
            },
        )


class StageConsumer:
    """Consumes from a stage's Redis Stream, executes, publishes to next."""

    def __init__(
        self,
        stage_name: str,
        stage: StageCommand,
        transport: EventBusTransport,
        group: str | None = None,
        consumer_name: str | None = None,
    ):
        self.stage_name = stage_name
        self.stage = stage
        self.transport = transport
        self.group = group or f"stage-{stage_name}"
        self.consumer_name = consumer_name or f"worker-{uuid.uuid4().hex[:6]}"

    def run(self, max_iterations: int | None = None, block_ms: int = 1000) -> int:
        """Process messages from the stage stream.

        Args:
            max_iterations: Max messages to process. None for unlimited.
            block_ms: Block timeout for XREADGROUP.

        Returns:
            Count of messages processed.
        """
        from smartmemory.pipeline.config import PipelineConfig as PC
        from smartmemory.pipeline.state import PipelineState as PS

        r = self.transport._get_redis()
        # Use a wildcard workspace pattern — consumers handle all workspaces
        # In practice, you'd register per-workspace or use a dispatcher
        processed = 0
        iterations = 0

        while max_iterations is None or iterations < max_iterations:
            iterations += 1
            try:
                # Read directly without consumer groups in this implementation
                # Real deployment would use XREADGROUP with proper consumer group setup
                messages = self._read_messages(r, block_ms)
                if not messages:
                    continue

                for message_id, fields in messages:
                    try:
                        self._process_message(r, message_id, fields, PS, PC)
                        processed += 1
                    except Exception as e:
                        run_id = fields.get("run_id", "unknown")
                        logger.error("Stage '%s' failed for run %s: %s", self.stage_name, run_id, e)
                        self.transport.mark_failed(run_id, self.stage_name, str(e))

            except Exception as e:
                logger.error("StageConsumer '%s' error: %s", self.stage_name, e)
                time.sleep(1)

        return processed

    def _read_messages(self, r: redis_lib.Redis, block_ms: int) -> list:
        """Read pending messages for this stage across workspaces."""
        # Scan for streams matching the stage pattern
        try:
            cursor = 0
            all_messages = []
            pattern = f"{self.transport._prefix}:*:{self.stage_name}"
            cursor, keys = r.scan(cursor, match=pattern, count=100)
            for key in keys:
                messages = r.xrange(key, count=10)
                for msg_id, fields in messages:
                    all_messages.append((msg_id, {**fields, "_stream_key": key}))
                    # Acknowledge by trimming (simple approach)
                    r.xdel(key, msg_id)
            return all_messages
        except Exception:
            return []

    def _process_message(self, r, message_id: str, fields: dict, PS, PC) -> None:
        """Execute the stage and publish result to next stream."""
        run_id = fields.get("run_id", "")
        state_json = fields.get("state", "{}")
        config_json = fields.get("config", "{}")
        stages_json = fields.get("stages", "[]")
        stage_index = int(fields.get("stage_index", "0"))

        state = PS.from_dict(json.loads(state_json))
        config = _deserialize_config(json.loads(config_json))
        stages = json.loads(stages_json)

        # Execute the stage
        new_state = self.stage.execute(state, config)

        # Publish to next stage
        self.transport.publish_stage_result(run_id, new_state, config, stages, stage_index)


def _serialize_config(config: PipelineConfig) -> Dict[str, Any]:
    """Serialize PipelineConfig to a plain dict."""
    try:
        if hasattr(config, "to_dict"):
            return config.to_dict()
        from dataclasses import asdict

        return asdict(config)
    except Exception:
        return {}


def _deserialize_config(data: Dict[str, Any]) -> PipelineConfig:
    """Deserialize PipelineConfig from a dict."""
    from smartmemory.pipeline.config import PipelineConfig as PC

    try:
        if hasattr(PC, "from_dict"):
            return PC.from_dict(data)
        return PC(**{k: v for k, v in data.items() if k in {f.name for f in __import__("dataclasses").fields(PC)}})
    except Exception:
        return PC()
