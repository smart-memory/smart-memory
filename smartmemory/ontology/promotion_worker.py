"""Background promotion worker — Redis Stream consumer for promotion candidates.

Follows the same pattern as enrichment/grounding workers.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smartmemory.graph.ontology_graph import OntologyGraph
    from smartmemory.observability.events import RedisStreamQueue
    from smartmemory.pipeline.config import PromotionConfig

logger = logging.getLogger(__name__)


class PromotionWorker:
    """Consume promotion candidates from Redis Stream and evaluate them.

    Usage::

        worker = PromotionWorker(ontology_graph, promotion_config)
        worker.run(max_iterations=100)  # or run() for indefinite
    """

    def __init__(self, ontology_graph: OntologyGraph, config: PromotionConfig, queue: RedisStreamQueue | None = None):
        self._ontology = ontology_graph
        self._config = config
        self._queue = queue

    def _get_queue(self) -> RedisStreamQueue:
        if self._queue is not None:
            return self._queue
        from smartmemory.observability.events import RedisStreamQueue

        self._queue = RedisStreamQueue.for_promote(group="promote-workers")
        self._queue.ensure_group()
        return self._queue

    def run(self, max_iterations: int | None = None) -> int:
        """Process promotion candidates. Returns count of processed messages.

        Args:
            max_iterations: Stop after this many iterations. None = run until queue is empty.
        """
        from smartmemory.ontology.promotion import PromotionCandidate, PromotionEvaluator

        queue = self._get_queue()
        evaluator = PromotionEvaluator(self._ontology, self._config)
        processed = 0
        iterations = 0

        while max_iterations is None or iterations < max_iterations:
            iterations += 1
            messages = queue.read_group(block_ms=500, count=10)
            if not messages:
                if max_iterations is None:
                    continue
                break

            for msg_id, fields in messages:
                try:
                    payload = json.loads(fields.get("payload", "{}"))
                    candidate = PromotionCandidate(
                        entity_name=payload["entity_name"],
                        entity_type=payload["entity_type"],
                        confidence=float(payload.get("confidence", 0.5)),
                        source_memory_id=payload.get("source_memory_id"),
                    )

                    result = evaluator.evaluate(candidate)
                    if result.promoted:
                        evaluator.promote(candidate)

                    queue.ack(msg_id)
                    processed += 1
                except Exception as e:
                    logger.warning("Failed to process promotion candidate %s: %s", msg_id, e)
                    try:
                        queue.move_to_dlq(msg_id, fields, reason=str(e))
                        queue.ack(msg_id)
                    except Exception:
                        pass

        return processed
